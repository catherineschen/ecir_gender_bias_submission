"""Build M/F/N attention-head steering vectors from the MS MARCO Fair
construction-query doc-pairs.

Vector-construction stage only (no alpha sweep, no evaluate_steering_aggr, no
plotting) -- companion to steering_vectors_embed_msmarco_fair.py but operating on
attention-head ("z" hook) activations at a fixed set of identified heads per model,
instead of full-embedding hook_full_embed activations.

The head-activation collection logic (run_with_cache on the "z" hook, per-doc mean
over swapped positions per head, concatenation across heads) and
split_direction_to_heads (unit-normalize each head's slice) are copied as-is from
the GREP-BiasIR reference script (submission_code_gender_bias_emnlp/steering_attn.py),
only adapted to construction_pairs.jsonl's field names and to group per query_id
instead of flattening across all doc-pairs. get_gendered_token_indices and the
per-query pooling logic (pooled_direction_per_query) are shared with the embedding
script via swap_pair_utils.py.

Per-query nested averaging: the concatenated (all-heads) diff vectors are grouped
by query_id as they're collected; pooled_direction_per_query computes each query's
own mean, then pools via an equal-weight mean of those per-query means. Only after
pooling is the pooled vector split into per-head slices and unit-normalized
(split_direction_to_heads) -- same per-head normalization as the reference script,
just applied to the pooled-not-flattened direction.

Steering level: heads only. The reference script's "resid" branch and its dead
projection/PCA branch are not ported here.

Two things about these vectors that matter at application time
(apply_steering_and_evaluate.py):

  1. Construction and application use different position scopes, exactly as in the
     reference. The direction is DERIVED from activations at gendered token positions
     (that's the only place an m/f/n swap changes anything), but it is APPLIED to all
     token positions of a steered document -- which is what lets it reach the CLS token
     on CLS-pooled models, where a final-layer z edit at non-CLS positions cannot.
     Which DOCUMENTS get steered is still gated on containing >=1 gendered term.
  2. The per-head slices in per_head_vectors.json are unit-normalized (that is what
     split_direction_to_heads does, matching the reference). The embedding-level vectors
     are raw, unnormalized mean-diffs with norm ~9-16. The two levels are therefore on
     different scales and need separate coefficient grids -- hence ALPHA_* vs BETA_* over
     in apply_steering_and_evaluate.py. vector.pt here holds the raw pooled (concatenated,
     un-normalized) direction if the raw per-head magnitudes are ever wanted instead.

Usage (from the repo root):
    conda run -n gender_bias python gender_bias_steering_utility/steering_vectors_attn_msmarco_fair.py
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import transformer_lens.utils as utils
from mechir import Dot
from tqdm import tqdm

from swap_pair_utils import get_best_device, get_gendered_token_indices, pooled_direction_per_query

# ---- paths ----
REPO_ROOT = Path(__file__).resolve().parent.parent
# SEED must match the SEED build_construction_pairs.py was run with. At the default (42)
# every path below is unchanged from before this was made overridable; other seeds read
# from and write to their own sibling _seed{SEED} directories. See run_steering_sweep.sh's
# "OTHER SEEDS" section.
SEED = int(os.environ.get("SEED", 42))
SEED_SUFFIX = "" if SEED == 42 else f"_seed{SEED}"
DATA_PATH = REPO_ROOT / f"gender_bias_steering_utility/preprocess/vector_construction{SEED_SUFFIX}/construction_pairs.jsonl"
SAVE_DIR = REPO_ROOT / f"gender_bias_steering_utility/experiments/vector_construction{SEED_SUFFIX}/attn"

DEVICE = get_best_device()

# MODEL CONFIGS -- final-layer heads only, originally identified by the EMNLP
# activation-patching work. Imported rather than re-inlined so the head lists can't drift
# from the copy build_construction_pairs.py reads.
from model_configs import MODEL_CONFIGS  # noqa: E402

DIRECTIONS = ["mf", "mn", "fn"]


def load_bi(model_name_or_path):
    cfg = MODEL_CONFIGS[model_name_or_path]
    return Dot(model_name_or_path, pooling_type=cfg["pool"], sim_func_type=cfg["sim_func"])


def load_construction_pairs(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ============================================================
# head-activation collection logic, copied as-is from
# submission_code_gender_bias_emnlp/steering_attn.py::collect_head_activation_differences,
# adapted to construction_pairs.jsonl's field names and grouped by query_id instead of
# flattened into one global list.
# ============================================================
def collect_per_doc_head_diffs(records, tl_model, tokenizer, heads):
    """Returns:
        per_query: {query_id: {"mf": [concat_diff_vec, ...], "mn": [...], "fn": [...]}}
        skipped_by_query: {query_id: count of doc-pairs skipped for tokenization-length mismatch}
        skipped_no_n: count of doc-pairs skipped because n_text was unavailable
    """
    per_query = defaultdict(lambda: {"mf": [], "mn": [], "fn": []})
    skipped_by_query = defaultdict(int)
    skipped_no_n = 0

    layers = sorted(set(layer for layer, _ in heads))
    names_filter = [utils.get_act_name("z", layer) for layer in layers]

    for rec in tqdm(records, desc="Collecting head activations"):
        query_id = rec["query_id"]
        doc_m = rec["m_text"]
        doc_f = rec["f_text"]
        doc_n = rec["n_text"]

        if doc_n is None:
            skipped_no_n += 1
            continue

        tok_m, tok_f, tok_n, idx_diff = get_gendered_token_indices(doc_m, doc_f, doc_n, tokenizer)

        if idx_diff is None or len(idx_diff) == 0:
            skipped_by_query[query_id] += 1
            continue

        with torch.no_grad():
            input_m = torch.tensor(tok_m).unsqueeze(0).to(DEVICE)
            input_f = torch.tensor(tok_f).unsqueeze(0).to(DEVICE)
            input_n = torch.tensor(tok_n).unsqueeze(0).to(DEVICE)

            _, cache_m = tl_model.run_with_cache(input_m, names_filter=lambda name: name in names_filter)
            _, cache_f = tl_model.run_with_cache(input_f, names_filter=lambda name: name in names_filter)
            _, cache_n = tl_model.run_with_cache(input_n, names_filter=lambda name: name in names_filter)

        head_acts_m = []
        head_acts_f = []
        head_acts_n = []

        for layer, head_idx in heads:
            hook_name = "_model." + utils.get_act_name("z", layer)
            act_m = cache_m[hook_name][0, idx_diff, head_idx, :].cpu().mean(dim=0)
            act_f = cache_f[hook_name][0, idx_diff, head_idx, :].cpu().mean(dim=0)
            act_n = cache_n[hook_name][0, idx_diff, head_idx, :].cpu().mean(dim=0)

            head_acts_m.append(act_m)
            head_acts_f.append(act_f)
            head_acts_n.append(act_n)

        emb_m_avg = torch.cat(head_acts_m)
        emb_f_avg = torch.cat(head_acts_f)
        emb_n_avg = torch.cat(head_acts_n)

        diff_mf = (emb_m_avg - emb_f_avg).numpy()
        diff_mn = (emb_m_avg - emb_n_avg).numpy()
        diff_fn = (emb_f_avg - emb_n_avg).numpy()

        per_query[query_id]["mf"].append(diff_mf)
        per_query[query_id]["mn"].append(diff_mn)
        per_query[query_id]["fn"].append(diff_fn)

    return per_query, skipped_by_query, skipped_no_n


# ============================================================
# split concatenated direction into per-head directions, copied as-is from
# submission_code_gender_bias_emnlp/steering_attn.py::split_direction_to_heads
# ============================================================
def split_direction_to_heads(direction, heads, d_head):
    """Split a concatenated direction vector back into per-head directions,
    unit-normalizing each head's slice."""
    directions = {}
    for i, (layer, head_idx) in enumerate(heads):
        head_dir = direction[i * d_head : (i + 1) * d_head]
        head_dir = head_dir / head_dir.norm()
        directions[(layer, head_idx)] = head_dir.to(DEVICE)
    return directions


# ============================================================
# per-query nested averaging (grouping already done by collect_per_doc_head_diffs;
# the per-query-mean-then-pool step is the shared pooled_direction_per_query)
# ============================================================
def compute_per_query_and_pooled(per_query, direction, query_ids):
    per_query_means = {}
    per_query_counts = {}
    for query_id in query_ids:
        vecs = per_query[query_id][direction]
        per_query_counts[query_id] = len(vecs)
        if vecs:
            per_query_means[query_id] = np.stack(vecs).mean(axis=0)

    if not per_query_means:
        return None, per_query_means, per_query_counts

    diffs_by_query = {qid: np.stack(per_query[qid][direction]) for qid in query_ids if per_query_counts[qid] > 0}
    pooled = pooled_direction_per_query(diffs_by_query)
    return pooled, per_query_means, per_query_counts


def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Save directory: {SAVE_DIR}")

    records = load_construction_pairs(DATA_PATH)
    print(f"{len(records)} doc-pair records loaded from {DATA_PATH}")

    query_ids = sorted({r["query_id"] for r in records})
    print(f"{len(query_ids)} distinct queries: {query_ids}")

    all_summary_rows = {}  # model_name -> list of (direction, query_id, count, norm, cosine)

    for model_name, model_cfg in MODEL_CONFIGS.items():
        print(f"\n{'#' * 60}\nMODEL: {model_name}\n{'#' * 60}")
        model = load_bi(model_name)
        tl_model = model._model
        tokenizer = model.tokenizer
        heads = model_cfg["heads"]
        d_head = tl_model.cfg.d_head

        print(f"Identified heads: {heads}  last_layer: {model_cfg['last_layer']}  d_head: {d_head}")

        short_name = model_name.split("/")[-1]
        model_save_dir = SAVE_DIR / short_name

        per_query, skipped_by_query, skipped_no_n = collect_per_doc_head_diffs(records, tl_model, tokenizer, heads)
        total_skipped_length_mismatch = sum(skipped_by_query.values())
        print(
            f"skipped {total_skipped_length_mismatch} doc-pairs for tokenization-length "
            f"mismatch across m/f/n (expect 0); {skipped_no_n} doc-pairs skipped for missing n_text"
        )

        model_summary_rows = []
        for direction in DIRECTIONS:
            pooled, per_query_means, per_query_counts = compute_per_query_and_pooled(per_query, direction, query_ids)
            out_dir = model_save_dir / direction
            out_dir.mkdir(parents=True, exist_ok=True)

            pooled_tensor = torch.tensor(pooled, dtype=torch.float32)
            torch.save(pooled_tensor, out_dir / "vector.pt")

            per_head_dirs = split_direction_to_heads(pooled_tensor.to(DEVICE), heads, d_head)
            with open(out_dir / "per_head_vectors.json", "w", encoding="utf-8") as f:
                json.dump(
                    {f"L{layer}H{head_idx}": vec.cpu().tolist() for (layer, head_idx), vec in per_head_dirs.items()},
                    f,
                )

            with open(out_dir / "per_query_vectors.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        qid: {
                            "mean_diff_vector": per_query_means[qid].tolist() if qid in per_query_means else None,
                            "doc_pair_count": per_query_counts.get(qid, 0),
                        }
                        for qid in query_ids
                    },
                    f,
                )

            per_query_diag = {}
            for qid in query_ids:
                vec = per_query_means.get(qid)
                per_query_diag[qid] = {
                    "vector_norm": float(np.linalg.norm(vec)) if vec is not None else None,
                    "cosine_to_pooled": cosine_sim(vec, pooled) if vec is not None else None,
                    "doc_pair_count": per_query_counts.get(qid, 0),
                    "skipped_length_mismatch": skipped_by_query.get(qid, 0),
                }
                model_summary_rows.append((
                    direction, qid, per_query_counts.get(qid, 0),
                    per_query_diag[qid]["vector_norm"], per_query_diag[qid]["cosine_to_pooled"],
                ))

            diagnostics = {
                "pooled_vector_norm": float(np.linalg.norm(pooled)),
                "total_skipped_length_mismatch": total_skipped_length_mismatch,
                "total_skipped_no_n_text": skipped_no_n,
                "per_query": per_query_diag,
            }
            with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as f:
                json.dump(diagnostics, f, indent=2)

        all_summary_rows[model_name] = model_summary_rows

        del model, tl_model, tokenizer
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()

    # ---- final summary table: 5 queries x 3 directions x 5 models ----
    print("\n" + "=" * 100)
    print("SUMMARY (rows = 5 queries x 3 directions x 5 models)")
    print("=" * 100)
    for model_name, rows in all_summary_rows.items():
        print(f"\n{model_name}")
        print(f"  {'dir':<4} {'query':<10} {'#pairs':>7} {'norm':>10} {'cos_to_pooled':>14}")
        for direction, qid, count, norm, cos in rows:
            norm_str = f"{norm:>10.4f}" if norm is not None else f"{'n/a':>10}"
            cos_str = f"{cos:>14.4f}" if cos is not None else f"{'n/a':>14}"
            print(f"  {direction:<4} {qid:<10} {count:>7} {norm_str} {cos_str}")

    # ---- flag whether query 59199 shows the same low-cosine outlier pattern seen at the
    # embedding level (steering_vectors_embed_msmarco_fair.py) ----
    print("\n" + "=" * 100)
    print("query 59199 outlier check (vs. other 4 queries, per model/direction)")
    print("=" * 100)
    for model_name, rows in all_summary_rows.items():
        by_direction = defaultdict(dict)
        for direction, qid, count, norm, cos in rows:
            by_direction[direction][qid] = cos
        for direction, cos_by_qid in by_direction.items():
            if cos_by_qid.get("59199") is None:
                continue
            is_lowest = cos_by_qid["59199"] == min(v for v in cos_by_qid.values() if v is not None)
            others = [c for qid, c in cos_by_qid.items() if qid != "59199" and c is not None]
            flag = "  <-- LOWEST cosine of the 5 queries (matches embedding-level pattern)" if is_lowest else ""
            print(
                f"  {model_name:<55} {direction:<4} 59199 cos={cos_by_qid['59199']:.4f} "
                f"(others mean={np.mean(others):.4f}){flag}"
            )

    print(f"\nAll outputs saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()

"""Build M/F/N steering vectors from the MS MARCO Fair construction-query doc-pairs.

This is the vector-construction stage only (no alpha sweep, no evaluation, no
plotting) -- it takes construction_pairs.jsonl (built by build_construction_pairs.py)
and produces, per model and per direction (mf/mn/fn), a single pooled steering
vector plus the per-query means and diagnostics that went into it.

The per-doc embedding-collection logic (hook_full_embed extraction + per-doc mean
over swapped token positions) is copied as-is from the GREP-BiasIR reference script
(submission_code_gender_bias_emnlp/steering_embed.py::collect_embedding_differences),
only adapted to construction_pairs.jsonl's field names and to group per query_id
instead of flattening across all doc-pairs. The token-diff logic
(get_gendered_token_indices) and the per-query pooling logic
(pooled_direction_per_query) live in swap_pair_utils.py, shared with the
attention-level script (steering_vectors_attn_msmarco_fair.py).

Per-query nested averaging (the one thing this script does differently from the
reference): each of the 5 construction queries gets its own mean diff vector
(mean over that query's doc-pairs), and the final pooled steering vector is the
mean of those 5 per-query means -- equal weight per query, regardless of how many
doc-pairs a query contributed. Per-doc-first averaging over swapped token
positions is unchanged from the reference.

Usage (from the repo root):
    conda run -n gender_bias python gender_bias_steering_utility/steering_vectors_embed_msmarco_fair.py
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
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
# "/embed" mirrors the attn script's "/attn" suffix below -- apply_steering_and_evaluate.py's
# EMBED_VECTOR_DIR expects it (seed 42's existing vectors already live there, from an earlier
# on-disk reorg that this constant had fallen out of sync with).
SAVE_DIR = REPO_ROOT / f"gender_bias_steering_utility/experiments/vector_construction{SEED_SUFFIX}/embed"

DEVICE = get_best_device()

MODEL_NAMES = {
    # note: TAS-B here is the mechir/transformer_lens-compatible checkpoint, not the
    # sentence-transformers one used in the dense baseline table -- known, accepted mismatch.
    "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco": {"pool": "cls", "sim_func": "dot"},
    "sentence-transformers/multi-qa-distilbert-dot-v1": {"pool": "cls", "sim_func": "dot"},
    "sentence-transformers/multi-qa-MiniLM-L6-dot-v1": {"pool": "cls", "sim_func": "dot"},
    "sentence-transformers/msmarco-bert-base-dot-v5": {"pool": "mean", "sim_func": "dot"},
    "sentence-transformers/msmarco-distilbert-dot-v5": {"pool": "mean", "sim_func": "dot"},
}
DIRECTIONS = ["mf", "mn", "fn"]


def load_bi(model_name_or_path):
    pool = MODEL_NAMES[model_name_or_path]["pool"]
    sim_func = MODEL_NAMES[model_name_or_path]["sim_func"]
    return Dot(model_name_or_path, pooling_type=pool, sim_func_type=sim_func)


def load_construction_pairs(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def collect_per_doc_diffs(records, tl_model, tokenizer):
    """Per-doc embedding-collection logic (hook_full_embed extraction + per-doc mean
    over swapped positions), copied as-is from steering_embed.collect_embedding_differences,
    adapted to construction_pairs.jsonl's field names and grouped by query_id instead of
    flattened into one global list.

    Returns:
        per_query: {query_id: {"mf": [diff_vec, ...], "mn": [...], "fn": [...]}}
        skipped_by_query: {query_id: count of doc-pairs skipped for tokenization-length mismatch}
        skipped_no_n: count of doc-pairs skipped because n_text was unavailable
    """
    per_query = defaultdict(lambda: {"mf": [], "mn": [], "fn": []})
    skipped_by_query = defaultdict(int)
    skipped_no_n = 0

    for rec in tqdm(records, desc="Collecting embeddings"):
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

            _, cache_m = tl_model.run_with_cache(input_m, names_filter="hook_full_embed")
            _, cache_f = tl_model.run_with_cache(input_f, names_filter="hook_full_embed")
            _, cache_n = tl_model.run_with_cache(input_n, names_filter="hook_full_embed")

            emb_m = cache_m["_model.hook_full_embed"][0, idx_diff, :].cpu()
            emb_f = cache_f["_model.hook_full_embed"][0, idx_diff, :].cpu()
            emb_n = cache_n["_model.hook_full_embed"][0, idx_diff, :].cpu()

        emb_m_avg = emb_m.mean(dim=0)
        emb_f_avg = emb_f.mean(dim=0)
        emb_n_avg = emb_n.mean(dim=0)

        diff_mf = (emb_m_avg - emb_f_avg).numpy()
        diff_mn = (emb_m_avg - emb_n_avg).numpy()
        diff_fn = (emb_f_avg - emb_n_avg).numpy()

        per_query[query_id]["mf"].append(diff_mf)
        per_query[query_id]["mn"].append(diff_mn)
        per_query[query_id]["fn"].append(diff_fn)

    return per_query, skipped_by_query, skipped_no_n


# ============================================================
# per-query nested averaging (this is the part that differs from the reference,
# which flattens all doc-pairs into one list and takes a single flat mean).
# Grouping (step 1) happens in collect_per_doc_diffs above; the per-query-mean-then-
# pool step (steps 2-3) is the shared pooled_direction_per_query from swap_pair_utils.
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
    print(f"schema (keys of one record): {sorted(records[0].keys())}")

    query_ids = sorted({r["query_id"] for r in records})
    print(f"{len(query_ids)} distinct queries: {query_ids}")

    all_summary_rows = {}  # model_name -> list of (direction, query_id, count, norm, cosine)

    for model_name in MODEL_NAMES:
        print(f"\n{'#' * 60}\nMODEL: {model_name}\n{'#' * 60}")
        model = load_bi(model_name)
        tl_model = model._model
        tokenizer = model.tokenizer

        short_name = model_name.split("/")[-1]
        model_save_dir = SAVE_DIR / short_name

        per_query, skipped_by_query, skipped_no_n = collect_per_doc_diffs(records, tl_model, tokenizer)
        total_skipped_length_mismatch = sum(skipped_by_query.values())
        print(
            f"skipped {total_skipped_length_mismatch} doc-pairs for tokenization-length "
            f"mismatch across m/f/n (expected 0, per earlier length validation); "
            f"{skipped_no_n} doc-pairs skipped for missing n_text"
        )

        model_summary_rows = []
        for direction in DIRECTIONS:
            pooled, per_query_means, per_query_counts = compute_per_query_and_pooled(per_query, direction, query_ids)
            out_dir = model_save_dir / direction
            out_dir.mkdir(parents=True, exist_ok=True)

            torch.save(torch.tensor(pooled, dtype=torch.float32), out_dir / "vector.pt")

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

    # ---- final summary tables ----
    print("\n" + "=" * 100)
    print("SUMMARY (rows = 5 queries x 3 directions, per model)")
    print("=" * 100)
    for model_name, rows in all_summary_rows.items():
        print(f"\n{model_name}")
        print(f"  {'dir':<4} {'query':<10} {'#pairs':>7} {'norm':>10} {'cos_to_pooled':>14}")
        for direction, qid, count, norm, cos in rows:
            norm_str = f"{norm:>10.4f}" if norm is not None else f"{'n/a':>10}"
            cos_str = f"{cos:>14.4f}" if cos is not None else f"{'n/a':>14}"
            print(f"  {direction:<4} {qid:<10} {count:>7} {norm_str} {cos_str}")

    print(f"\nAll outputs saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()

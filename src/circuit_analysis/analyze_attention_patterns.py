import argparse
import os
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ablation_experiments import MODEL_NAMES, load_bi


CATEGORIES = ["cls", "sep", "gendered", "matching", "non_matching"]
CATEGORY_COLORS = {
    "cls": "#4C72B0",
    "sep": "#55A868",
    "gendered": "#C44E52",
    "matching": "#8172B2",
    "non_matching": "#E0E0E0",
}

DOC_GROUP_IDS_NEUTRAL_OVERLAP = [
    12, 30, 31, 34, 35, 38, 39, 50, 51, 62, 63, 68, 69,
    184, 185, 188, 189, 192, 193, 194, 195,
]
SUBSET_FOLDERS = {
    "overlap": ["neutral_lexical_overlap"],
    "non_overlap": ["non_lexical_overlap"],
    "all": ["neutral_lexical_overlap", "non_lexical_overlap"],
}


def load_tokens_by_row(path: str) -> Dict[int, List[str]]:
    out = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            row_idx = int(parts[0])
            tokens = parts[1:]
            out[row_idx] = tokens
    return out


def find_special_positions(tokens: List[str], cls_token: str, sep_token: str):
    cls_pos = 0 if tokens[0] == cls_token else None
    sep_pos = None
    for i, t in enumerate(tokens):
        if i > 0 and t == sep_token:
            sep_pos = i
            break
    return cls_pos, sep_pos


def build_category_mask(
    tokens: List[str],
    paired_tokens: List[str],
    query_token_ids: Set[int],
    doc_token_ids: List[int],
    cls_token: str,
    sep_token: str,
    stopword_ids: Optional[Set[int]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    seq_len = len(tokens)
    cls_pos, sep_pos = find_special_positions(tokens, cls_token, sep_token)
    paired_cls_pos, paired_sep_pos = find_special_positions(paired_tokens, cls_token, sep_token)

    if sep_pos is None or paired_sep_pos is None or sep_pos != paired_sep_pos:
        return None

    a = np.array(tokens[1:sep_pos])
    b = np.array(paired_tokens[1:sep_pos])
    diff_offsets = np.where(a != b)[0]
    gendered_positions = set((diff_offsets + 1).tolist())

    matching_positions = set()
    for i in range(1, sep_pos):
        tok_id = doc_token_ids[i]
        if tok_id in query_token_ids:
            if stopword_ids is not None and tok_id in stopword_ids:
                continue
            matching_positions.add(i)

    masks = {cat: np.zeros(seq_len, dtype=bool) for cat in CATEGORIES}
    if cls_pos is not None:
        masks["cls"][cls_pos] = True
    masks["sep"][sep_pos] = True
    for i in range(1, sep_pos):
        if i in gendered_positions:
            masks["gendered"][i] = True
        elif i in matching_positions:
            masks["matching"][i] = True
        else:
            masks["non_matching"][i] = True
    return masks


def get_source_mask(
    tokens: List[str],
    masks: Dict[str, np.ndarray],
    pad_token: str,
    pooling: str,
    source_type: str,
) -> Optional[np.ndarray]:
    """
    source_type:
      - "pooled": CLS for CLS-pooled, all non-padding for mean-pooled
      - "gendered": gendered token positions
      - "matching": query-matching token positions
    Returns None if no valid source positions for this row.
    """
    seq_len = len(tokens)
    if source_type == "pooled":
        mask = np.zeros(seq_len, dtype=bool)
        if pooling == "cls":
            mask[0] = True
        elif pooling == "mean":
            for i, t in enumerate(tokens):
                if t != pad_token:
                    mask[i] = True
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
    elif source_type == "gendered":
        mask = masks["gendered"].copy()
    elif source_type == "matching":
        mask = masks["matching"].copy()
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    if not mask.any():
        return None
    return mask


def filter_meta_by_subset(meta: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "all":
        return meta
    elif subset == "overlap":
        return meta[meta["doc_group_id"].isin(DOC_GROUP_IDS_NEUTRAL_OVERLAP)]
    elif subset == "non_overlap":
        return meta[~meta["doc_group_id"].isin(DOC_GROUP_IDS_NEUTRAL_OVERLAP)]
    else:
        raise ValueError(f"Unknown subset: {subset}")


def analyze_model(
    model_name: str,
    dump_dir: str,
    csv_path: str,
    subset: str,
    source_type: str,
    stopword_ids: Optional[Set[int]] = None,
):
    pooling = MODEL_NAMES[model_name]["pool"]
    heads = MODEL_NAMES[model_name]["heads_to_ablate"]
    safe_name = model_name.replace("/", "_")

    model, _ = load_bi(model_name)
    tokenizer = model.tokenizer
    cls_token, sep_token, pad_token = tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token

    src_df = pd.read_csv(csv_path)[["doc_group_id", "q_id", "relevant"]].drop_duplicates()

    all_rows_out = []
    for subfolder in SUBSET_FOLDERS[subset]:
        model_dir = os.path.join(dump_dir, subfolder, safe_name)
        if not os.path.isdir(model_dir):
            print(f"  skipping {subfolder}: {model_dir} not found")
            continue
        if not os.path.isdir(model_dir):
            print(f"  skipping {subfolder}: {model_dir} not found")
            continue

        attn_dir = os.path.join(model_dir, "attn")
        labels_dir = os.path.join(model_dir, "labels")

        meta = pd.read_parquet(os.path.join(model_dir, "meta.parquet"))
        meta = meta.merge(src_df, on=["doc_group_id", "q_id"], how="left")
        if meta["relevant"].isna().any():
            n_missing = int(meta["relevant"].isna().sum())
            print(f"  warning: {n_missing} rows missing `relevant` after CSV join in {subfolder}")

        doc_tokens_by_row = load_tokens_by_row(os.path.join(labels_dir, "doc_tokens_by_row.txt"))
        query_tokens_by_row = load_tokens_by_row(os.path.join(labels_dir, "query_tokens_by_row.txt"))

        # Pairing uses meta from this subfolder (since the paired variant lives in the same dump).
        meta_by_group = meta.groupby("doc_group_id")

        row_masks: Dict[int, Optional[Dict[str, np.ndarray]]] = {}
        row_source_masks: Dict[int, np.ndarray] = {}
        skipped_groups = []

        for row_idx in tqdm(meta["row_idx"].tolist(), desc=f"{safe_name}/{subfolder}: masks"):
            row_meta = meta.loc[meta["row_idx"] == row_idx].iloc[0]
            doc_group_id = int(row_meta["doc_group_id"])
            variant = row_meta["variant"]

            try:
                group_rows = meta_by_group.get_group(doc_group_id)
            except KeyError:
                row_masks[row_idx] = None
                skipped_groups.append((doc_group_id, "no group in meta"))
                continue

            paired = group_rows[group_rows["variant"] != variant]
            if len(paired) == 0:
                row_masks[row_idx] = None
                skipped_groups.append((doc_group_id, "no paired variant"))
                continue
            paired_row_idx = int(paired.iloc[0]["row_idx"])

            if row_idx not in doc_tokens_by_row:
                print(f"row_idx {row_idx} missing. token file has {len(doc_tokens_by_row)} entries")
                print(f"sample keys: {sorted(doc_tokens_by_row.keys())[:10]}")
                print(f"meta size: {len(meta)}, row_idx range in meta: {meta['row_idx'].min()}–{meta['row_idx'].max()}")
                raise KeyError(row_idx)

            tokens = doc_tokens_by_row[row_idx]
            paired_tokens = doc_tokens_by_row[paired_row_idx]

            q_tokens = query_tokens_by_row[row_idx]
            q_token_ids = tokenizer.convert_tokens_to_ids(
                [t for t in q_tokens if t not in (cls_token, sep_token, pad_token)]
            )
            query_token_ids = set(q_token_ids)
            doc_token_ids = tokenizer.convert_tokens_to_ids(tokens)

            masks = build_category_mask(
                tokens, paired_tokens, query_token_ids, doc_token_ids,
                cls_token, sep_token, stopword_ids,
            )
            if masks is None:
                row_masks[row_idx] = None
                skipped_groups.append((doc_group_id, "content length mismatch"))
                continue

            source_mask = get_source_mask(tokens, masks, pad_token, pooling, source_type)
            if source_mask is None:
                row_masks[row_idx] = None
                skipped_groups.append((doc_group_id, f"no source positions for source_type={source_type}"))
                continue

            row_masks[row_idx] = masks
            row_source_masks[row_idx] = source_mask

        if skipped_groups:
            skip_path = os.path.join(model_dir, f"skipped_groups.txt")
            with open(skip_path, "w") as f:
                for gid, reason in skipped_groups:
                    f.write(f"{gid}\t{reason}\n")
            print(f"  skipped {len(skipped_groups)} entries in {subfolder} — see {skip_path}")

        for layer, head in tqdm(heads, desc=f"{safe_name}/{subfolder}: heads"):
            attn = np.load(os.path.join(attn_dir, f"d_attn_L{layer}_H{head}.npy"))

            for _, row_meta in meta.iterrows():
                row_idx = int(row_meta["row_idx"])
                masks = row_masks.get(row_idx)
                if masks is None:
                    continue
                source_mask = row_source_masks[row_idx]

                pattern = attn[row_idx]
                target_dist = pattern[source_mask].mean(axis=0)

                for cat in CATEGORIES:
                    n_tokens = int(masks[cat].sum())
                    bucket_total = float(target_dist[masks[cat]].sum())
                    bucket_per_token = bucket_total / n_tokens if n_tokens > 0 else 0.0

                    all_rows_out.append({
                        "model": model_name,
                        "subfolder": subfolder,
                        "layer": int(layer),
                        "head": int(head),
                        "row_idx": row_idx,
                        "doc_group_id": int(row_meta["doc_group_id"]),
                        "q_id": int(row_meta["q_id"]),
                        "variant": str(row_meta["variant"]),
                        "relevant": int(row_meta["relevant"]) if not pd.isna(row_meta["relevant"]) else -1,
                        "category": cat,
                        "attention": bucket_total,
                        "attention_per_token": bucket_per_token,
                        "n_tokens": n_tokens,
                    })

    out_df = pd.DataFrame(all_rows_out)
    # Output CSV at dump_dir level, since `all` spans both subfolders
    os.makedirs(dump_dir, exist_ok=True)
    # out_path = os.path.join(dump_dir, f"attention_by_category_{safe_name}_{subset}.csv")
    out_path = os.path.join(dump_dir, f"attention_by_category_{safe_name}_{subset}_src_{source_type}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  saved: {out_path} ({len(out_df)} rows)")


def plot_attention_by_category(
    csv_path: str,
    out_dir: str,
    subset: str,
    source_type: str,
    stratify_by_variant: bool = False,
):
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"  no rows in {csv_path}, skipping plot")
        return

    model = df["model"].iloc[0]
    safe_model = model.replace("/", "_")

    model_out_dir = os.path.join(out_dir, subset, source_type, safe_model)
    os.makedirs(model_out_dir, exist_ok=True)

    group_cols = ["model", "layer", "head", "relevant", "category"]
    if stratify_by_variant:
        group_cols.append("variant")

    agg_total = df.groupby(group_cols, as_index=False)["attention"].mean()
    agg_per_tok = df.groupby(group_cols, as_index=False)["attention_per_token"].mean()

    relevance_labels = {1: "relevant", 0: "non_relevant"}
    suffix = "_by_variant" if stratify_by_variant else ""

    for rel_val, rel_label in relevance_labels.items():
        sub_total = agg_total[agg_total["relevant"] == rel_val]
        sub_per_tok = agg_per_tok[agg_per_tok["relevant"] == rel_val]
        if sub_total.empty:
            continue

        if stratify_by_variant:
            variants = sorted(sub_total["variant"].unique())
            n_heads = sub_total[["layer", "head"]].drop_duplicates().shape[0]
            fig, axes = plt.subplots(
                2, len(variants),
                figsize=(max(5, 1.2 * n_heads) * len(variants), 9),
                sharey="row",
            )
            if len(variants) == 1:
                axes = axes[:, np.newaxis]
            for col_idx, variant in enumerate(variants):
                _plot_stacked_total(
                    sub_total[sub_total["variant"] == variant],
                    axes[0, col_idx],
                    title=f"variant={variant}",
                )
                _plot_per_token(
                    sub_per_tok[sub_per_tok["variant"] == variant],
                    axes[1, col_idx],
                    title=None,
                )
        else:
            n_heads = sub_total[["layer", "head"]].drop_duplicates().shape[0]
            fig, axes = plt.subplots(
                2, 1,
                figsize=(max(5, 1.2 * n_heads), 9),
                sharex=True,
            )
            _plot_stacked_total(sub_total, axes[0], title="attention proportion (bucket totals)")
            _plot_per_token(sub_per_tok, axes[1], title="attention per token (concentration)")

        fig.suptitle(f"{model}\n{rel_label}", fontsize=10)
        fig.tight_layout()
        out_path = os.path.join(model_out_dir, f"{rel_label}{suffix}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  saved plot: {out_path}")


def _plot_stacked_total(sub_df: pd.DataFrame, ax, title: Optional[str]):
    """Stacked bars of bucket totals — sums to 1.0 per head."""
    head_keys = sorted(sub_df[["layer", "head"]].drop_duplicates().itertuples(index=False))
    head_labels = [f"L{l}H{h}" for l, h in head_keys]

    bottom = np.zeros(len(head_keys))
    for cat in CATEGORIES:
        vals = []
        for layer, head in head_keys:
            row = sub_df[(sub_df["layer"] == layer) & (sub_df["head"] == head) & (sub_df["category"] == cat)]
            vals.append(row["attention"].iloc[0] if len(row) else 0.0)
        ax.bar(head_labels, vals, bottom=bottom, label=cat, color=CATEGORY_COLORS[cat])
        bottom += np.array(vals)

    ax.set_ylabel("attention proportion")
    ax.set_ylim(0, 1.0)
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", fontsize=7, ncol=1)
    ax.tick_params(axis="x", rotation=30)


def _plot_per_token(sub_df: pd.DataFrame, ax, title: Optional[str]):
    """Grouped bars of per-token attention — comparable across categories of different sizes."""
    head_keys = sorted(sub_df[["layer", "head"]].drop_duplicates().itertuples(index=False))
    head_labels = [f"L{l}H{h}" for l, h in head_keys]

    n_heads = len(head_keys)
    n_cats = len(CATEGORIES)
    bar_width = 0.8 / n_cats
    x = np.arange(n_heads)

    for ci, cat in enumerate(CATEGORIES):
        vals = []
        for layer, head in head_keys:
            row = sub_df[(sub_df["layer"] == layer) & (sub_df["head"] == head) & (sub_df["category"] == cat)]
            vals.append(row["attention_per_token"].iloc[0] if len(row) else 0.0)
        offset = (ci - (n_cats - 1) / 2) * bar_width
        ax.bar(x + offset, vals, bar_width, label=cat, color=CATEGORY_COLORS[cat])

    ax.set_xticks(x)
    ax.set_xticklabels(head_labels, rotation=30)
    ax.set_ylabel("attention per token")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", fontsize=7, ncol=1)


def _plot_single(sub_df: pd.DataFrame, ax, title: str):
    """Stacked bar for a single (model, relevance, [variant]) panel."""
    # Order heads by (layer, head).
    head_keys = sorted(sub_df[["layer", "head"]].drop_duplicates().itertuples(index=False))
    head_labels = [f"L{l}H{h}" for l, h in head_keys]

    bottom = np.zeros(len(head_keys))
    for cat in CATEGORIES:
        vals = []
        for layer, head in head_keys:
            row = sub_df[(sub_df["layer"] == layer) & (sub_df["head"] == head) & (sub_df["category"] == cat)]
            vals.append(row["attention"].iloc[0] if len(row) else 0.0)
        ax.bar(head_labels, vals, bottom=bottom, label=cat, color=CATEGORY_COLORS[cat])
        bottom += np.array(vals)

    ax.set_ylabel("attention proportion")
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", fontsize=7, ncol=1)
    ax.tick_params(axis="x", rotation=30)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze")
    a.add_argument("--dump_dir", "-dd", required=True)
    a.add_argument("--csv_path", default="data/grep_bias_ir_mechir_format.csv")
    a.add_argument("--subset", choices=["all", "overlap", "non_overlap"], default="all")
    a.add_argument(
        "--source_type", choices=["pooled", "gendered", "matching"], default="pooled",
        help="Which positions to use as the attention source.",
    )

    p = sub.add_parser("plot")
    p.add_argument("--attn_results_dir", "-ad", required=True)
    p.add_argument("--out_dir", "-od", required=True)
    p.add_argument("--subset", choices=["all", "overlap", "non_overlap"], default="all")
    p.add_argument("--source_type", choices=["pooled", "gendered", "matching"], default="pooled")
    p.add_argument("--stratify_by_variant", action="store_true")

    args = parser.parse_args()

    if args.cmd == "analyze":
        for model_name in MODEL_NAMES:
            analyze_model(model_name, args.dump_dir, args.csv_path, args.subset, args.source_type)
    elif args.cmd == "plot":
        for model_name in MODEL_NAMES:
            safe_model = model_name.replace("/", "_")
            csv_path = os.path.join(
                args.attn_results_dir,
                f"attention_by_category_{safe_model}_{args.subset}_src_{args.source_type}.csv",
            )
            if not os.path.exists(csv_path):
                print(f"  skipping {model_name}: {csv_path} not found")
                continue
            plot_attention_by_category(
                csv_path, args.out_dir, args.subset, args.source_type, args.stratify_by_variant
            )

if __name__ == "__main__":
    main()

# python analyze_attention_patterns.py analyze -dd results_mechir_attn_pattern --subset all
# python analyze_attention_patterns.py plot -ad results_mechir_attn_pattern -od result_figures_attn --subset all
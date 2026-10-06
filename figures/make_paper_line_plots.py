"""
Generate a per-model-pair line-plot figure for a paper, showing steering
effects on gender preference as a function of steering strength.

Layout
------
Rows:
  One row per model in the pair (2 rows for a pair of 2 models).

Columns:
  0 — Male-Female  (MvF)
  1 — Male-Neutral  (MvN)
  2 — Female-Neutral  (FvN)

Each subplot shows two lines for a single model/direction:
  - Embedding-level steering (ROW_CONFIGS[0]), solid line
  - Attention-level steering (ROW_CONFIGS[1]), dashed line
plotted against every alpha value actually present in that line's results
JSON (no fixed/hardcoded alpha grid). The y-axis is dual: the left axis
shows % of samples preferring direction["group_labels"][0], and the right
axis is a reversed mirror (0-100, inverted) labelled with
direction["group_labels"][1], since the two percentages are complements.

Assumption (unverified against the actual data): alpha keys in
steering_results.json are strings that parse directly as floats (e.g.
"0.0", "-5.0"), matching the lookup behavior already used by
compute_split() in make_paper_stacked_bars.py. get_available_alphas()
below assumes any JSON key that fails float() conversion is not an alpha
key and skips it.
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Reused as-is from make_paper_stacked_bars.py
MODELS = [
    "msmarco-distilbert-dot-v5",
    "distilbert-dot-tas_b-b256-msmarco",
    "multi-qa-distilbert-dot-v1",
    "multi-qa-MiniLM-L6-dot-v1",
    "msmarco-bert-base-dot-v5",
]

ROW_CONFIGS = [
    {
        "label": "Embedding",
        "path": "results/pointwise_steering/results_steering_embed_with_scores",
        "level": "",
        "no_steering_alpha": 0.0,
    },
    {
        "label": "Late-layer attention",
        "path": "results/pointwise_steering/results_steering_aggregation_with_scores",
        "level": "heads",
        "no_steering_alpha": 0.0,
    },
]

DIRECTIONS = [
    {"key": "mf", "label": "Male–Female",    "comp": "MvF", "group_labels": ("Male",   "Female"),  "colors": ("#8da0cb", "#fc8d62")},
    {"key": "mn", "label": "Male–Neutral",   "comp": "MvN", "group_labels": ("Male",   "Neutral"), "colors": ("#8da0cb", "#66c2a5")},
    {"key": "fn", "label": "Female–Neutral", "comp": "FvN", "group_labels": ("Female", "Neutral"), "colors": ("#fc8d62", "#66c2a5")},
]

# Pairs of models to render together, one figure per pair.
MODEL_PAIRS = [
    ["multi-qa-distilbert-dot-v1", "msmarco-distilbert-dot-v5"],
]

# Paper-style display names for row labels (filenames still use the full
# model identifiers above, to stay unique and filesystem-safe). The part
# after "_" is rendered as a subscript via render_model_name() below.
NAME_MAP = {
    "distilbert-dot-tas_b-b256-msmarco": "DistilBERT-TAS-B_ms-marco",
    "msmarco-distilbert-dot-v5": "DistilBERT_ms-marco",
    "multi-qa-distilbert-dot-v1": "DistilBERT_multi-qa",
    "multi-qa-MiniLM-L6-dot-v1": "MiniLM_multi-qa",
    "msmarco-bert-base-dot-v5": "BERT_ms-marco",
}


def render_model_name(model: str) -> str:
    """Render a NAME_MAP entry's "Base_subscript" form as mathtext, e.g.
    "DistilBERT_ms-marco" -> "DistilBERT$_{\\mathrm{ms‐marco}}$".

    The subscript is wrapped in \\mathrm so it isn't italicized, and any
    hyphen in it is swapped for a plain Unicode hyphen (U+2010) instead of
    a bare "-" — mathtext's math grammar treats a literal "-" as a binary
    minus operator and pads it with extra spacing, which renders as an
    oversized dash even inside \\mathrm.
    """
    name = NAME_MAP.get(model, model)
    if "_" not in name:
        return name
    base, sub = name.split("_", 1)
    sub = sub.replace("-", "‐")
    sub_math = "\\mathrm{" + sub + "}"
    return f"{base}$_{{{sub_math}}}$"

# One color per ROW_CONFIGS level, applied consistently across all subplots.
# Assumption: arbitrary but colorblind-distinguishable pair, independent of
# the per-group colors used in the original stacked-bar script.
LEVEL_STYLES = {
    "Embedding": {"color": "#3699d6", "linestyle": "-"},
    "Late-layer attention": {"color": "#a2263b", "linestyle": "-"},
}


def load_results(row_config, model, direction_key):
    level = row_config["level"]
    if level:
        path = os.path.join(row_config["path"], model, level, direction_key, "actadd", "steering_results.json")
    else:
        path = os.path.join(row_config["path"], model, direction_key, "actadd", "steering_results.json")
    if not os.path.exists(path):
        print(f"[warn] missing: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_split(results, comp, alpha):
    alpha_key = str(alpha)
    if alpha_key not in results:
        alpha_key = next((k for k in results if abs(float(k) - alpha) < 1e-9), None)
    if alpha_key is None or comp not in results[alpha_key]:
        return None
    diffs = np.array(results[alpha_key][comp])
    if diffs.size == 0:
        return None
    n_pos = int(np.sum(diffs > 0))
    n_neg = int(np.sum(diffs < 0))
    n_tie = int(np.sum(diffs == 0))
    total = diffs.size
    return (n_pos + 0.5 * n_tie) / total * 100.0, (n_neg + 0.5 * n_tie) / total * 100.0


def get_available_alphas(results) -> List[float]:
    """Return every alpha value present as a key in `results`, sorted numerically."""
    alphas = []
    for k in results:
        try:
            alphas.append(float(k))
        except (TypeError, ValueError):
            continue
    return sorted(alphas)


def compute_line(row_cfg, model, direction) -> Optional[np.ndarray]:
    """Return an (N, 2) array of [alpha, pct_group0] for every alpha present, or None."""
    results = load_results(row_cfg, model, direction["key"])
    if results is None:
        return None
    alphas = get_available_alphas(results)
    if not alphas:
        return None
    points = []
    for alpha in alphas:
        split = compute_split(results, direction["comp"], alpha)
        if split is not None:
            points.append((alpha, split[0]))
    if not points:
        return None
    return np.array(points)


def build_figure(model_pair: List[str]) -> Optional[plt.Figure]:
    n_rows, n_cols = len(model_pair), len(DIRECTIONS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.8 * n_rows), dpi=300)
    has_any_data = False

    legend_handles: Dict[str, object] = {}

    for row_idx, model in enumerate(model_pair):
        for col_idx, direction in enumerate(DIRECTIONS):
            ax = axes[row_idx][col_idx]
            ax2 = ax.twinx()

            for row_cfg in ROW_CONFIGS:
                style = LEVEL_STYLES[row_cfg["label"]]
                line = compute_line(row_cfg, model, direction)
                if line is None:
                    continue
                has_any_data = True
                handle, = ax.plot(
                    line[:, 0], line[:, 1],
                    color=style["color"], linestyle=style["linestyle"],
                    marker="o", markersize=3, linewidth=1.8,
                    label=row_cfg["label"],
                )
                if row_cfg["label"] not in legend_handles:
                    legend_handles[row_cfg["label"]] = handle

            ax.set_box_aspect(1)
            ax2.set_box_aspect(1)
            ax.set_ylim(0, 100)
            ax2.set_ylim(0, 100)
            ax2.invert_yaxis()
            ax.grid(False)
            ax2.grid(False)

            ax.tick_params(axis="both", labelsize=11)
            ax2.tick_params(axis="y", labelsize=11)

            if row_idx == 0:
                ax.set_title(direction["label"], fontsize=13, fontweight="bold")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Steering strength (α)", fontsize=14)
            else:
                ax.tick_params(axis="x", labelbottom=False)

            ax.set_ylabel(f"% {direction['group_labels'][0]}", fontsize=12)
            ax2.set_ylabel(f"% {direction['group_labels'][1]}", fontsize=12)

            if col_idx == 0:
                ax.text(
                    -0.38, 0.5, render_model_name(model),
                    transform=ax.transAxes,
                    fontsize=17, fontweight="bold",
                    ha="center", va="center", rotation=90,
                )

    if not has_any_data:
        plt.close(fig)
        return None

    fig.tight_layout()
    fig.subplots_adjust(top=0.85, left=0.15, hspace=0.05)

    if legend_handles:
        fig.legend(
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=len(legend_handles),
            fontsize=13,
            frameon=False,
        )

    return fig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="paper_figures")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for model_pair in MODEL_PAIRS:
        print(f"[info] processing pair: {model_pair}")
        fig = build_figure(model_pair)
        if fig is None:
            print(f"[skip] no data for pair {model_pair}")
            continue
        safe_names = [m.replace(" ", "_").replace("/", "-") for m in model_pair]
        base = "line_grid_" + "_".join(safe_names)
        for fmt in ("png", "svg"):
            p = os.path.join(args.outdir, f"{base}.{fmt}")
            fig.savefig(p, format=fmt, dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
            print(f"[ok] {p}")
        plt.close(fig)

    print("[done]")


if __name__ == "__main__":
    main()

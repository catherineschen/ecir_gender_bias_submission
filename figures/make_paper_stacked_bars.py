"""
Generate a 2x3 static stacked-bar figure per model for a paper.

Layout
------
Rows:
  0 — Embedding | Mean diff + ActAdd  (results_steering_embed_with_scores)
  1 — Aggregation (Heads) | Mean diff + ActAdd  (results_steering_aggregation_with_scores / heads)

Columns:
  0 — Male-Female  (MvF)
  1 — Male-Neutral  (MvN)
  2 — Female-Neutral  (FvN)

Each subplot shows 7 stacked bars for alpha in [-5, -3, -1, 0, 1, 3, 5].
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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

DEFAULT_ALPHAS = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]


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


def build_figure(model, alphas):
    n_rows, n_cols = len(ROW_CONFIGS), len(DIRECTIONS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows), dpi=300, sharey=True)
    x = np.arange(len(alphas))
    x_labels = [f"{a:.0f}" for a in alphas]
    has_any_data = False

    # Collect legend handles keyed by label so we build one shared legend.
    legend_handles: Dict[str, object] = {}

    for row_idx, row_cfg in enumerate(ROW_CONFIGS):
        for col_idx, direction in enumerate(DIRECTIONS):
            ax = axes[row_idx][col_idx]
            results = load_results(row_cfg, model, direction["key"])
            pf_list, ps_list, valid_x = [], [], []

            if results is not None:
                for i, alpha in enumerate(alphas):
                    split = compute_split(results, direction["comp"], alpha)
                    if split is not None:
                        pf_list.append(split[0])
                        ps_list.append(split[1])
                        valid_x.append(i)
                        has_any_data = True

            if valid_x:
                vx, pf, ps = np.array(valid_x), np.array(pf_list), np.array(ps_list)
                bar1 = ax.bar(vx, pf, width=0.7, color=direction["colors"][0], label=direction["group_labels"][0])
                bar2 = ax.bar(vx, ps, width=0.7, bottom=pf, color=direction["colors"][1], label=direction["group_labels"][1])
                for i in range(len(vx)):
                    if pf[i] >= 8:
                        ax.text(vx[i], pf[i] / 2, f"{pf[i]:.0f}%", ha="center", va="center", fontsize=9)
                    if ps[i] >= 8:
                        ax.text(vx[i], pf[i] + ps[i] / 2, f"{ps[i]:.0f}%", ha="center", va="center", fontsize=9)
                # Register handles for shared legend (first occurrence wins)
                if direction["group_labels"][0] not in legend_handles:
                    legend_handles[direction["group_labels"][0]] = bar1
                if direction["group_labels"][1] not in legend_handles:
                    legend_handles[direction["group_labels"][1]] = bar2
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color="gray")

            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=11)
            ax.set_ylim(0, 100)
            ax.tick_params(axis="y", labelsize=11)
            ax.grid(False)
            if row_idx == 0:
                ax.set_title(direction["label"], fontsize=13, fontweight="bold")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Steering strength (α)", fontsize=14)
            if col_idx == 0:
                ax.set_ylabel("% samples", fontsize=14)
                ax.text(
                    -0.42, 0.5, row_cfg["label"],
                    transform=ax.transAxes,
                    fontsize=14, fontweight="bold",
                    ha="center", va="center", rotation=90,
                )

    if not has_any_data:
        plt.close(fig)
        return None

    fig.tight_layout()
    # Reserve space at the top for the suptitle/legend and on the left for row labels.
    fig.subplots_adjust(top=0.82, left=0.22)

    # Shared legend centred above all subplots, below the suptitle.
    if legend_handles:
        fig.legend(
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            ncol=len(legend_handles),
            fontsize=13,
            frameon=False,
        )

    fig.suptitle(model, fontsize=13, fontweight="bold", y=0.97)
    return fig


def build_single_direction_figure(model: str, direction_key: str, alphas: List[float], orientation: str) -> Optional[plt.Figure]:
    """Build a figure with one subplot per ROW_CONFIG for a single direction.

    orientation: "horizontal"  → 1 row × N cols
                 "vertical"    → N rows × 1 col
    """
    direction = next(d for d in DIRECTIONS if d["key"] == direction_key)
    n = len(ROW_CONFIGS)

    if orientation == "horizontal":
        nrows, ncols = 1, n
        figsize = (4.5 * n, 4.5)
    else:
        nrows, ncols = n, 1
        figsize = (5.5, 4.0 * n)

    fig, axes_raw = plt.subplots(nrows, ncols, figsize=figsize, dpi=300, sharey=True)
    axes = list(axes_raw) if n > 1 else [axes_raw]

    x = np.arange(len(alphas))
    x_labels = [f"{a:.0f}" for a in alphas]
    legend_handles: Dict[str, object] = {}
    has_any_data = False

    for idx, row_cfg in enumerate(ROW_CONFIGS):
        ax = axes[idx]
        results = load_results(row_cfg, model, direction_key)
        pf_list, ps_list, valid_x = [], [], []

        if results is not None:
            for i, alpha in enumerate(alphas):
                split = compute_split(results, direction["comp"], alpha)
                if split is not None:
                    pf_list.append(split[0])
                    ps_list.append(split[1])
                    valid_x.append(i)
                    has_any_data = True

        if valid_x:
            vx, pf, ps = np.array(valid_x), np.array(pf_list), np.array(ps_list)
            bar1 = ax.bar(vx, pf, width=0.7, color=direction["colors"][0], label=direction["group_labels"][0])
            bar2 = ax.bar(vx, ps, width=0.7, bottom=pf, color=direction["colors"][1], label=direction["group_labels"][1])
            for i in range(len(vx)):
                if pf[i] >= 8:
                    ax.text(vx[i], pf[i] / 2, f"{pf[i]:.0f}%", ha="center", va="center", fontsize=9)
                if ps[i] >= 8:
                    ax.text(vx[i], pf[i] + ps[i] / 2, f"{ps[i]:.0f}%", ha="center", va="center", fontsize=9)
            if direction["group_labels"][0] not in legend_handles:
                legend_handles[direction["group_labels"][0]] = bar1
            if direction["group_labels"][1] not in legend_handles:
                legend_handles[direction["group_labels"][1]] = bar2
        else:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(False)

        if orientation == "horizontal":
            ax.set_xlabel("Steering strength (α)", fontsize=14)
            ax.set_title(row_cfg["label"], fontsize=14, fontweight="bold")
            if idx == 0:
                ax.set_ylabel("% samples", fontsize=14)
        else:
            # x-axis label only on the bottom subplot
            if idx == n - 1:
                ax.set_xlabel("Steering strength (α)", fontsize=14)
            # row label bold to the left, % samples as ylabel (closer to axis)
            ax.set_ylabel("% samples", fontsize=14)
            ax.text(
                -0.28, 0.5, row_cfg["label"],
                transform=ax.transAxes,
                fontsize=14, fontweight="bold",
                ha="center", va="center", rotation=90,
            )

    if not has_any_data:
        plt.close(fig)
        return None

    fig.tight_layout()
    if orientation == "vertical":
        fig.subplots_adjust(top=0.88, left=0.28)
        legend_y = 0.94
        suptitle_y = 0.99
    else:
        fig.subplots_adjust(top=0.72, left=0.10)
        legend_y = 0.94
        suptitle_y = 1.0

    if legend_handles:
        fig.legend(
            handles=list(legend_handles.values()),
            labels=list(legend_handles.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=len(legend_handles),
            fontsize=13,
            frameon=False,
        )

    fig.suptitle(f"{model} | {direction['label']}", fontsize=13, fontweight="bold", y=suptitle_y)
    return fig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", nargs="*", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--outdir", default="paper_figures")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    alphas = sorted(args.alphas)
    os.makedirs(args.outdir, exist_ok=True)
    for model in MODELS:
        print(f"[info] processing: {model}")
        fig = build_figure(model, alphas)
        if fig is None:
            print(f"[skip] no data for {model}")
            continue
        safe = model.replace(" ", "_").replace("/", "-")
        for fmt in ("png", "svg"):
            p = os.path.join(args.outdir, f"stacked_bar_grid_{safe}.{fmt}")
            fig.savefig(p, format=fmt, dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
            print(f"[ok] {p}")
        plt.close(fig)

    # Single-direction figures for msmarco-distilbert-dot-v5 / Male-Female
    single_model = "msmarco-distilbert-dot-v5"
    single_dir = "mf"
    for orientation in ("horizontal", "vertical"):
        print(f"[info] single-direction {orientation}: {single_model} | {single_dir}")
        fig = build_single_direction_figure(single_model, single_dir, alphas, orientation)
        if fig is None:
            print(f"[skip] no data")
            continue
        safe = single_model.replace(" ", "_").replace("/", "-")
        for fmt in ("png", "svg"):
            p = os.path.join(args.outdir, f"stacked_bar_{safe}_{single_dir}_{orientation}.{fmt}")
            fig.savefig(p, format=fmt, dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
            print(f"[ok] {p}")
        plt.close(fig)

    print("[done]")


if __name__ == "__main__":
    main()
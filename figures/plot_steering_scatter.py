# plot_steering_scatter_main.py
import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Color palette — purple for negative α, green for positive α, grey baseline.
COLOR_NEG = "#7B5BA6"   # muted purple
COLOR_POS = "#5B9279"   # muted teal-green
COLOR_BASELINE = "#999999"

AXIS_LIMITS = (-6, 6)
TICK_STEP = 2
ALPHA_NEG = -5
ALPHA_POS = 5
ALPHA_BASELINE = 0
ALPHA_ROW = 5  # which alpha to use in the row layout (positive or negative)

EMBEDDING_PATH = "results/pointwise_steering/results_steering_embed_with_scores"
AGGREGATION_PATH = "results/pointwise_steering/results_steering_aggregation_with_scores"
AGGREGATION_LEVEL = "heads"
DIRECTION = "mf"
METHOD_SUBDIR = "actadd"


def load_results(base_path, model, direction, level=None):
    if level:
        path = os.path.join(base_path, model, level, direction, METHOD_SUBDIR, "steering_results.json")
    else:
        path = os.path.join(base_path, model, direction, METHOD_SUBDIR, "steering_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Steering results not found: {path}")
    with open(path) as f:
        return json.load(f)


def get_mn_fn(results, alpha):
    key = str(float(alpha))
    if key not in results:
        # try without trailing .0
        key = str(alpha)
    if key not in results:
        raise KeyError(f"alpha={alpha} not in results (available: {list(results.keys())})")
    m = np.array(results[key]["m_scores"])
    f = np.array(results[key]["f_scores"])
    n = np.array(results[key]["n_scores"])
    return m - n, f - n  # m_rel_n, f_rel_n


def plot_panel(ax, x_vals, y_vals, color, axis_limits, show_xlabel, show_ylabel):
    sns.scatterplot(
        x=x_vals, y=y_vals,
        ax=ax,
        alpha=0.6,
        s=20,
        color=color,
    )
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_xticks(range(AXIS_LIMITS[0], AXIS_LIMITS[1] + 1, TICK_STEP))
    ax.set_yticks(range(AXIS_LIMITS[0], AXIS_LIMITS[1] + 1, TICK_STEP))
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axline((0, 0), slope=1, linestyle=":", linewidth=1, color="red")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=9)
    if show_xlabel:
        ax.set_xlabel("F score relative to N", fontsize=11)
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel("M score relative to N", fontsize=11)
    else:
        ax.set_yticklabels([])

def plot_row_layout(panels, axis_limits, out_path):
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.6))

    layout = [
        ("emb", "neg", COLOR_NEG, f"Embedding (α = {ALPHA_NEG})"),
        ("emb", "pos", COLOR_POS, f"Embedding (α = {ALPHA_POS})"),
        ("emb", "base", COLOR_BASELINE, "Baseline (α = 0)"),
        ("agg", "neg", COLOR_NEG, f"Attention (α = {ALPHA_NEG})"),
        ("agg", "pos", COLOR_POS, f"Attention (α = {ALPHA_POS})"),
    ]

    for c, (intervention, key, color, title) in enumerate(layout):
        m_rel_n, f_rel_n = panels[(intervention, key)]
        plot_panel(
            axes[c], f_rel_n, m_rel_n, color,
            axis_limits=axis_limits,
            show_xlabel=True,
            show_ylabel=(c == 0),
        )
        axes[c].set_title(title, fontsize=12, pad=4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_grid_layout(panels, axis_limits, out_path):
    # (the existing 2x3 grid code, factored out into its own function)
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.6))

    layout = [
        [("emb", "neg", COLOR_NEG), ("emb", "base", COLOR_BASELINE), ("emb", "pos", COLOR_POS)],
        [("agg", "neg", COLOR_NEG), ("agg", "base", COLOR_BASELINE), ("agg", "pos", COLOR_POS)],
    ]
    for r in range(2):
        for c in range(3):
            row_key, col_key, color = layout[r][c]
            m_rel_n, f_rel_n = panels[(row_key, col_key)]
            plot_panel(
                axes[r, c], f_rel_n, m_rel_n, color,
                axis_limits=axis_limits,
                show_xlabel=(r == 1),
                show_ylabel=(c == 0),
            )

    axes[0, 0].set_title(f"α = {ALPHA_NEG}", fontsize=13, pad=4)
    axes[0, 1].set_title(f"α = {ALPHA_BASELINE} (baseline)", fontsize=13, pad=4)
    axes[0, 2].set_title(f"α = {ALPHA_POS}", fontsize=13, pad=4)

    fig.text(0.02, 0.72, "Embedding", fontsize=13, rotation=90, va="center", ha="center")
    fig.text(0.02, 0.30, "Late-layer\nattention", fontsize=13, rotation=90, va="center", ha="center")

    plt.tight_layout(rect=[0.04, 0.0, 1.0, 1.0])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="msmarco-distilbert-dot-v5")
    parser.add_argument("--out_path", "-o", default="figures/steering_scatter_main.pdf")
    parser.add_argument("--layout", choices=["grid", "row"], default="grid")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    embedding_results = load_results(EMBEDDING_PATH, args.model, DIRECTION)
    aggregation_results = load_results(AGGREGATION_PATH, args.model, DIRECTION, level=AGGREGATION_LEVEL)

    panels = {
        ("emb", "neg"): get_mn_fn(embedding_results, ALPHA_NEG),
        ("emb", "base"): get_mn_fn(embedding_results, ALPHA_BASELINE),
        ("emb", "pos"): get_mn_fn(embedding_results, ALPHA_POS),
        ("agg", "neg"): get_mn_fn(aggregation_results, ALPHA_NEG),
        ("agg", "base"): get_mn_fn(aggregation_results, ALPHA_BASELINE),
        ("agg", "pos"): get_mn_fn(aggregation_results, ALPHA_POS),
    }

    axis_limits = (-6, 6)

    if args.layout == "grid":
        plot_grid_layout(panels, axis_limits, args.out_path)
    else:
        plot_row_layout(panels, axis_limits, args.out_path)


if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse

CATEGORIES = ["sep", "gendered", "matching", "cls", "non_matching"]
CATEGORY_COLORS = {
    "cls": "#7FB3D5",        # dusty blue
    "sep": "#82C09A",        # sage green
    "gendered": "#E89999",   # coral pink
    "matching": "#B19CD9",   # soft purple
    "non_matching": "#E0E0E0",
}
CATEGORY_LABELS = {
    "cls": "CLS",
    "sep": "SEP",
    "gendered": "Gendered",
    "matching": "Query-matching",
    "non_matching": "Non-matching",
}

MODELS = {
    "sentence-transformers/multi-qa-distilbert-dot-v1": {
        "row_label": r"DistilBERT$_{\rm multi\text{-}qa}$",
        "source_type": "pooled",
    },
    "sentence-transformers/msmarco-distilbert-dot-v5": {
        "row_label": r"DistilBERT$_{\rm ms\text{-}marco}$",
        "source_type": "gendered",
    },
}

RELEVANCE_COLS = {1: "Relevant", 0: "Non-relevant"}


def load_csv(attn_results_dir, model_name, subset, source_type):
    safe_model = model_name.replace("/", "_")
    path = os.path.join(
        attn_results_dir,
        f"attention_by_category_{safe_model}_{subset}_src_{source_type}.csv",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def get_head_vals(df, rel_val, model_name):
    sub = df[(df["model"] == model_name) & (df["relevant"] == rel_val)]
    agg = sub.groupby(["layer", "head", "category"], as_index=False)["attention"].mean()
    head_keys = sorted(agg[["layer", "head"]].drop_duplicates().itertuples(index=False))
    return head_keys, agg


def plot_stacked(ax, head_keys, agg, show_xticklabels=True, bar_width_frac=0.5, spacing=0.3):
    head_labels = [f"{l}.{h}" for l, h in head_keys]
    bottom = np.zeros(len(head_keys))
    x = np.arange(len(head_keys)) * spacing  # compress bar centers together
    bar_width = spacing * bar_width_frac 
    bars = {}
    for cat in CATEGORIES:
        vals = []
        for layer, head in head_keys:
            row = agg[
                (agg["layer"] == layer) &
                (agg["head"] == head) &
                (agg["category"] == cat)
            ]
            vals.append(row["attention"].iloc[0] if len(row) else 0.0)
        b = ax.bar(
            x, vals, width=bar_width, bottom=bottom,
            label=CATEGORY_LABELS[cat],
            color=CATEGORY_COLORS[cat],
        )
        bars[cat] = b
        bottom += np.array(vals)

    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xlim(-0.5 * spacing, (len(head_keys) - 1) * spacing + 0.5 * spacing)
    if show_xticklabels:
        ax.set_xticklabels(head_labels, rotation=0, ha="center", fontsize=8)
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="y", labelsize=8)
    return bars

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn_results_dir", "-ad", required=True)
    parser.add_argument("--subset", choices=["all", "overlap", "non_overlap"], default="non_overlap")
    parser.add_argument("--out_path", "-o", default="figures/attention_main.pdf")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    n_rows = len(MODELS)
    n_cols = len(RELEVANCE_COLS)

    fig = plt.figure(figsize=(4.5, 5.5))

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        top=0.84, bottom=0.16,
        left=0.24, right=0.97,
        hspace=0.20, wspace=0.10,
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]

    # Column headers
    for c, rel_label in enumerate(RELEVANCE_COLS.values()):
        axes[0][c].set_title(rel_label, fontsize=9, pad=4)

    # Row labels (y-axis label on leftmost column)
    for r, (model_name, meta) in enumerate(MODELS.items()):
        axes[r][0].set_ylabel(meta["row_label"], fontsize=12, labelpad=6)

    # Share y-axis within each row
    for r in range(n_rows):
        axes[r][1].sharey(axes[r][0])
        axes[r][1].tick_params(labelleft=False)

    # Only bottom row gets x-tick labels
    handle_cache = None
    for r, (model_name, meta) in enumerate(MODELS.items()):
        source_type = meta["source_type"]
        df = load_csv(args.attn_results_dir, model_name, args.subset, source_type)

        for c, rel_val in enumerate(RELEVANCE_COLS.keys()):
            ax = axes[r][c]
            show_x = (r == n_rows - 1)
            head_keys, agg = get_head_vals(df, rel_val, model_name)
            bars = plot_stacked(ax, head_keys, agg, show_xticklabels=show_x)
            if handle_cache is None:
                handle_cache = bars

    # Shared legend at top center
    LEGEND_ORDER = ["cls", "gendered", "matching", "sep", "non_matching"]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[cat], label=CATEGORY_LABELS[cat])
        for cat in LEGEND_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.1, 0.92, 0.8, 0.05),  # x, y, width, height
        mode="expand",
    )

    fig.savefig(args.out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out_path}")


if __name__ == "__main__":
    main()

# python plot_attention_paper_fig.py -ad results_attn_pattern --subset all -o result_figures_attn/attention_fig.png
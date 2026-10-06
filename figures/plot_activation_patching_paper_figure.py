"""
Main-text activation patching summary figure.

Produces a grid of line plots overlaying two models: one CLS-pooled and one
mean-pooled. The grid is N (components) x 2 (token positions), where N is
either 3 (residual stream, attention output, MLP output) or 2 (residual
stream, attention output) depending on --components.

Two versions:

  - Version A: single line per model, using MvF only.
  - Version B: mean line per model (averaged over MvF/MvN/FvN) with a
    min-max shaded band across the three gender comparisons.

Input CSVs are assumed to have columns:
    comparison, doc_group_id, component, layer,
    [CLS]_value, gendered terms_value, neutral terms_value, [SEP]_value

Usage:
    python plot_main_figure.py \\
        --cls-csv path/to/cls_pooled_model.csv \\
        --mean-csv path/to/mean_pooled_model.csv \\
        --cls-label "DistilBERT MultiQA (CLS)" \\
        --mean-label "DistilBERT MSMARCO (mean)" \\
        --save-dir figures/ \\
        --components resid attn_out         # 2x2 (main text)
        # or --components resid attn_out mlp  # 3x2 (appendix)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ----- config -----------------------------------------------------------------

ALL_COMPONENTS = ["resid", "attn_out", "mlp"]
COMPONENT_DISPLAY = {
    "resid": "Residual stream",
    "attn_out": "Attention output",
    "mlp": "MLP output",
}

# columns to plot as the two main-text token positions
TOKEN_COLS = ["[CLS]_value", "gendered terms_value"]
TOKEN_DISPLAY = {
    "[CLS]_value": "[CLS] token",
    "gendered terms_value": "Gendered terms",
}

GENDER_COMPARISONS = ["MvF", "MvN", "FvN"]

# colors chosen to be distinct and colorblind-friendly (Okabe-Ito-ish)
COLOR_CLS = "#0072B2"   # blue
COLOR_MEAN = "#D55E00"  # vermilion


# ----- aggregation ------------------------------------------------------------

def aggregate_over_docs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse doc_group_id by mean, giving one value per
    (comparison, component, layer, token_col).
    """
    value_cols = [c for c in df.columns if c.endswith("_value")]
    group_cols = ["comparison", "component", "layer"]
    return df.groupby(group_cols, as_index=False)[value_cols].mean()


def get_mvf_only(df_agg: pd.DataFrame, component: str, token_col: str):
    """Return (layers, values) for MvF only."""
    sub = df_agg[(df_agg["comparison"] == "MvF") & (df_agg["component"] == component)]
    sub = sub.sort_values("layer")
    return sub["layer"].to_numpy(), sub[token_col].to_numpy()


def get_mean_and_band(df_agg: pd.DataFrame, component: str, token_col: str):
    """
    Return (layers, mean_over_comparisons, min_over_comparisons, max_over_comparisons)
    across MvF/MvN/FvN.
    """
    sub = df_agg[
        (df_agg["comparison"].isin(GENDER_COMPARISONS))
        & (df_agg["component"] == component)
    ].copy()
    # pivot so rows are layers, columns are comparisons
    pivot = sub.pivot_table(
        index="layer", columns="comparison", values=token_col, aggfunc="mean"
    )
    # make sure all three comparisons are present; if not, fall back to whatever is there
    present = [c for c in GENDER_COMPARISONS if c in pivot.columns]
    pivot = pivot[present].sort_index()
    layers = pivot.index.to_numpy()
    mean_vals = pivot.mean(axis=1).to_numpy()
    min_vals = pivot.min(axis=1).to_numpy()
    max_vals = pivot.max(axis=1).to_numpy()
    return layers, mean_vals, min_vals, max_vals


# ----- plotting ---------------------------------------------------------------

def _setup_grid(n_rows: int, n_cols: int, orientation: str = "vertical"):
    # slightly different proportions for each orientation so panels stay readable
    if orientation == "horizontal":
        panel_w, panel_h = 3.6, 3.0
    else:
        panel_w, panel_h = 4.2, 3.2
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_w * n_cols, panel_h * n_rows),
        sharey="row",
        sharex=True,
    )
    axs = np.atleast_2d(axs)
    return fig, axs


def _decorate_axes(axs, n_layers: int, row_labels: list, col_labels: list, xlabel: str = "Layer"):
    """Shared decoration: titles, axis labels, grid."""
    n_rows, n_cols = axs.shape
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axs[r, c]
            # ax.grid(True, alpha=0.3)
            ax.axhline(0, color="black", linewidth=0.6) #, alpha=0.5
            ax.set_xticks(range(n_layers))
            # column titles on top row
            if r == 0:
                ax.set_title(col_labels[c], fontsize=11)
            # row labels on left column
            if c == 0:
                ax.set_ylabel(f"{row_labels[r]}\nPatch effect", fontsize=10)
            # x-axis label on bottom row
            if r == n_rows - 1:
                ax.set_xlabel(xlabel, fontsize=10)


def _add_legend(fig, cls_label: str, mean_label: str, include_band: bool):
    handles = [
        Line2D([0], [0], color=COLOR_CLS, lw=2, label=cls_label),
        Line2D([0], [0], color=COLOR_MEAN, lw=2, label=mean_label),
    ]
    if include_band:
        # a separate faint handle describing the band
        handles.append(
            Line2D(
                [0],
                [0],
                color="gray",
                lw=8,
                alpha=0.25,
                label="min–max across gender comparisons",
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.0),
    )


def plot_version_a(
    df_cls: pd.DataFrame,
    df_mean: pd.DataFrame,
    cls_label: str,
    mean_label: str,
    save_path: str,
    orientation: str = "vertical",
    components: list = None,
):
    """Version A: single line per model, MvF only.

    orientation: 'vertical' -> rows=components, cols=tokens (Nx2)
                 'horizontal' -> rows=tokens, cols=components (2xN)
    components: list of components to plot, in order. Defaults to all three.
    """
    if components is None:
        components = ALL_COMPONENTS

    df_cls_agg = aggregate_over_docs(df_cls)
    df_mean_agg = aggregate_over_docs(df_mean)

    n_layers = int(df_cls_agg["layer"].max()) + 1

    if orientation == "vertical":
        row_items, col_items = components, TOKEN_COLS
        row_labels = [COMPONENT_DISPLAY[c] for c in components]
        col_labels = [TOKEN_DISPLAY[t] for t in TOKEN_COLS]
    elif orientation == "horizontal":
        row_items, col_items = TOKEN_COLS, components
        row_labels = [TOKEN_DISPLAY[t] for t in TOKEN_COLS]
        col_labels = [COMPONENT_DISPLAY[c] for c in components]
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    fig, axs = _setup_grid(len(row_items), len(col_items), orientation)

    for r, row_key in enumerate(row_items):
        for c, col_key in enumerate(col_items):
            ax = axs[r, c]

            comp = row_key if orientation == "vertical" else col_key
            tok = col_key if orientation == "vertical" else row_key

            x_cls, y_cls = get_mvf_only(df_cls_agg, comp, tok)
            x_mean, y_mean = get_mvf_only(df_mean_agg, comp, tok)

            ax.plot(x_cls, y_cls, color=COLOR_CLS, marker="o",
                    linewidth=2, markersize=5, label=cls_label)
            ax.plot(x_mean, y_mean, color=COLOR_MEAN, marker="s",
                    linewidth=2, markersize=5, label=mean_label)

    _decorate_axes(axs, n_layers, row_labels, col_labels)
    _add_legend(fig, cls_label, mean_label, include_band=False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return save_path


def plot_version_b(
    df_cls: pd.DataFrame,
    df_mean: pd.DataFrame,
    cls_label: str,
    mean_label: str,
    save_path: str,
    orientation: str = "vertical",
    components: list = None,
):
    """Version B: mean line per model with min-max band across MvF/MvN/FvN.

    orientation: 'vertical' -> rows=components, cols=tokens (Nx2)
                 'horizontal' -> rows=tokens, cols=components (2xN)
    components: list of components to plot, in order. Defaults to all three.
    """
    if components is None:
        components = ALL_COMPONENTS

    df_cls_agg = aggregate_over_docs(df_cls)
    df_mean_agg = aggregate_over_docs(df_mean)

    n_layers = int(df_cls_agg["layer"].max()) + 1

    if orientation == "vertical":
        row_items, col_items = components, TOKEN_COLS
        row_labels = [COMPONENT_DISPLAY[c] for c in components]
        col_labels = [TOKEN_DISPLAY[t] for t in TOKEN_COLS]
    elif orientation == "horizontal":
        row_items, col_items = TOKEN_COLS, components
        row_labels = [TOKEN_DISPLAY[t] for t in TOKEN_COLS]
        col_labels = [COMPONENT_DISPLAY[c] for c in components]
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    fig, axs = _setup_grid(len(row_items), len(col_items), orientation)

    for r, row_key in enumerate(row_items):
        for c, col_key in enumerate(col_items):
            ax = axs[r, c]

            comp = row_key if orientation == "vertical" else col_key
            tok = col_key if orientation == "vertical" else row_key

            xc, mc, loc, hic = get_mean_and_band(df_cls_agg, comp, tok)
            xm, mm, lom, him = get_mean_and_band(df_mean_agg, comp, tok)

            ax.fill_between(xc, loc, hic, color=COLOR_CLS, alpha=0.2, linewidth=0)
            ax.plot(xc, mc, color=COLOR_CLS, marker="o",
                    linewidth=2, markersize=5, label=cls_label)

            ax.fill_between(xm, lom, him, color=COLOR_MEAN, alpha=0.2, linewidth=0)
            ax.plot(xm, mm, color=COLOR_MEAN, marker="s",
                    linewidth=2, markersize=5, label=mean_label)

    _decorate_axes(axs, n_layers, row_labels, col_labels)
    _add_legend(fig, cls_label, mean_label, include_band=True)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return save_path


# ----- driver -----------------------------------------------------------------

def _components_tag(components: list) -> str:
    """Short tag describing the component selection, for filenames."""
    if components == ALL_COMPONENTS:
        return "all"
    return "_".join(components)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls-csv", required=True, help="Path to CLS-pooled model CSV")
    parser.add_argument("--mean-csv", required=True, help="Path to mean-pooled model CSV")
    parser.add_argument(
        "--cls-label",
        default="CLS-pooled",
        help="Legend label for CLS-pooled model",
    )
    parser.add_argument(
        "--mean-label",
        default="Mean-pooled",
        help="Legend label for mean-pooled model",
    )
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["a", "b"],
        choices=["a", "b"],
        help="Which versions to produce",
    )
    parser.add_argument(
        "--orientations",
        nargs="+",
        default=["vertical", "horizontal"],
        choices=["vertical", "horizontal"],
        help="Which orientations to produce",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=ALL_COMPONENTS,
        choices=ALL_COMPONENTS,
        help=(
            "Which components to plot, in order. Use 'resid attn_out' for the "
            "2x2 main-text figure, or 'resid attn_out mlp' for the 3x2 appendix "
            "figure (default)."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    df_cls = pd.read_csv(args.cls_csv)
    df_mean = pd.read_csv(args.mean_csv)

    comp_tag = _components_tag(args.components)

    outputs = []
    for version in args.versions:
        plot_fn = plot_version_a if version == "a" else plot_version_b
        tag = "mvf" if version == "a" else "band"
        for orient in args.orientations:
            fname = f"main_figure_version_{version}_{tag}_{orient}_{comp_tag}.png"
            out = plot_fn(
                df_cls,
                df_mean,
                args.cls_label,
                args.mean_label,
                os.path.join(args.save_dir, fname),
                orientation=orient,
                components=args.components,
            )
            outputs.append(out)
            print(f"Saved Version {version.upper()} ({orient}, {comp_tag}) → {out}")

    return outputs


if __name__ == "__main__":
    main()

# Main-text 2x2 (resid + attention only):
# python plot_activation_patching_paper_figure.py --cls-csv result_figures/activation_patch/sentence-transformers-multi-qa-distilbert-dot-v1/patch_effect_df.csv --mean-csv result_figures_mechir_bi/activation_patch/sentence-transformers-msmarco-distilbert-dot-v5/patch_effect_df.csv --cls-label 'DistilBERT$_{\mathrm{multi-qa}}$ (CLS)' --mean-label 'DistilBERT$_{\mathrm{ms-marco}}$ (mean)' --save-dir result_figures/activation_patch --versions a b --orientations horizontal --components resid attn_out
#
# Appendix 3x2 (with MLP):
# python plot_activation_patching_paper_figure.py --cls-csv result_figures/activation_patch/sentence-transformers-multi-qa-distilbert-dot-v1/patch_effect_df.csv --mean-csv result_figures_mechir_bi/activation_patch/sentence-transformers-msmarco-distilbert-dot-v5/patch_effect_df.csv --cls-label "DistilBERT MultiQA (CLS)" --mean-label "DistilBERT MSMARCO (mean)" --save-dir result_figures/activation_patch --versions a b --orientations horizontal --components resid attn_out mlp
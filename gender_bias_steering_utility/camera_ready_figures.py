"""Static matplotlib paper figures, built from the exact same functions as
camera_ready_dashboard.py (all data logic lives in camera_ready_analysis.py). Use this once
you've settled on a coefficient threshold/mode in the interactive dashboard -- pass the same
arguments here to get camera-ready PNGs (or JPGs) instead of Plotly HTML.

plot_frontier's row/column label placement and margins (axes-fraction row labels via ax.text,
tight_layout + subplots_adjust to reserve fixed margins, not points-offset annotations) mirror
gender_bias_steering/make_paper_stacked_bars.py's established styling for this same paper, not a
fresh design -- keep them in sync if that file's conventions change. No axes carry gridlines, and
every baseline reference (unsteered dense, BM25 where applicable) has its own legend entry --
never just an unlabeled line.

Usage (from the repo root, or gender_bias_steering_utility/):
    conda run -n gender_bias python camera_ready_figures.py \\
        --dataset msmarco_fair --models msmarco-distilbert-dot-v5,multi-qa-distilbert-dot-v1 --threshold 0.02

Or import the functions directly:
    import camera_ready_figures as crf
    fig, axes = crf.plot_frontier("msmarco_fair", "msmarco-distilbert-dot-v5", threshold=0.02)
    fig.savefig("frontier_msmarco-distilbert-dot-v5.png")
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import camera_ready_analysis as cra

LEVEL_COLORS = {"embed": "#2a78d6", "attn": "#e34948"}
BASELINE_COLOR = "#8c8c8c"
BASELINE_MARKER_COLOR = "#000000"
# Frontier scatter: opacity floor for points at coefficient=0 (least steered, least interesting);
# opacity ramps linearly up to 1.0 at each row's most extreme coefficient.
_MIN_MARKER_ALPHA = 0.30

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
})


def _load_agg(dataset_key: str, model: str, arab_variant: str, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, seeds_found = cra.load_all_seeds(dataset_key, model, seeds)
    if df.empty:
        raise ValueError(f"No sweep data for {model} on {dataset_key}.")
    agg = cra.aggregate_across_seeds(df, arab_variant)
    return df, agg


_UTILITY_METRICS = {"nDCG@10": "ndcg10", "MRR@10": "mrr10"}


def _model_baseline(dataset_key: str, model: str, agg: pd.DataFrame, level: str, direction: str, metric: str = "nDCG@10"):
    bl = cra.unsteered_baseline(dataset_key, model)
    if bl.found:
        return getattr(bl, _UTILITY_METRICS[metric]), bl.nfairr10, None
    zp = cra.sweep_zero_point(agg, level, direction)
    return (zp[metric], zp["NFaiRR@10"], None) if zp is not None else (np.nan, np.nan, None)


# Paper layout for plot_frontier only (deliberately not touched elsewhere -- plot_arab_curves
# keeps the open-spine, no-legend-clutter Tufte style via the module-level rcParams above).
# "Directions abbreviated" per the brief, kept local rather than changed on cra.DIRECTIONS since
# that constant's " − " form is also used in the dashboard captions and LaTeX table labels.
_FRONTIER_DIRECTION_TITLES = {"mf": "Male-Female", "mn": "Male-Neutral", "fn": "Female-Neutral"}
_FRONTIER_ROW_LEVELS = ["embed", "attn"]


def _no_white_diverging_cmap(name: str = "PRGn", lightest: float = 0.35, n: int = 128):
    """PRGn (or any diverging colormap) with its near-white center band removed: the negative
    half is resampled from [0, lightest] and the positive half from [1-lightest, 1], concatenated
    directly -- so the lightest color used at either pole is still a visibly tinted purple/green,
    never white, with one hard step exactly at the zero boundary instead of fading through white."""
    base = plt.get_cmap(name)
    purple_half = base(np.linspace(0.0, lightest, n))
    green_half = base(np.linspace(1.0 - lightest, 1.0, n))
    return mcolors.LinearSegmentedColormap.from_list(f"{name}_no_white", np.vstack([purple_half, green_half]))


def _nice_ticks(lo: float, hi: float, target_n: int = 6, min_step: float = 0.01) -> np.ndarray:
    """"Nice" round ticks using steps that are 1, 2, or 5 x a power of ten (0.01, 0.02, 0.05, 0.1,
    0.2, 0.5, 1, 2, 5, ...) spanning at least [lo, hi]. `min_step` (default 0.01, matching the
    axes' 2-decimal tick labels) floors the step so a narrow data span -- e.g. TREC DL19's
    NFaiRR@10 values, often within 0.02 of each other -- never produces a step finer than the
    display precision, which would otherwise round two adjacent ticks to the same label (e.g. both
    showing "0.96"); a "1x" step is safe to allow again once that floor exists (0.01, not something
    finer, e.g. 0.005, that would still collide). The start isn't forced to an even multiple of a
    round number either -- e.g. a 0.01 step over TREC DL19's narrow range can and often does start
    on an odd hundredth (0.95, 0.97, ...), whatever floor(lo/step)*step lands on. The caller sets
    the axis limits to (ticks[0], ticks[-1]) so the first/last tick still lands exactly on the
    plot's edge, just at a round value instead of the raw (padded) data min/max."""
    span = hi - lo
    if span <= 0:
        return np.array([lo, hi])
    raw_step = max(span / max(target_n - 1, 1), min_step)
    magnitude = 10 ** np.floor(np.log10(raw_step))
    candidates = sorted(m * magnitude for m in (1, 2, 5, 10, 20, 50))
    step = next(c for c in candidates if c >= raw_step)
    start = np.floor(lo / step) * step
    end = np.ceil(hi / step) * step
    return np.round(np.arange(start, end + step / 2, step), 10)


# Font sizes for plot_frontier only, bumped for print legibility (a screen-sized default reads
# fine interactively but is too small once shrunk to a paper column).
_FRONTIER_FONTS = dict(suptitle=22, title=19, axis_label=18, row_label=19, tick=15, legend=17, colorbar=19)


def _resolve_frontier_cmap(colormap: str):
    """"prgn_no_white" (the default) is the paper's own locked-in palette -- PRGn with its
    near-white center band removed (see _no_white_diverging_cmap) so near-zero-coefficient points
    don't fade into the white panel background. Any other name is looked up as an ordinary
    matplotlib diverging colormap (e.g. "coolwarm") and used as-is, center band included."""
    if colormap == "prgn_no_white":
        return _no_white_diverging_cmap()
    return plt.get_cmap(colormap)


def plot_frontier(
    dataset_key: str, model: str, threshold: float, mode: str = "asymmetric",
    arab_variant: str = cra.DEFAULT_ARAB_VARIANT, seeds: list[int] = cra.SEEDS, zoom: bool = False,
    colorbar_orientation: str = "vertical", metric: str = "nDCG@10", marker_size: float = 60,
    colormap: str = "prgn_no_white",
):
    """`metric` (utility metric for the y-axis/frontier, default "nDCG@10") vs NFaiRR@10 for one
    model: a fixed 2 (row: embedding/attention) x 3 (column: comparison direction) grid, sharing
    ONE axis range across all six panels (tick LABELS shown only on the left column / bottom row,
    via explicit tick_params -- the panels' own tick MARKS and limits are still shared/linked via
    sharex=sharey=True). Black-bordered axes, a diverging coefficient colorbar (see `colormap`),
    and markers whose opacity ramps up from coefficient=0 to each row's most extreme coefficient.
    `metric`: "nDCG@10" or "MRR@10" -- this only changes what's plotted on the y-axis and the
    baseline marker's y-value; the operating range itself (used by `zoom=True` and `threshold`) is
    always defined by nDCG@10 degradation regardless of `metric`, matching
    cra.find_operating_range's convention.
    `colormap`: "prgn_no_white" (default -- purple/negative <-> green/positive, no white at its
    center) or any other matplotlib diverging colormap name, e.g. "coolwarm", used as-is.
    `colorbar_orientation`: "vertical" (default, at the right, embed ticks on its left side / attn
    on its right) or "horizontal" (along the bottom, embed ticks below the bar / attn above it).
    `marker_size`: scatter marker area (matplotlib's `s`, in points^2) for the sweep's data points
    only -- smaller values let overlapping/closely-spaced coefficients along the curve stay visually
    distinguishable instead of merging into a single blob. The baseline "x" marker is unaffected
    (always a fixed, prominent size, since it's a single reference point, not a density to read).
    `zoom=True` restricts points to the operating range at `threshold` but keeps the SAME axis
    limits as the full (zoom=False) view -- call both with identical args (only zoom differs) so
    the two figures' axes match, i.e. zoom is a visual subset, not a rescale."""
    if metric not in _UTILITY_METRICS:
        raise ValueError(f"metric must be one of {list(_UTILITY_METRICS)}, got {metric!r}")
    directions = list(cra.DIRECTIONS)
    horizontal = colorbar_orientation == "horizontal"
    # A horizontal colorbar competes with the subplot rows for vertical space rather than the
    # right margin, so it needs the figure to grow taller (not just a bigger bottom-margin
    # fraction of a fixed-height figure) -- otherwise the two rows compress until their rotated
    # row-group labels (below) are taller than the row itself and collide with each other.
    fig_height = (4.3 * 2 + 2.6) if horizontal else (4.3 * 2)
    # Per-column width multiplier, calibrated empirically (not derived in closed form -- the
    # colorbar/legend/row-label margins below interact nonlinearly) so each PANEL (not the whole
    # figure) renders at a deliberately non-square, taller-than-wide ratio (~0.70 width/height,
    # landing the figure at less overall width for the paper's column), measured via
    # ax.get_position() at candidate multipliers. horizontal=True keeps the original 5.0 (square
    # would be ~2.76, a 0.70 ratio ~1.93) -- at narrower widths the fixed-size row-group labels and
    # column titles (sized for the wider original layout) collide/overlap, so reflowing that
    # orientation would need its own font/margin retune, not just this multiplier.
    panel_width_mult = 5.0 if horizontal else 4.7
    fig, axes = plt.subplots(2, len(directions), figsize=(panel_width_mult * len(directions), fig_height),
                              squeeze=False, sharex=True, sharey=True)
    cmap = _resolve_frontier_cmap(colormap)

    _, agg_full = _load_agg(dataset_key, model, arab_variant, seeds)
    # PER-ROW color normalization: embed and attn sweep different coefficient grids (e.g. alpha in
    # [-10,10], beta in [-14,14]), and each row's points should use its OWN full color range (its
    # most extreme coefficient reaches full saturation) rather than being desaturated by the other
    # row's wider grid -- what makes the one shared, dual-labeled colorbar below valid.
    row_max = {lvl: agg_full.loc[agg_full.level == lvl, "coefficient"].abs().max() for lvl in _FRONTIER_ROW_LEVELS}
    row_norm = {lvl: plt.Normalize(vmin=-row_max[lvl], vmax=row_max[lvl]) for lvl in _FRONTIER_ROW_LEVELS}

    # baseline_ndcg is ALWAYS nDCG@10 (never the chosen `metric`) because it's what
    # find_operating_range below needs -- the operating range is defined by nDCG@10 degradation
    # regardless of which metric is being plotted. baseline_metric is the chosen metric's own
    # baseline value, used only for the y-axis/baseline marker; the two coincide when metric is
    # "nDCG@10" (the default), so no second lookup happens in that case.
    baseline_ndcg, baseline_nfairr, _ = _model_baseline(dataset_key, model, agg_full, "embed", directions[0])
    baseline_metric = baseline_ndcg if metric == "nDCG@10" else \
        _model_baseline(dataset_key, model, agg_full, "embed", directions[0], metric=metric)[0]

    # ONE shared axis range across all 6 panels (not per row): sharex=sharey=True only links the
    # tick marks/limits, it doesn't compute them -- this is what "only edge panels need tick
    # labels" actually depends on, since two different ranges would make hiding interior labels
    # misleading.
    pad_y = (agg_full[metric].max() - agg_full[metric].min()) * 0.08 or 0.01
    pad_x = (agg_full["NFaiRR@10"].max() - agg_full["NFaiRR@10"].min()) * 0.08 or 0.01
    raw_ylim = (agg_full[metric].min() - pad_y, agg_full[metric].max() + pad_y)
    raw_xlim = (agg_full["NFaiRR@10"].min() - pad_x, agg_full["NFaiRR@10"].max() + pad_x)
    # X: ticks first, axis limits derived from them (not the other way around) -- this is what
    # makes the tick values themselves round (multiples of 1, 2, or 5 x a power of ten) while the
    # first and last tick still sit exactly on the panel's edge.
    xticks = _nice_ticks(*raw_xlim, target_n=6)
    xlim = (xticks[0], xticks[-1])
    # Y: the other way around -- nice_ticks' floor/ceil rounds its OWN endpoints out to the next
    # full step, which on top of the padding already in raw_ylim could silently donate up to one
    # entire extra tick step of blank margin (e.g. data topping out at 0.2194 with target 0.02
    # ticks rounds the padded 0.2247 up to 0.24, not 0.22, for ~4x the intended ~8% padding).
    # Keeping raw_ylim as the actual view limits avoids that; yticks is used only to pick which
    # round numbers get labeled, and any that land outside raw_ylim simply aren't drawn.
    yticks = _nice_ticks(*raw_ylim, target_n=6)

    for ri, level in enumerate(_FRONTIER_ROW_LEVELS):
        level_agg = agg_full[agg_full.level == level]

        for ci, d in enumerate(directions):
            ax = axes[ri, ci]
            sub = level_agg[level_agg.direction == d].sort_values("coefficient")
            if zoom:
                orr = cra.find_operating_range(agg_full, level, d, baseline_ndcg, threshold, mode)
                sub = sub[(sub.coefficient >= orr.left_bound) & (sub.coefficient <= orr.right_bound)] if (orr is not None and not orr.empty) else sub.iloc[0:0]
            ax.plot(sub["NFaiRR@10"], sub[metric], "-", color="0.75", lw=1, zorder=1)
            # Per-point opacity, not a single scalar alpha: points near coefficient=0 (the least
            # interesting, closest to unsteered) fade toward MIN_MARKER_ALPHA, points at the
            # sweep's extremes are fully opaque. Requires precomputing RGBA per point (scatter's
            # `alpha` kwarg only accepts one value for the whole collection) and passing that as
            # `c` instead of raw coefficients + cmap/norm.
            rgba = cmap(row_norm[level](sub["coefficient"].to_numpy()))
            frac_from_zero = np.abs(sub["coefficient"].to_numpy()) / (row_max[level] or 1.0)
            rgba[:, 3] = _MIN_MARKER_ALPHA + (1.0 - _MIN_MARKER_ALPHA) * frac_from_zero
            ax.scatter(sub["NFaiRR@10"], sub[metric], c=rgba, s=marker_size, zorder=2, edgecolor="white", linewidth=0.6)
            star = ax.scatter([baseline_nfairr], [baseline_metric], marker="x", s=110, color=BASELINE_MARKER_COLOR,
                               linewidth=2.0, zorder=3, label="Unsteered baseline")

            # Black axis borders (all four spines) -- local to this function; plot_arab_curves
            # keeps the module-wide open-spine style.
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(1.0)
            ax.tick_params(colors="black", labelsize=_FRONTIER_FONTS["tick"])
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels([f"{v:.2f}" for v in xticks], rotation=35, ha="right")
            ax.set_yticklabels([f"{v:.2f}" for v in yticks])
            ax.set_ylim(raw_ylim)

            if ri == 0:
                ax.set_title(_FRONTIER_DIRECTION_TITLES[d], fontsize=_FRONTIER_FONTS["title"], fontweight="bold")
            if ci == 0:
                ax.set_ylabel(metric, fontsize=_FRONTIER_FONTS["axis_label"])
                # Bold, rotated row-group label, in AXES-fraction coordinates (not a fixed-point
                # offset) so it sits at a consistent distance from the axis regardless of figure
                # size -- same technique as the paper's stacked-bar figures
                # (gender_bias_steering/make_paper_stacked_bars.py's row_cfg["label"] text).
                ax.text(-0.55, 0.5, cra.LEVELS[level], transform=ax.transAxes,
                        fontsize=_FRONTIER_FONTS["row_label"], fontweight="bold", ha="center", va="center", rotation=90)
            if ri == 1:
                ax.set_xlabel("NFaiRR@10", fontsize=_FRONTIER_FONTS["axis_label"])
            ax.grid(False)
            # Tick LABELS only on the left column / bottom row (marks/limits stay linked via
            # sharex/sharey regardless). Explicit tick_params rather than ax.label_outer(), which
            # left every panel's labels visible here -- set_xticklabels/set_yticklabels above
            # create fresh Text artists per axes, decoupled from the shared-axis label-visibility
            # bookkeeping label_outer() relies on.
            if ci != 0:
                ax.tick_params(labelleft=False)
            if ri != 1:
                ax.tick_params(labelbottom=False)

    # Reserve fixed margins for the row labels (left), suptitle/legend (top), and -- for a
    # horizontal colorbar -- extra room at the bottom, all BEFORE creating the colorbar below
    # (same idea as make_paper_stacked_bars.py's subplots_adjust call, but order matters here in a
    # way it doesn't there: that figure has no colorbar. subplots_adjust re-applies ALL FOUR
    # margins from the figure's gridspec, including whichever "right"/"bottom" value is already in
    # effect -- calling it AFTER fig.colorbar(..., ax=axes) discarded the margin shrink that call
    # had just made room with, re-expanding the panels over the colorbar. Doing it first means the
    # colorbar below only has to shrink whatever margin is left.
    if horizontal:
        fig.subplots_adjust(top=0.87, left=0.19, wspace=0.28, hspace=0.35, bottom=0.24)
    else:
        fig.subplots_adjust(top=0.84, left=0.19, wspace=0.28)

    # One shared colorbar, canonical range [-1, 1], with TWO label sets on its two sides: the
    # primary axis (left for vertical, bottom for horizontal) shows embed's own raw coefficient
    # values, the twin axis (right / top) shows attn's -- both placed at their row's coefficient /
    # row_max position, which is exactly what row_norm above maps each row's scatter onto, so the
    # tick positions line up with the colors actually plotted.
    canonical_sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-1, vmax=1), cmap=cmap)
    canonical_sm.set_array([])
    embed_ticks_raw = np.linspace(-row_max["embed"], row_max["embed"], 5)
    attn_ticks_raw = np.linspace(-row_max["attn"], row_max["attn"], 5)

    if horizontal:
        cb = fig.colorbar(canonical_sm, ax=axes, orientation="horizontal", shrink=0.7, pad=0.30, aspect=30)
        cb.ax.tick_params(labelsize=_FRONTIER_FONTS["colorbar"])
        cb.ax.xaxis.set_ticks_position("bottom")
        cb.ax.xaxis.set_label_position("bottom")
        cb.set_ticks(attn_ticks_raw / row_max["attn"])
        cb.set_ticklabels([f"{v:g}" for v in attn_ticks_raw])
        cb.ax.set_xlabel("Attention steering strength (\u03b1)", fontsize=_FRONTIER_FONTS["colorbar"])

        cax2 = cb.ax.twiny()
        cax2.set_xlim(cb.ax.get_xlim())
        cax2.set_xticks(embed_ticks_raw / row_max["embed"])
        cax2.set_xticklabels([f"{v:g}" for v in embed_ticks_raw])
        cax2.tick_params(labelsize=_FRONTIER_FONTS["colorbar"])
        cax2.set_xlabel("Embedding steering strength (\u03b1)", fontsize=_FRONTIER_FONTS["colorbar"])
    else:
        cb = fig.colorbar(canonical_sm, ax=axes, shrink=0.85, pad=0.14)
        cb.ax.tick_params(labelsize=_FRONTIER_FONTS["colorbar"])
        cb.ax.yaxis.set_ticks_position("left")
        cb.ax.yaxis.set_label_position("left")
        cb.set_ticks(embed_ticks_raw / row_max["embed"])
        cb.set_ticklabels([f"{v:g}" for v in embed_ticks_raw])
        cb.ax.set_ylabel("Embedding steering strength (\u03b1)", fontsize=_FRONTIER_FONTS["colorbar"])

        cax2 = cb.ax.twinx()
        cax2.set_ylim(cb.ax.get_ylim())
        cax2.set_yticks(attn_ticks_raw / row_max["attn"])
        cax2.set_yticklabels([f"{v:g}" for v in attn_ticks_raw])
        cax2.tick_params(labelsize=_FRONTIER_FONTS["colorbar"])
        cax2.set_ylabel("Attention steering strength (\u03b1)", fontsize=_FRONTIER_FONTS["colorbar"])

    fig.legend(handles=[star], loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=1, frameon=False, fontsize=_FRONTIER_FONTS["legend"])
    suffix = " (operating range)" if zoom else ""
    fig.suptitle(f"{model}{suffix}", fontsize=_FRONTIER_FONTS["suptitle"], fontweight="bold", y=0.99)
    return fig, axes


def plot_arab_curves(
    dataset_key: str, models: list[str], threshold: float, mode: str = "asymmetric",
    arab_variant: str = cra.DEFAULT_ARAB_VARIANT, seeds: list[int] = cra.SEEDS,
):
    """ARaB@10 vs coefficient, one row per model, one column per direction; embed/attn overlaid
    per panel. Faint per-seed lines, bold mean, shaded std band, peak-within-operating-range star
    marker, and a legend-labeled unsteered-baseline reference line."""
    signed_col, _ = cra.arab_col_names(arab_variant)
    directions = list(cra.DIRECTIONS)

    fig, axes = plt.subplots(len(models), len(directions), figsize=(4.2 * len(directions), 3.6 * len(models)), squeeze=False)
    legend_handles = None

    for ri, model in enumerate(models):
        df, agg = _load_agg(dataset_key, model, arab_variant, seeds)

        for ci, d in enumerate(directions):
            ax = axes[ri, ci]
            for level in cra.LEVELS:
                per_seed = df[(df.level == level) & (df.direction == d)]
                for seed in seeds:
                    s = per_seed[per_seed.seed == seed].sort_values("coefficient")
                    if s.empty or signed_col not in s.columns:
                        continue
                    ax.plot(s["coefficient"], s[signed_col], color=LEVEL_COLORS[level], alpha=0.15, lw=0.8)

                level_agg = agg[(agg.level == level) & (agg.direction == d)].sort_values("coefficient")
                if level_agg.empty or "ARaB@10" not in level_agg.columns:
                    continue
                mean, std = level_agg["ARaB@10"], level_agg["ARaB@10_std"].fillna(0)
                ax.plot(level_agg["coefficient"], mean, color=LEVEL_COLORS[level], lw=2, label=cra.LEVELS[level])
                ax.fill_between(level_agg["coefficient"], mean - std, mean + std, color=LEVEL_COLORS[level], alpha=0.15, lw=0)

                baseline_ndcg, _, _ = _model_baseline(dataset_key, model, agg, level, d)
                orr = cra.find_operating_range(agg, level, d, baseline_ndcg, threshold, mode)
                peak = cra.find_arab_peak(level_agg, level, d, orr) if orr is not None else None
                if peak is not None and peak.coefficient is not None:
                    ax.plot(peak.coefficient, peak.arab_signed, marker="*", color=LEVEL_COLORS[level], markersize=14, markeredgecolor="white", markeredgewidth=0.5, zorder=5)
                    ax.annotate(f"{peak.arab_signed:.3f} @ α={peak.coefficient:g}", (peak.coefficient, peak.arab_signed),
                                 textcoords="offset points", xytext=(4, 6), fontsize=7, color=LEVEL_COLORS[level])

            baseline_arab = None
            zp = cra.sweep_zero_point(agg, "embed", d)
            if zp is not None and pd.notna(zp.get("ARaB@10")):
                baseline_arab = float(zp["ARaB@10"])
                ax.axhline(baseline_arab, ls=":", color=BASELINE_COLOR, lw=1.2, label="Unsteered baseline")
            ax.axhline(0, color="0.9", lw=1)
            if ri == 0:
                ax.set_title(cra.DIRECTIONS[d], fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"{model}\nARaB@10 (signed)", fontsize=9)
            if ri == len(models) - 1:
                ax.set_xlabel("Coefficient")
            ax.grid(False)
            if legend_handles is None:
                handles, labels = ax.get_legend_handles_labels()
                seen = dict(zip(labels, handles))
                legend_handles = list(seen.items())

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    if legend_handles:
        fig.legend([h for _, h in legend_handles], [lbl for lbl, _ in legend_handles],
                   loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(legend_handles), frameon=False, fontsize=8)
    fig.suptitle(f"ARaB@10 vs coefficient ({cra.ARAB_SIGN_NOTE})", fontsize=8, y=0.995, wrap=True)
    return fig, axes


def save_summary_table(
    dataset_keys: list[str], models: list[str], direction: str, threshold: float, out_dir: Path,
    arab_variant: str = cra.DEFAULT_ARAB_VARIANT, mode: str = "asymmetric",
):
    """Writes summary_<direction>.md and .tex (rows include a Model column), and returns the raw
    numeric DataFrame."""
    display_df, raw_df = cra.build_summary_table_multi_model(dataset_keys, models, direction, threshold, arab_variant, mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"summary_{direction}.md").write_text(cra.to_markdown_table(display_df))
    (out_dir / f"summary_{direction}.tex").write_text(
        cra.to_latex_booktabs(display_df, caption=f"{cra.DIRECTIONS[direction]}", label=f"tab:{direction}")
    )
    return raw_df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=list(cra.DATASETS), default="msmarco_fair")
    p.add_argument("--models", default=",".join(cra.MAIN_TEXT_MODELS), help="Comma-separated model short names.")
    p.add_argument("--threshold", type=float, default=0.02)
    p.add_argument("--mode", choices=["asymmetric", "symmetric"], default="asymmetric")
    p.add_argument("--arab-variant", choices=list(cra.ARAB_VARIANTS), default=cra.DEFAULT_ARAB_VARIANT)
    p.add_argument("--frontier-metrics", default="nDCG@10", choices=["nDCG@10", "MRR@10", "both"],
                    help="Utility metric(s) for the frontier plot's y-axis. 'both' writes one figure per metric.")
    p.add_argument("--marker-size", type=float, default=60, help="Frontier scatter marker size (matplotlib's `s`); smaller values show overlapping/dense points better.")
    p.add_argument("--colormap", default="prgn_no_white", help="Frontier coefficient colormap: 'prgn_no_white' (default, paper palette) or any matplotlib diverging colormap name, e.g. 'coolwarm'.")
    p.add_argument("--out-dir", default="paper_figures", help="Figures land in <out-dir>/<dataset>/.")
    p.add_argument("--format", choices=["png", "jpg"], default="png", help="Raster format (png is lossless; jpg is smaller but can fuzz thin lines/text).")
    args = p.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ext = args.format
    frontier_metrics = list(_UTILITY_METRICS) if args.frontier_metrics == "both" else [args.frontier_metrics]

    out_dir = Path(args.out_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        for metric in frontier_metrics:
            # "nDCG@10" keeps the original bare filename (no behavior change for existing callers);
            # any other metric gets its own suffix so multiple metrics' figures coexist in out_dir.
            metric_suffix = "" if metric == "nDCG@10" else f"_{metric.split('@')[0].lower()}"
            colormap_suffix = "" if args.colormap == "prgn_no_white" else f"_{args.colormap.lower()}"
            for zoom in (False, True):
                for orientation in ("vertical", "horizontal"):
                    fig, _ = plot_frontier(args.dataset, model, args.threshold, args.mode, args.arab_variant,
                                            zoom=zoom, colorbar_orientation=orientation, metric=metric,
                                            marker_size=args.marker_size, colormap=args.colormap)
                    suffix = metric_suffix + colormap_suffix + ("_zoom" if zoom else "") + ("_horizontal_colorbar" if orientation == "horizontal" else "")
                    fig.savefig(out_dir / f"frontier_{model}{suffix}.{ext}", bbox_inches="tight")
                    plt.close(fig)

    fig, _ = plot_arab_curves(args.dataset, models, args.threshold, args.mode, args.arab_variant)
    fig.savefig(out_dir / f"arab_curves.{ext}", bbox_inches="tight")
    plt.close(fig)

    for direction in cra.DIRECTIONS:
        raw = save_summary_table([args.dataset], models, direction, args.threshold, out_dir, args.arab_variant, args.mode)
        print(f"--- {direction} ---")
        print(raw.to_string())
    print(f"Wrote figures + summary tables to {out_dir}/")


if __name__ == "__main__":
    main()

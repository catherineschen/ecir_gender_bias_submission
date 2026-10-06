import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json
from typing import Optional, Sequence


NAME_MAP = {
    "sentence-transformers/msmarco-distilbert-base-tas-b": "DistilBERT-TAS-B_msmarco",
    "sentence-transformers/msmarco-distilbert-dot-v5": "DistilBERT_msmarco",
    "sentence-transformers/multi-qa-distilbert-dot-v1": "DistilBERT_mqa",
    "sentence-transformers/multi-qa-MiniLM-L6-dot-v1": "MiniLM_mqa",
    "sentence-transformers/msmarco-bert-base-dot-v5": "BERT_msmarco",
}
COMP_ORDER = ["MvF", "MvN", "FvN"]
BASELINE_STYLE = {
    "Baseline": dict(color="blue", linestyle="-"),
    "Full Ablation": dict(color="red", linestyle="-"),
}


def _left_out_head_to_label(x) -> str:
    """
    Convert JSON string like "[10, 6]" into "10.6".
    Falls back gracefully if parsing fails.
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    try:
        arr = json.loads(s)
        if isinstance(arr, (list, tuple)) and len(arr) == 2:
            return f"{int(arr[0])}.{int(arr[1])}"
    except Exception:
        pass
    # fallback: keep original string
    return s


def ecdf(x: np.ndarray):
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def add_gap_columns(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    df = df.copy()
    if "gap_abs" not in df.columns:
        if "score_diff" not in df.columns:
            raise ValueError("Need either gap_abs or score_diff in CSV.")
        df["gap_abs"] = df["score_diff"].abs()

    # Percent gap (symmetric normalization by average magnitude)
    if "score_a" in df.columns and "score_b" in df.columns:
        denom = (df["score_a"].abs() + df["score_b"].abs()) / 2.0
        df["gap_pct"] = (df["gap_abs"] / (denom + eps)) * 100.0
    else:
        df["gap_pct"] = np.nan  # filled only if available
    return df


def load_and_combine(
    orig_csv: str,
    abl_csv: str,
    eps: float = 1e-8,
    *,
    loo_csv: str | None = None,
    loo_heads: list[str] | None = None,
) -> pd.DataFrame:
    orig = pd.read_csv(orig_csv)
    abl  = pd.read_csv(abl_csv)
    orig["condition"] = "Baseline"
    abl["condition"]  = "Full Ablation"

    frames = [orig, abl]

    if loo_csv is not None:
        loo = pd.read_csv(loo_csv).copy()
        if "left_out_head" not in loo.columns:
            raise ValueError("loo_csv must contain 'left_out_head'.")

        # Convert "[10, 6]" -> "10.6"
        loo["head_label"] = loo["left_out_head"].apply(_left_out_head_to_label)

        # Optional filter only if user provided it
        if loo_heads is not None and len(loo_heads) > 0:
            keep = set(str(h).strip() for h in loo_heads)
            loo = loo[loo["head_label"].isin(keep)].copy()

        loo["condition"] = loo["head_label"].apply(lambda h: f"LOO {h}")
        loo = loo.drop(columns=["head_label"], errors="ignore")

        frames.append(loo)

    df = pd.concat(frames, ignore_index=True)
    df = add_gap_columns(df, eps=eps)
    return df



def filter_top_fraction(
    df: pd.DataFrame,
    value_col: str,
    top_frac: float = 0.30,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Keep only the top `top_frac` fraction of rows by `value_col`.
    If group_cols is provided, do this *within each group*.
    """
    if not (0 < top_frac < 1):
        raise ValueError("top_frac must be between 0 and 1 (e.g., 0.30 for top 30%).")

    q = 1.0 - top_frac  # e.g., 0.70 cutoff for top 30%

    df = df.copy()

    if group_cols is None or len(group_cols) == 0:
        cutoff = df[value_col].quantile(q)
        return df[df[value_col] >= cutoff].copy()

    # group-wise cutoff (preferred)
    cutoffs = (
        df.groupby(group_cols)[value_col]
          .quantile(q)
          .rename("cutoff")
          .reset_index()
    )
    df2 = df.merge(cutoffs, on=group_cols, how="left")
    out = df2[df2[value_col] >= df2["cutoff"]].drop(columns=["cutoff"])
    return out.copy()

def filter_top_fraction_by_baseline(
    df: pd.DataFrame,
    value_col: str,
    top_frac: float = 0.30,
    *,
    baseline_condition: str = "Baseline",
    key_cols: list[str] = ["model", "comparison", "qid", "doc_group_id"],
    within_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Keep the top `top_frac` fraction of *baseline* (Original) rows by `value_col`,
    then apply that selection (by keys) to all conditions.

    - within_group_cols: columns to compute the cutoff within (e.g., ["model","comparison"]).
      If None, compute cutoff globally over all baseline rows.
    """
    if not (0 < top_frac < 1):
        raise ValueError("top_frac must be between 0 and 1 (e.g., 0.30 for top 30%).")

    q = 1.0 - top_frac

    df = df.copy()

    base = df[df["condition"] == baseline_condition].copy()
    if base.empty:
        raise ValueError(f"No rows found for baseline_condition='{baseline_condition}'.")

    # safety: baseline must have the value col
    if value_col not in base.columns or base[value_col].isna().all():
        raise ValueError(f"Baseline rows have missing/empty value_col='{value_col}'.")

    # compute which baseline instances are in the top fraction
    if within_group_cols is None or len(within_group_cols) == 0:
        cutoff = base[value_col].quantile(q)
        base_keep = base[base[value_col] >= cutoff]
    else:
        cutoffs = (
            base.groupby(within_group_cols)[value_col]
                .quantile(q)
                .rename("cutoff")
                .reset_index()
        )
        base2 = base.merge(cutoffs, on=within_group_cols, how="left")
        base_keep = base2[base2[value_col] >= base2["cutoff"]]

    keep_keys = base_keep[key_cols].drop_duplicates()

    # apply to all conditions
    out = df.merge(keep_keys, on=key_cols, how="inner")
    return out


def filter_out_neutral(
    df: pd.DataFrame,
    group_ids_to_filter: list,
):
    neutral_comparisons = ["FvN", "MvN"]
    return df[
        ~(
            df["comparison"].isin(neutral_comparisons)
            & df["doc_group_id"].isin(group_ids_to_filter)
        )
    ]

KEY_COLS = ["model", "comparison", "doc_group_id", "qid"]  # join keys for pairing

def parse_left_out_head_to_str(x: str) -> str:
    """'[10, 6]' -> '10.6'"""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    try:
        arr = json.loads(s)
        if isinstance(arr, (list, tuple)) and len(arr) == 2:
            return f"{int(arr[0])}.{int(arr[1])}"
    except Exception:
        pass
    # fallback: best-effort digit parse
    digits = [t for t in "".join(ch if (ch.isdigit() or ch == " ") else " " for ch in s).split() if t]
    if len(digits) >= 2:
        return f"{int(digits[0])}.{int(digits[1])}"
    return s


def choose_topk_loo_heads_by_rebound(df_full: pd.DataFrame, df_loo: pd.DataFrame, gap_col: str, k: int) -> list[str]:
    """
    Pick top-k LOO heads by mean rebound vs full ablation:
        rebound_i(h) = gap_abs(LOO(h)) - gap_abs(full)
    averaged over matched samples within model×comparison.
    """
    # index for pairing
    full = df_full.set_index(KEY_COLS)[[gap_col]].rename(columns={gap_col: "gap_full"})
    loo = df_loo.copy()
    loo["head"] = loo["left_out_head"].apply(parse_left_out_head_to_str)
    loo = loo.set_index(KEY_COLS + ["head"])[[gap_col]].rename(columns={gap_col: "gap_loo"})

    joined = loo.join(full, how="inner")
    joined["rebound"] = joined["gap_loo"] - joined["gap_full"]

    # average rebound per head across all models/comparisons/samples
    # (simple heuristic; good enough for picking a few heads to overlay)
    head_scores = (
        joined.reset_index()
        .groupby("head")["rebound"]
        .mean()
        .sort_values(ascending=False)
    )

    return head_scores.head(k).index.tolist()


def plot_facets(
    df: pd.DataFrame,
    model_col: str,
    plot_type: str,
    gap_col: str,
    hist_mode: str,
    x_mode: str,
    bins: int,
    # clip_pct: float | None,
    xmax: float | None,
):
    """
    Facet plots of gap distributions per (model, comparison).

    Backward compatible:
      - If df contains only conditions {"Baseline","Ablated"}, behavior matches old code.
    Extended:
      - If df also contains conditions like "LOO 11.3", these are overlaid as additional curves/hist lines.
      - Condition plotting order: Original, Ablated, then LOO heads sorted by (layer, head).
    """
    # choose x values label
    if gap_col == "gap_abs":
        x_label = r"$|\Delta \mathrm{score}|$"
    elif gap_col == "gap_pct":
        x_label = r"% $|\Delta \mathrm{score}|$"
    else:
        x_label = gap_col

    # optional clipping
    # if clip_pct is not None:
    #     hi = np.nanpercentile(df[gap_col].to_numpy(), clip_pct)
    #     df = df.copy()
    #     df[gap_col] = df[gap_col].clip(upper=hi)

    models = sorted(df[model_col].unique().tolist())
    comps = [c for c in COMP_ORDER if c in df["comparison"].unique().tolist()]

    nrows, ncols = len(models), len(comps)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 2.8 * nrows),
        sharex=(x_mode == "shared"),
        sharey=(plot_type == "ecdf"),
    )

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    # shared bins if x_mode=shared and plot_type=hist
    if plot_type == "hist" and x_mode == "shared":
        all_vals = df[gap_col].dropna().to_numpy()
        x_max = float(np.nanmax(all_vals)) if xmax is None else float(xmax)
        bin_edges_shared = np.linspace(0.0, x_max, bins + 1)
    else:
        bin_edges_shared = None

    def _loo_sort_key(cond: str):
        # cond like "LOO 10.6"
        try:
            head = cond.split(" ", 1)[1]
            layer_s, head_s = head.split(".", 1)
            return (int(layer_s), int(head_s))
        except Exception:
            return (10**9, 10**9)

    for i, m in enumerate(models):
        for j, c in enumerate(comps):
            ax = axes[i, j]
            sub = df[(df[model_col] == m) & (df["comparison"] == c)]

            # Determine condition order for this panel
            conds = sub["condition"].dropna().unique().tolist()
            ordered = []
            if "Baseline" in conds:
                ordered.append("Baseline")
            if "Full Ablation" in conds:
                ordered.append("Full Ablation")

            loo_conds = [x for x in conds if isinstance(x, str) and x.startswith("LOO ")]
            ordered += sorted([x for x in loo_conds if x not in ordered], key=_loo_sort_key)

            # Any other conditions (just in case)
            ordered += [x for x in conds if x not in ordered]

            # Collect data arrays
            vals = {}
            for cond in ordered:
                arr = sub[sub["condition"] == cond][gap_col].dropna().to_numpy()
                if len(arr) > 0:
                    vals[cond] = arr

            if not vals:
                ax.axis("off")
                continue

            if plot_type == "ecdf":
                for cond in ordered:
                    if cond in vals:
                        xs, ys = ecdf(vals[cond])
                        style = BASELINE_STYLE.get(cond, {})
                        # lw = 2.6 if cond in BASELINE_STYLE else 1.6

                        ax.plot(
                            xs,
                            ys,
                            # linewidth=lw,
                            label=cond,
                            **style,
                        )


                ax.set_ylim(0, 1)
                ax.set_ylabel("ECDF")

            elif plot_type == "hist":
                # pick bins
                if x_mode == "shared":
                    bin_edges = bin_edges_shared
                    if xmax is not None:
                        ax.set_xlim(0.0, float(xmax))
                else:
                    all_panel_vals = np.concatenate(list(vals.values()))
                    x_max = float(np.nanmax(all_panel_vals))
                    if xmax is not None:
                        x_max = min(x_max, float(xmax))
                        ax.set_xlim(0.0, float(xmax))
                    bin_edges = np.linspace(0.0, x_max, bins + 1)

                density = (hist_mode == "density")
                for cond in ordered:
                    if cond in vals:
                        style = BASELINE_STYLE.get(cond, {})
                        lw = 2.6 if cond in BASELINE_STYLE else 1.6

                        ax.hist(
                            vals[cond],
                            bins=bin_edges,
                            density=density,
                            histtype="step",
                            linewidth=lw,
                            label=cond,
                            **style,
                        )


                ax.set_ylabel("Density" if density else "Count")

            else:
                raise ValueError(f"Unknown plot_type: {plot_type}")

            if i == 0:
                ax.set_title(c)
            if j == 0:
                ax.set_ylabel(f"{m}\n{ax.get_ylabel()}")

            ax.set_xlabel(x_label)

            # # legend once (top-right panel)
            # if i == 0 and j == ncols - 1:
            #     ax.legend(frameon=False)
            if j == ncols - 1:
                ax.legend(frameon=False, fontsize=8)


    fig.tight_layout()
    return fig

def plot_loo_rebound_facets(
    df: pd.DataFrame,
    model_col: str,
    gap_col: str,
    plot_type: str = "ecdf",             # "ecdf" or "hist"
    hist_mode: str = "density",          # "counts" or "density"
    x_mode: str = "local",               # "local" or "shared" (hist only)
    bins: int = 40,
    clip_pct: float | None = None,
    xmax: float | None = None,
    head_labels: list[str] | None = None,  # e.g., ["11.3","11.11"]; if None, plot all heads present
    key_cols: list[str] = ["qid", "doc_group_id"],  # pairing keys inside each model×comparison
):
    """
    Plots distribution of per-sample rebound:
        rebound_i(h) = gap_abs(LOO h) - gap_abs(Full Ablation)
    faceted by model × comparison, with one curve per head.

    Requires df to contain:
      - condition == "Full Ablation" for full ablation
      - condition like "LOO 11.3" for leave-one-out heads
      - gap_col (e.g., gap_abs)
    """
    # label
    if gap_col == "gap_abs":
        x_label = r"$|\Delta|_{\mathrm{LOO}} - |\Delta|_{\mathrm{Full}}$"
    elif gap_col == "gap_pct":
        x_label = r"\% $|\Delta|_{\mathrm{LOO}} - |\Delta|_{\mathrm{Full}}$"
    else:
        x_label = f"{gap_col} (LOO - Full)"

    df = df.copy()

    # Filter to only Full + LOO
    is_full = df["condition"] == "Full Ablation"
    is_loo = df["condition"].astype(str).str.startswith("LOO ")
    subdf = df[is_full | is_loo].copy()

    if subdf.empty:
        raise ValueError("No rows with condition 'Ablated' and/or 'LOO ...' found in df.")

    # extract head label from condition
    subdf["head"] = subdf["condition"].astype(str).str.replace(r"^LOO\s+", "", regex=True)
    subdf.loc[subdf["condition"] == "Full Ablation", "head"] = "__FULL__"

    # optional head filter
    if head_labels is not None:
        keep = set(str(h) for h in head_labels)
        subdf = subdf[(subdf["head"] == "__FULL__") | (subdf["head"].isin(keep))].copy()

    # optional clipping on rebound later (after compute)
    models = sorted(subdf[model_col].unique().tolist())
    comps = [c for c in COMP_ORDER if c in subdf["comparison"].unique().tolist()]

    nrows, ncols = len(models), len(comps)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 2.8 * nrows),
        sharex=(x_mode == "shared"),
        sharey=(plot_type == "ecdf"),
    )
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    # shared bins for hist
    if plot_type == "hist" and x_mode == "shared":
        # compute all rebounds first (global), then bins
        all_rebounds = []
        for m in models:
            for c in comps:
                panel = subdf[(subdf[model_col] == m) & (subdf["comparison"] == c)]
                if panel.empty:
                    continue
                full = panel[panel["head"] == "__FULL__"].set_index(key_cols)[gap_col]
                for h in panel["head"].unique():
                    if h == "__FULL__":
                        continue
                    loo = panel[panel["head"] == h].set_index(key_cols)[gap_col]
                    joined = pd.concat([full.rename("full"), loo.rename("loo")], axis=1).dropna()
                    if not joined.empty:
                        all_rebounds.append((joined["loo"] - joined["full"]).to_numpy())
        if all_rebounds:
            all_vals = np.concatenate(all_rebounds)
            if clip_pct is not None:
                hi = np.nanpercentile(all_vals, clip_pct)
                all_vals = np.clip(all_vals, None, hi)
            x_max = float(np.nanmax(all_vals)) if xmax is None else float(xmax)
            x_min = float(np.nanmin(all_vals))
            bin_edges_shared = np.linspace(x_min, x_max, bins + 1)
        else:
            bin_edges_shared = None
    else:
        bin_edges_shared = None

    def _loo_sort_key(h: str):
        # h like "10.6"
        try:
            layer_s, head_s = h.split(".", 1)
            return (int(layer_s), int(head_s))
        except Exception:
            return (10**9, 10**9)

    for i, m in enumerate(models):
        for j, c in enumerate(comps):
            ax = axes[i, j]
            panel = subdf[(subdf[model_col] == m) & (subdf["comparison"] == c)]
            if panel.empty:
                ax.axis("off")
                continue

            # full gaps (index by keys)
            full = panel[panel["head"] == "__FULL__"].set_index(key_cols)[gap_col]

            # each head -> rebound array
            head_list = sorted([h for h in panel["head"].unique() if h != "__FULL__"], key=_loo_sort_key)
            rebounds = {}
            for h in head_list:
                loo = panel[panel["head"] == h].set_index(key_cols)[gap_col]
                joined = pd.concat([full.rename("full"), loo.rename("loo")], axis=1).dropna()
                if joined.empty:
                    continue
                r = (joined["loo"] - joined["full"]).to_numpy()
                if clip_pct is not None:
                    hi = np.nanpercentile(r, clip_pct)
                    r = np.clip(r, None, hi)
                rebounds[h] = r

            if not rebounds:
                ax.axis("off")
                continue

            if plot_type == "ecdf":
                for h in head_list:
                    if h in rebounds:
                        xs, ys = ecdf(rebounds[h])
                        ax.plot(xs, ys, linewidth=1.6, label=f"LOO {h}")
                ax.set_ylim(0, 1)
                ax.set_ylabel("ECDF")

            elif plot_type == "hist":
                if x_mode == "shared" and bin_edges_shared is not None:
                    bin_edges = bin_edges_shared
                    if xmax is not None:
                        ax.set_xlim(None, float(xmax))
                else:
                    all_vals = np.concatenate(list(rebounds.values()))
                    if clip_pct is not None:
                        hi = np.nanpercentile(all_vals, clip_pct)
                        all_vals = np.clip(all_vals, None, hi)
                    x_max = float(np.nanmax(all_vals)) if xmax is None else float(xmax)
                    x_min = float(np.nanmin(all_vals))
                    bin_edges = np.linspace(x_min, x_max, bins + 1)

                density = (hist_mode == "density")
                for h in head_list:
                    if h in rebounds:
                        ax.hist(
                            rebounds[h],
                            bins=bin_edges,
                            density=density,
                            histtype="step",
                            linewidth=1.6,
                            label=f"LOO {h}",
                        )
                ax.set_ylabel("Density" if density else "Count")

            else:
                raise ValueError(f"Unknown plot_type: {plot_type}")

            if i == 0:
                ax.set_title(c)
            if j == 0:
                ax.set_ylabel(f"{m}\n{ax.get_ylabel()}")

            ax.set_xlabel(x_label)

            # legend per row (so head sets can differ by model)
            if j == ncols - 1:
                ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    return fig



def main():
    p = argparse.ArgumentParser(description="Facet plots of score-gap distributions (Original vs Ablated).")
    p.add_argument("--orig_csv", type=str, required=True)
    p.add_argument("--abl_csv", type=str, required=True)
    p.add_argument("--out_png", type=str, required=True)

    p.add_argument("--plot_type", choices=["ecdf", "hist"], default="ecdf")
    p.add_argument("--gap_metric", choices=["abs", "pct"], default="pct",
                   help="abs uses gap_abs/|score_diff|; pct uses percent gap based on score_a/score_b.")
    p.add_argument("--hist_mode", choices=["counts", "density"], default="counts")
    p.add_argument("--x_mode", choices=["local", "shared"], default="local",
                   help="local = each subplot gets its own x-range/bins; shared = all subplots share x-range/bins (hist only).")
    p.add_argument("--bins", type=int, default=30)
    p.add_argument("--clip_pct", type=float, default=None,
                   help="Clip gaps at this percentile for readability, e.g., 99.")
    p.add_argument("--xmax", type=float, default=None,
                   help="Optional hard cap on x-axis max (applies to hist; for local x it caps per panel).")
    p.add_argument("--top_frac", type=float, default=None,
                   help="If set (e.g., 0.30), filter to top fraction of largest gaps before plotting.")
    p.add_argument("--top_within_group", action="store_true",
                   help="If set, top-fraction filtering is done within each model×comparison×condition.")
    p.add_argument("--use_short_names", action="store_true")
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--filter_out_neutral", default=False, action="store_true")
    p.add_argument("--loo_csv", type=str, default=None,
                   help="Optional CSV of leave-one-out runs (contains left_out_head).")
    p.add_argument("--loo_heads", type=str, nargs="*", default=None,
                   help="Optional list of head labels like: 11.3 11.11 10.6 (overlaid as extra curves).")
    p.add_argument("--loo_topk", type=int, default=None,
                   help="If set, automatically select top-k LOO heads by mean rebound vs full ablation.")
    p.add_argument("--plot_loo_rebound", action="store_true", default=False)


    args = p.parse_args()

    df = load_and_combine(
        args.orig_csv,
        args.abl_csv,
        eps=args.eps,
        loo_csv=args.loo_csv,
        loo_heads=args.loo_heads,
        # loo_topk=args.loo_topk,
    )


    if args.gap_metric == "abs":
        gap_col = "gap_abs"
    elif args.gap_metric == "pct":
        gap_col = "gap_pct"
    else:
        raise ValueError(f"Unknown gap_metric: {args.gap_metric}")

    # safety check (especially for pct)
    if gap_col not in df.columns or df[gap_col].isna().all():
        raise ValueError(
            f"{gap_col} is missing/empty. "
            "If you used --gap_metric pct, make sure score_a and score_b exist."
        )

    model_col = "model"
    if args.use_short_names:
        df["model_short"] = df["model"].map(NAME_MAP).fillna(df["model"])
        model_col = "model_short"

    if args.top_frac is not None:
        within = None
        if args.top_within_group:
            within = [model_col, "comparison"]  # IMPORTANT: no "condition"

        df = filter_top_fraction_by_baseline(
            df,
            value_col=gap_col,
            top_frac=args.top_frac,
            baseline_condition="Baseline",
            key_cols=[model_col, "comparison", "qid", "doc_group_id"],
            within_group_cols=within,
        )


    if args.filter_out_neutral:
        df = filter_out_neutral(
            df,
            group_ids_to_filter=[12,30,31,34,35,38,39,50,51,62,63,68,69,184,185,188,189,192,193,194,195],
        )

    if args.plot_loo_rebound:
        fig = plot_loo_rebound_facets(
            df=df,
            model_col=model_col,
            gap_col="gap_abs",
            plot_type="ecdf",
            head_labels=None,  # or ["11.3","11.11"]
        )
    else:

        # If shared x_mode requested for ecdf, it's fine (sharex True/False doesn't matter much),
        # but shared bins only apply to hist.
        fig = plot_facets(
            df=df,
            model_col=model_col,
            plot_type=args.plot_type,
            gap_col=gap_col,
            hist_mode=args.hist_mode,
            x_mode=args.x_mode,
            bins=args.bins,
            # clip_pct=args.clip_pct,
            xmax=args.xmax,
        )

    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


# python plot_ablation_results.py --orig_csv results_baseline/baseline_score_diffs_all_dot_models_with_abs_gap.csv --abl_csv results_ablation/mean/ablated_score_diffs_all_dot_models.csv --out_png result_figures_mechir_bi/ablation/mean/reduction_gaps_ecdf.png --plot_type ecdf --use_short_names --gap_metric abs
# python plot_ablation_results.py --orig_csv results_baseline/baseline_score_diffs_all_dot_models_with_abs_gap.csv --abl_csv results_ablation/mean/ablated_score_diffs_all_dot_models.csv --out_png result_figures_mechir_bi/ablation/mean/reduction_gaps_ecdf_tail30.png --plot_type ecdf --use_short_names --gap_metric abs --top_frac 0.30 --top_within_group
# python plot_ablation_results.py --orig_csv results_baseline/baseline_score_diffs_all_dot_models_with_abs_gap.csv --abl_csv results_ablation/mean/ablated_score_diffs_all_dot_models.csv --out_png result_figures_mechir_bi/ablation/mean/reduction_gaps_ecdf_tail30_filter_out_neutral.png --plot_type ecdf --use_short_names --gap_metric abs --top_frac 0.30 --top_within_group --filter_out_neutral
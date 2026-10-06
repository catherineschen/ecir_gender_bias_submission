"""Core, Streamlit-free analysis functions for the ECIR camera-ready steering dashboard.

Pure pandas/numpy over the same sweep_results.csv / arab_results.csv / baseline TSVs that
steering_dashboard.py already reads. ARaB's sign convention is unchanged from that file:
POSITIVE ARaB@10 MEANS MALE-LEANING (step3's male-minus-female convention -- see
compute_arab_from_runs.py's module docstring, lines 20-31). Deliberately has no Streamlit or
plotting import so the exact same functions back both the interactive dashboard
(camera_ready_dashboard.py) and the static matplotlib paper figures (camera_ready_figures.py) --
nothing about "what the operating range is" or "what goes in the summary table" should be
re-derived differently in the two places.

Directory / file layout this reads (all under experiments/):
    steering_heldout_final[_seed{29,59,104,996}][_trecdl19]/<model_short_name>/
        sweep_results.csv   -- model, level, direction, coefficient, MRR@10, nDCG@10, Recall@10, NFaiRR@10
        arab_results.csv    -- model, level, direction, coefficient, n_queries, ARaB@10_{tc,tf,bool}, |ARaB|@10_{tc,tf,bool}
    baseline/all_baselines_all_seeds.tsv
        dataset, model, pool, seed, MRR@10, nDCG@10, Recall@10, NFaiRR@{5,10,20,50}
        -- MS MARCO Fair rows are seed in {42,29,59,104,996,"mean","std"}; TREC DL19 rows have
        seed "NA" (a single row -- TREC DL19 is a fixed 30-query zero-shot eval, unaffected by
        the construction-query resampling seed, per the project's own steering-sweep convention).

No suffix = seed 42 (steering_heldout_final's run_config.json has "seed": 42); _seed{n} dirs hold
the other 4 seeds. This mirrors steering_dashboard.py's find_runs() regex exactly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
BASELINE_DIR = EXPERIMENTS_DIR / "baseline"
ALL_BASELINES_TSV = BASELINE_DIR / "all_baselines_all_seeds.tsv"

SUPERSEDED_SUFFIX = "_v0"

SEEDS = [42, 29, 59, 104, 996]


def _seed_suffix(seed: int) -> str:
    return "" if seed == 42 else f"_seed{seed}"


DATASETS = {
    "msmarco_fair": {
        "label": "MS MARCO Fair (heldout)",
        "sweep_prefix": "steering_heldout_final",
        "baseline_key": "msmarco_fair_heldout",
        "seed_varies_baseline": True,
    },
    "trecdl19_fair": {
        "label": "TREC DL19 Fair",
        "sweep_prefix": "steering_heldout_final_trecdl19",
        "baseline_key": "trecdl19_fair_all30",
        "seed_varies_baseline": False,
    },
}

DIRECTIONS = {"mf": "Male − Female", "mn": "Male − Neutral", "fn": "Female − Neutral"}
LEVELS = {"embed": "Embedding", "attn": "Attention"}
ARAB_VARIANTS = {"tf": "tf (log term freq)", "tc": "tc (term count)", "bool": "bool (boolean)"}
DEFAULT_ARAB_VARIANT = "tf"

ARAB_SIGN_NOTE = (
    "ARaB@10 > 0 = male-leaning ranking, ARaB@10 < 0 = female-leaning ranking "
    "(step3's male-minus-female convention; see compute_arab_from_runs.py module docstring)."
)

METRIC_COLS = ["MRR@10", "nDCG@10", "Recall@10", "NFaiRR@10"]

# The sweep's own model short_name -> the full HF id that all_baselines_all_seeds.tsv keys on.
# Explicit, not string-derived: one of the five genuinely diverges. The swept TAS-B checkpoint is
# sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco (see its run_config.json), but the only
# TAS-B baseline ever computed is for a DIFFERENT checkpoint, sentence-transformers/
# msmarco-distilbert-base-tas-b. That mismatch is real project state, not a bug here --
# baseline_row() returns found=False for it and the diagnostics panel surfaces it as a flag
# rather than silently pairing the sweep with the wrong baseline.
MODEL_TO_BASELINE_ID = {
    "msmarco-distilbert-dot-v5": "sentence-transformers/msmarco-distilbert-dot-v5",
    "msmarco-bert-base-dot-v5": "sentence-transformers/msmarco-bert-base-dot-v5",
    "multi-qa-distilbert-dot-v1": "sentence-transformers/multi-qa-distilbert-dot-v1",
    "multi-qa-MiniLM-L6-dot-v1": "sentence-transformers/multi-qa-MiniLM-L6-dot-v1",
    "distilbert-dot-tas_b-b256-msmarco": None,
}

# The two models slated for the ECIR paper's main text (both "distilbert" checkpoints, hence
# "both distilbert models" in the brief) -- used only as the dashboard's default model
# selection; every function above still takes an explicit model argument.
MAIN_TEXT_MODELS = ["msmarco-distilbert-dot-v5", "multi-qa-distilbert-dot-v1"]


def sweep_dir(dataset_key: str, seed: int) -> Path:
    prefix = DATASETS[dataset_key]["sweep_prefix"]
    return EXPERIMENTS_DIR / f"{prefix}{_seed_suffix(seed)}"


def discover_models(dataset_key: str, seed: int = 42) -> list[str]:
    d = sweep_dir(dataset_key, seed)
    if not d.is_dir():
        return []
    out = []
    for sub in sorted(d.iterdir()):
        if sub.is_dir() and (sub / "sweep_results.csv").exists():
            out.append(sub.name)
    return out


def arab_col_names(variant: str) -> tuple[str, str]:
    return f"ARaB@10_{variant}", f"|ARaB|@10_{variant}"


def load_seed_run(dataset_key: str, model: str, seed: int) -> pd.DataFrame | None:
    """Merged sweep_results.csv + arab_results.csv for one (dataset, model, seed).

    Mirrors steering_dashboard.py's load_run(): left-merge on (level, direction, coefficient
    rounded to 4dp) so a sweep with no arab_results.csv yet still loads (ARaB columns come back
    all-NaN instead of raising).
    """
    run_dir = sweep_dir(dataset_key, seed) / model
    sweep_csv = run_dir / "sweep_results.csv"
    if not sweep_csv.exists():
        return None
    df = pd.read_csv(sweep_csv)
    df["_coef_key"] = df["coefficient"].round(4)

    arab_csv = run_dir / "arab_results.csv"
    if arab_csv.exists():
        adf = pd.read_csv(arab_csv)
        adf["_coef_key"] = adf["coefficient"].round(4)
        keep = ["level", "direction", "_coef_key"] + [
            c for c in adf.columns if c.startswith("ARaB@10_") or c.startswith("|ARaB|@10_")
        ]
        df = df.merge(adf[keep], on=["level", "direction", "_coef_key"], how="left")

    df = df.drop(columns=["_coef_key"])
    df["dataset"] = dataset_key
    df["model"] = model
    df["seed"] = seed
    return df


def load_all_seeds(dataset_key: str, model: str, seeds: list[int] = SEEDS) -> tuple[pd.DataFrame, list[int]]:
    """Concatenate every available seed's run. Returns (df, seeds_actually_found)."""
    frames, found = [], []
    for seed in seeds:
        df = load_seed_run(dataset_key, model, seed)
        if df is not None:
            frames.append(df)
            found.append(seed)
    if not frames:
        return pd.DataFrame(), []
    return pd.concat(frames, ignore_index=True), found


def aggregate_across_seeds(df: pd.DataFrame, arab_variant: str = DEFAULT_ARAB_VARIANT) -> pd.DataFrame:
    """Group by (level, direction, coefficient) -> mean/std/n over seeds.

    Output columns are named "<metric>" (mean) and "<metric>_std"; ARaB's variant-specific
    columns are renamed to the generic "ARaB@10" / "|ARaB|@10" here since the caller picked the
    variant already -- downstream code never needs to know which variant was chosen.
    """
    if df.empty:
        return df
    signed_col, mag_col = arab_col_names(arab_variant)
    metrics = [c for c in METRIC_COLS if c in df.columns]
    arab_metrics = [c for c in (signed_col, mag_col) if c in df.columns]
    all_metrics = metrics + arab_metrics

    grouped = df.groupby(["level", "direction", "coefficient"], as_index=False)
    agg = grouped[all_metrics].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    agg = agg.reset_index() if "level" not in agg.columns else agg
    keys = df.groupby(["level", "direction", "coefficient"], as_index=False).size().drop(columns="size")
    out = pd.concat([keys, agg], axis=1) if "level" not in agg.columns else agg

    rename = {}
    for m in all_metrics:
        rename[f"{m}_mean"] = m
        rename[f"{m}_std"] = f"{m}_std"
        rename[f"{m}_count"] = "n_seeds"
    out = out.rename(columns=rename)
    if signed_col in out.columns:
        out = out.rename(columns={signed_col: "ARaB@10", f"{signed_col}_std": "ARaB@10_std"})
    if mag_col in out.columns:
        out = out.rename(columns={mag_col: "|ARaB|@10", f"{mag_col}_std": "|ARaB|@10_std"})
    # n_seeds got duplicated once per metric by the rename above; collapse to one column.
    n_seed_cols = [c for c in out.columns if c == "n_seeds"]
    if len(n_seed_cols) > 1:
        out = out.loc[:, ~out.columns.duplicated()]
    return out.sort_values(["level", "direction", "coefficient"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

_baselines_cache: pd.DataFrame | None = None


def load_baselines() -> pd.DataFrame:
    global _baselines_cache
    if _baselines_cache is None:
        df = pd.read_csv(ALL_BASELINES_TSV, sep="\t")
        _baselines_cache = df
    return _baselines_cache


@dataclass
class BaselineValue:
    found: bool
    ndcg10: float | None = None
    ndcg10_std: float | None = None
    nfairr10: float | None = None
    nfairr10_std: float | None = None
    mrr10: float | None = None
    recall10: float | None = None
    note: str = ""


def baseline_row(dataset_key: str, model_id: str | None) -> BaselineValue:
    """model_id: the full baseline-table key ('BM25' or 'sentence-transformers/...'), or None."""
    if model_id is None:
        return BaselineValue(found=False, note="No baseline computed for this checkpoint (see MODEL_TO_BASELINE_ID).")
    df = load_baselines()
    dkey = DATASETS[dataset_key]["baseline_key"]
    sub = df[(df["dataset"] == dkey) & (df["model"] == model_id)]
    if sub.empty:
        return BaselineValue(found=False, note=f"No baseline row for {model_id} on {dkey}.")

    if DATASETS[dataset_key]["seed_varies_baseline"]:
        mean_row = sub[sub["seed"] == "mean"]
        std_row = sub[sub["seed"] == "std"]
        if mean_row.empty:
            return BaselineValue(found=False, note=f"No mean row for {model_id} on {dkey}.")
        m, s = mean_row.iloc[0], (std_row.iloc[0] if not std_row.empty else None)
        return BaselineValue(
            found=True,
            ndcg10=float(m["nDCG@10"]), ndcg10_std=float(s["nDCG@10"]) if s is not None else None,
            nfairr10=float(m["NFaiRR@10"]), nfairr10_std=float(s["NFaiRR@10"]) if s is not None else None,
            mrr10=float(m["MRR@10"]), recall10=float(m["Recall@10"]),
        )
    else:
        row = sub.iloc[0]
        return BaselineValue(
            found=True,
            ndcg10=float(row["nDCG@10"]), ndcg10_std=None,
            nfairr10=float(row["NFaiRR@10"]), nfairr10_std=None,
            mrr10=float(row["MRR@10"]), recall10=float(row["Recall@10"]),
            note="TREC DL19 is a fixed 30-query eval; no seed variation to report.",
        )


def bm25_baseline(dataset_key: str) -> BaselineValue:
    return baseline_row(dataset_key, "BM25")


def unsteered_baseline(dataset_key: str, model: str) -> BaselineValue:
    return baseline_row(dataset_key, MODEL_TO_BASELINE_ID.get(model))


def sweep_zero_point(agg_df: pd.DataFrame, level: str, direction: str) -> pd.Series | None:
    """The aggregated coefficient=0 row for (level, direction) -- the in-sweep unsteered proxy.

    Used as the ARaB reference for "unsteered": ARaB is never computed for baseline .trec files
    outside the steering pipeline, but coefficient=0 IS literally "no steering vector applied",
    so its aggregated ARaB is the unsteered value by construction. steering_dashboard.py uses this
    same convention for its own baseline reference lines.
    """
    sub = agg_df[(agg_df["level"] == level) & (agg_df["direction"] == direction)]
    if sub.empty:
        return None
    idx = (sub["coefficient"] - 0.0).abs().idxmin()
    return sub.loc[idx]


# ---------------------------------------------------------------------------
# 1. Operating-range finder
# ---------------------------------------------------------------------------

@dataclass
class OperatingRange:
    level: str
    direction: str
    threshold: float
    mode: str
    baseline_ndcg: float
    left_bound: float
    right_bound: float
    empty: bool
    trivial: bool
    grid_step: float


def find_operating_range(
    agg_df: pd.DataFrame,
    level: str,
    direction: str,
    baseline_ndcg: float,
    threshold: float,
    mode: str = "asymmetric",
) -> OperatingRange | None:
    """Largest coefficient interval around 0 for which nDCG@10 stays within `threshold`
    (relative, e.g. 0.02 = 2%) of `baseline_ndcg`, walking outward from the grid point nearest 0
    and stopping at the first violation on each side (so a later recovery past a dip doesn't get
    included -- the range must be *contiguous* from 0).

    mode="symmetric": the interval is clamped to +/- the smaller of the two one-sided margins.
    mode="asymmetric" (default): left and right bounds are independent.
    """
    sub = agg_df[(agg_df["level"] == level) & (agg_df["direction"] == direction)].sort_values("coefficient")
    if sub.empty or "nDCG@10" not in sub.columns:
        return None
    coefs = sub["coefficient"].to_numpy()
    ndcgs = sub["nDCG@10"].to_numpy()
    zero_idx = int(np.argmin(np.abs(coefs)))
    tol = threshold * abs(baseline_ndcg)

    def within(v: float) -> bool:
        return abs(v - baseline_ndcg) <= tol

    right_bound = coefs[zero_idx]
    for i in range(zero_idx, len(coefs)):
        if within(ndcgs[i]):
            right_bound = coefs[i]
        else:
            break
    left_bound = coefs[zero_idx]
    for i in range(zero_idx, -1, -1):
        if within(ndcgs[i]):
            left_bound = coefs[i]
        else:
            break

    if mode == "symmetric":
        m = min(right_bound - coefs[zero_idx], coefs[zero_idx] - left_bound)
        right_bound = coefs[zero_idx] + m
        left_bound = coefs[zero_idx] - m

    diffs = np.diff(np.sort(np.unique(coefs)))
    grid_step = float(np.min(diffs)) if len(diffs) else 0.0
    empty = (right_bound == coefs[zero_idx]) and (left_bound == coefs[zero_idx])
    width = (right_bound - left_bound)
    trivial = (not empty) and grid_step > 0 and width <= grid_step * 2

    return OperatingRange(
        level=level, direction=direction, threshold=threshold, mode=mode,
        baseline_ndcg=baseline_ndcg, left_bound=float(left_bound), right_bound=float(right_bound),
        empty=bool(empty), trivial=bool(trivial), grid_step=grid_step,
    )


def operating_range_table(
    dataset_key: str, model: str, agg_by_level_direction: dict[str, pd.DataFrame],
    threshold: float, mode: str = "asymmetric",
) -> pd.DataFrame:
    """agg_by_level_direction: {"embed": agg_df, "attn": agg_df} (already seed-aggregated)."""
    rows = []
    for level, agg_df in agg_by_level_direction.items():
        for direction in DIRECTIONS:
            bl = unsteered_baseline(dataset_key, model)
            if not bl.found:
                zp = sweep_zero_point(agg_df, level, direction)
                baseline_ndcg = float(zp["nDCG@10"]) if zp is not None else np.nan
            else:
                baseline_ndcg = bl.ndcg10
            orr = find_operating_range(agg_df, level, direction, baseline_ndcg, threshold, mode)
            if orr is None:
                continue
            rows.append({
                "dataset": dataset_key, "model": model, "level": level, "direction": direction,
                "baseline_nDCG@10": orr.baseline_ndcg, "threshold": threshold,
                "left_bound": orr.left_bound, "right_bound": orr.right_bound,
                "width": orr.right_bound - orr.left_bound,
                "empty": orr.empty, "trivial": orr.trivial,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. ARaB peak within operating range
# ---------------------------------------------------------------------------

@dataclass
class ArabPeak:
    coefficient: float | None
    arab_signed: float | None
    arab_abs: float | None
    arab_abs_std: float | None
    in_range: bool


def find_arab_peak(agg_df: pd.DataFrame, level: str, direction: str, orr: OperatingRange) -> ArabPeak:
    sub = agg_df[(agg_df["level"] == level) & (agg_df["direction"] == direction)]
    if orr is not None and not orr.empty:
        sub = sub[(sub["coefficient"] >= orr.left_bound) & (sub["coefficient"] <= orr.right_bound)]
    if sub.empty or "|ARaB|@10" not in sub.columns or sub["|ARaB|@10"].isna().all():
        return ArabPeak(None, None, None, None, in_range=False)
    idx = sub["|ARaB|@10"].idxmax()
    row = sub.loc[idx]
    return ArabPeak(
        coefficient=float(row["coefficient"]),
        arab_signed=float(row["ARaB@10"]) if "ARaB@10" in row and pd.notna(row["ARaB@10"]) else None,
        arab_abs=float(row["|ARaB|@10"]),
        arab_abs_std=float(row["|ARaB|@10_std"]) if "|ARaB|@10_std" in row and pd.notna(row["|ARaB|@10_std"]) else None,
        in_range=True,
    )


# ---------------------------------------------------------------------------
# 4. Summary table
# ---------------------------------------------------------------------------

def _fmt(mean, std, digits=4):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "—"
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def build_summary_table(
    dataset_keys: list[str], model: str, direction: str, threshold: float,
    arab_variant: str = DEFAULT_ARAB_VARIANT, mode: str = "asymmetric",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (display_df, raw_df). display_df has formatted "mean ± std" string cells
    (ready for markdown/LaTeX); raw_df keeps numeric mean/std columns for programmatic use
    (e.g. the matplotlib figure script)."""
    display_rows, raw_rows = [], []
    for dataset_key in dataset_keys:
        # NB: agg[level] must be filtered to that level. An earlier version aggregated the full
        # (both-level) sweep and stored the SAME unfiltered result under both "embed" and "attn"
        # keys; because embed's and attn's coefficient grids overlap (attn's range is typically a
        # superset), the boundary-row lookup below would then silently match whichever level's
        # row sorted first for a shared coefficient value ("attn" < "embed" alphabetically),
        # mislabeling attn's metrics as the "Embedding-steered" row whenever the operating-range
        # boundary coefficient also happened to be a valid point on attn's grid -- which was
        # nearly always. Filtering here is what prevents that.
        df, _ = load_all_seeds(dataset_key, model)
        agg_all = aggregate_across_seeds(df, arab_variant) if not df.empty else pd.DataFrame()
        agg = {level: (agg_all[agg_all["level"] == level].reset_index(drop=True) if not agg_all.empty else pd.DataFrame()) for level in LEVELS}

        bm25 = bm25_baseline(dataset_key)
        dense = unsteered_baseline(dataset_key, model)
        baseline_ndcg = dense.ndcg10 if dense.found else (
            sweep_zero_point(agg["embed"], "embed", direction)["nDCG@10"] if not agg["embed"].empty else np.nan
        )

        # Unsteered dense ARaB from coefficient=0 (see sweep_zero_point docstring).
        zp = sweep_zero_point(agg["embed"], "embed", direction) if not agg["embed"].empty else None
        dense_arab = float(zp["ARaB@10"]) if zp is not None and pd.notna(zp.get("ARaB@10")) else None

        method_rows = [
            ("BM25", bm25.mrr10, None, bm25.ndcg10, bm25.ndcg10_std, bm25.nfairr10, bm25.nfairr10_std, None, None, bm25.found),
            ("Unsteered dense",
             dense.mrr10 if dense.found else (zp["MRR@10"] if zp is not None else None), None,
             dense.ndcg10 if dense.found else (zp["nDCG@10"] if zp is not None else None),
             dense.ndcg10_std if dense.found else (zp.get("nDCG@10_std") if zp is not None else None),
             dense.nfairr10 if dense.found else (zp["NFaiRR@10"] if zp is not None else None),
             dense.nfairr10_std if dense.found else (zp.get("NFaiRR@10_std") if zp is not None else None),
             dense_arab, zp.get("ARaB@10_std") if zp is not None else None, dense.found),
        ]
        for label, level in [("Embedding-steered", "embed"), ("Attention-steered", "attn")]:
            level_agg = agg[level]
            if level_agg.empty:
                method_rows.append((label, None, None, None, None, None, None, None, None, False))
                continue
            orr = find_operating_range(level_agg, level, direction, baseline_ndcg, threshold, mode)
            if orr is None or orr.empty:
                method_rows.append((label, None, None, None, None, None, None, None, None, False))
                continue
            # "Boundary-of-operating-range coefficient (the strongest effect before degradation)":
            # compare |ARaB@10| at the two EDGES of the operating range (not the interior peak --
            # that's a different question, answered separately by find_arab_peak for the curve-chart
            # marker in section 3) and take whichever edge has the larger magnitude.
            left_row = level_agg[(level_agg["coefficient"] - orr.left_bound).abs() < 1e-9]
            right_row = level_agg[(level_agg["coefficient"] - orr.right_bound).abs() < 1e-9]
            candidates = [r for r in (left_row, right_row) if not r.empty and "|ARaB|@10" in r.columns and pd.notna(r.iloc[0]["|ARaB|@10"])]
            if not candidates:
                method_rows.append((label, None, None, None, None, None, None, None, None, False))
                continue
            row = max(candidates, key=lambda r: r.iloc[0]["|ARaB|@10"])
            r = row.iloc[0]
            method_rows.append((
                f"{label} (α={float(r['coefficient']):g})",
                float(r["MRR@10"]) if "MRR@10" in r and pd.notna(r.get("MRR@10")) else None,
                float(r.get("MRR@10_std", np.nan)),
                float(r["nDCG@10"]), float(r.get("nDCG@10_std", np.nan)),
                float(r["NFaiRR@10"]), float(r.get("NFaiRR@10_std", np.nan)),
                float(r["ARaB@10"]) if pd.notna(r.get("ARaB@10")) else None,
                float(r.get("ARaB@10_std", np.nan)), True,
            ))

        for label, mrr_m, mrr_s, ndcg_m, ndcg_s, nf_m, nf_s, arab_m, arab_s, found in method_rows:
            display_rows.append({
                "Method": label, "Dataset": DATASETS[dataset_key]["label"],
                "MRR@10": _fmt(mrr_m, mrr_s) if mrr_m is not None else "—",
                "nDCG@10": _fmt(ndcg_m, ndcg_s), "NFaiRR@10": _fmt(nf_m, nf_s),
                "ARaB@10": _fmt(arab_m, arab_s) if arab_m is not None else "—",
                "found": found,
            })
            raw_rows.append({
                "Method": label, "Dataset": DATASETS[dataset_key]["label"],
                "MRR@10_mean": mrr_m, "MRR@10_std": mrr_s,
                "nDCG@10_mean": ndcg_m, "nDCG@10_std": ndcg_s,
                "NFaiRR@10_mean": nf_m, "NFaiRR@10_std": nf_s,
                "ARaB@10_mean": arab_m, "ARaB@10_std": arab_s, "found": found,
            })
    return pd.DataFrame(display_rows), pd.DataFrame(raw_rows)


def build_summary_table_multi_model(
    dataset_keys: list[str], models: list[str], direction: str, threshold: float,
    arab_variant: str = DEFAULT_ARAB_VARIANT, mode: str = "asymmetric",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs build_summary_table per model and stacks the results with a leading "Model" column.
    BM25 doesn't depend on the dense model, so only the first model's BM25 row is kept (per
    dataset) rather than duplicating it once per model."""
    display_parts, raw_parts = [], []
    for i, model in enumerate(models):
        disp, raw = build_summary_table(dataset_keys, model, direction, threshold, arab_variant, mode)
        if i > 0:
            disp = disp[disp["Method"] != "BM25"].copy()
            raw = raw[raw["Method"] != "BM25"].copy()
        disp.insert(0, "Model", np.where(disp["Method"] == "BM25", "—", model))
        raw.insert(0, "Model", np.where(raw["Method"] == "BM25", "—", model))
        display_parts.append(disp)
        raw_parts.append(raw)
    return pd.concat(display_parts, ignore_index=True), pd.concat(raw_parts, ignore_index=True)


def to_markdown_table(display_df: pd.DataFrame) -> str:
    """Hand-rolled (no `tabulate` dependency, which pandas.to_markdown requires and this repo's
    conda env doesn't have)."""
    cols = [c for c in display_df.columns if c != "found"]
    rows = display_df[cols].astype(str).values.tolist()
    widths = [max(len(cols[i]), *(len(r[i]) for r in rows)) if rows else len(cols[i]) for i in range(len(cols))]
    header = "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = ["| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) + " |" for r in rows]
    return "\n".join([header, sep] + body)


def _latex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
             .replace("&", r"\&").replace("#", r"\#")
             .replace("±", r"$\pm$").replace("—", "--")
             .replace("−", "-").replace("α", r"$\alpha$"))


def to_latex_booktabs(display_df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Hand-rolled booktabs table (no Jinja2 dependency, which pandas 3.x's to_latex now requires
    via DataFrame.style and this repo's conda env doesn't have). Assumes \\usepackage{booktabs}
    in the paper preamble."""
    cols = [c for c in display_df.columns if c != "found"]
    rows = display_df[cols].astype(str).values.tolist()
    colspec = "l" + "r" * (len(cols) - 1)
    lines = ["\\begin{table}", "\\centering", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    lines.append(" & ".join(_latex_escape(c) for c in cols) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(" & ".join(_latex_escape(v) for v in r) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if caption:
        lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Seed stability
# ---------------------------------------------------------------------------

def seed_stability_table(
    dataset_key: str, model: str, level: str, direction: str, arab_variant: str = DEFAULT_ARAB_VARIANT,
    aggregate_range: OperatingRange | None = None,
) -> pd.DataFrame:
    """Per-seed peak |ARaB@10| location and magnitude, over that seed's full sweep (not
    restricted to any operating range -- the point is to see raw seed-to-seed variance).
    If `aggregate_range` is given, flags whether each seed's peak coefficient falls inside the
    AGGREGATE (mean-curve) operating range, which is the range actually used in the paper table.
    """
    df, seeds_found = load_all_seeds(dataset_key, model)
    signed_col, mag_col = arab_col_names(arab_variant)
    rows = []
    for seed in seeds_found:
        sub = df[(df["seed"] == seed) & (df["level"] == level) & (df["direction"] == direction)]
        if sub.empty or mag_col not in sub.columns or sub[mag_col].isna().all():
            rows.append({"seed": seed, "peak_coefficient": None, "peak_arab_abs": None, "peak_arab_signed": None, "in_aggregate_range": None})
            continue
        idx = sub[mag_col].idxmax()
        r = sub.loc[idx]
        in_range = None
        if aggregate_range is not None and not aggregate_range.empty:
            in_range = aggregate_range.left_bound <= r["coefficient"] <= aggregate_range.right_bound
        rows.append({
            "seed": seed, "peak_coefficient": float(r["coefficient"]),
            "peak_arab_abs": float(r[mag_col]),
            "peak_arab_signed": float(r[signed_col]) if signed_col in r and pd.notna(r[signed_col]) else None,
            "in_aggregate_range": in_range,
        })
    out = pd.DataFrame(rows)
    if not out.empty and out["peak_coefficient"].notna().any():
        out.attrs["coef_spread"] = float(out["peak_coefficient"].max() - out["peak_coefficient"].min())
        out.attrs["arab_spread"] = float(out["peak_arab_abs"].max() - out["peak_arab_abs"].min())
    return out


# ---------------------------------------------------------------------------
# 6. Diagnostics
# ---------------------------------------------------------------------------

def diagnostics(dataset_key: str, model: str, threshold: float, arab_variant: str = DEFAULT_ARAB_VARIANT, mode: str = "asymmetric") -> list[dict]:
    flags = []
    baseline_id = MODEL_TO_BASELINE_ID.get(model)
    if baseline_id is None:
        flags.append({
            "severity": "warning", "scope": f"{dataset_key}/{model}",
            "message": (
                f"No official baseline computed for '{model}' -- the swept checkpoint diverges "
                "from the only TAS-B baseline on file (sentence-transformers/msmarco-distilbert-base-tas-b "
                "vs. sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco). Falling back to the "
                "sweep's own coefficient=0 row as the unsteered reference."
            ),
        })

    for level in LEVELS:
        df, seeds_found = load_all_seeds(dataset_key, model)
        if df.empty:
            flags.append({"severity": "error", "scope": f"{dataset_key}/{model}/{level}", "message": "No sweep data found at all."})
            continue
        if len(seeds_found) < len(SEEDS):
            missing = sorted(set(SEEDS) - set(seeds_found))
            flags.append({
                "severity": "warning", "scope": f"{dataset_key}/{model}/{level}",
                "message": f"Only {len(seeds_found)}/{len(SEEDS)} seeds found (missing: {missing}).",
            })
        agg = aggregate_across_seeds(df, arab_variant)
        bl = unsteered_baseline(dataset_key, model)
        for direction in DIRECTIONS:
            level_agg = agg[agg["level"] == level]
            if level_agg.empty:
                continue
            baseline_ndcg = bl.ndcg10 if bl.found else (
                sweep_zero_point(agg, level, direction)["nDCG@10"] if sweep_zero_point(agg, level, direction) is not None else np.nan
            )
            orr = find_operating_range(agg, level, direction, baseline_ndcg, threshold, mode)
            if orr is None:
                continue
            if orr.empty:
                flags.append({
                    "severity": "critical", "scope": f"{dataset_key}/{model}/{level}/{direction}",
                    "message": f"Operating range is EMPTY at threshold={threshold:.1%}: nDCG@10 degrades past threshold at the very first coefficient step away from 0. This is itself a finding.",
                })
            elif orr.trivial:
                flags.append({
                    "severity": "warning", "scope": f"{dataset_key}/{model}/{level}/{direction}",
                    "message": f"Operating range is trivially small ({orr.left_bound:g} to {orr.right_bound:g}, width {orr.right_bound-orr.left_bound:g}, only ~2 grid steps) at threshold={threshold:.1%}.",
                })

    # ARaB sign-convention check, reported DESCRIPTIVELY rather than as a pass/fail assertion.
    # A naive "positive coefficient should always push ARaB more male-leaning" check was tried
    # and dropped: on the real embed/mf curve, ARaB@10 is NOT monotonic in coefficient -- it
    # peaks near coefficient=0 and reverses sign at large |alpha| on BOTH sides (steering is
    # applied to every candidate document with a gendered term, per apply_steering_and_evaluate.py,
    # not selectively to male-vs-female docs, so overshoot/saturation at large alpha is plausible
    # real behavior, not a bug). A hard-coded "should" here would misfire on genuine non-monotonic
    # science. Instead this reports the observed LOCAL slope at coefficient=0 (the one place the
    # actadd hook's effect is closest to linear) so the user can check it against their own
    # expectation for the mf direction (diff_mf = male_embedding - female_embedding, per
    # steering_vectors_embed_msmarco_fair.py) -- this is the "labeled example" check point 6 of
    # the brief asked for; the direction itself was left as a placeholder in the brief, so this
    # surfaces the number rather than asserting a verdict.
    df, _ = load_all_seeds(dataset_key, model)
    if not df.empty:
        agg = aggregate_across_seeds(df, arab_variant)
        sub = agg[(agg["level"] == "embed") & (agg["direction"] == "mf")].sort_values("coefficient")
        if not sub.empty and "ARaB@10" in sub.columns and sub["ARaB@10"].notna().any():
            zero_row = sweep_zero_point(agg, "embed", "mf")
            pos_near = sub[sub["coefficient"] > 0].sort_values("coefficient")
            neg_near = sub[sub["coefficient"] < 0].sort_values("coefficient", ascending=False)
            if zero_row is not None and len(pos_near) and len(neg_near):
                arab0 = zero_row["ARaB@10"]
                arab_pos1 = pos_near.iloc[0]["ARaB@10"]
                arab_neg1 = neg_near.iloc[0]["ARaB@10"]
                slope_sign = "increases" if arab_pos1 > arab0 else "decreases"
                flags.append({
                    "severity": "info", "scope": f"{dataset_key}/{model}/embed/mf",
                    "message": (
                        f"ARaB sign reference point: at coefficient=0, ARaB@10={arab0:.4f}; the "
                        f"first step toward positive (male-direction, diff_mf=male-female) coefficient "
                        f"gives ARaB@10={arab_pos1:.4f} ({slope_sign}); the first step toward negative "
                        f"gives ARaB@10={arab_neg1:.4f}. {ARAB_SIGN_NOTE} Confirm this local direction "
                        f"matches your own expectation for the mf comparison -- the full curve is NOT "
                        f"monotonic (it can reverse sign at large |coefficient|), so only this near-zero "
                        f"comparison is a meaningful sign check."
                    ),
                })
    return flags

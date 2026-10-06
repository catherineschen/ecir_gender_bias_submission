"""Compile every baseline (BM25 + 5 dense models) into one TSV: MS MARCO Fair heldout queries for
all 5 seeds (with a mean and std row per model across seeds) and TREC DL 2019 Fair (one run, all
30 queries as heldout, so no seed variation and no mean/std).

The std is the sample standard deviation (ddof=1) across the 5 seeds. MS MARCO numbers are the
heldout-only summaries (compute_heldout_baseline_summary.py), i.e. the same queries the steered
results are averaged over. TREC DL uses rel>=2 for MRR/Recall; MS MARCO uses rel>=1.

Usage (from the repo root):
    python gender_bias_steering_utility/compile_all_baselines.py
"""
import csv
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE_DIR = HERE / "experiments/baseline"
SEEDS = [42, 29, 59, 104, 996]
METRICS = ["MRR@10", "nDCG@10", "Recall@10", "NFaiRR@5", "NFaiRR@10", "NFaiRR@20", "NFaiRR@50"]
OUT_PATH = BASELINE_DIR / "all_baselines_all_seeds.tsv"


def read_summary(path):
    with open(path, encoding="utf-8") as f:
        return {row["model"]: row for row in csv.DictReader(f, delimiter="\t")}


def fmt(x):
    return f"{x:.4f}"


def main():
    rows = []  # (dataset, model, pool, seed, {metric: str})

    per_seed = {}
    for seed in SEEDS:
        suffix = "" if seed == 42 else f"_seed{seed}"
        per_seed[seed] = read_summary(BASELINE_DIR / f"sentence-transformers{suffix}/all_baselines_summary_heldout.tsv")
    models = list(per_seed[SEEDS[0]])  # BM25 first, then the 5 dense models
    for model in models:
        pool = per_seed[SEEDS[0]][model]["pool"]
        for seed in SEEDS:
            rows.append(("msmarco_fair_heldout", model, pool, str(seed), {m: per_seed[seed][model][m] for m in METRICS}))
        vals = {m: [float(per_seed[s][model][m]) for s in SEEDS] for m in METRICS}
        rows.append(("msmarco_fair_heldout", model, pool, "mean", {m: fmt(statistics.mean(vals[m])) for m in METRICS}))
        rows.append(("msmarco_fair_heldout", model, pool, "std", {m: fmt(statistics.stdev(vals[m])) for m in METRICS}))

    trec = read_summary(BASELINE_DIR / "trecdl19/sentence-transformers/all_baselines_summary.tsv")
    for model, row in trec.items():
        rows.append(("trecdl19_fair_all30", model, row["pool"], "NA", {m: row[m] for m in METRICS}))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("dataset\tmodel\tpool\tseed\t" + "\t".join(METRICS) + "\n")
        for dataset, model, pool, seed, m in rows:
            f.write(f"{dataset}\t{model}\t{pool}\t{seed}\t" + "\t".join(m[k] for k in METRICS) + "\n")
    print(f"wrote {len(rows)} rows to {OUT_PATH}")
    for dataset, model, _pool, seed, m in rows:
        if seed in ("mean", "std", "NA"):
            print(f"{dataset:<22} {model.replace('sentence-transformers/', ''):<38} {seed:<5} " + " ".join(f"{m[k]:>9}" for k in METRICS))


if __name__ == "__main__":
    main()

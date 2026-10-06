"""Heldout-only (and validation-only) baseline summaries for MS MARCO Fair.

The steered results that get reported (experiments/steering_heldout_final/) are evaluated on
the 195 heldout test queries, while all_baselines_summary.tsv averages over all 210 test
queries (heldout + the 15 validation queries used to pick alpha). This restricts the existing
per-query baseline values to the heldout (and, separately, validation) qids -- no re-encoding
needed -- so baselines and steered numbers are averaged over the same queries.

Also cross-checks that each dense model's heldout MRR@10/NFaiRR@10 equals its alpha=0 row in
the steering sweep (steering at alpha=0 must reproduce the unsteered baseline).

Usage (from the repo root):
    python gender_bias_steering_utility/compute_heldout_baseline_summary.py
    SEED=29 python gender_bias_steering_utility/compute_heldout_baseline_summary.py   # after compute_dense_baselines.py SEED=29
"""
import csv
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = int(os.environ.get("SEED", 42))  # other seeds read compute_dense_baselines.py's sentence-transformers_seed{SEED}/ outputs
SEED_SUFFIX = "" if SEED == 42 else f"_seed{SEED}"
BASELINE_DIR = HERE / "experiments/baseline"
DENSE_DIR = BASELINE_DIR / f"sentence-transformers{SEED_SUFFIX}"
SPLIT_DIR = HERE / f"preprocess/n5_seed{SEED}"
STEERING_DIR = HERE / f"experiments/steering_heldout_final{SEED_SUFFIX}"

METRIC_LABELS = ["MRR@10", "nDCG@10", "Recall@10", "NFaiRR@5", "NFaiRR@10", "NFaiRR@20", "NFaiRR@50"]
# baseline model name -> steering output dir name (TAS-B's checkpoint is named differently in the two scripts)
STEERING_DIR_NAME = {"msmarco-distilbert-base-tas-b": "distilbert-dot-tas_b-b256-msmarco"}


def read_qids(path):
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        return {line.split("\t")[0] for line in f if line.strip()}


def restrict_and_average(perquery, qids):
    """perquery = {"retrieval_effectiveness": {...}, "fairness_NFaiRR": {...}} -> {metric: (mean, n)}"""
    out = {}
    for section in ("retrieval_effectiveness", "fairness_NFaiRR"):
        for label, per_q in perquery[section].items():
            if label == "query_set":
                continue
            values = [v for q, v in per_q.items() if q in qids]
            out[label] = (sum(values) / len(values), len(values))
    return out


def main():
    dense = json.load(open(DENSE_DIR / "dense_baseline_perquery.json", encoding="utf-8"))
    dense_meta = {}
    with open(DENSE_DIR / "dense_baseline_metrics.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            dense_meta[row["model"].rsplit("/", 1)[-1]] = (row["model"], row["pool"])
    bm25_path = BASELINE_DIR / "bm25_baseline_perquery_v2.json" if SEED == 42 else DENSE_DIR / "bm25_baseline_perquery.json"
    bm25 = json.load(open(bm25_path, encoding="utf-8"))

    for split in ("heldout", "validation"):
        qids = read_qids(SPLIT_DIR / f"test_queries_{split}.tsv")
        rows = [("BM25", "-", restrict_and_average(bm25, qids))]
        rows += [(dense_meta[name][0], dense_meta[name][1], restrict_and_average(pq, qids)) for name, pq in dense.items()]

        out_path = DENSE_DIR / f"all_baselines_summary_{split}.tsv"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("model\tpool\t" + "\t".join(METRIC_LABELS) + "\n")
            for model, pool, m in rows:
                f.write(f"{model}\t{pool}\t" + "\t".join(f"{m[l][0]:.4f}" for l in METRIC_LABELS) + "\n")
        n_ret, n_fair = rows[0][2]["MRR@10"][1], rows[0][2]["NFaiRR@10"][1]
        print(f"\n{split} ({len(qids)} queries; {n_ret} scored for retrieval, {n_fair} for NFaiRR) -> {out_path}")
        print(f"{'model':<52} {'pool':<5} " + " ".join(f"{l:>10}" for l in METRIC_LABELS))
        for model, pool, m in rows:
            print(f"{model:<52} {pool:<5} " + " ".join(f"{m[l][0]:>10.4f}" for l in METRIC_LABELS))

    # cross-check against steering alpha=0 (heldout only)
    heldout = read_qids(SPLIT_DIR / "test_queries_heldout.tsv")
    print("\nalpha=0 steering vs heldout baseline:")
    ok = True
    for name, pq in dense.items():
        base = restrict_and_average(pq, heldout)
        sweep = STEERING_DIR / STEERING_DIR_NAME.get(name, name) / "sweep_results.csv"
        if not sweep.exists():
            print(f"  {name}: no steering sweep at {sweep} yet, skipped")
            continue
        with open(sweep, encoding="utf-8") as f:
            zero = [r for r in csv.DictReader(f) if float(r["coefficient"]) == 0]
        for r in zero:  # one per direction; all identical at alpha=0
            for label in ("MRR@10", "NFaiRR@10"):
                diff = abs(float(r[label]) - base[label][0])
                if diff > 1e-4:
                    ok = False
                    print(f"  MISMATCH {name} {r['direction']} {label}: steering={float(r[label]):.4f} baseline={base[label][0]:.4f}")
        print(f"  {name}: checked {len(zero)} direction rows")
    print("ALL MATCH" if ok else "MISMATCHES FOUND")


if __name__ == "__main__":
    main()

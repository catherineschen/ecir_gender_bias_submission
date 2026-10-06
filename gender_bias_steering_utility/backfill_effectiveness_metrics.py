"""One-off backfill: add nDCG@10/Recall@10 to a sweep_results.csv written before those columns
existed in SWEEP_CSV_FIELDS.

Nothing is re-ranked and no model is loaded -- MRR@10/nDCG@10/Recall@10 are pure functions of a
run.trec file plus the qrels, so this recomputes them directly from the run.trec files the
original sweep already wrote (the same read-only pattern compute_arab_from_runs.py uses for
ARaB). MRR@10 is recomputed too, purely as a safety check: it is asserted to match the existing
value to a tight tolerance before anything is written, which is what catches a wrong SEED/
TEST_QUERIES_SPLIT/MODEL_KEY (i.e. pointing this at the wrong run) before it can corrupt the file.
NFaiRR@10 needs no backfill (it was already in SWEEP_CSV_FIELDS) and is carried over unchanged.

Same env-var resolution as the sweep that wrote the runs, imported from
apply_steering_and_evaluate.py rather than duplicated (see that module's SEED/TEST_QUERIES_SPLIT/
OUTPUT_DIR_NAME/MODEL_KEY docs) -- point this at exactly the same OUTPUT_DIR_NAME/SEED/
TEST_QUERIES_SPLIT/MODEL_KEY combination that produced the sweep_results.csv to fix:

    OUTPUT_DIR_NAME=steering_validation_coarse TEST_QUERIES_SPLIT=validation MODEL_KEY=tas_b \\
        conda run -n gender_bias python gender_bias_steering_utility/backfill_effectiveness_metrics.py

Idempotent: a sweep_results.csv that already has nDCG@10 and Recall@10 is left untouched (and
says so) rather than recomputed.
"""
import contextlib
import csv
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Same env-var resolution, same paths, same folder naming as the sweep that wrote the runs.
# Importing it loads torch/mechir but does NOT load a model or touch a GPU (everything in that
# module below the constants lives inside functions), so this stays a read-only metric script.
from apply_steering_and_evaluate import (  # noqa: E402
    DATASET,
    MIN_REL,
    MODEL_KEY,
    OUTPUT_DIR,
    QRELS_PATH,
    SEED,
    TEST_QUERIES_PATH,
    coef_folder,
    load_construction_qids,
    load_test_queries,
)
from compute_baseline_metrics import compute_retrieval_effectiveness  # noqa: E402

SWEEP_CSV_FIELDS = ["model", "level", "direction", "coefficient", "MRR@10", "nDCG@10", "Recall@10", "NFaiRR@10"]
MRR_TOLERANCE = 1e-6  # the safety check: a mismatch here means the wrong seed/split was pointed at this dir


def main():
    if DATASET != "msmarco":
        sys.exit(f"This backfill is for MS MARCO sweeps only; DATASET={DATASET!r}. "
                 "TREC DL sweeps already have nDCG@10/Recall@10 (they were run after those columns existed).")

    csv_path = OUTPUT_DIR / "sweep_results.csv"
    if not csv_path.exists():
        sys.exit(f"No sweep_results.csv at {csv_path} -- check MODEL_KEY/OUTPUT_DIR_NAME/SEED.")

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{csv_path} has no rows.")

    if "nDCG@10" in rows[0] and "Recall@10" in rows[0] and all(r.get("nDCG@10") not in (None, "") for r in rows):
        print(f"{csv_path} already has nDCG@10/Recall@10 for every row -- nothing to do.")
        return

    construction_qids = load_construction_qids()
    queries = load_test_queries(TEST_QUERIES_PATH, construction_qids)
    test_qids = set(queries)
    print(f"Model: {MODEL_KEY}  Seed: {SEED}  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"{len(test_qids)} test queries loaded from {TEST_QUERIES_PATH.name} "
          f"(construction queries confirmed disjoint)")

    updated = []
    mismatches = []
    for i, row in enumerate(rows):
        level, direction, coef = row["level"], row["direction"], float(row["coefficient"])
        run_path = OUTPUT_DIR / level / direction / coef_folder(level, coef) / "run.trec"
        if not run_path.exists():
            sys.exit(f"Expected run.trec missing: {run_path} -- this sweep_results.csv doesn't "
                     "match the run.trec files on disk under this OUTPUT_DIR.")
        # compute_retrieval_effectiveness prints a one-line query summary per call, and this loop
        # makes one call per row (up to ~294 for a full embed+attn sweep) -- only the first is shown,
        # the counts are identical for every call against the same test_qids.
        sink = contextlib.redirect_stdout(io.StringIO()) if i else contextlib.nullcontext()
        with sink:
            aggregate, _ = compute_retrieval_effectiveness(run_path, QRELS_PATH, test_qids, min_rel=MIN_REL)

        old_mrr = float(row["MRR@10"])
        if abs(aggregate["MRR@10"] - old_mrr) > MRR_TOLERANCE:
            mismatches.append((level, direction, coef, old_mrr, aggregate["MRR@10"]))
            continue

        new_row = dict(row)
        new_row["MRR@10"] = aggregate["MRR@10"]
        new_row["nDCG@10"] = aggregate["nDCG@10"]
        new_row["Recall@10"] = aggregate["Recall@10"]
        updated.append(new_row)

    if mismatches:
        lines = "\n".join(f"  {lvl}/{d} coef={c}: existing MRR@10={old:.6f} vs recomputed={new:.6f}"
                          for lvl, d, c, old, new in mismatches[:10])
        sys.exit(
            f"Refusing to write: {len(mismatches)} row(s) have a recomputed MRR@10 that doesn't match "
            f"the existing value (tolerance {MRR_TOLERANCE}) -- this usually means SEED/"
            f"TEST_QUERIES_SPLIT/MODEL_KEY don't match what actually produced {csv_path}:\n{lines}"
        )

    tmp_path = csv_path.with_suffix(".csv.tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_CSV_FIELDS)
        writer.writeheader()
        for row in updated:
            writer.writerow({k: row.get(k, "") for k in SWEEP_CSV_FIELDS})
    tmp_path.replace(csv_path)
    print(f"Backfilled nDCG@10/Recall@10 for {len(updated)} row(s) in {csv_path} "
          "(MRR@10/NFaiRR@10 verified unchanged).")


if __name__ == "__main__":
    main()

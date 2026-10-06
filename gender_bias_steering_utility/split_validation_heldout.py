"""Split a seed's test queries (preprocess/n5_seed{SEED}/test_queries.tsv, written by
split_queries_dedupe.py) into a validation subset (for alpha-range selection) and a
held-out subset (for final, report-once evaluation), so alpha is never selected on the
same queries it gets reported on -- see apply_steering_and_evaluate.py's TEST_QUERIES_SPLIT.

Seed 42's split (preprocess/n5_seed42/test_queries_validation.tsv /
test_queries_heldout.tsv) already exists on disk, generated ad hoc before this script
existed. This script refuses to overwrite an existing validation/held-out split (see
main()) so it can never silently replace seed 42's split -- every sweep run so far was
evaluated against that exact 15/195 partition, and regenerating it would invalidate them.

Usage (from the repo root), for a new seed already produced by split_queries_dedupe.py:
    SEED=123 conda run -n gender_bias python gender_bias_steering_utility/split_validation_heldout.py
"""
import os
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = int(os.environ.get("SEED", 42))
N_VALIDATION = int(os.environ.get("N_VALIDATION", 15))

SPLIT_DIR = REPO_ROOT / "gender_bias_steering_utility/preprocess" / f"n5_seed{SEED}"
TEST_QUERIES_PATH = SPLIT_DIR / "test_queries.tsv"
VALIDATION_PATH = SPLIT_DIR / "test_queries_validation.tsv"
HELDOUT_PATH = SPLIT_DIR / "test_queries_heldout.tsv"


def load_queries(path):
    queries = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            if not line.strip():
                continue
            qid, text = line.rstrip("\n").split("\t")[:2]
            queries[qid] = text
    return queries


def write_queries_tsv(qids, queries, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("query_id\tquery_text\n")
        for qid in qids:
            f.write(f"{qid}\t{queries[qid]}\n")


def main():
    if VALIDATION_PATH.exists() or HELDOUT_PATH.exists():
        raise FileExistsError(
            f"{VALIDATION_PATH} or {HELDOUT_PATH} already exists -- refusing to overwrite an "
            "existing validation/held-out split (every sweep run against it would be invalidated). "
            "Delete both first if you really mean to regenerate it."
        )
    if not TEST_QUERIES_PATH.exists():
        raise FileNotFoundError(
            f"{TEST_QUERIES_PATH} not found -- run split_queries_dedupe.py with SEED={SEED} first."
        )

    queries = load_queries(TEST_QUERIES_PATH)
    if N_VALIDATION >= len(queries):
        raise ValueError(f"N_VALIDATION={N_VALIDATION} >= {len(queries)} test queries")

    rng = random.Random(SEED)  # separate instance, same pattern as split_queries_dedupe.sample_construction_qids
    ordered = sorted(queries, key=int)  # sort first so the shuffle is reproducible regardless of input order
    shuffled = ordered[:]
    rng.shuffle(shuffled)

    validation_qids = sorted(shuffled[:N_VALIDATION], key=int)
    heldout_qids = sorted(shuffled[N_VALIDATION:], key=int)

    write_queries_tsv(validation_qids, queries, VALIDATION_PATH)
    write_queries_tsv(heldout_qids, queries, HELDOUT_PATH)

    print(f"{len(queries)} test queries loaded from {TEST_QUERIES_PATH}")
    print(f"wrote {len(validation_qids)} validation queries -> {VALIDATION_PATH}")
    print(f"wrote {len(heldout_qids)} held-out queries -> {HELDOUT_PATH}")


if __name__ == "__main__":
    main()

"""Split fairness queries into construction/test sets and dedupe docs shared between them.

Construction queries are sampled (seeded) from queries with enough gendered-term docs;
the excluded low-gendered-term queries can still serve as test queries, just never as
construction queries. Docs whose candidate lists overlap between the two resulting
splits are then removed from whichever side is the minority effect, per the <5%/>=5%
rule below, so no single doc can leak between a construction and a test query.

No model is loaded; this only reads the outputs of annotate_gendered_candidates.py.

Usage (from the repo root), and how to rerun for a sweep:
    conda run -n gender_bias python gender_bias_steering_utility/split_queries_dedupe.py
    # then edit N_CONSTRUCTION and/or SEED below and rerun for a different split
"""
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
import random

# ---- paths (edit here, or copy this script and change these constants) ----
REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESS_DIR = REPO_ROOT / "gender_bias_steering_utility/preprocess"
ANNOTATED_CANDIDATES_PATH = PREPROCESS_DIR / "annotated_candidates.tsv"
QUERY_STATS_PATH = PREPROCESS_DIR / "query_stats.tsv"
ANNOTATION_SUMMARY_PATH = PREPROCESS_DIR / "annotation_summary.json"
DOC_OVERLAP_PATH = PREPROCESS_DIR / "doc_overlap.json"  # not used for the split decision (see note below); kept for reference/logging
# annotated_candidates.tsv/query_stats.tsv carry qids but not query text, so the original
# queries file is needed too to write out {query_id, query_text} for each split.
QUERIES_PATH = REPO_ROOT / "data/FairnessRetrievalResults/dataset/msmarco_passage.dev.fair.FULL_ANNOTATION.tsv"
OUTPUT_DIR = PREPROCESS_DIR

# ---- sweep parameters: N_CONSTRUCTION stays fixed at 5 by convention (the n5_seed{SEED}
# directory naming bakes that in); SEED is overridable via env var so the same script
# reruns unmodified for other seeds -- see run_steering_sweep.sh's "OTHER SEEDS" section --
# without ever needing to hand-edit this file (which would risk changing seed 42's output). ----
N_CONSTRUCTION = 5
SEED = int(os.environ.get("SEED", 42))

TEST_OVERLAP_PCT_THRESHOLD = 5.0  # below this, drop overlapping docs from test; at/above, drop from construction

CANDIDATE_COLUMNS = ["qid", "q0", "docid", "rank", "score", "tag", "has_gendered_terms", "gendered_terms_found", "gendered_term_positions"]


def load_queries_text(path):
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, text = line.rstrip("\n").split("\t")[:2]
            if qid == "QID":
                continue
            queries[qid] = text
    return queries


def load_all_qids(query_stats_path):
    qids = []
    with open(query_stats_path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            if line.strip():
                qids.append(line.split("\t", 1)[0])
    return qids


def load_excluded_qids(annotation_summary_path):
    summary = json.loads(annotation_summary_path.read_text())
    return set(summary["queries_with_fewer_than_5_gendered_docs"]["query_ids"])


def load_candidates(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            if not line.strip():
                continue
            values = line.rstrip("\n").split("\t")
            row = dict(zip(CANDIDATE_COLUMNS, values))
            row["rank"] = int(row["rank"])
            rows.append(row)
    return rows


def write_queries_tsv(qids, queries_text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("query_id\tquery_text\n")
        for qid in sorted(qids, key=int):
            f.write(f"{qid}\t{queries_text.get(qid, '')}\n")


def write_candidates_tsv(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(CANDIDATE_COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in CANDIDATE_COLUMNS) + "\n")


def sample_construction_qids(eligible_qids, n, seed):
    rng = random.Random(seed)  # separate Random instance so this doesn't disturb global RNG state
    ordered = sorted(eligible_qids, key=int)  # sort first so the sample is reproducible regardless of input order
    return set(rng.sample(ordered, n))


def find_overlap_and_removed_rows(construction_rows, test_rows):
    """Docs whose docid appears in both splits' candidate lists. Returns the overlap docid set and,
    per removal-decision branch, which exact rows would be dropped from each side."""
    construction_docids = {r["docid"] for r in construction_rows}
    test_docids = {r["docid"] for r in test_rows}
    overlap_docids = construction_docids & test_docids

    test_affected_rows = [r for r in test_rows if r["docid"] in overlap_docids]
    construction_affected_rows = [r for r in construction_rows if r["docid"] in overlap_docids]
    return overlap_docids, test_affected_rows, construction_affected_rows


def summarize_removed(removed_rows):
    by_doc = defaultdict(list)
    for r in removed_rows:
        by_doc[r["docid"]].append({"query_id": r["qid"], "rank": r["rank"]})
    return [{"doc_id": doc_id, "occurrences": occ} for doc_id, occ in sorted(by_doc.items(), key=lambda kv: kv[0])]


def main():
    excluded_qids = load_excluded_qids(ANNOTATION_SUMMARY_PATH)
    all_qids = load_all_qids(QUERY_STATS_PATH)
    queries_text = load_queries_text(QUERIES_PATH)
    candidates = load_candidates(ANNOTATED_CANDIDATES_PATH)
    print(f"{len(all_qids)} total queries, {len(excluded_qids)} excluded from construction eligibility, {len(candidates)} candidate rows")

    eligible_qids = [q for q in all_qids if q not in excluded_qids]
    if N_CONSTRUCTION > len(eligible_qids):
        raise ValueError(f"N_CONSTRUCTION={N_CONSTRUCTION} exceeds {len(eligible_qids)} eligible queries")
    construction_qids = sample_construction_qids(eligible_qids, N_CONSTRUCTION, SEED)
    test_qids = set(all_qids) - construction_qids  # eligible-but-not-sampled + excluded

    construction_rows = [c for c in candidates if c["qid"] in construction_qids]
    test_rows = [c for c in candidates if c["qid"] in test_qids]
    construction_before, test_before = len(construction_rows), len(test_rows)

    overlap_docids, test_affected_rows, construction_affected_rows = find_overlap_and_removed_rows(construction_rows, test_rows)
    test_affected_pct = (len(test_affected_rows) / test_before * 100) if test_before else 0.0

    if test_affected_pct < TEST_OVERLAP_PCT_THRESHOLD:
        removed_from = "test"
        removed_rows = test_affected_rows
        test_rows = [r for r in test_rows if r["docid"] not in overlap_docids]
    else:
        removed_from = "construction"
        removed_rows = construction_affected_rows
        construction_rows = [r for r in construction_rows if r["docid"] not in overlap_docids]

    run_output_dir = OUTPUT_DIR / f"n{N_CONSTRUCTION}_seed{SEED}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    write_queries_tsv(construction_qids, queries_text, run_output_dir / "construction_queries.tsv")
    write_queries_tsv(test_qids, queries_text, run_output_dir / "test_queries.tsv")
    write_candidates_tsv(construction_rows, run_output_dir / "construction_candidates.tsv")
    write_candidates_tsv(test_rows, run_output_dir / "test_candidates.tsv")

    removed_docs_summary = summarize_removed(removed_rows)
    removed_ranks = [occ["rank"] for d in removed_docs_summary for occ in d["occurrences"]]
    rank_stats = (
        {"min": min(removed_ranks), "median": statistics.median(removed_ranks), "max": max(removed_ranks)}
        if removed_ranks else {"min": None, "median": None, "max": None}
    )

    split_stats = {
        "n_construction": N_CONSTRUCTION,
        "seed": SEED,
        "construction_query_count": len(construction_qids),
        "test_query_count": len(test_qids),
        "candidate_counts": {
            "construction_before_dedup": construction_before,
            "construction_after_dedup": len(construction_rows),
            "test_before_dedup": test_before,
            "test_after_dedup": len(test_rows),
        },
        "overlap": {
            "overlapping_doc_count": len(overlap_docids),
            "test_candidates_affected_count": len(test_affected_rows),
            "test_candidates_affected_pct": round(test_affected_pct, 2),
            "removed_from": removed_from,
            "removed_row_count": len(removed_rows),
        },
        "removed_docs": removed_docs_summary,
        "removed_doc_rank_stats": rank_stats,
    }
    with open(run_output_dir / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=2)

    print(f"wrote outputs to {run_output_dir}")
    print(f"N={N_CONSTRUCTION}, seed={SEED}")
    print(f"construction queries: {len(construction_qids)}, test queries: {len(test_qids)}")
    print(f"construction candidates: {construction_before} -> {len(construction_rows)} after dedup")
    print(f"test candidates: {test_before} -> {len(test_rows)} after dedup")
    print(f"overlapping docs: {len(overlap_docids)}; {len(test_affected_rows)} test candidates affected ({test_affected_pct:.2f}%); removed from {removed_from}")
    print(f"removed doc rank (min/median/max): {rank_stats['min']} / {rank_stats['median']} / {rank_stats['max']}")


if __name__ == "__main__":
    main()

"""Sanity-check NFaiRR by running the original paper's BM25 run through our corrected pipeline.

Our own BM25 baseline's NFaiRR (0.7999/0.8180/0.8395/0.8588 @5/10/20/50, after the
full-collection-background and punctuation-stripped-tokenization fixes) is still above
Rekabsaz & Schedl's reported ~0.7227. This substitutes their own released BM25 run
(different candidate docs, since it's a different BM25 toolkit/params) into our unmodified
metric pipeline -- if that closes the gap, our candidates were the cause, not the metric
code; if it doesn't, something in the pipeline still needs a look.

Read-only against the original run file; fairr_full_collection.py is reused as-is.

Usage (from the repo root):
    conda run -n gender_bias python gender_bias_steering_utility/sanity_check_original_bm25_run.py
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

ORIGINAL_RUN_PATH = REPO_ROOT / "data/FairnessRetrievalResults/measurement/sample_trec_runs/msmarco_passage/BM25.run"
TEST_QUERIES_PATH = HERE / "preprocess/n5_seed42/test_queries.tsv"
V2_NEUTRALITY_PATH = HERE / "experiments/baseline/collection_neutralityscores_v2.tsv"
OUR_METRICS_PATH = HERE / "experiments/baseline/bm25_baseline_metrics_v2.tsv"
OUTPUT_DIR = HERE / "experiments/baseline/sanity_check"

PAPER_REPORTED_NFAIRR = 0.7227  # Rekabsaz & Schedl's reported BM25 NFaiRR on this query set, for comparison only
CLOSE_ENOUGH_TOLERANCE = 0.02  # NFaiRR points; below this, treat the gap as explained

from compute_baseline_metrics import FAIRR_THRESHOLDS, make_headerless_qids_file, read_old_metrics  # noqa: E402
from evaluate_run import read_qids  # noqa: E402  (same directory as this script)
from fairr_full_collection import FaiRRMetricHelper, load_fairr_metric, compute_nfairr_full_collection_background  # noqa: E402


def inspect_run_file(path):
    qids, depths, non_integer_docids = set(), {}, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 6:
                continue
            qid, docid = parts[0], parts[2]
            qids.add(qid)
            depths[qid] = depths.get(qid, 0) + 1
            if not re.fullmatch(r"\d+", docid):
                non_integer_docids += 1
    depth_values = list(depths.values())
    print(f"original run file: {path}")
    print(f"  format: 6 space-separated fields (qid Q0 docid rank score tag), matches read_retrievalresults_from_runfile's expectation")
    print(f"  unique queries: {len(qids)}")
    print(f"  depth per query: min={min(depth_values)} max={max(depth_values)}")
    print(f"  non-integer docids: {non_integer_docids} (0 expected -- doc IDs should be plain integers matching our collection's IDs)")
    return qids


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_qids = inspect_run_file(ORIGINAL_RUN_PATH)

    qids_path = OUTPUT_DIR / "_test_qids.tsv"
    make_headerless_qids_file(TEST_QUERIES_PATH, qids_path)
    our_test_qids = read_qids(qids_path)  # reused as-is from evaluate_run.py
    print(f"\nour test queries: {len(our_test_qids)}")

    in_both = our_test_qids & original_qids
    only_ours = our_test_qids - original_qids
    only_theirs = original_qids - our_test_qids
    print(f"query coverage: {len(in_both)} of our {len(our_test_qids)} test queries are present in their run file")
    print(f"  in ours but not theirs: {len(only_ours)} ({sorted(only_ours) if only_ours else 'none'})")
    print(f"  in theirs but not ours: {len(only_theirs)} (expected -- their run covers all 215 fairness queries, including our 5 construction queries)")
    if not our_test_qids | original_qids:
        print("  WARNING: no queries in either set")

    print(f"\nevaluating on the intersection: {len(in_both)} queries")

    helper = FaiRRMetricHelper()
    retrievalresults_all = helper.read_retrievalresults_from_runfile(str(ORIGINAL_RUN_PATH))
    in_both_int = {int(q) for q in in_both}
    retrievalresults = {q: v for q, v in retrievalresults_all.items() if q in in_both_int}
    print(f"{len(retrievalresults)} of {len(in_both_int)} intersected queries have candidates in the original run file")

    fairr_metric = load_fairr_metric(V2_NEUTRALITY_PATH)
    result = compute_nfairr_full_collection_background(fairr_metric, retrievalresults, FAIRR_THRESHOLDS)
    original_run_nfairr = {th: result["metrics_avg"]["NFaiRR"][th] for th in FAIRR_THRESHOLDS}

    our_metrics = read_old_metrics(OUR_METRICS_PATH)  # our own already-computed numbers; carried over, not rerun
    our_query_count = len(our_test_qids)

    with open(OUTPUT_DIR / "sanity_check_original_run_metrics.tsv", "w", encoding="utf-8") as f:
        f.write("run\tquery_count\t" + "\t".join(f"NFaiRR@{th}" for th in FAIRR_THRESHOLDS) + "\n")
        f.write(f"our_bm25_run\t{our_query_count}\t" + "\t".join(f"{our_metrics[f'NFaiRR@{th}']:.4f}" for th in FAIRR_THRESHOLDS) + "\n")
        f.write(f"original_paper_bm25_run\t{len(in_both)}\t" + "\t".join(f"{original_run_nfairr[th]:.4f}" for th in FAIRR_THRESHOLDS) + "\n")

    print("\ncomparison (both scored with our corrected pipeline: full-collection background, punctuation-stripped tokenization):")
    print(f"{'':>26} {'n queries':>10} " + " ".join(f"@{th:>8}" for th in FAIRR_THRESHOLDS))
    print(f"{'our BM25 run':>26} {our_query_count:>10} " + " ".join(f"{our_metrics[f'NFaiRR@{th}']:>9.4f}" for th in FAIRR_THRESHOLDS))
    print(f"{'original paper BM25 run':>26} {len(in_both):>10} " + " ".join(f"{original_run_nfairr[th]:>9.4f}" for th in FAIRR_THRESHOLDS))

    at50 = original_run_nfairr[50]
    gap = abs(at50 - PAPER_REPORTED_NFAIRR)
    print(f"\noriginal-run NFaiRR@50 = {at50:.4f} vs. paper-reported ~{PAPER_REPORTED_NFAIRR} (gap = {gap:.4f})")
    if gap <= CLOSE_ENOUGH_TOLERANCE:
        print("VERDICT: close enough (within "
              f"{CLOSE_ENOUGH_TOLERANCE}) -- the remaining gap between our BM25 baseline and the paper's ~0.72 is attributable to "
              "BM25 candidate composition (different toolkit/params retrieving different docs), not our metric pipeline.")
    else:
        print(f"VERDICT: still {gap:.4f} above the paper's reported value even with their own candidates -- a pipeline discrepancy "
              "likely remains (e.g. document neutrality computation, position-bias formula, or query-set/threshold conventions) "
              "and is worth investigating further before trusting downstream NFaiRR comparisons.")


if __name__ == "__main__":
    main()

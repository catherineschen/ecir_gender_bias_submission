"""Validate NFaiRR using only the original paper's BM25 run, as both retrievalresults and
background_doc_set -- mirroring the paper's own SET_Top200 setup (the same 200-per-query pool
serves as both the reranked/evaluated set and the IFaiRR background). This removes both our
BM25 candidates AND our full-collection-background methodology change as variables, checking
FaiRRMetric's implementation in isolation, with only the v2 (punctuation-fixed) neutrality
scores as our contribution.

Uses the original, unmodified FaiRRMetric class directly -- not fairr_full_collection.py's
full-collection-background wrapper. Read-only against the original run file.

Usage (from the repo root):
    conda run -n gender_bias python gender_bias_steering_utility/sanity_check_original_run_correct_background.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

ORIGINAL_RUN_PATH = REPO_ROOT / "data/FairnessRetrievalResults/measurement/sample_trec_runs/msmarco_passage/BM25.run"
TEST_QUERIES_PATH = HERE / "preprocess/n5_seed42/test_queries.tsv"
V2_NEUTRALITY_PATH = HERE / "experiments/baseline/collection_neutralityscores_v2.tsv"
FAIRNESS_CODE_DIR = REPO_ROOT / "data/FairnessRetrievalResults/adversarial_mitigation/fairness_measurement"
OUTPUT_DIR = HERE / "experiments/baseline/sanity_check"

FAIRR_THRESHOLDS = [5, 10, 20, 50]
CUT_OFF = 200  # full SET_Top200 depth
PAPER_NFAIRR_AT_10 = 0.786  # Rekabsaz & Schedl's reported BM25 NFaiRR@10 on MSMARCO Fair
CLOSE_ENOUGH_TOLERANCE = 0.02  # NFaiRR points

sys.path.insert(0, str(FAIRNESS_CODE_DIR))
from metrics_fairness import FaiRRMetric, FaiRRMetricHelper  # noqa: E402  (original, unmodified)

from compute_baseline_metrics import make_headerless_qids_file  # noqa: E402
from evaluate_run import read_qids  # noqa: E402  (same directory as this script)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    qids_path = OUTPUT_DIR / "_test_qids.tsv"
    make_headerless_qids_file(TEST_QUERIES_PATH, qids_path)
    our_test_qids = {int(q) for q in read_qids(qids_path)}  # reused as-is from evaluate_run.py
    print(f"{len(our_test_qids)} test queries loaded")

    helper = FaiRRMetricHelper()
    retrievalresults_all = helper.read_retrievalresults_from_runfile(str(ORIGINAL_RUN_PATH), cut_off=CUT_OFF)
    background_doc_set_all = helper.read_documentset_from_retrievalresults(str(ORIGINAL_RUN_PATH))  # same run file, both roles -- mirrors SET_Top200

    retrievalresults = {q: v for q, v in retrievalresults_all.items() if q in our_test_qids}
    background_doc_set = {q: v for q, v in background_doc_set_all.items() if q in our_test_qids}
    print(f"{len(retrievalresults)} of {len(our_test_qids)} test queries have candidates in the original run file "
          f"(intersection, as already confirmed clean: all 210 present)")

    fairr_metric = FaiRRMetric(str(V2_NEUTRALITY_PATH), background_doc_set)  # original, unmodified class
    result = fairr_metric.calc_FaiRR_retrievalresults(retrievalresults, thresholds=FAIRR_THRESHOLDS)

    nfairr_avg = result["metrics_avg"]["NFaiRR"]
    nfairr_perq = result["metrics_perq"]["NFaiRR"]

    with open(OUTPUT_DIR / "sanity_check_original_run_correct_background_v2.tsv", "w", encoding="utf-8") as f:
        f.write("metric\tvalue\n")
        for th in FAIRR_THRESHOLDS:
            f.write(f"NFaiRR@{th}\t{nfairr_avg[th]:.4f}\n")

    perquery_out = {
        "query_set": f"test ({len(retrievalresults)} queries); original paper BM25 run as both retrievalresults and background_doc_set (SET_Top200 setup)",
        "neutrality_scores": str(V2_NEUTRALITY_PATH),
        **{f"NFaiRR@{th}": {str(qid): v for qid, v in nfairr_perq[th].items()} for th in FAIRR_THRESHOLDS},
    }
    with open(OUTPUT_DIR / "sanity_check_original_run_correct_background_perquery_v2.json", "w", encoding="utf-8") as f:
        json.dump(perquery_out, f, indent=2)

    print(f"\nwrote outputs to {OUTPUT_DIR}")
    print("NFaiRR (original paper BM25 run as both retrievalresults and background_doc_set):")
    for th in FAIRR_THRESHOLDS:
        print(f"  NFaiRR@{th}: {nfairr_avg[th]:.4f}")

    ours_at_10 = nfairr_avg[10]
    gap = abs(ours_at_10 - PAPER_NFAIRR_AT_10)
    print(f"\nNFaiRR@10 = {ours_at_10:.4f} (this run) vs. {PAPER_NFAIRR_AT_10} (paper)  [gap = {gap:.4f}]")
    if gap <= CLOSE_ENOUGH_TOLERANCE:
        print(f"VERDICT: within {CLOSE_ENOUGH_TOLERANCE} of the paper's reported value -- the NFaiRR implementation is validated. "
              "Any remaining gap seen with our own BM25 run/background methodology traces to candidate composition and/or the "
              "full-collection-background design choice, not a bug in the metric code itself.")
    else:
        print(f"VERDICT: still {gap:.4f} off even with the paper's own run used for both retrievalresults and background "
              "(i.e. their exact SET_Top200 setup) -- an implementation issue independent of BM25 candidates and background "
              "methodology likely remains (e.g. document neutrality/tokenization differences, position-bias formula, or "
              "wordlist/threshold differences from the paper's own) and is worth investigating further.")


if __name__ == "__main__":
    main()

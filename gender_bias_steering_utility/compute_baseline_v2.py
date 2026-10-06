"""Recompute collection neutrality scores with punctuation-stripped tokenization (v2), and
rerun the BM25 baseline's NFaiRR with them -- fairr_full_collection.py's full-collection
background logic is reused unmodified. MRR@10/nDCG@10/Recall@10 aren't touched by neutrality
scores at all, so they're carried over from the existing v1 baseline output rather than rerun.

Nothing here overwrites the v1 files (collection_neutralityscores.tsv,
bm25_baseline_metrics.tsv, bm25_baseline_perquery.json) -- v2 is written alongside them so
both remain available for comparison. collection_neutralityscores_v2.tsv is the corrected
neutrality file and becomes the new default input for the 5 dense-model baselines and every
steered condition going forward.

Usage (from the repo root):
    conda run -n gender_bias python gender_bias_steering_utility/compute_baseline_v2.py
"""
import json
from pathlib import Path

from tqdm import tqdm

# import order matters here: document_neutrality_fixed.py points at the adversarial_mitigation
# copy of document_neutrality.py (per this task's given path) and, since Python caches modules
# by name, importing it first makes that the copy everything below resolves "document_neutrality"
# to. (In practice this is moot -- that copy and the measurement/ one compute_baseline_metrics.py
# uses are byte-identical except trailing whitespace -- but this keeps the import path honest.)
from document_neutrality_fixed import PunctuationStrippedDocumentNeutrality  # noqa: E402
from compute_baseline_metrics import (  # noqa: E402
    BM25_RUN_PATH, TEST_QUERIES_PATH, OUTPUT_DIR, FAIRR_THRESHOLDS, COLLECTION_SIZE_HINT,
    COLLECTION_PATH, GENDERED_TERMS_PATH,
    make_headerless_qids_file, check_missing_neutrality_docs, read_old_metrics,
)
from evaluate_run import read_qids  # noqa: E402  (same directory as this script)
from fairr_full_collection import FaiRRMetricHelper, load_fairr_metric, compute_nfairr_full_collection_background  # noqa: E402

V1_NEUTRALITY_PATH = OUTPUT_DIR / "collection_neutralityscores.tsv"
V2_NEUTRALITY_PATH = OUTPUT_DIR / "collection_neutralityscores_v2.tsv"
V1_METRICS_PATH = OUTPUT_DIR / "bm25_baseline_metrics.tsv"
V2_METRICS_PATH = OUTPUT_DIR / "bm25_baseline_metrics_v2.tsv"
V1_PERQUERY_PATH = OUTPUT_DIR / "bm25_baseline_perquery.json"
V2_PERQUERY_PATH = OUTPUT_DIR / "bm25_baseline_perquery_v2.json"

# NFaiRR from the very first baseline run (top-100-pool background, raw split(' ') tokenization
# -- neither fix applied yet). Not reproducible from any file on disk any more: the full-collection-
# background fix and now this tokenization fix have each since overwritten bm25_baseline_metrics.tsv
# in place. Recorded here from that run's own verified output for the three-way comparison below.
ORIGINAL_TOP100BG_RAWTOK_NFAIRR = {5: 0.8290, 10: 0.8427, 20: 0.8609, 50: 0.8808}

RETRIEVAL_LABELS = ["MRR@10", "nDCG@10", "Recall@10"]  # unaffected by neutrality scores; carried over from v1, not rerun


def compute_or_load_neutrality_scores_v2(collection_path, wordlist_path, out_path):
    if out_path.exists():
        print(f"reusing existing v2 neutrality scores at {out_path}")
        return out_path

    print(f"computing PUNCTUATION-STRIPPED document neutrality scores over the full collection -> {out_path} "
          f"(one-time cost; this becomes the default neutrality file for the 5 dense-model baselines and every steered condition)")
    doc_neutrality = PunctuationStrippedDocumentNeutrality(representative_words_path=str(wordlist_path), threshold=1, groups_portion={"f": 0.5, "m": 0.5})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf8") as fw, open(collection_path, "r", encoding="utf8") as fr:
        for line in tqdm(fr, total=COLLECTION_SIZE_HINT, desc="neutrality scores v2", unit=" docs", unit_scale=True):
            vals = line.strip().split("\t")
            if len(vals) != 2:
                continue
            docid, doctext = vals
            fw.write("%s\t%f\n" % (docid, doc_neutrality.get_neutrality_from_text(doctext)))
    return out_path


def count_neutral(path):
    total, neutral = 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            if float(line.rstrip("\n").split("\t")[1]) == 1.0:
                neutral += 1
    return neutral, total


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    v2_path = compute_or_load_neutrality_scores_v2(COLLECTION_PATH, GENDERED_TERMS_PATH, V2_NEUTRALITY_PATH)

    v1_neutral, v1_total = count_neutral(V1_NEUTRALITY_PATH)
    v2_neutral, v2_total = count_neutral(v2_path)
    v1_pct, v2_pct = v1_neutral / v1_total * 100, v2_neutral / v2_total * 100
    print("\nneutral-doc shift (score == 1.0):")
    print(f"  v1 (raw split):      {v1_neutral:,}/{v1_total:,} ({v1_pct:.2f}%)")
    print(f"  v2 (punct-stripped): {v2_neutral:,}/{v2_total:,} ({v2_pct:.2f}%)")
    print(f"  absolute change: {v2_neutral - v1_neutral:,} fewer neutral docs ({v2_pct - v1_pct:+.2f} percentage points)")

    qids_path = OUTPUT_DIR / "_test_qids.tsv"
    make_headerless_qids_file(TEST_QUERIES_PATH, qids_path)
    test_qids = read_qids(qids_path)  # reused as-is from evaluate_run.py
    print(f"\n{len(test_qids)} test queries loaded from {TEST_QUERIES_PATH}")

    check_missing_neutrality_docs(BM25_RUN_PATH, v2_path)
    fairr_metric = load_fairr_metric(v2_path)

    retrievalresults_all = FaiRRMetricHelper().read_retrievalresults_from_runfile(str(BM25_RUN_PATH), cut_off=100)
    test_qids_int = {int(q) for q in test_qids}
    retrievalresults = {q: v for q, v in retrievalresults_all.items() if q in test_qids_int}
    print(f"NFaiRR (v2): {len(retrievalresults)} of {len(test_qids_int)} test queries have BM25 candidates")

    result = compute_nfairr_full_collection_background(fairr_metric, retrievalresults, FAIRR_THRESHOLDS)
    print("one-time IFaiRR v2 (full collection, punct-stripped tokenization, same for every query): "
          + ", ".join(f"@{th}={result['ideal_fairr'][th]:.4f}" for th in FAIRR_THRESHOLDS))

    nfairr_v2 = {f"NFaiRR@{th}": result["metrics_avg"]["NFaiRR"][th] for th in FAIRR_THRESHOLDS}
    nfairr_v2_perq = {f"NFaiRR@{th}": {str(qid): v for qid, v in result["metrics_perq"]["NFaiRR"][th].items()} for th in FAIRR_THRESHOLDS}

    v1_metrics = read_old_metrics(V1_METRICS_PATH)
    all_metrics_v2 = {label: v1_metrics[label] for label in RETRIEVAL_LABELS if label in v1_metrics}
    all_metrics_v2.update(nfairr_v2)
    with open(V2_METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("metric\tvalue\n")
        for label, value in all_metrics_v2.items():
            f.write(f"{label}\t{value:.4f}\n")

    v1_perquery = json.loads(V1_PERQUERY_PATH.read_text()) if V1_PERQUERY_PATH.exists() else {}
    perquery_v2 = {
        "retrieval_effectiveness": v1_perquery.get("retrieval_effectiveness", {}),  # unaffected by neutrality; carried over, not rerun
        "fairness_NFaiRR": {
            "query_set": f"test ({len(test_qids)} queries) retrieval results; full MS MARCO collection as background "
                         f"(punctuation-stripped tokenization, v2)",
            **nfairr_v2_perq,
        },
    }
    with open(V2_PERQUERY_PATH, "w", encoding="utf-8") as f:
        json.dump(perquery_v2, f, indent=2)

    print(f"\nwrote {V2_NEUTRALITY_PATH}, {V2_METRICS_PATH}, {V2_PERQUERY_PATH}")

    print("\nthree-way NFaiRR comparison:")
    header = f"{'':>10} {'orig (top100 bg, raw tok)':>28} {'full-collection bg only':>26} {'+ fixed tokenization (this fix)':>34}"
    print(header)
    for th in FAIRR_THRESHOLDS:
        orig = ORIGINAL_TOP100BG_RAWTOK_NFAIRR[th]
        bg_fix_only = v1_metrics.get(f"NFaiRR@{th}")
        both_fixes = nfairr_v2[f"NFaiRR@{th}"]
        bg_fix_str = f"{bg_fix_only:.4f}" if bg_fix_only is not None else "n/a"
        print(f"{'@' + str(th):>10} {orig:>28.4f} {bg_fix_str:>26} {both_fixes:>34.4f}")


if __name__ == "__main__":
    main()

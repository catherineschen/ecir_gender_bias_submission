"""Compute the BM25 baseline: MRR@10, nDCG@10, Recall@10, and NFaiRR@{5,10,20,50}.

Retrieval-effectiveness metrics reuse evaluate_run.py's own read_qids() helper and the
exact ir_measures calls its main() makes (that main() only prints/writes aggregates, so
its measures/averaging logic is mirrored here to also capture the per-query values for
the JSON output). Fairness metrics reuse document_neutrality.py's DocumentNeutrality
class (only if the neutrality score file needs (re)computing) and, via
fairr_full_collection.py, metrics_fairness.py's FaiRRMetric for the FaiRR numerator --
NFaiRR's denominator (IFaiRR) is computed from the full collection, not a per-query
background pool; see fairr_full_collection.py for why. All metrics are computed over the
same 210 test queries.

No new metric logic is implemented here; this only wires the existing scripts together.

Usage (from the repo root):
    conda run -n gender_bias python gender_bias_steering_utility/compute_baseline_metrics.py
"""
import json
import sys
from pathlib import Path

import ir_measures
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# ---- paths (edit here, or copy this script and change these constants) ----
BM25_RUN_PATH = HERE / "output/msmarco_passage.dev.fair.FULL_ANNOTATION.bm25.top100.trec"
TEST_QUERIES_PATH = HERE / "preprocess/n5_seed42/test_queries.tsv"
QRELS_PATH = REPO_ROOT / "data/msmarco/msmarco.qrels.dev.tsv"
COLLECTION_PATH = REPO_ROOT / "data/msmarco/collection.tsv"
GENDERED_TERMS_PATH = REPO_ROOT / "data/FairnessRetrievalResults/resources/wordlist_gender_representative.txt"
NEUTRALITY_CODE_DIR = REPO_ROOT / "data/FairnessRetrievalResults/measurement"  # DocumentNeutrality, for (re)computing the score file
# "gender_bias_steer_utility/experiments/baseline" as given looks like a typo for this
# repo's actual directory name (gender_bias_steering_utility, used everywhere else) --
# using the real directory here; see the chat reply for a note on this.
OUTPUT_DIR = HERE / "experiments/baseline"

COLLECTION_SIZE_HINT = 8841823  # only used to size the tqdm progress bar
FAIRR_THRESHOLDS = [5, 10, 20, 50]
K = 10  # cutoff for MRR / nDCG / Recall
MIN_REL = 1  # MS MARCO dev qrels are binary

sys.path.insert(0, str(NEUTRALITY_CODE_DIR))
from document_neutrality import DocumentNeutrality  # noqa: E402
from evaluate_run import read_qids  # noqa: E402  (same directory as this script)
from fairr_full_collection import FaiRRMetricHelper, load_fairr_metric, compute_nfairr_full_collection_background  # noqa: E402


def compute_or_load_neutrality_scores(collection_path, wordlist_path, out_path):
    """Mirrors calc_documents_neutrality.py's logic, adding the reuse-if-present check."""
    if out_path.exists():
        print(f"reusing existing neutrality scores at {out_path}")
        return out_path

    print(f"computing document neutrality scores over the full collection -> {out_path} (one-time cost, reused for every later model)")
    doc_neutrality = DocumentNeutrality(representative_words_path=str(wordlist_path), threshold=1, groups_portion={"f": 0.5, "m": 0.5})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf8") as fw, open(collection_path, "r", encoding="utf8") as fr:
        for line in tqdm(fr, total=COLLECTION_SIZE_HINT, desc="neutrality scores", unit=" docs", unit_scale=True):
            vals = line.strip().split("\t")
            if len(vals) != 2:
                continue
            docid, doctext = vals
            doctokens = doctext.lower().split(" ")  # matches calc_documents_neutrality.py's tokenization exactly
            fw.write("%s\t%f\n" % (docid, doc_neutrality.get_neutrality(doctokens)))
    return out_path


def make_headerless_qids_file(source_path, dest_path):
    """evaluate_run.py's read_qids() only recognizes a 'QID\\t...' header; test_queries.tsv's
    header is 'query_id\\t...', so passing it directly would smuggle in "query_id" as a bogus
    qid. Stripping the header keeps every remaining line a real qid, which read_qids handles
    correctly without needing any change to evaluate_run.py itself."""
    with open(source_path, encoding="utf-8") as f, open(dest_path, "w", encoding="utf-8") as g:
        next(f)  # header
        for line in f:
            if line.strip():
                g.write(line)


def check_missing_neutrality_docs(run_path, neutrality_scores_path):
    run_docids = set()
    with open(run_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 6:
                run_docids.add(int(parts[2]))
    scored_docids = set()
    with open(neutrality_scores_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scored_docids.add(int(line.split("\t")[0]))
    missing = run_docids - scored_docids
    if missing:
        print(f"WARNING: {len(missing)} of {len(run_docids)} unique docs in the run file are missing from the neutrality score file "
              f"(sample: {sorted(missing)[:5]})")
    else:
        print(f"all {len(run_docids)} unique docs in the run file have neutrality scores")
    return missing


def compute_retrieval_effectiveness(run_path, qrels_path, qids, min_rel=MIN_REL):
    """Same measures/averaging evaluate_run.py's main() uses (via its read_qids/ir_measures
    calls), just also keeping the per-query values it doesn't expose."""
    qrels = [q for q in ir_measures.read_trec_qrels(str(qrels_path)) if q.query_id in qids]
    run = [d for d in ir_measures.read_trec_run(str(run_path)) if d.query_id in qids]
    unjudged = qids - {q.query_id for q in qrels}
    no_results = qids - {d.query_id for d in run}
    evaluated = sorted(qids - unjudged)
    print(f"{len(evaluated)} queries evaluated ({len(unjudged)} without qrels, {len(no_results)} with no retrieved docs counted as 0)")

    measures = [ir_measures.RR(rel=min_rel) @ K, ir_measures.nDCG @ K, ir_measures.R(rel=min_rel) @ K]
    labels = {measures[0]: "MRR@10", measures[1]: "nDCG@10", measures[2]: "Recall@10"}
    per_query = {m: {} for m in measures}
    for metric in ir_measures.iter_calc(measures, qrels, run):
        per_query[metric.measure][metric.query_id] = metric.value

    aggregate, perquery_out = {}, {}
    for m in measures:
        label = labels[m]
        values = {qid: per_query[m].get(qid, 0.0) for qid in evaluated}
        perquery_out[label] = values
        aggregate[label] = sum(values.values()) / len(values)
    return aggregate, perquery_out


def compute_nfairr(run_path, neutrality_scores_path, test_qids):
    check_missing_neutrality_docs(run_path, neutrality_scores_path)

    fairr_metric = load_fairr_metric(neutrality_scores_path)
    neutral_count = sum(1 for v in fairr_metric.documents_neutrality.values() if v == 1.0)
    total_count = len(fairr_metric.documents_neutrality)
    print(f"collection neutrality: {neutral_count}/{total_count} docs are fully neutral (score 1.0), "
          f"{total_count - neutral_count} are <1.0")

    retrievalresults_all = FaiRRMetricHelper().read_retrievalresults_from_runfile(str(run_path), cut_off=100)
    test_qids_int = {int(q) for q in test_qids}
    retrievalresults = {q: v for q, v in retrievalresults_all.items() if q in test_qids_int}
    print(f"NFaiRR: {len(retrievalresults)} of {len(test_qids_int)} test queries have BM25 candidates")

    result = compute_nfairr_full_collection_background(fairr_metric, retrievalresults, FAIRR_THRESHOLDS)
    print(f"one-time IFaiRR (full collection, same for every query): "
          + ", ".join(f"@{th}={result['ideal_fairr'][th]:.4f}" for th in FAIRR_THRESHOLDS))

    aggregate = {f"NFaiRR@{th}": result["metrics_avg"]["NFaiRR"][th] for th in FAIRR_THRESHOLDS}
    perquery_out = {f"NFaiRR@{th}": {str(qid): v for qid, v in result["metrics_perq"]["NFaiRR"][th].items()} for th in FAIRR_THRESHOLDS}
    return aggregate, perquery_out


def read_old_metrics(path):
    if not path.exists():
        return {}
    old = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            if line.strip():
                label, value = line.rstrip("\n").split("\t")
                old[label] = float(value)
    return old


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "bm25_baseline_metrics.tsv"
    old_metrics = read_old_metrics(metrics_path)  # grabbed before overwriting, for the old-vs-new NFaiRR print below

    neutrality_scores_path = compute_or_load_neutrality_scores(COLLECTION_PATH, GENDERED_TERMS_PATH, OUTPUT_DIR / "collection_neutralityscores.tsv")

    qids_path = OUTPUT_DIR / "_test_qids.tsv"
    make_headerless_qids_file(TEST_QUERIES_PATH, qids_path)
    test_qids = read_qids(qids_path)  # reused as-is from evaluate_run.py
    print(f"{len(test_qids)} test queries loaded from {TEST_QUERIES_PATH}")

    retrieval_aggregate, retrieval_perquery = compute_retrieval_effectiveness(BM25_RUN_PATH, QRELS_PATH, test_qids)
    fairr_aggregate, fairr_perquery = compute_nfairr(BM25_RUN_PATH, neutrality_scores_path, test_qids)

    all_metrics = {**retrieval_aggregate, **fairr_aggregate}
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("metric\tvalue\n")
        for label, value in all_metrics.items():
            f.write(f"{label}\t{value:.4f}\n")

    perquery = {
        "retrieval_effectiveness": {"query_set": f"test ({len(test_qids)} queries)", **retrieval_perquery},
        "fairness_NFaiRR": {
            "query_set": f"test ({len(test_qids)} queries) retrieval results; full MS MARCO collection as background",
            **fairr_perquery,
        },
    }
    with open(OUTPUT_DIR / "bm25_baseline_perquery.json", "w", encoding="utf-8") as f:
        json.dump(perquery, f, indent=2)

    print(f"wrote outputs to {OUTPUT_DIR}")
    for label, value in all_metrics.items():
        print(f"{label}: {value:.4f}")

    print("\nold (top-100-pool background) vs new (full-collection background) NFaiRR:")
    for th in FAIRR_THRESHOLDS:
        label = f"NFaiRR@{th}"
        old_val = old_metrics.get(label)
        old_str = f"{old_val:.4f}" if old_val is not None else "n/a"
        print(f"{label}: {old_str} -> {all_metrics[label]:.4f}")


if __name__ == "__main__":
    main()

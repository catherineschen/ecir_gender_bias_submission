"""Unsteered baseline metrics (MRR@10/nDCG@10/Recall@10, NFaiRR@{5,10,20,50}) for the 5
sentence-transformers dense retrieval models, reranking the existing BM25 top-100 test
candidates.

Each unique test query and each unique candidate doc is encoded exactly once per model
(210 queries, ~20,374 docs) and looked up by qid/docid when scoring a query's candidates,
rather than re-encoding the same query or doc once per (query, candidate) pair -- 21,000
query-doc pairs would otherwise mean up to 21,000 redundant query encodes alone.

Retrieval-effectiveness metrics reuse compute_baseline_metrics.py's
compute_retrieval_effectiveness() (itself built on evaluate_run.py's read_qids() and
ir_measures calls) as-is. NFaiRR reuses fairr_full_collection.py's full-collection-background
implementation as-is, loading collection_neutralityscores_v2.tsv once and sharing that one
FaiRRMetric instance across all 5 models rather than reloading the ~8.8M-row file 5 times.

DATASET=trecdl runs the same baselines on TREC DL 2019 Fair: all 30 queries are the heldout
evaluation set (no candidate/validation/test split), reranking the existing BM25 top-100 run.
TREC DL qrels are graded (0-3), so MRR@10 and Recall@10 count rel>=2 as relevant (nDCG@10 uses
the grades as-is); the msmarco default keeps rel>=1. The BM25 row is computed here from the BM25
run (there is no precomputed BM25 metrics file for TREC DL). Default (DATASET unset) behavior is
unchanged.

Usage (from the repo root; PYTORCH_ENABLE_MPS_FALLBACK is also set defensively below, but
setting it in the environment is the documented, sure way to get it):
    PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n gender_bias python gender_bias_steering_utility/compute_dense_baselines.py
    PYTORCH_ENABLE_MPS_FALLBACK=1 SEED=29 conda run -n gender_bias python gender_bias_steering_utility/compute_dense_baselines.py   # other MS MARCO seed
    PYTORCH_ENABLE_MPS_FALLBACK=1 DATASET=trecdl conda run -n gender_bias python gender_bias_steering_utility/compute_dense_baselines.py
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # belt-and-suspenders; set it in the env too (see module docstring)

import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

DATASET = os.environ.get("DATASET", "msmarco")
# SEED selects which n5_seed{SEED} split's test candidates/queries to score (msmarco only). Seed 42
# (default) is byte-for-byte what it was; other seeds read preprocess/n5_seed{SEED}/ and write to
# their own sibling experiments/baseline/sentence-transformers_seed{SEED}/, since each seed's
# construction-doc dedup leaves a slightly different BM25 candidate pool.
SEED = int(os.environ.get("SEED", 42))
SEED_SUFFIX = "" if SEED == 42 else f"_seed{SEED}"
DATASET_CONFIGS = {
    "msmarco": {
        "candidates": HERE / f"preprocess/n5_seed{SEED}/test_candidates.tsv",
        "queries": HERE / f"preprocess/n5_seed{SEED}/test_queries.tsv",
        "qrels": REPO_ROOT / "data/msmarco/msmarco.qrels.dev.tsv",
        "min_rel": 1,
        "output_dir": HERE / f"experiments/baseline/sentence-transformers{SEED_SUFFIX}",
        "query_set": "test",
    },
    "trecdl": {
        "candidates": HERE / "data/trecdeep19_passage.fair.bm25.top100.trec",  # headerless 6-col TREC run
        "queries": REPO_ROOT / "data/FairnessRetrievalResults/dataset/trecdeep19_passage.fair.tsv",  # headerless
        "qrels": REPO_ROOT / "data/msmarco/qrels.trec-dl-2019.ir_datasets.tsv",
        "min_rel": 2,
        "output_dir": HERE / "experiments/baseline/trecdl19/sentence-transformers",
        "query_set": "TREC DL 2019 Fair, all queries as heldout",
    },
}
CFG = DATASET_CONFIGS[DATASET]
TEST_CANDIDATES_PATH = CFG["candidates"]
TEST_QUERIES_PATH = CFG["queries"]
QRELS_PATH = CFG["qrels"]
MIN_REL = CFG["min_rel"]
COLLECTION_PATH = REPO_ROOT / "data/msmarco/collection.tsv"
V2_NEUTRALITY_PATH = HERE / "experiments/baseline/collection_neutralityscores_v2.tsv"
BM25_METRICS_PATH = HERE / "experiments/baseline/bm25_baseline_metrics_v2.tsv"  # msmarco only
OUTPUT_DIR = CFG["output_dir"]

COLLECTION_SIZE_HINT = 8841823
BATCH_SIZE = 64
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MODELS = {
    "sentence-transformers/msmarco-distilbert-base-tas-b": "cls",
    "sentence-transformers/multi-qa-distilbert-dot-v1": "cls",
    "sentence-transformers/multi-qa-MiniLM-L6-dot-v1": "cls",
    "sentence-transformers/msmarco-bert-base-dot-v5": "mean",
    "sentence-transformers/msmarco-distilbert-dot-v5": "mean",
}  # all sim_func="dot" -> normalize_embeddings=False for every model; pool is informational only (ST bakes pooling into the model)

from compute_baseline_metrics import FAIRR_THRESHOLDS, check_missing_neutrality_docs, compute_retrieval_effectiveness  # noqa: E402
from fairr_full_collection import FaiRRMetricHelper, load_fairr_metric, compute_nfairr_full_collection_background  # noqa: E402


def short_name(model_name):
    return model_name.rsplit("/", 1)[-1]


def load_test_queries(path):
    """qid<TAB>text; a header row (test_queries.tsv's "query_id\\tquery_text") is skipped if present,
    the TREC DL queries file has none."""
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qid, text = line.rstrip("\n").split("\t")[:2]
                if not qid.isdigit():
                    continue
                queries[qid] = text
    return queries


def load_test_candidates(path):
    """Returns {qid: [(docid, bm25_rank), ...]} in BM25 rank order, and the set of unique docids.
    Handles test_candidates.tsv (tab-separated, header row, extra columns) and a plain headerless
    whitespace-separated TREC run."""
    by_qid, docids = {}, set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, _q0, docid, rank, *_ = line.split("\t") if "\t" in line else line.split()
            if not qid.isdigit():
                continue  # header row
            by_qid.setdefault(qid, []).append((docid, int(rank)))
            docids.add(docid)
    for qid in by_qid:
        by_qid[qid].sort(key=lambda x: x[1])
    return by_qid, docids


def fetch_doc_texts(collection_path, needed_docids):
    """Single targeted pass over the (large) collection file, keeping only the docids we need."""
    texts = {}
    remaining = set(needed_docids)
    with open(collection_path, encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            pid, _, text = line.partition("\t")
            if pid in remaining:
                texts[pid] = text.rstrip("\n")
                remaining.discard(pid)
    if remaining:
        print(f"WARNING: {len(remaining)} candidate docids were not found in the collection")
    return texts


def write_run_file(path, ranked_by_qid):
    with open(path, "w", encoding="utf-8") as f:
        for qid, ranked in ranked_by_qid.items():
            for rank, (docid, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} dense\n")


def score_model(model_name, device, queries, doc_texts, candidates_by_qid):
    print(f"\n=== {model_name} (device={device}) ===")
    st = SentenceTransformer(model_name, device=device)

    query_ids = sorted(queries, key=int)
    doc_ids = sorted(doc_texts, key=int)

    query_emb = st.encode([queries[q] for q in query_ids], batch_size=BATCH_SIZE, convert_to_numpy=True,
                           normalize_embeddings=False, show_progress_bar=True)
    doc_emb = st.encode([doc_texts[d] for d in doc_ids], batch_size=BATCH_SIZE, convert_to_numpy=True,
                         normalize_embeddings=False, show_progress_bar=True)
    query_vec = dict(zip(query_ids, query_emb))
    doc_vec = dict(zip(doc_ids, doc_emb))

    ranked_by_qid = {}
    for qid, candidates in candidates_by_qid.items():
        q = query_vec[qid]
        docids = [d for d, _rank in candidates]
        scores = np.array([doc_vec[d] for d in docids]) @ q
        order = np.lexsort((docids, -scores))  # score desc, ties broken by docid for determinism
        ranked_by_qid[qid] = [(docids[i], float(scores[i])) for i in order]

    del st
    return ranked_by_qid


def evaluate_run_file(run_path, retrievalresults, test_qids, fairr_metric):
    retrieval_aggregate, retrieval_perquery = compute_retrieval_effectiveness(run_path, QRELS_PATH, test_qids, min_rel=MIN_REL)
    check_missing_neutrality_docs(run_path, V2_NEUTRALITY_PATH)
    fairr_result = compute_nfairr_full_collection_background(fairr_metric, retrievalresults, FAIRR_THRESHOLDS)
    nfairr_aggregate = {f"NFaiRR@{th}": fairr_result["metrics_avg"]["NFaiRR"][th] for th in FAIRR_THRESHOLDS}
    nfairr_perquery = {f"NFaiRR@{th}": {str(qid): v for qid, v in fairr_result["metrics_perq"]["NFaiRR"][th].items()} for th in FAIRR_THRESHOLDS}
    return retrieval_aggregate, retrieval_perquery, nfairr_aggregate, nfairr_perquery


def compute_bm25_row(fairr_metric, test_qids, candidates_by_qid):
    """BM25 metrics from the candidate pool itself (TREC DL and non-42 seeds have no precomputed
    BM25 metrics file). Written out as a plain 6-col run so the same ir_measures path is used."""
    run_path = OUTPUT_DIR / "bm25_unsteered.trec"
    with open(run_path, "w", encoding="utf-8") as f:
        for qid, cands in candidates_by_qid.items():
            for docid, rank in cands:
                f.write(f"{qid} Q0 {docid} {rank} {1000 - rank:.6f} bm25\n")
    retrievalresults = {int(qid): [int(d) for d, _r in cands] for qid, cands in candidates_by_qid.items()}
    retrieval_aggregate, retrieval_perquery, nfairr_aggregate, nfairr_perquery = evaluate_run_file(run_path, retrievalresults, test_qids, fairr_metric)
    with open(OUTPUT_DIR / "bm25_baseline_perquery.json", "w", encoding="utf-8") as f:
        json.dump({"retrieval_effectiveness": {"query_set": f"{CFG['query_set']} ({len(test_qids)} queries)", **retrieval_perquery},
                   "fairness_NFaiRR": {"query_set": f"{CFG['query_set']} ({len(retrievalresults)} queries); full MS MARCO collection as background", **nfairr_perquery}}, f, indent=2)
    return {**retrieval_aggregate, **nfairr_aggregate}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    queries = load_test_queries(TEST_QUERIES_PATH)
    candidates_by_qid, needed_docids = load_test_candidates(TEST_CANDIDATES_PATH)
    print(f"{len(queries)} test queries, {len(candidates_by_qid)} with candidates, {len(needed_docids)} unique candidate docs")

    doc_texts = fetch_doc_texts(COLLECTION_PATH, needed_docids)

    test_qids = set(queries)  # same set evaluate_run.read_qids() returns for the headerless qids file
    if DATASET == "trecdl":
        assert set(candidates_by_qid) == test_qids, "every TREC DL query should have BM25 candidates"
        construction = set()
        for path in HERE.glob("preprocess/n5_seed*/construction_queries.tsv"):
            construction |= set(load_test_queries(path))
        assert not (construction & test_qids), f"TREC DL queries overlap construction queries: {construction & test_qids}"

    fairr_metric = load_fairr_metric(V2_NEUTRALITY_PATH)  # loaded once, shared across all 5 models (~8.8M rows)

    per_model_metrics = {}
    per_model_perquery = {}
    for model_name, pool in MODELS.items():
        name = short_name(model_name)
        ranked_by_qid = score_model(model_name, DEVICE, queries, doc_texts, candidates_by_qid)

        run_path = OUTPUT_DIR / f"{name}_unsteered.trec"
        write_run_file(run_path, ranked_by_qid)

        retrievalresults = {int(qid): [int(d) for d, _s in ranked] for qid, ranked in ranked_by_qid.items()}
        retrieval_aggregate, retrieval_perquery, nfairr_aggregate, nfairr_perquery = evaluate_run_file(run_path, retrievalresults, test_qids, fairr_metric)

        per_model_metrics[name] = {"model": model_name, "pool": pool, **retrieval_aggregate, **nfairr_aggregate}
        per_model_perquery[name] = {
            "retrieval_effectiveness": {"query_set": f"{CFG['query_set']} ({len(test_qids)} queries)", **retrieval_perquery},
            "fairness_NFaiRR": {"query_set": f"{CFG['query_set']} ({len(retrievalresults)} queries); full MS MARCO collection as background", **nfairr_perquery},
        }
        print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in per_model_metrics[name].items() if isinstance(v, float)))

    metric_labels = ["MRR@10", "nDCG@10", "Recall@10"] + [f"NFaiRR@{th}" for th in FAIRR_THRESHOLDS]
    with open(OUTPUT_DIR / "dense_baseline_metrics.tsv", "w", encoding="utf-8") as f:
        f.write("model\tpool\t" + "\t".join(metric_labels) + "\n")
        for name, m in per_model_metrics.items():
            f.write(f"{m['model']}\t{m['pool']}\t" + "\t".join(f"{m[label]:.4f}" for label in metric_labels) + "\n")

    with open(OUTPUT_DIR / "dense_baseline_perquery.json", "w", encoding="utf-8") as f:
        json.dump(per_model_perquery, f, indent=2)

    # combined summary: BM25 (carried over, not rerun) + the 5 dense models
    if DATASET == "msmarco" and SEED == 42:
        bm25_metrics = {}
        with open(BM25_METRICS_PATH, encoding="utf-8") as f:
            next(f)  # header
            for line in f:
                if line.strip():
                    label, value = line.rstrip("\n").split("\t")
                    bm25_metrics[label] = float(value)
    else:
        bm25_metrics = compute_bm25_row(fairr_metric, test_qids, candidates_by_qid)

    summary_rows = [("BM25", "-", bm25_metrics)] + [(m["model"], m["pool"], m) for m in per_model_metrics.values()]
    with open(OUTPUT_DIR / "all_baselines_summary.tsv", "w", encoding="utf-8") as f:
        f.write("model\tpool\t" + "\t".join(metric_labels) + "\n")
        for model, pool, m in summary_rows:
            f.write(f"{model}\t{pool}\t" + "\t".join(f"{m[label]:.4f}" for label in metric_labels) + "\n")

    print(f"\nwrote outputs to {OUTPUT_DIR}")
    print("\ncombined summary (BM25 + 5 dense models):")
    header = f"{'model':<45} {'pool':<6} " + " ".join(f"{l:>10}" for l in metric_labels)
    print(header)
    for model, pool, m in summary_rows:
        print(f"{model:<45} {pool:<6} " + " ".join(f"{m[label]:>10.4f}" for label in metric_labels))


if __name__ == "__main__":
    main()

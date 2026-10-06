"""Compute MRR@k, nDCG@k and Recall@k for a TREC run file against MS MARCO qrels.

Metrics are averaged over every query in --queries (default: the MS MARCO dev fairness
queries), so a query with no retrieved documents scores 0 instead of being dropped.

Usage (from the repo root):
    conda run -n gender_bias python gender_bias_steering_utility/evaluate_run.py
"""
import argparse
from pathlib import Path

import ir_measures
from ir_measures import Qrel, ScoredDoc

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
DEFAULT_RUN = HERE / "output/msmarco_passage.dev.fair.bm25.top100.trec"
DEFAULT_QRELS = DATA_DIR / "msmarco/qrels.dev.ir_datasets.tsv"  # ir_datasets msmarco-passage/dev
DEFAULT_QUERIES = DATA_DIR / "FairnessRetrievalResults/dataset/msmarco_passage.dev.fair.tsv"


def read_qids(path):
    with open(path, encoding="utf-8") as f:
        return {line.split("\t")[0] for line in f if line.strip() and not line.startswith("QID\t")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES, help="TSV whose first column is the QIDs to evaluate")
    parser.add_argument("--output", type=Path, default=None, help="also write the metrics to this TSV file")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-rel", type=int, default=1, help="minimum grade counted as relevant by MRR and recall (TREC DL uses 2)")
    args = parser.parse_args()

    qids = read_qids(args.queries)
    qrels = [q for q in ir_measures.read_trec_qrels(str(args.qrels)) if q.query_id in qids]
    run = [d for d in ir_measures.read_trec_run(str(args.run)) if d.query_id in qids]
    unjudged = qids - {q.query_id for q in qrels}
    no_results = qids - {d.query_id for d in run}

    measures = [ir_measures.RR(rel=args.min_rel) @ args.k, ir_measures.nDCG @ args.k, ir_measures.R(rel=args.min_rel) @ args.k]
    per_query = {m: {} for m in measures}
    for metric in ir_measures.iter_calc(measures, qrels, run):
        per_query[metric.measure][metric.query_id] = metric.value

    evaluated = sorted(qids - unjudged)
    summary = f"{len(evaluated)} queries evaluated ({len(unjudged)} without qrels, {len(no_results)} with no retrieved docs counted as 0)"
    print(summary)
    scores = []
    for m in measures:
        mean = sum(per_query[m].get(qid, 0.0) for qid in evaluated) / len(evaluated)
        label = f"M{m}" if str(m).startswith("RR") else str(m)  # mean of per-query RR is MRR
        scores.append((label, mean))
        print(f"{label}: {mean:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(f"# run: {args.run}\n# qrels: {args.qrels}\n# queries: {args.queries}\n# {summary}\n")
            f.write("metric\tvalue\n")
            for label, mean in scores:
                f.write(f"{label}\t{mean:.4f}\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

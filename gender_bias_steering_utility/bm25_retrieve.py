"""Retrieve the top-k MS MARCO passages for each fairness-sensitive query with PyTerrier BM25.

Queries come from the Fairness Retrieval dataset (QID<TAB>query; extra annotation
columns and a QID/Query header row, as in the FULL_ANNOTATION files, are ignored) and the
corpus is the MS MARCO passage collection (PID<TAB>passage, no header). The corpus is
indexed once with Terrier (default English stopwords + Porter stemming) and the index
is reused on later runs. Output is a standard TREC run file: `qid Q0 pid rank score tag`.

PyTerrier needs a JVM. If JAVA_HOME is unset, the Homebrew OpenJDK location is tried.

Usage (from the repo root):
    conda run -n gender_bias python gender_bias_steering_utility/bm25_retrieve.py
"""
import argparse
import os
import re
from pathlib import Path

import pandas as pd
import pyterrier as pt
from tqdm import tqdm

HOMEBREW_JAVA_HOME = "/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home"
if not os.environ.get("JAVA_HOME") and Path(HOMEBREW_JAVA_HOME).exists():
    os.environ["JAVA_HOME"] = HOMEBREW_JAVA_HOME

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
DEFAULT_QUERIES = DATA_DIR / "FairnessRetrievalResults/dataset/msmarco_passage.dev.fair.tsv"
DEFAULT_COLLECTION = DATA_DIR / "msmarco/collection.tsv"
DEFAULT_INDEX = HERE / "output/msmarco_passage_terrier_index"
DEFAULT_OUTPUT = HERE / "output/msmarco_passage.dev.fair.bm25.top100.trec"


def load_queries(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qid, text = line.rstrip("\n").split("\t")[:2]
                if qid == "QID":  # header row in the FULL_ANNOTATION files
                    continue
                # Terrier's query parser chokes on punctuation such as ? ' : ( )
                rows.append({"qid": qid, "query": re.sub(r"[^A-Za-z0-9]+", " ", text).strip()})
    return pd.DataFrame(rows)


def iter_collection(path, limit):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=limit or 8841823, desc="indexing", unit=" docs", unit_scale=True)):
            if limit is not None and i >= limit:
                return
            pid, text = line.rstrip("\n").split("\t", 1)
            yield {"docno": pid, "text": text}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Terrier index directory (built if missing)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k", type=int, default=100, help="candidates to retrieve per query")
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--threads", type=int, default=1, help="indexing threads")
    parser.add_argument("--limit-docs", type=int, default=None, help="only index the first N passages (for smoke tests)")
    args = parser.parse_args()

    pt.java.init()

    if not (args.index / "data.properties").exists():
        indexer = pt.terrier.IterDictIndexer(str(args.index), meta={"docno": 20}, threads=args.threads)
        indexer.index(iter_collection(args.collection, args.limit_docs))
    index = pt.terrier.IndexFactory.of(str(args.index))
    print(index.getCollectionStatistics().toString())

    queries = load_queries(args.queries)
    bm25 = pt.terrier.Retriever(
        index, wmodel="BM25", num_results=args.k, controls={"bm25.k_1": str(args.k1), "bm25.b": str(args.b)},
    )
    results = bm25.transform(queries)
    results = results.sort_values(["qid", "score"], ascending=[True, False], kind="stable")
    results["rank"] = results.groupby("qid").cumcount() + 1  # TREC ranks are 1-based

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out:
        for r in results.itertuples():
            out.write(f"{r.qid} Q0 {r.docno} {r.rank} {r.score:.6f} bm25\n")
    missing = set(queries.qid) - set(results.qid)
    print(f"{results.qid.nunique()}/{len(queries)} queries returned results, {len(results)} rows -> {args.output}")
    if missing:
        print(f"no results for qids: {sorted(missing)}")


if __name__ == "__main__":
    main()

"""Annotate BM25 candidate docs with gendered terms and compute cross-query doc overlap.

For each (query, candidate doc) pair in a TREC run file, this scans the doc's text for
whole-word matches against the gendered terms wordlist (whitespace-tokenized, so a term
that is a substring of another word, e.g. "he" inside "the", never matches). It then
reports, per query, how many of its candidates contain at least one gendered term, and,
across queries, which candidate docs are shared and by how many queries.

No model is loaded; this is plain text scanning against the MS MARCO passage collection.

Usage (from the repo root):
    conda run -n gender_bias python gender_bias_steering_utility/annotate_gendered_candidates.py
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

# ---- paths (edit here, or copy this script and change these constants) ----
REPO_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = REPO_ROOT / "gender_bias_steering_utility/output/msmarco_passage.dev.fair.FULL_ANNOTATION.bm25.top100.trec"
QUERIES_PATH = REPO_ROOT / "data/FairnessRetrievalResults/dataset/msmarco_passage.dev.fair.FULL_ANNOTATION.tsv"
GENDERED_TERMS_PATH = REPO_ROOT / "data/FairnessRetrievalResults/resources/wordlist_gender_representative.txt"
COLLECTION_PATH = REPO_ROOT / "data/msmarco/collection.tsv"
OUTPUT_DIR = REPO_ROOT / "gender_bias_steering_utility/preprocess"
COLLECTION_SIZE_HINT = 8841823  # only used to size the tqdm progress bar

FEW_GENDERED_DOCS_THRESHOLD = 5
STRIP_PUNCT_RE = re.compile(r"^\W+|\W+$")  # strips leading/trailing punctuation, not inner (e.g. "don't" stays whole)


def load_gendered_terms(path):
    """Returns {lowercased term: canonical term as listed in the wordlist}."""
    terms = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            term, _gender = line.split(",")
            terms[term.lower()] = term
    return terms


def load_queries(path):
    """Returns {qid: query text}, skipping the header row if present (FULL_ANNOTATION files)."""
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


def load_candidates(path):
    """Returns a list of dicts, one per TREC run line: qid, q0, docid, rank, score, tag."""
    candidates = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, q0, docid, rank, score, tag = line.split()
            candidates.append({"qid": qid, "q0": q0, "docid": docid, "rank": int(rank), "score": score, "tag": tag})
    return candidates


def fetch_doc_texts(collection_path, needed_docids):
    """Single pass over the (large) collection file, keeping only the docids we need."""
    texts = {}
    remaining = set(needed_docids)
    with open(collection_path, encoding="utf-8") as f:
        for line in tqdm(f, total=COLLECTION_SIZE_HINT, desc="scanning collection", unit=" docs", unit_scale=True):
            if not remaining:
                break
            pid, _, text = line.partition("\t")
            if pid in remaining:
                texts[pid] = text.rstrip("\n")
                remaining.discard(pid)
    if remaining:
        print(f"warning: {len(remaining)} candidate docids were not found in the collection")
    return texts


def find_gendered_terms(text, gendered_terms):
    """Whole-word, case-insensitive match against whitespace tokens. Returns (terms_found, positions), one entry per occurrence."""
    terms_found, positions = [], []
    for pos, token in enumerate(text.split()):
        normalized = STRIP_PUNCT_RE.sub("", token).lower()
        canonical = gendered_terms.get(normalized)
        if canonical is not None:
            terms_found.append(canonical)
            positions.append(pos)
    return terms_found, positions


def annotate_candidates(candidates, doc_texts, gendered_terms):
    for c in candidates:
        terms_found, positions = find_gendered_terms(doc_texts.get(c["docid"], ""), gendered_terms)
        c["has_gendered_terms"] = bool(terms_found)
        c["gendered_terms_found"] = terms_found
        c["gendered_term_positions"] = positions
    return candidates


def write_annotated_candidates(candidates, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("qid\tq0\tdocid\trank\tscore\ttag\thas_gendered_terms\tgendered_terms_found\tgendered_term_positions\n")
        for c in candidates:
            f.write(
                f"{c['qid']}\t{c['q0']}\t{c['docid']}\t{c['rank']}\t{c['score']}\t{c['tag']}\t"
                f"{c['has_gendered_terms']}\t{','.join(c['gendered_terms_found'])}\t"
                f"{','.join(map(str, c['gendered_term_positions']))}\n"
            )


def compute_query_stats(candidates, all_qids):
    by_qid = defaultdict(list)
    for c in candidates:
        by_qid[c["qid"]].append(c)

    stats = []
    for qid in all_qids:  # iterate every query, including ones with zero candidates
        docs = by_qid.get(qid, [])
        total = len(docs)
        with_gendered = sum(1 for d in docs if d["has_gendered_terms"])
        pct = (with_gendered / total * 100) if total else 0.0
        stats.append({"query_id": qid, "total_candidates": total, "candidates_with_gendered_terms": with_gendered, "percentage": round(pct, 2)})
    return stats


def write_query_stats(stats, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("query_id\ttotal_candidates\tcandidates_with_gendered_terms\tpercentage\n")
        for s in stats:
            f.write(f"{s['query_id']}\t{s['total_candidates']}\t{s['candidates_with_gendered_terms']}\t{s['percentage']}\n")


def compute_doc_overlap(candidates):
    by_doc = defaultdict(list)  # docid -> [(qid, rank), ...]
    for c in candidates:
        by_doc[c["docid"]].append((c["qid"], c["rank"]))

    shared_docs = []
    for docid, hits in by_doc.items():
        qids = sorted({qid for qid, _rank in hits})
        if len(qids) > 1:
            ranks = {qid: rank for qid, rank in hits}  # a doc has one rank per query (no duplicate candidate rows expected)
            shared_docs.append({"doc_id": docid, "query_ids": qids, "ranks": ranks})
    shared_docs.sort(key=lambda d: (-len(d["query_ids"]), d["doc_id"]))

    distribution = Counter(len(d["query_ids"]) for d in shared_docs)
    return shared_docs, {"total_shared_docs": len(shared_docs), "queries_per_doc_distribution": {str(k): v for k, v in sorted(distribution.items())}}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gendered_terms = load_gendered_terms(GENDERED_TERMS_PATH)
    queries = load_queries(QUERIES_PATH)
    candidates = load_candidates(CANDIDATES_PATH)
    print(f"{len(queries)} queries, {len(candidates)} candidate rows, {len(gendered_terms)} gendered terms")

    needed_docids = {c["docid"] for c in candidates}
    doc_texts = fetch_doc_texts(COLLECTION_PATH, needed_docids)

    candidates = annotate_candidates(candidates, doc_texts, gendered_terms)
    write_annotated_candidates(candidates, OUTPUT_DIR / "annotated_candidates.tsv")

    query_stats = compute_query_stats(candidates, sorted(queries))
    write_query_stats(query_stats, OUTPUT_DIR / "query_stats.tsv")
    few_gendered = [s for s in query_stats if s["candidates_with_gendered_terms"] < FEW_GENDERED_DOCS_THRESHOLD]

    shared_docs, overlap_summary = compute_doc_overlap(candidates)
    with open(OUTPUT_DIR / "doc_overlap.json", "w", encoding="utf-8") as f:
        json.dump({"shared_docs": shared_docs, "summary": overlap_summary}, f, indent=2)

    unique_docs = len(doc_texts)
    unique_docs_with_gendered = len({c["docid"] for c in candidates if c["has_gendered_terms"]})
    with_gendered = sum(1 for c in candidates if c["has_gendered_terms"])
    summary = {
        "total_candidates_annotated": len(candidates),
        "unique_docs_annotated": unique_docs,
        "candidates_with_gendered_terms": with_gendered,
        "candidates_with_gendered_terms_pct": round(with_gendered / len(candidates) * 100, 2) if candidates else 0.0,
        "unique_docs_with_gendered_terms": unique_docs_with_gendered,
        "docs_shared_across_queries": overlap_summary["total_shared_docs"],
        "queries_with_fewer_than_5_gendered_docs": {"count": len(few_gendered), "query_ids": [s["query_id"] for s in few_gendered]},
    }
    with open(OUTPUT_DIR / "annotation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote outputs to {OUTPUT_DIR}")
    print(f"total candidates: {len(candidates)}")
    print(f"candidates with gendered terms: {with_gendered} ({summary['candidates_with_gendered_terms_pct']}%)")
    print(f"docs shared across queries: {overlap_summary['total_shared_docs']}")
    print(f"queries with fewer than {FEW_GENDERED_DOCS_THRESHOLD} gendered-term docs: {len(few_gendered)}")


if __name__ == "__main__":
    main()

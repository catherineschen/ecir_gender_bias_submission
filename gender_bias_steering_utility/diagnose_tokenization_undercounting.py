"""Diagnose whether raw (unpretokenized) collection.tsv undercounts gendered terms.

document_neutrality.py's get_neutrality() expects pre-cleaned, pre-tokenized input and
only ever does doctext.lower().split(' ') -- no punctuation stripping. If collection.tsv
still has punctuation attached to tokens ("boy,", "husband.", "(she"), those tokens will
never exact-match the wordlist, undercounting the doc's gendered-term magnitude and,
since threshold=1 in the neutrality run, likely defaulting the doc to neutral (1.0).

This is read-only: it does not modify or rerun the full neutrality computation. It only
samples real gendered-term occurrences that have punctuation attached, shows the exact
raw-split tokenization actually used, compares it to a punctuation-stripped tokenization
using DocumentNeutrality.get_neutrality() (reused as-is) for both, and looks up each
doc's real precomputed score.

Usage (from the repo root):
    conda run -n gender_bias python gender_bias_steering_utility/diagnose_tokenization_undercounting.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

COLLECTION_PATH = REPO_ROOT / "data/msmarco/collection.tsv"
NEUTRALITY_SCORES_PATH = HERE / "experiments/baseline/collection_neutralityscores.tsv"
GENDERED_TERMS_PATH = REPO_ROOT / "data/FairnessRetrievalResults/resources/wordlist_gender_representative.txt"
NEUTRALITY_CODE_DIR = REPO_ROOT / "data/FairnessRetrievalResults/measurement"

SAMPLE_SIZE = 30
SCAN_LINE_CAP = 2_000_000  # safety cap; gendered pronouns with trailing punctuation should be common well before this
TEXT_PREVIEW_CHARS = 320

sys.path.insert(0, str(NEUTRALITY_CODE_DIR))
from document_neutrality import DocumentNeutrality  # noqa: E402

# Real-word occurrences (\b-bounded, so "the"/"mother"/"this" substring collisions are excluded)
# immediately followed by trailing punctuation, or immediately preceded by an opening paren/quote.
GENDER_WORDS = r"(he|she|his|her|him|himself|herself|husband|wife|mother|father|brother|sister|son|daughter|boy|girl|man|woman|men|women|boyfriend|girlfriend|grandmother|grandfather)"
TRAILING_RE = re.compile(rf"\b{GENDER_WORDS}\b('s|[,.;:!?)\"'])", re.IGNORECASE)
LEADING_RE = re.compile(rf"([(\"']){GENDER_WORDS}\b", re.IGNORECASE)

STRIP_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


def classify_match(m, kind):
    if kind == "trailing":
        suffix = m.group(2)
        category = "contraction" if suffix == "'s" else "trailing_punct"
        return category, m.group(0)
    return "leading_punct", m.group(0)


def find_candidates(collection_path, target_count, scan_cap):
    candidates = []
    seen_docids = set()
    with open(collection_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(candidates) >= target_count or i >= scan_cap:
                break
            docid, _, text = line.rstrip("\n").partition("\t")
            if not text or docid in seen_docids:
                continue

            m = TRAILING_RE.search(text) or LEADING_RE.search(text)
            if not m:
                continue
            category, snippet = classify_match(m, "trailing" if m.re is TRAILING_RE else "leading")

            candidates.append({"docid": docid, "text": text, "category": category, "snippet": snippet})
            seen_docids.add(docid)
    return candidates, i + 1


def fetch_neutrality_scores(path, needed_docids):
    scores = {}
    remaining = set(needed_docids)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            docid, _, score = line.rstrip("\n").partition("\t")
            if docid in remaining:
                scores[docid] = float(score)
                remaining.discard(docid)
    return scores, remaining


def format_tokens(tokens, wordlist_terms, limit=25):
    survived = sorted({t for t in tokens if t in wordlist_terms})
    shown = tokens if len(tokens) <= limit else tokens[:limit] + ["...", f"({len(tokens)} tokens total)"]
    return survived, shown


def main():
    doc_neutrality = DocumentNeutrality(representative_words_path=str(GENDERED_TERMS_PATH), threshold=1, groups_portion={"f": 0.5, "m": 0.5})
    wordlist_terms = set(doc_neutrality.representative_words["f"]) | set(doc_neutrality.representative_words["m"])
    print(f"{len(wordlist_terms)} gendered terms loaded from wordlist")

    candidates, lines_scanned = find_candidates(COLLECTION_PATH, SAMPLE_SIZE, SCAN_LINE_CAP)
    print(f"scanned {lines_scanned:,} lines of {COLLECTION_PATH} to find {len(candidates)} candidates containing a real gendered word "
          f"with attached punctuation\n")

    actual_scores, missing = fetch_neutrality_scores(NEUTRALITY_SCORES_PATH, {c["docid"] for c in candidates})
    if missing:
        print(f"WARNING: {len(missing)} sampled docids were not found in {NEUTRALITY_SCORES_PATH}: {sorted(missing)}\n")

    rows = []
    for i, c in enumerate(candidates, 1):
        docid, text, category, snippet = c["docid"], c["text"], c["category"], c["snippet"]
        tokens_raw = text.lower().split(" ")
        tokens_stripped = STRIP_PUNCT_RE.sub(" ", text.lower()).split()

        survived_raw, shown_raw = format_tokens(tokens_raw, wordlist_terms)
        survived_stripped, shown_stripped = format_tokens(tokens_stripped, wordlist_terms)

        score_actual = actual_scores.get(docid)
        score_recomputed_raw = doc_neutrality.get_neutrality(tokens_raw)
        score_stripped = doc_neutrality.get_neutrality(tokens_stripped)

        preview = text if len(text) <= TEXT_PREVIEW_CHARS else text[:TEXT_PREVIEW_CHARS] + "..."
        print(f"[{i}] docid={docid}  match=\"{snippet}\"  category={category}")
        print(f"    RAW TEXT: {preview}")
        print(f"    raw split(' ') wordlist terms surviving: {survived_raw or 'NONE'}")
        print(f"    punct-stripped wordlist terms surviving: {survived_stripped or 'NONE'}")
        print(f"    neutrality: actual(from file)={score_actual}  recomputed_raw={score_recomputed_raw:.4f}"
              f"{'  [MISMATCH vs file]' if score_actual is not None and abs(score_actual - score_recomputed_raw) > 1e-6 else ''}"
              f"  punct_stripped={score_stripped:.4f}")
        print()

        rows.append({
            "docid": docid, "category": category, "score_actual": score_actual,
            "score_recomputed_raw": score_recomputed_raw, "score_stripped": score_stripped,
            "survived_raw": survived_raw, "survived_stripped": survived_stripped,
        })

    n = len(rows)
    neutral_under_raw = [r for r in rows if (r["score_actual"] if r["score_actual"] is not None else r["score_recomputed_raw"]) == 1.0]
    recovered_by_stripping = [r for r in neutral_under_raw if r["score_stripped"] != 1.0]
    still_neutral_after_stripping = [r for r in neutral_under_raw if r["score_stripped"] == 1.0]
    mismatches = [r for r in rows if r["score_actual"] is not None and abs(r["score_actual"] - r["score_recomputed_raw"]) > 1e-6]

    from collections import Counter
    recovered_categories = Counter(r["category"] for r in recovered_by_stripping)

    print("=" * 70)
    print(f"sampled {n} passages, each containing a real gendered word with attached punctuation")
    print(f"  scored neutral (1.0) under the raw tokenization: {len(neutral_under_raw)}/{n}")
    print(f"    -> of those, would score non-neutral under punctuation-stripped tokenization: {len(recovered_by_stripping)} "
          f"(breakdown: {dict(recovered_categories)})")
    print(f"    -> of those, STILL score neutral even with clean tokenization "
          f"(insufficient/balanced magnitude, not a tokenization artifact): {len(still_neutral_after_stripping)}")
    print(f"  file-vs-recomputed raw score mismatches: {len(mismatches)}/{n} "
          f"({'file was computed from this exact collection.tsv' if not mismatches else 'WARNING: file may have used a different collection file'})")
    print("  capitalization: ruled out by construction -- both tokenizations apply .lower() before matching, "
          "so casing cannot explain any of the above")


if __name__ == "__main__":
    main()

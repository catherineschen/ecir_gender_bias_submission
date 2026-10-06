"""Build M/F/N swap-pair triplets from the construction-query candidate docs.

For each construction candidate doc flagged with gendered terms, this builds:
  - an M variant (every gendered term in the doc converted to its male form)
  - an F variant (every gendered term converted to its female form)
  - an N variant (every gendered term converted to a shared neutral form), when possible

M<->F swapping uses a hand-curated pairing table derived from the gendered-terms
wordlist (see M_TO_F / F_TO_M below). The table deliberately excludes:
  - "her", which is genuinely ambiguous when converting to the male side (it stands
    for both the object pronoun "him" and the possessive determiner "his", and
    without parsing grammatical role we can't tell which one is meant). This is
    resolved per-occurrence with a spaCy POS tagger instead of a fixed table entry
    (see resolve_her_occurrences): PRP -> "him", PRP$ -> "his". English collapses
    both male forms into the single female form "her", so the F variant never needs
    to change for this term -- only the M variant requires resolution.
  - proper names (no natural 1:1 name pairing exists)
  - a handful of wordlist terms with no natural single-word counterpart on the
    other side (e.g. "godfather", "lad"/"lads", "gentlemen", "gals", "grandmothers")
A doc containing any such unmapped term is dropped from M/F construction entirely
rather than guessing a swap. A doc blocked *solely* by "her" is recovered if every
occurrence tags unambiguously; if any occurrence gets an unexpected tag, or if more
than ~10% of all attempted "her" occurrences tag unexpectedly (signalling the
tagger is struggling on this text domain), the recovery is not applied and the
affected docs stay dropped.

N-variant construction is independent of M/F: it uses the separate neutral mapping
file, which covers every non-name wordlist term except "sir"/"madam" (no natural
single-word neutral substitute for those). A doc only gets an N variant if every
gendered term it contains has a neutral mapping entry.

Usage (from the repo root):
    conda run -n gender_bias python gender_bias_steering_utility/build_construction_pairs.py
"""
import csv
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import spacy
from transformers import AutoTokenizer

from model_configs import MODEL_CONFIGS

MODEL_NAMES = list(MODEL_CONFIGS)  # model_configs.py only defines the dict; this script just needs the keys

SPACY_MODEL = "en_core_web_sm"
HER_TAG_TO_M = {"PRP$": "his", "PRP": "him"}
HER_UNRESOLVED_RATE_THRESHOLD = 0.10

# ---- paths ----
REPO_ROOT = Path(__file__).resolve().parent.parent
# SEED selects which n5_seed{SEED} split (from split_queries_dedupe.py) to build construction
# pairs from. At the default (42) every path below is unchanged from before this was made
# overridable; other seeds get their own sibling preprocess/vector_construction_seed{SEED}
# directory so nothing about seed 42's existing output ever moves. See run_steering_sweep.sh's
# "OTHER SEEDS" section.
SEED = int(os.environ.get("SEED", 42))
SEED_SUFFIX = "" if SEED == 42 else f"_seed{SEED}"
CANDIDATES_PATH = REPO_ROOT / f"gender_bias_steering_utility/preprocess/n5_seed{SEED}/construction_candidates.tsv"
WORDLIST_PATH = REPO_ROOT / "data/FairnessRetrievalResults/resources/wordlist_gender_representative.txt"
NEUTRAL_PATH = REPO_ROOT / "gender_bias_steering_utility/data/wordlist_gender_representative_no_names_w_neutral.txt"
COLLECTION_PATH = REPO_ROOT / "data/msmarco/collection.tsv"
OUTPUT_DIR = REPO_ROOT / f"gender_bias_steering_utility/preprocess/vector_construction{SEED_SUFFIX}"
COLLECTION_SIZE_HINT = 8841823

PREFIX_RE = re.compile(r"^\W+")
SUFFIX_RE = re.compile(r"\W+$")

# ---- hand-curated M<->F swap table (lowercase; see module docstring for rationale) ----
M_TO_F = {
    "boy": "girl", "boys": "girls",
    "brother": "sister", "brothers": "sisters",
    "dad": "mom", "dads": "moms",
    "father": "mother", "fathers": "mothers",
    "fiance": "fiancee",
    "gentleman": "lady",
    "grandfather": "grandmother", "grandpa": "grandma",
    "grandson": "granddaughter", "grandsons": "granddaughters",
    "guy": "gal",
    "he": "she", "him": "her", "himself": "herself", "his": "her",
    "male": "female", "males": "females",
    "man": "woman", "men": "women",
    "sir": "madam",
    "son": "daughter", "sons": "daughters",
    "stepfather": "stepmother", "stepson": "stepdaughter",
}
F_TO_M = {
    "girl": "boy", "girls": "boys",
    "sister": "brother", "sisters": "brothers",
    "mom": "dad", "moms": "dads", "mama": "dad", "mommy": "dad",
    "mother": "father", "mothers": "fathers",
    "fiancee": "fiance",
    "lady": "gentleman",
    "grandmother": "grandfather", "grandma": "grandpa",
    "granddaughter": "grandson", "granddaughters": "grandsons",
    "gal": "guy",
    "she": "he", "hers": "his", "herself": "himself",
    "female": "male", "females": "males",
    "woman": "man", "women": "men",
    "madam": "sir",
    "daughter": "son", "daughters": "sons",
    "stepmother": "stepfather", "stepdaughter": "stepson",
    # "her" intentionally excluded: ambiguous (object pronoun "him" vs possessive "his")
}
AMBIGUOUS_F_TERMS = {"her"}


def load_gendered_terms(path):
    """Returns {lowercased term: gender ('m'/'f')}."""
    terms = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            term, gender = line.split(",")
            terms[term.lower()] = gender
    return terms


def load_neutral_map(path):
    """Returns {lowercased term: lowercased neutral replacement}."""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            term, neutral, _gender = line.split(",")
            mapping[term.lower()] = neutral.lower()
    return mapping


def load_construction_candidates(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def fetch_doc_texts(collection_path, needed_docids):
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
        print(f"warning: {len(remaining)} candidate docids were not found in the collection")
    return texts


def split_case(token):
    """Split a raw whitespace token into (prefix_punct, core, suffix_punct)."""
    prefix_m = PREFIX_RE.match(token)
    prefix = prefix_m.group(0) if prefix_m else ""
    rest = token[len(prefix):]
    suffix_m = SUFFIX_RE.search(rest)
    suffix = suffix_m.group(0) if suffix_m else ""
    core = rest[: len(rest) - len(suffix)] if suffix else rest
    return prefix, core, suffix


def apply_case(word, core):
    if core.isupper() and len(core) > 1:
        return word.upper()
    if core[:1].isupper():
        return word.capitalize()
    return word.lower()


def whitespace_token_spans(text):
    """Char (start, end) spans of each whitespace-split token, in text.split() order."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def resolve_her_occurrences(nlp_doc, token_spans, positions, terms_found):
    """For each "her" occurrence (by whitespace position), find the overlapping
    spaCy token and return its Penn Treebank tag. Returns a list of
    {"whitespace_pos", "tag"} in doc order (tag is None if no matching token found)."""
    resolutions = []
    for pos, term in zip(positions, terms_found):
        if term.lower() != "her":
            continue
        span_start, span_end = token_spans[pos]
        tag = None
        for tok in nlp_doc:
            tok_start = tok.idx
            tok_end = tok.idx + len(tok.text)
            if tok_start < span_end and tok_end > span_start and tok.text.lower() == "her":
                tag = tok.tag_
                break
        resolutions.append({"whitespace_pos": pos, "tag": tag})
    return resolutions


def build_variant(tokens, positions, replacement_fn):
    """Returns (variant_text, char_spans) where char_spans[i] = (start, end) of
    tokens[positions[i]] within variant_text (post-replacement)."""
    out_tokens = list(tokens)
    for pos in positions:
        prefix, core, suffix = split_case(tokens[pos])
        replacement = replacement_fn(pos, core.lower())
        if replacement is None:
            continue
        out_tokens[pos] = f"{prefix}{apply_case(replacement, core)}{suffix}"

    variant_text = " ".join(out_tokens)
    char_spans = {}
    cursor = 0
    for i, tok in enumerate(out_tokens):
        char_spans[i] = (cursor, cursor + len(tok))
        cursor += len(tok) + 1  # +1 for the joining space
    return variant_text, char_spans


def subword_indices_for_span(offset_mapping, char_start, char_end):
    idxs = []
    for i, (start, end) in enumerate(offset_mapping):
        if start == end == 0:
            continue  # special token
        if start < char_end and end > char_start:
            idxs.append(i)
    return idxs


def classify_her_blocking(terms_found, gendered_terms):
    """Returns (has_her, blocked_by_other) where blocked_by_other is True if the
    doc has a missing-mapping issue independent of "her" (name, godfather, etc.)."""
    distinct_terms_lower = {t.lower() for t in terms_found}
    m_terms_lower = {t for t in distinct_terms_lower if gendered_terms.get(t) == "m"}
    f_terms_lower = {t for t in distinct_terms_lower if gendered_terms.get(t) == "f"}
    missing_m_to_f = {t for t in m_terms_lower if t not in M_TO_F}
    missing_f_to_m = {t for t in (f_terms_lower - AMBIGUOUS_F_TERMS) if t not in F_TO_M}
    has_her = "her" in f_terms_lower
    blocked_by_other = bool(missing_m_to_f or missing_f_to_m)
    return has_her, blocked_by_other


def build_doc_pair(doc_id, query_id, text, terms_found, positions, gendered_terms, neutral_map, her_resolution=None):
    """her_resolution: optional {whitespace_pos: (tag, resolved_m_word)} covering
    every "her" occurrence in this doc. If the doc contains "her" and this doesn't
    cover every occurrence, the doc is dropped (still ambiguous)."""
    tokens = text.split()
    distinct_terms_lower = {t.lower() for t in terms_found}

    m_terms_lower = {t for t in distinct_terms_lower if gendered_terms.get(t) == "m"}
    f_terms_lower = {t for t in distinct_terms_lower if gendered_terms.get(t) == "f"}

    her_positions = [pos for pos, term in zip(positions, terms_found) if term.lower() == "her"]
    has_her = bool(her_positions)
    if has_her and (her_resolution is None or any(pos not in her_resolution for pos in her_positions)):
        return None  # still ambiguous / unresolved -> drop

    missing_m_to_f = {t for t in m_terms_lower if t not in M_TO_F}
    missing_f_to_m = {t for t in (f_terms_lower - AMBIGUOUS_F_TERMS) if t not in F_TO_M}

    if missing_m_to_f or missing_f_to_m:
        return None  # drop doc from M/F construction

    def m_variant_replacement(pos, core_lower):
        # M variant: convert F terms to M, keep M terms as-is.
        if core_lower == "her":
            return her_resolution[pos][1]  # resolved per-occurrence via POS tag
        if core_lower in f_terms_lower:
            return F_TO_M[core_lower]
        return None

    def f_variant_replacement(pos, core_lower):
        # F variant: convert M terms to F, keep F terms as-is ("her" is already
        # the correct F form for both PRP and PRP$ roles, so it's left unchanged).
        if core_lower in m_terms_lower:
            return M_TO_F[core_lower]
        return None

    m_text, m_spans = build_variant(tokens, positions, m_variant_replacement)
    f_text, f_spans = build_variant(tokens, positions, f_variant_replacement)

    n_coverage = all(t in neutral_map for t in distinct_terms_lower)
    n_text, n_spans = None, None
    if n_coverage:
        def n_variant_replacement(pos, core_lower):
            return neutral_map.get(core_lower)

        n_text, n_spans = build_variant(tokens, positions, n_variant_replacement)

    missing_neutral = distinct_terms_lower - set(neutral_map)
    excluded_solely_sir_madam = bool(missing_neutral) and missing_neutral <= {"sir", "madam"}

    her_resolved = None
    if has_her:
        her_resolved = [
            {"occurrence": i, "whitespace_pos": pos, "tag": her_resolution[pos][0], "resolved_to": her_resolution[pos][1]}
            for i, pos in enumerate(her_positions)
        ]

    return {
        "doc_id": doc_id,
        "query_id": query_id,
        "original_text": text,
        "m_text": m_text,
        "f_text": f_text,
        "n_text": n_text,
        "gendered_terms_found": terms_found,
        "n_coverage": n_coverage,
        "her_resolved": her_resolved,
        "excluded_solely_sir_madam": excluded_solely_sir_madam,
        "_positions": positions,
        "_m_spans": m_spans,
        "_f_spans": f_spans,
        "_n_spans": n_spans,
    }


def compute_model_validity_and_positions(pair, tokenizer, max_len):
    positions = pair["_positions"]

    def encode(text):
        enc = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    m_ids, m_offsets = encode(pair["m_text"])
    f_ids, f_offsets = encode(pair["f_text"])

    mvf_valid = len(m_ids) <= max_len and len(f_ids) <= max_len
    mvf_positions = [
        {
            "whitespace_pos": pos,
            "m_subword_idxs": subword_indices_for_span(m_offsets, *pair["_m_spans"][pos]),
            "f_subword_idxs": subword_indices_for_span(f_offsets, *pair["_f_spans"][pos]),
        }
        for pos in positions
    ]

    result = {"mvf": mvf_valid}
    swapped = {"mvf": mvf_positions}

    if pair["n_text"] is not None:
        n_ids, n_offsets = encode(pair["n_text"])
        mvn_valid = len(m_ids) <= max_len and len(n_ids) <= max_len
        fvn_valid = len(f_ids) <= max_len and len(n_ids) <= max_len
        result["mvn"] = mvn_valid
        result["fvn"] = fvn_valid
        swapped["mvn"] = [
            {
                "whitespace_pos": pos,
                "m_subword_idxs": subword_indices_for_span(m_offsets, *pair["_m_spans"][pos]),
                "n_subword_idxs": subword_indices_for_span(n_offsets, *pair["_n_spans"][pos]),
            }
            for pos in positions
        ]
        swapped["fvn"] = [
            {
                "whitespace_pos": pos,
                "f_subword_idxs": subword_indices_for_span(f_offsets, *pair["_f_spans"][pos]),
                "n_subword_idxs": subword_indices_for_span(n_offsets, *pair["_n_spans"][pos]),
            }
            for pos in positions
        ]
    else:
        result["mvn"] = False
        result["fvn"] = False

    return result, swapped


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gendered_terms = load_gendered_terms(WORDLIST_PATH)
    neutral_map = load_neutral_map(NEUTRAL_PATH)
    rows = load_construction_candidates(CANDIDATES_PATH)
    print(f"{len(rows)} candidate rows loaded")

    gendered_rows = [r for r in rows if r["has_gendered_terms"] == "True"]
    print(f"{len(gendered_rows)} rows with gendered terms")

    needed_docids = {r["docid"] for r in gendered_rows}
    doc_texts = fetch_doc_texts(COLLECTION_PATH, needed_docids)

    # per-query gendered doc listing (uses the raw filtered set, before M/F survival)
    by_query = defaultdict(list)
    for r in gendered_rows:
        by_query[r["qid"]].append(r)

    gendered_docs_by_query = {}
    for qid, docs in by_query.items():
        docs_sorted = sorted(docs, key=lambda d: int(d["rank"]))
        ranks = [int(d["rank"]) for d in docs_sorted]
        gendered_docs_by_query[qid] = {
            "query_id": qid,
            "num_gendered_docs": len(docs_sorted),
            "docs": [{"doc_id": d["docid"], "bm25_rank": int(d["rank"])} for d in docs_sorted],
            "min_rank": min(ranks),
            "median_rank": statistics.median(ranks),
            "max_rank": max(ranks),
        }
    with open(OUTPUT_DIR / "gendered_docs_by_query.json", "w", encoding="utf-8") as f:
        json.dump(gendered_docs_by_query, f, indent=2)

    # ---- POS-based resolution of "her" for docs blocked solely by that ambiguity ----
    print(f"loading spaCy model: {SPACY_MODEL}")
    nlp = spacy.load(SPACY_MODEL)

    her_only_docids = set()
    for r in gendered_rows:
        text = doc_texts.get(r["docid"])
        if text is None:
            continue
        terms_found = r["gendered_terms_found"].split(",") if r["gendered_terms_found"] else []
        has_her, blocked_by_other = classify_her_blocking(terms_found, gendered_terms)
        if has_her and not blocked_by_other:
            her_only_docids.add(r["docid"])
    print(f"{len(her_only_docids)} unique docs blocked solely by 'her' ambiguity (recovery candidates)")

    her_resolution_by_docid = {}  # docid -> {whitespace_pos: (tag, resolved_word)}
    prp_count = prp_dollar_count = unresolved_count = 0
    unresolved_docids = set()
    if her_only_docids:
        her_docid_list = list(her_only_docids)
        her_texts = [doc_texts[d] for d in her_docid_list]
        # first pass: tag every "her" occurrence across all recovery candidates
        raw_resolutions_by_docid = {}
        for docid, text, spacy_doc in zip(her_docid_list, her_texts, nlp.pipe(her_texts)):
            token_spans = whitespace_token_spans(text)
            # positions/terms_found are identical across rows sharing this docid
            r = next(row for row in gendered_rows if row["docid"] == docid)
            terms_found = r["gendered_terms_found"].split(",")
            positions = [int(p) for p in r["gendered_term_positions"].split(",")]
            occurrences = resolve_her_occurrences(spacy_doc, token_spans, positions, terms_found)
            raw_resolutions_by_docid[docid] = occurrences
            for occ in occurrences:
                if occ["tag"] == "PRP$":
                    prp_dollar_count += 1
                elif occ["tag"] == "PRP":
                    prp_count += 1
                else:
                    unresolved_count += 1

        total_her_occurrences = prp_count + prp_dollar_count + unresolved_count
        unresolved_rate = (unresolved_count / total_her_occurrences) if total_her_occurrences else 0.0
        recovery_enabled = unresolved_rate <= HER_UNRESOLVED_RATE_THRESHOLD
        if not recovery_enabled:
            print(
                f"WARNING: {unresolved_count}/{total_her_occurrences} her-occurrences "
                f"({unresolved_rate:.1%}) tagged unexpectedly -- exceeds the "
                f"{HER_UNRESOLVED_RATE_THRESHOLD:.0%} threshold. Skipping 'her' recovery "
                f"entirely; affected docs remain dropped as before."
            )
        else:
            for docid, occurrences in raw_resolutions_by_docid.items():
                if any(occ["tag"] not in HER_TAG_TO_M for occ in occurrences):
                    unresolved_docids.add(docid)
                    continue
                her_resolution_by_docid[docid] = {
                    occ["whitespace_pos"]: (occ["tag"], HER_TAG_TO_M[occ["tag"]]) for occ in occurrences
                }

    # build M/F(/N) pairs
    pairs = []
    dropped_ambiguous_or_unmapped = 0
    recovered_her_docs = set()
    terms_in_docs_lacking_n = Counter()
    for r in gendered_rows:
        docid = r["docid"]
        text = doc_texts.get(docid)
        if text is None:
            continue
        terms_found = r["gendered_terms_found"].split(",") if r["gendered_terms_found"] else []
        positions = [int(p) for p in r["gendered_term_positions"].split(",")] if r["gendered_term_positions"] else []
        her_resolution = her_resolution_by_docid.get(docid)
        pair = build_doc_pair(docid, r["qid"], text, terms_found, positions, gendered_terms, neutral_map, her_resolution)
        if pair is None:
            dropped_ambiguous_or_unmapped += 1
            continue
        if her_resolution is not None:
            recovered_her_docs.add(docid)
        if not pair["n_coverage"]:
            for t in {t.lower() for t in terms_found}:
                if t not in neutral_map:
                    terms_in_docs_lacking_n[t] += 1
        pairs.append(pair)

    unresolved_tag_dropped = len(unresolved_docids)
    print(
        f"{len(pairs)} docs kept for M/F construction; {dropped_ambiguous_or_unmapped} dropped "
        f"(ambiguous/unmapped terms); {len(recovered_her_docs)} recovered via 'her' POS resolution"
    )

    # tokenizer-based validity + swapped positions, per model
    tokenizers = {}
    for model_name in MODEL_NAMES:
        print(f"loading tokenizer: {model_name}")
        tok = AutoTokenizer.from_pretrained(model_name)
        max_len = tok.model_max_length
        if max_len is None or max_len > 100000:
            max_len = 512
        tokenizers[model_name] = (tok, max_len)

    valid_pair_counts = {m: {"mvf": 0, "mvn": 0, "fvn": 0} for m in MODEL_NAMES}

    for pair in pairs:
        per_model_valid = {}
        per_model_swapped_positions = {}
        for model_name, (tok, max_len) in tokenizers.items():
            validity, swapped = compute_model_validity_and_positions(pair, tok, max_len)
            per_model_valid[model_name] = validity
            per_model_swapped_positions[model_name] = swapped
            for comp in ("mvf", "mvn", "fvn"):
                if validity[comp]:
                    valid_pair_counts[model_name][comp] += 1
        pair["per_model_valid"] = per_model_valid
        pair["per_model_swapped_positions"] = per_model_swapped_positions

    # write construction_pairs.jsonl
    out_path = OUTPUT_DIR / "construction_pairs.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            row = {k: v for k, v in pair.items() if not k.startswith("_")}
            row.pop("excluded_solely_sir_madam", None)
            f.write(json.dumps(row) + "\n")

    # stats
    n_covered = sum(1 for p in pairs if p["n_coverage"])
    n_not_covered = len(pairs) - n_covered
    excluded_solely_sir_madam = sum(1 for p in pairs if p["excluded_solely_sir_madam"])

    prev_stats_path = OUTPUT_DIR / "pair_construction_stats.json"
    prev_stats = None
    if prev_stats_path.exists():
        with open(prev_stats_path, encoding="utf-8") as f:
            prev_stats = json.load(f)

    stats = {
        "total_docs": len(rows),
        "docs_with_gendered_terms": len(gendered_rows),
        "docs_valid_mf_pair": len(pairs),
        "docs_dropped_mf_ambiguous_or_unmapped": dropped_ambiguous_or_unmapped,
        "docs_with_n_coverage": n_covered,
        "docs_without_n_coverage": n_not_covered,
        "docs_excluded_from_n_solely_due_to_sir_madam": excluded_solely_sir_madam,
        "valid_pairs_per_model": valid_pair_counts,
        "her_pos_resolution": {
            "recovery_candidates": len(her_only_docids),
            "docs_recovered": len(recovered_her_docs),
            "docs_still_dropped_unexpected_tag": unresolved_tag_dropped,
            "occurrences_resolved_prp_object": prp_count,
            "occurrences_resolved_prp_dollar_possessive": prp_dollar_count,
            "occurrences_unresolved_unexpected_tag": unresolved_count,
        },
    }
    with open(OUTPUT_DIR / "pair_construction_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # ---- summary print ----
    print("\n=== summary ===")
    print(f"docs: {len(rows)} -> gendered docs: {len(gendered_rows)} -> valid M/F pairs: {len(pairs)}")
    n_rate = (n_covered / len(pairs) * 100) if pairs else 0.0
    print(f"N-coverage rate (of valid M/F docs): {n_covered}/{len(pairs)} ({n_rate:.1f}%)")
    print("valid pairs per model per comparison:")
    for model_name, counts in valid_pair_counts.items():
        print(f"  {model_name}: MvF={counts['mvf']} MvN={counts['mvn']} FvN={counts['fvn']}")

    print("\n'her' POS-based recovery:")
    print(f"  recovery candidates (blocked solely by 'her'): {len(her_only_docids)}")
    print(f"  docs recovered: {len(recovered_her_docs)}")
    print(f"  docs still dropped (unexpected tag): {unresolved_tag_dropped}")
    print(f"  occurrences resolved: PRP (object -> 'him')={prp_count}  PRP$ (possessive -> 'his')={prp_dollar_count}")
    if prp_count + prp_dollar_count + unresolved_count:
        rate = unresolved_count / (prp_count + prp_dollar_count + unresolved_count)
        flag = "  <-- exceeds 10% threshold, recovery skipped" if rate > HER_UNRESOLVED_RATE_THRESHOLD else ""
        print(f"  unresolved (unexpected tag): {unresolved_count} ({rate:.1%}){flag}")

    if prev_stats is not None:
        print("\ncompared to previous run:")
        print(f"  valid M/F pairs: {prev_stats.get('docs_valid_mf_pair')} -> {len(pairs)}")
        print(f"  N coverage: {prev_stats.get('docs_with_n_coverage')} -> {n_covered}")
        for model_name in MODEL_NAMES:
            prev_counts = prev_stats.get("valid_pairs_per_model", {}).get(model_name, {})
            new_counts = valid_pair_counts[model_name]
            print(
                f"  {model_name}: MvF {prev_counts.get('mvf')}->{new_counts['mvf']}, "
                f"MvN {prev_counts.get('mvn')}->{new_counts['mvn']}, "
                f"FvN {prev_counts.get('fvn')}->{new_counts['fvn']}"
            )

    print("\ntop 10 gendered terms found in docs lacking N coverage:")
    for term, count in terms_in_docs_lacking_n.most_common(10):
        flag = "" if term in ("sir", "madam") else "  <-- unexpected, check mapping"
        print(f"  {term}: {count}{flag}")

    print("\nper construction query:")
    for qid, info in sorted(gendered_docs_by_query.items(), key=lambda kv: kv[0]):
        print(
            f"  query {qid}: {info['num_gendered_docs']} gendered docs, "
            f"ranks {info['min_rank']}-{info['max_rank']} (median {info['median_rank']})"
        )

    print(f"\nwrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""Compute ARaB (Average Rank Bias) from the steering sweep's EXISTING run.trec files.

This script is READ-ONLY with respect to the sweep: it never loads the retrieval model,
never steers, never re-ranks. It globs the run.trec files apply_steering_and_evaluate.py
already wrote, reads the MS MARCO collection once for the document texts, and computes
ARaB@10. Nothing here can change MRR@10/NFaiRR@10 or sweep_results.csv.

PORTED FROM THE ORIGINAL ARaB REPO (data/GenderBias_IR, Rekabsaz & Schedl), not reinvented:

  step1_calculate_bias_documents.ipynb  -> load_gender_wordlists / get_tokens / get_bias
  step2_calculate_bias_runs.ipynb       -> read_run_docs_bias / calc_RaB_q / calc_ARaB_q
  step3_bias_metrics.ipynb              -> aggregate_arab (the query-level average)

The three document-bias variants of step1 are all kept:
  tc    term count:    (cnt_feml - cnt_male, cnt_feml, cnt_male)
  tf    log term freq: (log(cnt_feml+1) - log(cnt_male+1), log(cnt_feml+1), log(cnt_male+1))
  bool  boolean:       (sign(cnt_feml) - sign(cnt_male), sign(cnt_feml), sign(cnt_male))
('bool' is spelled the way the original repo's dict keys spell it -- it is the boolean variant.)

SIGN CONVENTION (this is the one thing easy to get backwards -- see aggregate_arab):
  step1 computes per-document bias as FEMALE MINUS MALE, so a document's bias[0] is
  POSITIVE for female-leaning text. step2 carries that through unchanged. But step3 does
  NOT average bias[0]; it recomputes the reported number as (male_component - feml_component):

      eval_results_bias[...] = np.mean([(_male_x - _feml_x) for _male_x, _feml_x in zip(...)])

  So the FINAL, REPORTED ARaB FLIPS THE SIGN relative to the per-document value:
      *** ARaB@10 > 0  ==  MALE-leaning ranking  ***
      *** ARaB@10 < 0  ==  FEMALE-leaning ranking ***
  This script reports step3's convention (positive = male-leaning), computed exactly the
  way step3 computes it, and never the per-document convention.

WORDLIST: data/GenderBias_IR/resources/wordlist_genderspecific.txt -- the ORIGINAL repo's
64-term male/female list, deliberately NOT the project's
data/wordlist_gender_representative_no_names_w_neutral.txt that the steering vectors were
built from. ARaB is being ported as a metric, so it keeps its own paper's term list; using
the steering wordlist would make the metric a function of the intervention's own vocabulary.
Which file was used is printed on every run.

DIFFERENCES FROM THE ORIGINAL NOTEBOOKS, all deliberate:
  * Document bias is computed only for the docids that appear in the runs, not for all 8.8M
    collection passages (per-document bias is independent of every other document, so this is
    identical to step1 restricted to those docids -- just vastly cheaper). Results are cached
    in documents_bias.<variant>.tsv and reused across invocations.
  * Only at_rank = K (10) is computed, to line up with the sweep's MRR@10 / NFaiRR@10, instead
    of the original's [5, 10, 20, 30, 40].
  * Queries are filtered to this seed's test-query split (the same construction-disjoint split
    apply_steering_and_evaluate.py evaluated), not to the original repo's
    resources/queries_gender_annotated.csv gender-neutral subset. Same role in the pipeline
    (step2's qryids_filter / step3's queries_effective), different query set, because this
    project's test queries are its own defined, held-out set.
  * |ARaB|@10 (mean over queries of |ARaB_q|) is an ADDITION -- step3 reports only the signed
    average. It is the mean of per-query magnitudes, not the absolute value of the signed
    average, so it measures how much gender skew each query's ranking has regardless of
    direction. (The absolute value of the signed column is trivially derivable from it, and
    would collapse to ~0 at a sign crossing rather than showing the residual per-query skew.)

Paths, the coefficient-folder naming, the test-query loader and the collection reader are
IMPORTED from apply_steering_and_evaluate.py rather than duplicated, so DATASET, OUTPUT_DIR_NAME,
SHORT_NAME (MODEL_KEY), SEED, TEST_QUERIES_SPLIT and N_TEST_QUERIES_SUBSET resolve from the
exact same env vars, and can never drift from the script that wrote the runs. DATASET=trecdl
therefore needs nothing special here: OUTPUT_DIR picks up the _trecdl19 suffix and
TEST_QUERIES_PATH the TREC DL query file through that same import.

The document-bias cache (experiments/arab_cache/documents_bias.*.tsv) is deliberately shared
across datasets as well as seeds: document bias is a property of the collection alone, and
TREC DL 2019 passages come from the same MS MARCO collection, so its candidates simply extend
the cache rather than needing one of their own.

Usage (from the repo root, same env vars as the sweep):
    conda run -n gender_bias python gender_bias_steering_utility/compute_arab_from_runs.py

    OUTPUT_DIR_NAME=steering_heldout_final TEST_QUERIES_SPLIT=heldout MODEL_KEY=tas_b \
        conda run -n gender_bias python gender_bias_steering_utility/compute_arab_from_runs.py

    DATASET=trecdl OUTPUT_DIR_NAME=steering_heldout_final MODEL_KEY=tas_b \
        conda run -n gender_bias python gender_bias_steering_utility/compute_arab_from_runs.py

Idempotent: results go to a SEPARATE arab_results.csv next to sweep_results.csv, keyed by
(model, level, direction, coefficient). Cells already in that file are skipped; --force
recomputes them (and the document-bias cache).
"""
import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Same env-var resolution, same paths, same folder naming as the script that wrote the runs.
# Importing it loads torch/mechir but does NOT load a model or touch a GPU (everything in that
# module below the constants lives inside functions), so this stays a read-only metric script.
from apply_steering_and_evaluate import (  # noqa: E402
    CANDIDATE_CUTOFF,
    CFG,
    COLLECTION_PATH,
    DATASET,
    DIRECTIONS,
    K,
    LEVELS,
    MODEL_NAME,
    N_TEST_QUERIES_SUBSET,
    OUTPUT_DIR,
    SEED,
    SHORT_NAME,
    TEST_QUERIES_PATH,
    coef_folder,
    fetch_doc_texts,
    load_construction_qids,
    load_test_queries,
)

# The original repo's own wordlist -- see the WORDLIST note in the module docstring.
WORDLIST_GENDERSPECIFIC_PATH = REPO_ROOT / "data/GenderBias_IR/resources/wordlist_genderspecific.txt"

# Document bias is a property of the collection alone -- not of a model, seed, sweep or
# coefficient -- so the cache is shared by every invocation instead of living under OUTPUT_DIR.
DOCS_BIAS_CACHE_DIR = Path(
    os.environ.get("ARAB_CACHE_DIR", REPO_ROOT / "gender_bias_steering_utility/experiments/arab_cache")
)
DOCS_BIAS_CACHE_NAME = "documents_bias.{variant}.tsv"

VARIANTS = ("tc", "tf", "bool")  # step1's three document-bias variants, its own dict keys
COEF_PREFIX = {"embed": "alpha", "attn": "beta"}  # inverse of coef_folder()

ARAB_CSV_FIELDS = (
    ["model", "level", "direction", "coefficient", "n_queries"]
    + [f"ARaB@{K}_{v}" for v in VARIANTS]
    + [f"|ARaB|@{K}_{v}" for v in VARIANTS]
)


# ============================================================
# step1: per-document bias  (port of step1_calculate_bias_documents.ipynb)
# ============================================================
def load_gender_wordlists(path):
    """Verbatim port of step1's wordlist parsing: 'term,m' / 'term,f' lines, lowercased."""
    genderwords_feml = []
    genderwords_male = []
    for l in open(path, encoding="utf-8"):
        vals = l.strip().lower().split(',')
        if len(vals) < 2:
            continue
        if vals[1] == 'f':
            genderwords_feml.append(vals[0])
        elif vals[1] == 'm':
            genderwords_male.append(vals[0])
    return set(genderwords_feml), set(genderwords_male)


def get_tokens(text):
    """step1's tokenizer, unchanged: lowercase and split on single spaces.

    Deliberately NOT the punctuation-stripping tokenization used elsewhere in this project
    (document_neutrality_fixed.STRIP_PUNCT_RE) -- that would change the metric's definition.
    """
    return text.lower().split(" ")


def get_bias(tokens, genderwords_feml, genderwords_male):
    """step1's get_bias, returning all three variants keyed as step1 keys them.

    (step1 also accumulates cnt_logfeml / cnt_logmale per word and then never uses them;
    they are dropped here because they enter none of its three returned tuples. Note the
    elif: a term counts as female or male, never both, female checked first.)
    """
    text_cnt = collections.Counter(tokens)

    cnt_feml = 0
    cnt_male = 0
    for word in text_cnt:
        if word in genderwords_feml:
            cnt_feml += text_cnt[word]
        elif word in genderwords_male:
            cnt_male += text_cnt[word]

    bias_tc = (float(cnt_feml - cnt_male), float(cnt_feml), float(cnt_male))
    bias_tf = (
        float(np.log(cnt_feml + 1) - np.log(cnt_male + 1)),
        float(np.log(cnt_feml + 1)),
        float(np.log(cnt_male + 1)),
    )
    bias_bool = (
        float(np.sign(cnt_feml) - np.sign(cnt_male)),
        float(np.sign(cnt_feml)),
        float(np.sign(cnt_male)),
    )
    return {"tc": bias_tc, "tf": bias_tf, "bool": bias_bool}


def load_docs_bias_cache(cache_dir):
    docs_bias = {v: {} for v in VARIANTS}
    for variant in VARIANTS:
        path = cache_dir / DOCS_BIAS_CACHE_NAME.format(variant=variant)
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            next(f, None)  # header
            for line in f:
                vals = line.rstrip("\n").split("\t")
                if len(vals) != 4:
                    continue
                docs_bias[variant][vals[0]] = (float(vals[1]), float(vals[2]), float(vals[3]))
    return docs_bias


def write_docs_bias_cache(cache_dir, docs_bias):
    """Temp file + rename, so an interrupted write can't leave a half-file where a cache was.

    Values are written with .17g, i.e. enough digits to round-trip the float exactly. That
    matters: the 'tf' variant stores logs, and rounding them (say to %.6f) would make a cell
    computed from a freshly scanned collection differ in the 7th decimal from the same cell
    recomputed later from the cache. ARaB@10 must not depend on whether the cache was warm.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        path = cache_dir / DOCS_BIAS_CACHE_NAME.format(variant=variant)
        tmp_path = path.with_suffix(".tsv.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            # bias == feml - male (step1's per-DOCUMENT convention; the reported ARaB flips it)
            f.write("docid\tbias\tbias_feml\tbias_male\n")
            for docid in sorted(docs_bias[variant], key=int):
                b, feml, male = docs_bias[variant][docid]
                f.write(f"{docid}\t{b:.17g}\t{feml:.17g}\t{male:.17g}\n")
        tmp_path.replace(path)


def build_docs_bias(needed_docids, genderwords_feml, genderwords_male, force):
    """Document bias for every docid appearing in the runs, cached across invocations."""
    # --force recomputes the docids THIS invocation needs, but still loads the cache first so
    # that entries belonging to other sweeps (other query splits, other seeds -- the cache is
    # shared) are carried over rather than dropped when the file is rewritten below.
    docs_bias = load_docs_bias_cache(DOCS_BIAS_CACHE_DIR)
    cached = set(docs_bias[VARIANTS[0]])
    for variant in VARIANTS[1:]:
        cached &= set(docs_bias[variant])  # only fully-cached docids count as cached
    if force:
        print(f"--force: recomputing document bias for this run's docids, keeping the "
              f"{len(cached - set(needed_docids))} unrelated cached docid(s) in {DOCS_BIAS_CACHE_DIR}")
        cached = set()
    else:
        print(f"document-bias cache: {len(cached)} docid(s) already in {DOCS_BIAS_CACHE_DIR}")

    missing = sorted(set(needed_docids) - cached, key=int)
    if not missing:
        print(f"all {len(needed_docids)} run docid(s) already cached -- not rescanning the collection")
        return docs_bias

    print(f"computing document bias for {len(missing)} docid(s) not in the cache")
    # Same collection reader and same COLLECTION_PATH as apply_steering_and_evaluate.py. (It
    # keeps everything after the first tab as the text; step1 treated a line with more than 2
    # tab-separated fields as empty. MS MARCO collection.tsv has exactly 2 fields per line.)
    doc_texts = fetch_doc_texts(COLLECTION_PATH, missing)

    empty_cnt = 0
    for docid in missing:
        text = doc_texts.get(docid, "")
        if not text:
            empty_cnt += 1  # step1's empty_cnt: bias is all-zeros for these, they are not dropped
        res = get_bias(get_tokens(text), genderwords_feml, genderwords_male)
        for variant in VARIANTS:
            docs_bias[variant][docid] = res[variant]
    print(f"documents with empty/missing text (bias 0, kept): {empty_cnt}")

    write_docs_bias_cache(DOCS_BIAS_CACHE_DIR, docs_bias)
    print(f"wrote document-bias cache for {len(docs_bias[VARIANTS[0]])} docid(s)")
    return docs_bias


# ============================================================
# step2: per-query RaB / ARaB from a run file  (port of step2_calculate_bias_runs.ipynb)
# ============================================================
def read_run_rows(run_path, qids_filter):
    """Read a 6-column TREC run into {qid: [(rank, docid), ...]} in FILE order.

    Same handling as step2: only 6-field lines are used, queries outside qids_filter are
    skipped, and a line whose qid differs from the previous one starts that qid's list over
    (so a run listing a qid in two separate blocks keeps only the last block, exactly as the
    original does). step2 ignores the rank column entirely and relies on the file being in
    rank order within each query; the rank is kept here only for the ordering check below.
    """
    rows = {}
    with open(run_path, encoding="utf-8") as fr:
        qryid_cur = None
        for line in fr:
            vals = line.strip().split(' ')
            if len(vals) == 6:
                qryid = vals[0]
                docid = vals[2]
                rank = int(vals[3])

                if qryid not in qids_filter:
                    continue

                if qryid != qryid_cur:
                    rows[qryid] = []
                    qryid_cur = qryid
                rows[qryid].append((rank, docid))
    return rows


def run_docs_bias_lists(rows, docs_bias, run_path):
    """{variant: {qid: [bias tuple per ranked doc]}} -- step2's runs_docs_bias, per run file."""
    out = {v: {} for v in VARIANTS}
    for qid, qrows in rows.items():
        if [r for r, _ in qrows] != sorted(r for r, _ in qrows):
            # Never expected from write_trec_run (it writes rank-ascending); step2's file-order
            # assumption would silently mis-rank if it happened, so repair it and say so.
            print(f"  warning: {run_path} is not rank-ordered for qid {qid} -- sorting by rank")
            qrows = sorted(qrows)
        for variant in VARIANTS:
            out[variant][qid] = [docs_bias[variant][docid] for _, docid in qrows]
    return out


def calc_RaB_q(bias_list, at_rank):
    """step2's calc_RaB_q, verbatim."""
    bias_val = np.mean([x[0] for x in bias_list[:at_rank]])
    bias_feml_val = np.mean([x[1] for x in bias_list[:at_rank]])
    bias_male_val = np.mean([x[2] for x in bias_list[:at_rank]])

    return bias_val, bias_feml_val, bias_male_val


def calc_ARaB_q(bias_list, at_rank):
    """step2's calc_ARaB_q, verbatim: the mean of RaB@1..RaB@at_rank.

    The `len(bias_list) >= t+1` guard is the original's: a query with fewer than at_rank
    retrieved documents averages over however many prefixes it has.
    """
    _vals = []
    _feml_vals = []
    _male_vals = []
    for t in range(at_rank):
        if len(bias_list) >= t + 1:
            _val_RaB, _feml_val_RaB, _male_val_RaB = calc_RaB_q(bias_list, t + 1)
            _vals.append(_val_RaB)
            _feml_vals.append(_feml_val_RaB)
            _male_vals.append(_male_val_RaB)

    bias_val = np.mean(_vals)
    bias_feml_val = np.mean(_feml_vals)
    bias_male_val = np.mean(_male_vals)

    return bias_val, bias_feml_val, bias_male_val


# ============================================================
# step3: aggregation over queries  (port of step3_bias_metrics.ipynb)
# ============================================================
def aggregate_arab(per_query, queries_effective):
    """step3's query-level average, including its SIGN FLIP.

    step3 builds _feml_list / _male_list from the per-query tuples' components [1] and [2] and
    reports np.mean([(_male_x - _feml_x) ...]) -- i.e. MALE MINUS FEMALE, the opposite sign to
    the per-document bias[0] that step1 computed. Hence, in the returned value:
        > 0  ==  MALE-leaning     < 0  ==  FEMALE-leaning
    Queries with no entry for this run are skipped, as in step3. queries_effective must be an
    ORDERED sequence, not a set: np.mean's summation order is what it is, and iterating a set of
    qids makes the last float bits depend on PYTHONHASHSEED, so the same run.trec would produce
    a slightly different number in each process. (step3 iterates a dict, so it is already
    insertion-ordered.)

    |ARaB| (see the module docstring) is this script's addition: the mean of the per-query
    magnitudes |male - feml|, not the absolute value of the signed mean.
    """
    _feml_list = []
    _male_list = []
    for qryid in queries_effective:
        if qryid in per_query:
            _feml_list.append(per_query[qryid][1])
            _male_list.append(per_query[qryid][2])

    if not _male_list:
        return float("nan"), float("nan"), 0

    signed = float(np.mean([(_male_x - _feml_x) for _male_x, _feml_x in zip(_male_list, _feml_list)]))
    magnitude = float(np.mean([abs(_male_x - _feml_x) for _male_x, _feml_x in zip(_male_list, _feml_list)]))
    return signed, magnitude, len(_male_list)


# ============================================================
# run discovery (inverse of apply_steering_and_evaluate.coef_folder)
# ============================================================
def parse_coef_folder(level, folder_name):
    """'alpha_-9.00' -> -9.0 (embed), 'beta_0.25' -> 0.25 (attn). None if not such a folder."""
    prefix = COEF_PREFIX[level] + "_"
    if not folder_name.startswith(prefix):
        return None
    try:
        return float(folder_name[len(prefix):])
    except ValueError:
        return None


def discover_runs():
    """[(level, direction, coef, run_path)] for every OUTPUT_DIR/{level}/{dir}/{coef}/run.trec."""
    found = []
    for level in LEVELS:
        for direction in DIRECTIONS:
            direction_dir = OUTPUT_DIR / level / direction
            if not direction_dir.is_dir():
                continue
            for coef_dir in sorted(direction_dir.iterdir()):
                if not coef_dir.is_dir():
                    continue
                coef = parse_coef_folder(level, coef_dir.name)
                if coef is None:
                    continue
                if coef_folder(level, coef) != coef_dir.name:
                    print(f"note: {coef_dir} does not round-trip through coef_folder() -- "
                          f"reading it as coefficient {coef}")
                run_path = coef_dir / "run.trec"
                if not run_path.exists():
                    print(f"note: {coef_dir} has no run.trec -- skipping (not creating it)")
                    continue
                found.append((level, direction, coef, run_path))
    return sorted(found, key=lambda x: (x[0], x[1], x[2]))


def report_expected_but_missing(found):
    """Compare what's on disk against the grids run_config.json says were swept.

    Advisory only: a directory that was swept more than once (e.g. a coarse pass and then a
    fine pass) keeps only the last run's config, so a 'missing' cell here can simply be a
    coefficient of an earlier grid. Nothing is ever created in response.
    """
    config_path = OUTPUT_DIR / "run_config.json"
    if not config_path.exists():
        print(f"note: no {config_path.name} -- cannot check the discovered runs against an expected grid")
        return
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"note: {config_path.name} is not valid JSON -- skipping the expected-grid check")
        return
    levels_cfg = cfg.get("levels") or {lvl: cfg for lvl in cfg.get("run_levels", [])}

    have = {(lvl, d, round(c, 2)) for lvl, d, c, _ in found}
    for level, lvl_cfg in levels_cfg.items():
        if level not in LEVELS:
            continue
        lo_key, hi_key, step_key = (
            ("alpha_min", "alpha_max", "alpha_step") if level == "embed" else ("beta_min", "beta_max", "beta_step")
        )
        if any(lvl_cfg.get(k) is None for k in (lo_key, hi_key, step_key)):
            continue
        lo, hi, step = float(lvl_cfg[lo_key]), float(lvl_cfg[hi_key]), float(lvl_cfg[step_key])
        if step <= 0:
            continue
        grid = [round(v, 4) for v in np.arange(lo, hi + step / 2, step)]
        for direction in DIRECTIONS:
            absent = [c for c in grid if (level, direction, round(c, 2)) not in have]
            if absent:
                print(f"note: {level}/{direction}: {len(absent)}/{len(grid)} cell(s) of the "
                      f"run_config grid ({lo} to {hi} step {step}) have no run.trec, e.g. "
                      f"{[coef_folder(level, c) for c in absent[:5]]}")


# ============================================================
# results CSV (separate from sweep_results.csv, which is never touched)
# ============================================================
def load_existing_arab_rows(csv_path):
    if not csv_path.exists():
        return {}
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    existing = {}
    for row in rows:
        try:
            key = (row["level"], row["direction"], round(float(row["coefficient"]), 2))
        except (KeyError, TypeError, ValueError):
            continue
        existing[key] = row
    return existing


def write_arab_results(csv_path, rows_by_key):
    tmp_path = csv_path.with_suffix(".csv.tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ARAB_CSV_FIELDS)
        writer.writeheader()
        for key in sorted(rows_by_key, key=lambda k: (k[0], k[1], k[2])):
            row = rows_by_key[key]
            writer.writerow({k: row.get(k, "") for k in ARAB_CSV_FIELDS})
    tmp_path.replace(csv_path)


def load_sweep_results(csv_path):
    """Existing NFaiRR@10 / MRR@10, read only -- never recomputed, never rewritten."""
    if not csv_path.exists():
        print(f"note: {csv_path} not found -- the combined table will show no NFaiRR@10")
        return {}
    out = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = (row["level"], row["direction"], round(float(row["coefficient"]), 2))
            except (KeyError, TypeError, ValueError):
                continue
            out[key] = row
    return out


def _fmt(value, width=10, prec=4):
    try:
        return f"{float(value):>{width}.{prec}f}"
    except (TypeError, ValueError):
        return f"{'-':>{width}}"


def print_combined_table(rows_by_key, sweep):
    print(f"\nARaB@{K} (positive = MALE-leaning) joined with the existing NFaiRR@{K} / MRR@{K} "
          f"from sweep_results.csv:")
    header = (f"{'level':<6} {'dir':<4} {'coef':>7} {'NFaiRR@10':>10} {'MRR@10':>8}"
              + "".join(f"{'ARaB_' + v:>10}{'|ARaB|_' + v:>11}" for v in VARIANTS))
    print(header)
    print("-" * len(header))
    for key in sorted(rows_by_key, key=lambda k: (k[0], k[1], k[2])):
        level, direction, coef = key
        row = rows_by_key[key]
        sweep_row = sweep.get(key, {})
        line = (f"{level:<6} {direction:<4} {coef:>7.2f} "
                f"{_fmt(sweep_row.get('NFaiRR@10'), 10)} {_fmt(sweep_row.get('MRR@10'), 8)}")
        for v in VARIANTS:
            line += f"{_fmt(row.get(f'ARaB@{K}_{v}'), 10)}{_fmt(row.get(f'|ARaB|@{K}_{v}'), 11)}"
        print(line)


# ============================================================
# sanity checks
# ============================================================
def _signed(rows_by_key, key, variant):
    row = rows_by_key.get(key)
    if row is None:
        return None
    try:
        return float(row[f"ARaB@{K}_{variant}"])
    except (KeyError, TypeError, ValueError):
        return None


def _magnitude(rows_by_key, key, variant):
    row = rows_by_key.get(key)
    if row is None:
        return None
    try:
        return float(row[f"|ARaB|@{K}_{variant}"])
    except (KeyError, TypeError, ValueError):
        return None


def check_zero_coefficient_agreement(rows_by_key):
    """At coefficient 0 nothing is steered, so all three directions read the SAME run -- their
    ARaB must be identical, not merely close. Any difference means the runs, the query filter
    or the doc-bias lookup disagree somewhere."""
    print("\n[check] coefficient 0 is the unsteered run: ARaB must be identical across mf/mn/fn")
    all_values = []
    for level in LEVELS:
        for variant in VARIANTS:
            vals = {d: _signed(rows_by_key, (level, d, 0.0), variant) for d in DIRECTIONS}
            present = {d: v for d, v in vals.items() if v is not None}
            if len(present) < 2:
                print(f"  {level:<6} {variant:<5} skipped: coefficient 0 present for "
                      f"{sorted(present) or 'no direction'}")
                continue
            spread = max(present.values()) - min(present.values())
            all_values.extend(present.values())
            status = "OK  " if spread <= 1e-12 else "FLAG"
            print(f"  {status} {level:<6} {variant:<5} {', '.join(f'{d}={v:+.6f}' for d, v in present.items())}"
                  + ("" if spread <= 1e-12 else f"   spread={spread:.3e} -- the unsteered runs disagree!"))
    if not all_values:
        print("  (no coefficient-0 cell found at all)")


def _describe_shape(mag_vals):
    """(shape, min_coef, min_mag, max_coef, max_mag) for a |ARaB| curve: 'U', 'inverted-U',
    'monotone' or 'flat'. Interior means strictly inside the grid, not at either end."""
    coefs = [c for c, _ in mag_vals]
    mags = [m for _, m in mag_vals]
    min_coef, min_mag = min(mag_vals, key=lambda cm: cm[1])
    max_coef, max_mag = max(mag_vals, key=lambda cm: cm[1])
    ends = (mags[0], mags[-1])
    if max_mag - min_mag <= 1e-12:
        shape = "flat"
    elif min_coef not in (coefs[0], coefs[-1]) and all(e > min_mag for e in ends):
        shape = "U"
    elif max_coef not in (coefs[0], coefs[-1]) and all(e < max_mag for e in ends):
        shape = "inverted-U"
    else:
        shape = "monotone"
    return shape, min_coef, min_mag, max_coef, max_mag


def check_mf_embed_trend(rows_by_key):
    """embed/mf is the direction that should sweep the ranking from one gender to the other:
    signed ARaB@10 trending monotonically through zero across the alpha grid, with |ARaB|@10
    U-shaped -- bottoming out where the sign changes. This reports what the data actually does
    and FLAGS any departure; it never adjusts anything.

    A frequent, and legitimate, departure: |ARaB| comes out INVERTED-U (peaking near alpha=0
    and collapsing toward 0 at both grid ends). That is the signature of large |alpha| pushing
    every gendered document out of the top K -- a top-10 of gender-free documents scores 0 on
    both the signed and the magnitude metric -- rather than of the ranking flipping gender. The
    flag says so explicitly, because the two look identical in the signed column alone.
    """
    print(f"\n[check] embed/mf: signed ARaB@{K} should trend through zero; |ARaB|@{K} U-shaped near the crossing")
    coefs = sorted(c for (lvl, d, c) in rows_by_key if lvl == "embed" and d == "mf")
    if len(coefs) < 5:
        print(f"  skipped: only {len(coefs)} embed/mf coefficient(s) available")
        return

    for variant in VARIANTS:
        signed = [(c, _signed(rows_by_key, ("embed", "mf", c), variant)) for c in coefs]
        signed = [(c, v) for c, v in signed if v is not None]
        if len(signed) < 5:
            print(f"  {variant:<5} skipped: too few values")
            continue
        values = [v for _, v in signed]
        flags = []

        # --- does the signed curve pass through zero, and where? ---
        crossings = [(a, b) for (a, va), (b, vb) in zip(signed, signed[1:]) if va * vb < 0]
        if min(values) < 0 < max(values):
            cross_coef = (crossings[0][0] + crossings[0][1]) / 2 if crossings else \
                min(signed, key=lambda cv: abs(cv[1]))[0]
            cross_txt = (f"crosses zero {len(crossings)}x, first between alpha="
                         f"{crossings[0][0]:+.2f} and {crossings[0][1]:+.2f}" if crossings
                         else f"crosses zero near alpha={cross_coef:+.2f}")
            if len(crossings) > 1:
                flags.append(f"crosses zero {len(crossings)} times, not once -- the signed curve "
                             f"is noisy around 0, so 'the' crossing is not well defined")
        else:
            cross_coef = min(signed, key=lambda cv: abs(cv[1]))[0]
            cross_txt = f"never changes sign (closest to 0 at alpha={cross_coef:+.2f})"
            flags.append(f"no sign change at all (range {min(values):+.4f} .. {max(values):+.4f})")

        # --- is the signed curve a consistent trend in alpha? ---
        diffs = [b - a for a, b in zip(values, values[1:])]
        nonzero = [d for d in diffs if d != 0]
        trend = 0.0
        if nonzero:
            ups = sum(1 for d in nonzero if d > 0)
            trend = max(ups, len(nonzero) - ups) / len(nonzero)
            if trend < 0.8:
                flags.append(f"not a monotone trend -- only {trend:.0%} of the {len(nonzero)} grid "
                             f"steps move the same way")

        # --- is |ARaB| U-shaped, and does its minimum sit near the crossing? ---
        mag_vals = [(c, _magnitude(rows_by_key, ("embed", "mf", c), variant)) for c, _ in signed]
        mag_vals = [(c, m) for c, m in mag_vals if m is not None]
        mag_txt = ""
        if mag_vals:
            shape, min_coef, min_mag, max_coef, max_mag = _describe_shape(mag_vals)
            step = (coefs[-1] - coefs[0]) / max(len(coefs) - 1, 1)
            mag_txt = (f"; |ARaB| {shape}: min {min_mag:.4f} at alpha={min_coef:+.2f}, "
                       f"max {max_mag:.4f} at alpha={max_coef:+.2f}, "
                       f"ends {mag_vals[0][1]:.4f}/{mag_vals[-1][1]:.4f}")
            if shape != "U":
                flags.append(f"|ARaB| is {shape}, not U-shaped")
                if shape == "inverted-U":
                    flags.append(f"  -> |ARaB| peaks at alpha={max_coef:+.2f} and falls to ~0 at both "
                                 f"grid ends: the top {K} is losing its gendered documents "
                                 f"altogether at large |alpha|, rather than flipping gender")
            elif abs(min_coef - cross_coef) > 2 * step:
                flags.append(f"|ARaB| minimum (alpha={min_coef:+.2f}) is more than 2 grid steps "
                             f"from the sign change (alpha={cross_coef:+.2f})")

        summary = (f"signed {values[0]:+.4f} at alpha={signed[0][0]:+.2f} -> {values[-1]:+.4f} at "
                   f"alpha={signed[-1][0]:+.2f}, {cross_txt}, trend consistency {trend:.0%}{mag_txt}")
        print(f"  {'FLAG' if flags else 'OK  '} {variant:<5} {summary}")
        for flag in flags:
            print(f"       - {flag}")


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true",
                        help="recompute every discovered cell instead of reusing what is already "
                             "in arab_results.csv, and recompute this run's document bias from the "
                             "collection instead of reading it from the cache (cached docids that "
                             "belong to other sweeps are kept)")
    args = parser.parse_args()

    print(f"Model: {MODEL_NAME}  (short name: {SHORT_NAME})")
    print(f"Dataset: {DATASET} -- {CFG['query_set']}")
    print(f"Seed: {SEED}   OUTPUT_DIR: {OUTPUT_DIR}")
    split_note = ("no split -- every query is evaluation data" if DATASET == "trecdl"
                  else f"TEST_QUERIES_SPLIT={os.environ.get('TEST_QUERIES_SPLIT', 'full')}")
    print(f"Test queries: {TEST_QUERIES_PATH.name} ({split_note})")
    print(f"Wordlist: {WORDLIST_GENDERSPECIFIC_PATH}")
    print(f"Sign convention (from step3): ARaB@{K} > 0 == MALE-leaning, < 0 == FEMALE-leaning")
    print(f"Variants: {', '.join(VARIANTS)} ('bool' = boolean)   at_rank K={K}")
    print("READ-ONLY: no model is loaded, no steering is run, no run.trec is created or modified")
    if N_TEST_QUERIES_SUBSET:
        print(f"N_TEST_QUERIES_SUBSET={N_TEST_QUERIES_SUBSET} -- averaging over a query subset, not the full split")

    if not OUTPUT_DIR.is_dir():
        sys.exit(f"OUTPUT_DIR does not exist: {OUTPUT_DIR}\n"
                 "Check MODEL_KEY / OUTPUT_DIR_NAME / SEED -- this script only reads existing runs.")

    genderwords_feml, genderwords_male = load_gender_wordlists(WORDLIST_GENDERSPECIFIC_PATH)
    print(f"{len(genderwords_feml)} female / {len(genderwords_male)} male wordlist terms")

    # Same test-query loading as the sweep, via the same imported helpers: the dataset's queries,
    # asserted disjoint from the vector-construction queries, optionally truncated by
    # N_TEST_QUERIES_SUBSET.
    construction_qids = load_construction_qids()
    queries = load_test_queries(TEST_QUERIES_PATH, construction_qids, limit=N_TEST_QUERIES_SUBSET)
    test_qids = set(queries)               # membership filter (step2's qryids_filter)
    queries_effective = list(queries)      # ORDERED average (step3's queries_effective); see aggregate_arab
    print(f"{len(queries)} test queries loaded (construction queries confirmed disjoint)")

    found = discover_runs()
    print(f"\ndiscovered {len(found)} run.trec file(s) under {OUTPUT_DIR}")
    if not found:
        sys.exit("no run.trec files found -- nothing to do (this script never creates runs)")
    report_expected_but_missing(found)

    csv_path = OUTPUT_DIR / "arab_results.csv"
    rows_by_key = {} if args.force else load_existing_arab_rows(csv_path)
    if rows_by_key:
        print(f"{len(rows_by_key)} cell(s) already in {csv_path.name} -- skipping them (use --force to recompute)")
    todo = [(lvl, d, c, p) for lvl, d, c, p in found if (lvl, d, round(c, 2)) not in rows_by_key]
    print(f"{len(todo)} cell(s) to compute")

    if todo:
        # Pass 1: which docids do the cells we are about to compute actually contain?
        print("scanning the run files for the docids they contain...")
        needed_docids = set()
        for _lvl, _d, _c, run_path in todo:
            for qid, qrows in read_run_rows(run_path, test_qids).items():
                needed_docids.update(docid for _rank, docid in qrows)
        print(f"{len(needed_docids)} distinct docid(s) across those runs "
              f"(runs hold up to {CANDIDATE_CUTOFF} docs/query; only the top {K} enter ARaB@{K})")

        docs_bias = build_docs_bias(needed_docids, genderwords_feml, genderwords_male, args.force)

        # Pass 2: step2 per-query ARaB, then step3 aggregation, per cell.
        print("\ncomputing ARaB per cell...")
        warned_incomplete = False
        for lvl, direction, coef, run_path in todo:
            rows = read_run_rows(run_path, test_qids)
            if len(rows) != len(test_qids) and not warned_incomplete:
                missing_qids = sorted(test_qids - set(rows), key=int)
                print(f"  note: {run_path} covers {len(rows)}/{len(test_qids)} test queries "
                      f"(missing e.g. {missing_qids[:5]}); averaging over the queries present, "
                      f"as step3 does. Reported once.")
                warned_incomplete = True
            per_run = run_docs_bias_lists(rows, docs_bias, run_path)

            row = {"model": SHORT_NAME, "level": lvl, "direction": direction, "coefficient": coef}
            n_queries = 0
            for variant in VARIANTS:
                per_query = {
                    qid: calc_ARaB_q(bias_list, K) for qid, bias_list in per_run[variant].items()
                }
                signed, magnitude, n_queries = aggregate_arab(per_query, queries_effective)
                row[f"ARaB@{K}_{variant}"] = signed
                row[f"|ARaB|@{K}_{variant}"] = magnitude
            row["n_queries"] = n_queries
            rows_by_key[(lvl, direction, round(coef, 2))] = row

        write_arab_results(csv_path, rows_by_key)
        print(f"\nwrote {csv_path} ({len(rows_by_key)} row(s))")
    else:
        print("nothing to compute -- reporting the existing arab_results.csv")

    print_combined_table(rows_by_key, load_sweep_results(OUTPUT_DIR / "sweep_results.csv"))
    check_zero_coefficient_agreement(rows_by_key)
    check_mf_embed_trend(rows_by_key)
    print(f"\nWordlist used: {WORDLIST_GENDERSPECIFIC_PATH}")
    print(f"Sign convention: ARaB@{K} > 0 == MALE-leaning (step3's male-minus-female average)")


if __name__ == "__main__":
    main()

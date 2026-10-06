"""Apply the M/F/N steering vectors (embedding-level and attention-level) to one
model's retrieval of an evaluation query set, and evaluate the result
(MRR@10, nDCG@10, Recall@10, NFaiRR@10) across the full alpha/beta sweep.

DATASET picks the evaluation set (see DATASET_CONFIGS):
  msmarco (default) -- the 210 MS MARCO Fair test queries of n5_seed{SEED}, sliced by
                       TEST_QUERIES_SPLIT into the validation (15) / held-out (195) passes.
  trecdl            -- TREC DL 2019 Fair, all 30 queries, as a ZERO-SHOT evaluation: the
                       steering vectors are still the MS MARCO-derived ones built from
                       n5_seed{SEED}, every TREC DL query is evaluation data (there is no
                       validation/held-out split and no coefficient is ever selected on it),
                       and the coefficient grid defaults to the same one the MS MARCO
                       held-out sweep used so the two sets of curves are comparable.

Scope: ONE model per run (MODEL_KEY env var selects which of the 5 vector-construction
models; default "tas_b" -- see MODEL_CONFIGS_ALL for the others), both steering levels
(embed, attn), all 3 directions (mf/mn/fn). Output nests under OUTPUT_DIR_NAME/SHORT_NAME
so different models' runs never collide. This is correctness-first: every (level,
direction, coefficient) run
does a full, uncached forward pass per candidate doc -- no reuse of a doc's
unsteered/baseline embedding even when it has no gendered positions (where steering
is provably a no-op). That's a real, meaningful speedup left for a documented
follow-up, not applied here, to keep this first pass simple to verify.

(Tokenizing each candidate doc and finding its gendered token positions IS done once
up front, not once per coefficient -- that's deterministic text preprocessing with no
model forward pass involved, not the "cached baseline embedding" shortcut above.)

find_gendered_token_positions (new here) plays the same role for a single doc that
get_gendered_token_indices (swap_pair_utils.py) plays for an m/f/n triplet: identify
which of a model's tokenizer's positions correspond to a gendered word, using the
same punctuation-stripping tokenization as document_neutrality_fixed.py (imported,
not reimplemented) matched against the neutral-mapped wordlist (the same term set the
steering vectors themselves were built from -- no names, no sir/madam).

The embedding-level actadd hook (hook_full_embed) and the attention-level per-head
actadd hook (z-activation at each identified head) follow the same pattern as the
GREP-BiasIR reference scripts' steering_hook/actadd_hook and aggr_actadd_hook, just
written fresh here since this is application+evaluation, not vector construction.

The two levels are independent -- RUN_LEVELS picks embed, attn, or both, and running one
never disturbs the other's results (sweep_results.csv and run_config.json are merged, not
truncated). Each has its own coefficient grid (ALPHA_* / BETA_*), because the embedding
vectors are raw mean-diffs while the per-head attention vectors are unit-normalized.

At BOTH levels, only candidate documents containing at least one gendered term are steered.
At the embedding level that falls out of the empty-idx_diff slice; at the attention level
it is an explicit gate in compute_steered_doc_embedding. Within a steered document, which
token positions the attention hook edits is ATTN_STEER_SCOPE's job -- all positions
(default, as in the reference) or gendered positions only. See that constant's comment for
why the narrow option cannot work for CLS-pooled models.

Usage (from the repo root):
    conda run -n gender_bias python gender_bias_steering_utility/apply_steering_and_evaluate.py

Quick smoke test on a handful of queries before committing to the full
41-point, 2-level, 3-direction sweep:
    N_TEST_QUERIES_SUBSET=3 conda run -n gender_bias python gender_bias_steering_utility/apply_steering_and_evaluate.py

TREC DL 2019 zero-shot pass (all 30 queries, grid defaults to the MS MARCO held-out grid):
    DATASET=trecdl conda run -n gender_bias python gender_bias_steering_utility/apply_steering_and_evaluate.py

Coarse-then-fine alpha search: run a wide, coarse-stepped pass on a small query subset
first (own OUTPUT_DIR_NAME so it doesn't collide with a real run) to find the useful
range before committing to the full fine-grained sweep there:
    N_TEST_QUERIES_SUBSET=10 ALPHA_MIN=-50 ALPHA_MAX=50 ALPHA_STEP=5 OUTPUT_DIR_NAME=steering_coarse \\
        conda run -n gender_bias python gender_bias_steering_utility/apply_steering_and_evaluate.py
"""
import contextlib
import csv
import io
import json
import os
import re
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import transformer_lens.utils as utils
from mechir import Dot
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from document_neutrality_fixed import STRIP_PUNCT_RE  # noqa: E402  (reused, not reimplemented)
# The SAME effectiveness function the unsteered baselines use (compute_dense_baselines.py imports
# it too), so an alpha=0 sweep row is directly comparable to all_baselines_summary.tsv instead of
# being computed by a near-copy that could drift. Importing it loads no model and touches no GPU.
from compute_baseline_metrics import compute_retrieval_effectiveness  # noqa: E402
from fairr_full_collection import load_fairr_metric, compute_nfairr_full_collection_background  # noqa: E402
from swap_pair_utils import get_best_device  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "data/FairnessRetrievalResults/adversarial_mitigation/fairness_measurement"))
from metrics_fairness import FaiRRMetricHelper  # noqa: E402  (unmodified; only its run-file reader is used)

# ---- model (one model per run; MODEL_KEY selects which). Same 5 models the vector-
# construction scripts built vectors for -- this just consumes those, doesn't rebuild them. ----
MODEL_CONFIGS_ALL = {
    "tas_b": {"name": "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco", "pool": "cls", "sim_func": "dot"},
    "multi-qa-distilbert": {"name": "sentence-transformers/multi-qa-distilbert-dot-v1", "pool": "cls", "sim_func": "dot"},
    "multi-qa-minilm": {"name": "sentence-transformers/multi-qa-MiniLM-L6-dot-v1", "pool": "cls", "sim_func": "dot"},
    "msmarco-bert-base": {"name": "sentence-transformers/msmarco-bert-base-dot-v5", "pool": "mean", "sim_func": "dot"},
    "msmarco-distilbert": {"name": "sentence-transformers/msmarco-distilbert-dot-v5", "pool": "mean", "sim_func": "dot"},
}
MODEL_KEY = os.environ.get("MODEL_KEY", "tas_b")
_model_cfg = MODEL_CONFIGS_ALL[MODEL_KEY]
MODEL_NAME = _model_cfg["name"]
MODEL_POOL, MODEL_SIM_FUNC = _model_cfg["pool"], _model_cfg["sim_func"]
SHORT_NAME = MODEL_NAME.split("/")[-1]

# ---- paths ----
# SEED selects which n5_seed{SEED} split's construction pairs / steering VECTORS to apply. At the
# default (42) every path below is byte-for-byte what it was before this was made overridable --
# nothing about seed 42's existing files moves. Other seeds read from and write to their own
# sibling _seed{SEED} directories end-to-end. See run_steering_sweep.sh's "OTHER SEEDS" section.
#
# Under DATASET=msmarco the seed also selects the evaluation query set, since that set IS the
# seed's own test split. Under DATASET=trecdl it does not: the query set is fixed (all 30 TREC DL
# queries) and the seed only decides which MS MARCO-derived vectors get applied to it.
SEED = int(os.environ.get("SEED", 42))
SEED_SUFFIX = "" if SEED == 42 else f"_seed{SEED}"
CONSTRUCTION_PAIRS_PATH = REPO_ROOT / f"gender_bias_steering_utility/preprocess/vector_construction{SEED_SUFFIX}/construction_pairs.jsonl"
EMBED_VECTOR_DIR = REPO_ROOT / f"gender_bias_steering_utility/experiments/vector_construction{SEED_SUFFIX}/embed" / SHORT_NAME
ATTN_VECTOR_DIR = REPO_ROOT / f"gender_bias_steering_utility/experiments/vector_construction{SEED_SUFFIX}/attn" / SHORT_NAME

## TEST_QUERIES_SPLIT selects which split of the seed's MS MARCO test queries to evaluate on. Use
## the validation subset (15 queries) to search/select a coefficient, and only ever evaluate the
## held-out subset (195 queries) ONCE, at the single chosen coefficient, for the number that
## actually gets reported -- using the same queries to both pick alpha and report the final metric
## would overfit the hyperparameter to the eval set. Override via env var:
##   TEST_QUERIES_SPLIT=validation | heldout | full   (default: full, i.e. all 210 -- set
##   this explicitly once you're doing coefficient selection or final reporting)
## It is an MS MARCO concept only; see DATASET_CONFIGS["trecdl"] below.
_TEST_QUERIES_FILES = {
    "full": "test_queries.tsv",
    "validation": "test_queries_validation.tsv",
    "heldout": "test_queries_heldout.tsv",
}
DATASET = os.environ.get("DATASET", "msmarco")
TEST_QUERIES_SPLIT = os.environ.get("TEST_QUERIES_SPLIT", "full")
# Pinned to "full" for the msmarco config entry when DATASET=trecdl: the dict literal below is
# evaluated whichever dataset is selected, so a leftover TEST_QUERIES_SPLIT (run_steering_sweep.sh
# exports one unconditionally) must not be able to KeyError a trecdl run.
_MSMARCO_SPLIT = TEST_QUERIES_SPLIT if DATASET == "msmarco" else "full"
assert _MSMARCO_SPLIT in _TEST_QUERIES_FILES, (
    f"TEST_QUERIES_SPLIT must be one of {sorted(_TEST_QUERIES_FILES)}, got {TEST_QUERIES_SPLIT!r}"
)

# ---- evaluation dataset ----
# Same switch shape as compute_dense_baselines.py's DATASET_CONFIGS, and deliberately pointed at
# the same three TREC DL files, so the steered numbers and the unsteered baselines in
# experiments/baseline/trecdl19/ are computed over exactly the same queries, pool and qrels.
#
# The msmarco BM25 pool is NOT the external
# data/FairnessRetrievalResults/measurement/sample_trec_runs/.../BM25.run -- that's a different
# BM25 system (different index/parameters) than the one the rest of this pipeline is built on
# (dense baselines, vector construction, annotation), confirmed by checking a shared query:
# completely different top candidate and score scale, not just rounding differences.
# test_candidates.tsv is this project's own PyTerrier BM25 pool (see bm25_retrieve.py) and is what
# all_baselines_summary.tsv's numbers are computed against -- using anything else means alpha=0
# here can never match that reference baseline. The TREC DL pool is that dataset's own published
# top-100 BM25 run, which is likewise what its baseline summary was computed against.
#
# alpha_grid/beta_grid are the DEFAULT (min, max, step) per level; ALPHA_*/BETA_* override them.
# trecdl's are the grid the MS MARCO held-out sweep actually ran (experiments/steering_heldout_final:
# embed -10..10 step 0.5, attn -14..14 step 0.5), so a bare DATASET=trecdl submission is
# point-for-point comparable with the held-out curves by construction rather than by the submitter
# remembering to pass the grid. msmarco's are unchanged from what they have always been.
DATASET_CONFIGS = {
    "msmarco": {
        "queries": REPO_ROOT / f"gender_bias_steering_utility/preprocess/n5_seed{SEED}" / _TEST_QUERIES_FILES[_MSMARCO_SPLIT],
        "candidates": REPO_ROOT / f"gender_bias_steering_utility/preprocess/n5_seed{SEED}/test_candidates.tsv",
        "qrels": REPO_ROOT / "data/msmarco/msmarco.qrels.dev.tsv",
        "min_rel": 1,  # MS MARCO dev qrels are binary
        "dir_suffix": "",
        "split_label": TEST_QUERIES_SPLIT,
        "query_set": f"MS MARCO Fair, n5_seed{SEED} {TEST_QUERIES_SPLIT} split",
        "alpha_grid": (-5, 5, 0.25),
        "beta_grid": (-5, 5, 0.25),
    },
    "trecdl": {
        # 30 queries, ALL of them evaluation data. The FULL_ANNOTATION file has a header row and
        # extra annotation columns (Categories, Domains, ...); load_test_queries reads only the
        # first two tab-separated columns and skips the header via its isdigit() check.
        "queries": REPO_ROOT / "data/FairnessRetrievalResults/dataset/trecdeep19_passage.fair.FULL_ANNOTATION.tsv",
        "candidates": HERE / "data/trecdeep19_passage.fair.bm25.top100.trec",  # headerless 6-col TREC run
        "qrels": REPO_ROOT / "data/msmarco/qrels.trec-dl-2019.ir_datasets.tsv",
        "min_rel": 2,  # graded 0-3; grade >=2 counts as relevant for MRR/Recall, nDCG uses raw grades
        "dir_suffix": "_trecdl19",
        # NOT "heldout": query_ranking_dashboard.py selects runs by that exact label and then
        # renders them against hard-coded MS MARCO qrels and query/candidate files. A distinct
        # label keeps trecdl runs out of a dashboard that cannot read them, and steering_dashboard.py
        # falls back to showing the raw string.
        "split_label": "trecdl19_all",
        "query_set": "TREC DL 2019 Fair, all 30 queries as zero-shot evaluation",
        "alpha_grid": (-10, 10, 0.5),
        "beta_grid": (-14, 14, 0.5),
    },
}
assert DATASET in DATASET_CONFIGS, f"DATASET must be one of {sorted(DATASET_CONFIGS)}, got {DATASET!r}"
CFG = DATASET_CONFIGS[DATASET]
TEST_QUERIES_PATH = CFG["queries"]
BM25_RUN_PATH = CFG["candidates"]
QRELS_PATH = CFG["qrels"]

WORDLIST_PATH = REPO_ROOT / "gender_bias_steering_utility/data/wordlist_gender_representative_no_names_w_neutral.txt"
# Collection-wide, not query-set-dependent -- shared across every seed AND both datasets (TREC DL
# 2019 passages are the same MS MARCO passage collection), never suffixed.
NEUTRALITY_SCORES_V2_PATH = REPO_ROOT / "gender_bias_steering_utility/experiments/baseline/collection_neutralityscores_v2.tsv"
COLLECTION_PATH = REPO_ROOT / "data/msmarco/collection.tsv"
# The dataset suffix goes BEFORE the seed suffix so steering_dashboard.py's _seed(\d+)$ regex still
# strips the seed correctly (steering_trecdl19_seed29 -> sweep "steering_trecdl19", seed 29). It is
# applied here rather than left to OUTPUT_DIR_NAME so a trecdl run can never overwrite MS MARCO rows.
OUTPUT_DIR_NAME = os.environ.get("OUTPUT_DIR_NAME", "steering")
OUTPUT_DIR = REPO_ROOT / "gender_bias_steering_utility/experiments" / f"{OUTPUT_DIR_NAME}{CFG['dir_suffix']}{SEED_SUFFIX}" / SHORT_NAME
# The MS MARCO run this trecdl run should be grid-comparable with; used only for the warning in
# check_grid_parity(). None under DATASET=msmarco.
MSMARCO_COUNTERPART_CONFIG = (
    REPO_ROOT / "gender_bias_steering_utility/experiments" / f"{OUTPUT_DIR_NAME}{SEED_SUFFIX}" / SHORT_NAME / "run_config.json"
    if DATASET == "trecdl" else None
)

COLLECTION_SIZE_HINT = 8841823
CANDIDATE_CUTOFF = 100
K = 10  # MRR/nDCG/Recall/NFaiRR cutoff
MIN_REL = CFG["min_rel"]  # grade at/above which a qrel counts as relevant, for MRR and Recall
NFAIRR_THRESHOLDS = [10]
DIRECTIONS = ["mf", "mn", "fn"]
LEVELS = ["embed", "attn"]

# The reference scripts' -5..5 step-0.25 range was tuned for the GREP-BiasIR paper's own
# embedding scale, not verified against ours (our pooled vector norms are ~9-16, so alpha=5
# already means a perturbation of magnitude ~50-80 at each gendered position -- quite possibly
# past the point where retrieval quality collapses). ALPHA_MIN/MAX/STEP let a coarse pass over
# a wider range find the useful region cheaply before committing to a fine full sweep there.
# The unset default comes from the dataset's alpha_grid (see DATASET_CONFIGS).
_ALPHA_LO, _ALPHA_HI, _ALPHA_ST = CFG["alpha_grid"]
ALPHA_MIN = float(os.environ.get("ALPHA_MIN", _ALPHA_LO))
ALPHA_MAX = float(os.environ.get("ALPHA_MAX", _ALPHA_HI))
ALPHA_STEP = float(os.environ.get("ALPHA_STEP", _ALPHA_ST))
ALPHA_VALUES = [round(v, 4) for v in np.arange(ALPHA_MIN, ALPHA_MAX + ALPHA_STEP / 2, ALPHA_STEP)]

# Beta (attention level) gets its OWN grid rather than reusing ALPHA_VALUES: the two levels'
# vectors are on different scales. Embedding vectors are raw mean-diffs (norm ~9-16); the
# per-head attention vectors in per_head_vectors.json are unit-normalized, the way the
# GREP-BiasIR reference normalizes each head's slice in split_direction_to_heads even for
# actadd. So the same numeric coefficient means a ~10x smaller perturbation at the attention
# level, and a shared grid would put the attention sweep in the wrong part of the range.
# Run a wide coarse pass (e.g. BETA_MIN=-20 BETA_MAX=20 BETA_STEP=2) to find the useful
# region before committing to a fine sweep there.
_BETA_LO, _BETA_HI, _BETA_ST = CFG["beta_grid"]
BETA_MIN = float(os.environ.get("BETA_MIN", _BETA_LO))
BETA_MAX = float(os.environ.get("BETA_MAX", _BETA_HI))
BETA_STEP = float(os.environ.get("BETA_STEP", _BETA_ST))
BETA_VALUES = [round(v, 4) for v in np.arange(BETA_MIN, BETA_MAX + BETA_STEP / 2, BETA_STEP)]

N_TEST_QUERIES_SUBSET = os.environ.get("N_TEST_QUERIES_SUBSET")
N_TEST_QUERIES_SUBSET = int(N_TEST_QUERIES_SUBSET) if N_TEST_QUERIES_SUBSET else None

# Which steering levels to run. The two are independent and either can be run alone:
# RUN_LEVELS=embed (default, unchanged), RUN_LEVELS=attn, or RUN_LEVELS=embed,attn.
RUN_LEVELS = os.environ.get("RUN_LEVELS", "embed").split(",")

# Which token positions the attention-level z-hook edits within a steered document.
#
#   all_positions      (default) every position, including CLS -- what the GREP-BiasIR
#                      reference does (steering_attn.py::aggr_actadd_hook).
#   gendered_positions only the document's gendered token positions.
#
# gendered_positions is an architectural NO-OP for the CLS-pooled models (tas_b,
# multi-qa-distilbert, multi-qa-minilm): everything after the last layer's attention
# sublayer -- W_O, both residual adds, both LayerNorms, the MLP -- is position-wise, so a
# non-CLS z edit at the final layer can never reach the CLS token, and there is no later
# attention layer to carry it there (verified empirically: bit-identical output even at
# beta=+-1000, 200x the sweep's max). It is retained for comparison, and is meaningful for
# the mean-pooled models (msmarco-bert-base, msmarco-distilbert) where the edited positions
# do contribute to the pool.
#
# Either way, WHICH DOCUMENTS get steered is the same as at the embedding level: only
# candidates containing at least one gendered term. See compute_steered_doc_embedding.
ATTN_STEER_SCOPE = os.environ.get("ATTN_STEER_SCOPE", "all_positions")
assert ATTN_STEER_SCOPE in ("all_positions", "gendered_positions"), (
    f"ATTN_STEER_SCOPE must be 'all_positions' or 'gendered_positions', got {ATTN_STEER_SCOPE!r}"
)
CLS_POOLED_MODELS = {"tas_b", "multi-qa-distilbert", "multi-qa-minilm"}


# ============================================================
# gendered token positions in a single doc (companion to swap_pair_utils.get_gendered_token_indices,
# which needs an m/f/n triplet -- this needs just one doc)
# ============================================================
def load_wordlist_terms(path):
    """The set of lowercased terms this steering setup can act on -- the same
    neutral-mapped, non-name wordlist the steering vectors themselves were built
    from (no names, no sir/madam)."""
    terms = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            terms.add(line.split(",")[0].lower())
    return terms


def find_gendered_token_positions(doc_text, tokenizer, wordlist):
    """Whole-word match doc_text against wordlist, reusing document_neutrality_fixed's
    punctuation-stripping regex (STRIP_PUNCT_RE) rather than rewriting the matching
    logic, then map each matched word's character span onto the tokenizer's subword
    token indices that overlap it. Returns a sorted list of subword token indices
    (empty if the doc has no gendered terms -- steering is then a no-op for it, via
    the actadd hooks' empty-index no-op, not a special case here)."""
    cleaned = STRIP_PUNCT_RE.sub(" ", doc_text.lower())
    word_spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", cleaned)]
    matched_spans = [(s, e) for s, e in word_spans if cleaned[s:e] in wordlist]
    if not matched_spans:
        return []

    encoded = tokenizer(doc_text, add_special_tokens=True, return_offsets_mapping=True)
    offset_mapping = encoded["offset_mapping"]

    positions = set()
    for span_start, span_end in matched_spans:
        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end == 0:
                continue  # special token
            if tok_start < span_end and tok_end > span_start:
                positions.add(i)
    return sorted(positions)


# ============================================================
# actadd hooks -- same pattern as the GREP-BiasIR reference scripts'
# actadd_hook / aggr_actadd_hook, written fresh here for application+evaluation
# ============================================================
def actadd_hook_embed(value, hook, idx_diff, steering_vector, alpha):
    value[:, idx_diff, :] += alpha * steering_vector.unsqueeze(0).unsqueeze(0)
    return value


def actadd_hook_head(value, hook, idx_diff, head_idx, steering_vector, alpha):
    """Gendered token positions only (ATTN_STEER_SCOPE=gendered_positions).

    NOTE: a no-op for CLS-pooled models -- see the ATTN_STEER_SCOPE comment above for why.
    Meaningful for the mean-pooled models, where the edited positions contribute to the pool.
    """
    value[:, idx_diff, head_idx, :] += alpha * steering_vector.unsqueeze(0).unsqueeze(0)
    return value


def actadd_hook_head_all_positions(value, hook, head_idx, steering_vector, alpha):
    """Every token position, including CLS (ATTN_STEER_SCOPE=all_positions, the default) --
    matches the GREP-BiasIR reference's aggr_actadd_hook, which is why that work's attention
    steering moves relevance scores for CLS-pooled models where the narrowed hook above cannot.

    Only reached for documents that contain at least one gendered term (the document-level
    gate in compute_steered_doc_embedding), so untargeted documents are never perturbed --
    the same targeting the embedding level gets from its empty-idx_diff slice.
    """
    value[:, :, head_idx, :] += alpha * steering_vector.unsqueeze(0).unsqueeze(0)
    return value


# ============================================================
# data loading
# ============================================================
def load_construction_qids():
    """Queries that must never appear in an evaluation set, because the steering vectors being
    applied were built from them.

    Always includes this seed's construction_pairs.jsonl -- the authoritative record of what went
    into the vectors. Under DATASET=trecdl it additionally unions EVERY seed's
    preprocess/n5_seed*/construction_queries.tsv, matching compute_dense_baselines.py's TREC DL
    guard: 23 qids across the 5 seeds versus 5 for one seed, so it is strictly stricter and it is
    free. TREC DL is not any seed's test split, so unlike the MS MARCO case nothing about the
    split construction already guarantees disjointness -- the check has to be made explicitly.
    """
    qids = set()
    with open(CONSTRUCTION_PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            qids.add(json.loads(line)["query_id"])
    if DATASET == "trecdl":
        for path in sorted((HERE / "preprocess").glob("n5_seed*/construction_queries.tsv")):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    qid = line.split("\t")[0]
                    if qid.isdigit():
                        qids.add(qid)
    return qids


def load_test_queries(path, construction_qids, limit=None):
    """qid<TAB>text. A header row (test_queries.tsv's "query_id\\tquery_text") is skipped by the
    isdigit() test rather than by an unconditional next(f): the TREC DL queries file is headerless
    and its first line is a real query (573724), which a blind first-line skip silently dropped.
    Same approach as compute_dense_baselines.load_test_queries."""
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, text = line.rstrip("\n").split("\t")[:2]
            if not qid.isdigit():  # header row
                continue
            queries[qid] = text
    overlap = set(queries) & construction_qids
    assert not overlap, f"test queries overlap with construction queries: {overlap}"
    qids = sorted(queries, key=int)
    if limit:
        qids = qids[:limit]
    return {qid: queries[qid] for qid in qids}


def load_bm25_candidates(path, test_qids, cutoff=CANDIDATE_CUTOFF):
    """Returns {qid: [(docid, rank), ...]} sorted by rank, capped at cutoff, for qids in test_qids.
    Handles both a plain 6-column TREC run and test_candidates.tsv's 9-column annotated TSV
    (header row + extra has_gendered_terms/gendered_terms_found/gendered_term_positions columns)
    -- only the first 6 columns (qid, q0, docid, rank, score, tag) are used either way."""
    candidates = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 6 or parts[0] == "qid":  # header row (test_candidates.tsv)
                continue
            qid, _q0, docid, rank = parts[0], parts[1], parts[2], parts[3]
            if qid not in test_qids:
                continue
            rank = int(rank)
            if rank <= cutoff:
                candidates[qid].append((docid, rank))
    for qid in candidates:
        candidates[qid].sort(key=lambda x: x[1])
    return candidates


def fetch_doc_texts(collection_path, needed_docids):
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


def load_embed_vector(direction_dir):
    return torch.load(direction_dir / "vector.pt")


def load_attn_per_head_vectors(direction_dir):
    """Returns {(layer, head_idx): unit-normalized tensor(d_head)}, parsed from the
    "L{layer}H{head}" keys per_head_vectors.json was written with."""
    with open(direction_dir / "per_head_vectors.json", encoding="utf-8") as f:
        raw = json.load(f)
    heads = {}
    for key, vec in raw.items():
        m = re.match(r"L(\d+)H(\d+)", key)
        layer, head_idx = int(m.group(1)), int(m.group(2))
        heads[(layer, head_idx)] = torch.tensor(vec, dtype=torch.float32)
    return heads


# ============================================================
# scoring
# ============================================================
def compute_steered_doc_embedding(model, tl_model, input_ids, attention_mask, idx_diff, level, coef, steering_data):
    if level == "embed":
        hook_fn = partial(actadd_hook_embed, idx_diff=idx_diff, steering_vector=steering_data, alpha=coef)
        fwd_hooks = [("hook_full_embed", hook_fn)]
    else:
        # Document-level gate: a candidate with no gendered terms is not steered at all.
        # This is the same targeting the embedding level gets for free (its
        # value[:, [], :] slice is a no-op), made explicit here because the
        # all_positions hook would otherwise perturb every document in the ranked list.
        fwd_hooks = []
        if len(idx_diff) > 0:
            for (layer, head_idx), vec in steering_data.items():
                hook_name = utils.get_act_name("z", layer)
                if ATTN_STEER_SCOPE == "all_positions":
                    hook_fn = partial(actadd_hook_head_all_positions, head_idx=head_idx, steering_vector=vec, alpha=coef)
                else:
                    hook_fn = partial(actadd_hook_head, idx_diff=idx_diff, head_idx=head_idx, steering_vector=vec, alpha=coef)
                fwd_hooks.append((hook_name, hook_fn))

    with torch.no_grad():
        encoder_output = tl_model.run_with_hooks(input_ids, fwd_hooks=fwd_hooks, return_type="embeddings")

    pooled = model._pooling(encoder_output, attention_mask)
    if model._normalize:
        from mechir.util import normalize_outputs
        pooled = normalize_outputs(pooled)
    return pooled


def coef_folder(level, coef):
    prefix = "alpha" if level == "embed" else "beta"
    return f"{prefix}_{coef:.2f}"


def write_trec_run(path, run_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, docid, rank, score in run_rows:
            f.write(f"{qid} Q0 {docid} {rank} {score:.6f} steer\n")


EFFECTIVENESS_FIELDS = ["MRR@10", "nDCG@10", "Recall@10"]
_EFFECTIVENESS_LOGGED = False


def evaluate_effectiveness(run_path, test_qids):
    """{MRR@10, nDCG@10, Recall@10} via compute_baseline_metrics.compute_retrieval_effectiveness --
    the same function (so the same measures, the same min_rel semantics, and the same
    judged-queries-only denominator) that the unsteered baselines use, which is what makes an
    alpha=0 sweep row directly comparable to all_baselines_summary.tsv. Verified to return MRR@10
    bit-identical to the evaluate_mrr() this replaced, so existing sweep numbers are unchanged.

    MIN_REL applies to MRR and Recall only; nDCG@10 uses the raw qrel grades, which is what makes
    it meaningful on TREC DL's graded 0-3 judgements.

    That function prints a one-line query summary per call, and a sweep makes 3 directions x
    len(grid) x len(levels) calls (123 for a 41-point embed grid, 300 with the 57-point attn one),
    which would bury the sweep's own output. Only the first call's line is shown -- the counts are
    identical for every call in a run. stdout only; tqdm writes to stderr, so progress bars are
    unaffected.
    """
    global _EFFECTIVENESS_LOGGED
    sink = contextlib.nullcontext() if not _EFFECTIVENESS_LOGGED else contextlib.redirect_stdout(io.StringIO())
    with sink:
        aggregate, _ = compute_retrieval_effectiveness(run_path, QRELS_PATH, test_qids, min_rel=MIN_REL)
    _EFFECTIVENESS_LOGGED = True
    return aggregate


SWEEP_CSV_FIELDS = ["model", "level", "direction", "coefficient", *EFFECTIVENESS_FIELDS, "NFaiRR@10"]


def write_sweep_results(csv_path, new_rows, levels_written):
    """Merge new_rows into any existing sweep_results.csv rather than truncating it.

    sweep_results.csv lives at the model-directory root and is shared across levels, so an
    attn-only run into a directory that already holds an embed sweep would otherwise erase
    the embed rows (and vice versa). Rows whose level is in levels_written are replaced;
    every other level's rows are carried over. Written via temp file + rename so an
    interrupted write can't leave a half-file where results used to be.

    row.get(k, "") rather than row[k]: a carried-over row may predate a column later added to
    SWEEP_CSV_FIELDS (nDCG@10 and Recall@10 were), and a legacy row is worth keeping with blanks
    in the new columns rather than crashing the write of the rows that do have them.
    """
    kept = []
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if r.get("level") not in levels_written]
        if kept:
            print(f"preserving {len(kept)} existing row(s) from other level(s) in {csv_path.name}")

    tmp_path = csv_path.with_suffix(".csv.tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_CSV_FIELDS)
        writer.writeheader()
        for row in kept + new_rows:
            writer.writerow({k: row.get(k, "") for k in SWEEP_CSV_FIELDS})
    tmp_path.replace(csv_path)


def write_run_config(config_path, run_config, levels_written):
    """Same merge treatment as write_sweep_results, for the run config.

    Stays FLAT at the top level (this run's config verbatim) so existing readers keep
    working -- steering_dashboard.py's find_runs() reads cfg["test_queries_split"] and
    cfg["short_name"] directly. A "levels" sub-dict carries each level's own config, so an
    attn run into a directory that already holds an embed run doesn't lose what the embed
    sweep was run with. A pre-merge run_config.json (no "levels" key) is migrated first.
    """
    levels = {}
    if config_path.exists():
        try:
            prior = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior = {}
        levels = prior.get("levels") or {lvl: prior for lvl in prior.get("run_levels", [])}
    for level in levels_written:
        levels[level] = run_config

    merged = dict(run_config, levels=levels)
    tmp_path = config_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    tmp_path.replace(config_path)


def assert_steering_has_effect(model, tl_model, doc_cache, level, grid, steering_data):
    """Fail loudly if steering provably cannot move the output, instead of discovering it
    after a full sweep. Compares one gated document's pooled embedding at coefficient 0
    against the largest-magnitude coefficient in the grid.

    Skipped where a no-op is the EXPECTED result: ATTN_STEER_SCOPE=gendered_positions on a
    CLS-pooled model (see the ATTN_STEER_SCOPE comment for why that combination cannot
    reach the pool).
    """
    if level == "attn" and ATTN_STEER_SCOPE == "gendered_positions" and MODEL_KEY in CLS_POOLED_MODELS:
        print(f"  [{level}] effect check skipped: gendered_positions on a CLS-pooled model is an expected no-op")
        return

    probe = next(((ids, idx) for ids, idx in doc_cache.values() if len(idx) > 0), None)
    if probe is None:
        print(f"  [{level}] effect check skipped: no candidate document has a gendered token position")
        return
    input_ids, idx_diff = probe
    attention_mask = torch.ones_like(input_ids)
    extreme = max(grid, key=abs)
    if extreme == 0:
        print(f"  [{level}] effect check skipped: grid contains only 0")
        return

    base = compute_steered_doc_embedding(model, tl_model, input_ids, attention_mask, idx_diff, level, 0.0, steering_data)
    steered = compute_steered_doc_embedding(model, tl_model, input_ids, attention_mask, idx_diff, level, extreme, steering_data)
    max_abs_diff = (steered - base).abs().max().item()
    if max_abs_diff < 1e-6:
        raise RuntimeError(
            f"{level} steering is a no-op: pooled embedding is unchanged (max abs diff {max_abs_diff:.3e}) "
            f"between coefficient 0 and {extreme} on a document with {len(idx_diff)} gendered position(s). "
            f"ATTN_STEER_SCOPE={ATTN_STEER_SCOPE}, model={MODEL_KEY} ({MODEL_POOL}-pooled). "
            "Refusing to run a sweep that cannot produce an effect."
        )
    print(f"  [{level}] effect check passed: max abs diff {max_abs_diff:.3e} at coefficient {extreme}")


def check_grid_parity():
    """Warn if a trecdl sweep isn't on the same coefficient grid as its MS MARCO counterpart.

    TREC DL is a zero-shot transfer evaluation, so its curves are only worth much next to the
    MS MARCO ones at the same coefficients. The grid defaults (DATASET_CONFIGS[...]["alpha_grid"])
    already make that the case for a bare submission; this catches an ALPHA_*/BETA_* override that
    silently breaks the correspondence.

    A warning, not an assert: the MS MARCO counterpart directory need not exist on the machine
    running this (the cluster may only have the trecdl half), and a deliberately wider grid is a
    legitimate thing to want.
    """
    if MSMARCO_COUNTERPART_CONFIG is None or not MSMARCO_COUNTERPART_CONFIG.exists():
        return
    try:
        prior = json.loads(MSMARCO_COUNTERPART_CONFIG.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    levels = prior.get("levels") or {}
    here = {"embed": (ALPHA_MIN, ALPHA_MAX, ALPHA_STEP), "attn": (BETA_MIN, BETA_MAX, BETA_STEP)}
    keys = {"embed": ("alpha_min", "alpha_max", "alpha_step"), "attn": ("beta_min", "beta_max", "beta_step")}
    for level in RUN_LEVELS:
        cfg = levels.get(level) or prior
        theirs = tuple(cfg.get(k) for k in keys[level])
        if None in theirs:
            continue
        if tuple(float(v) for v in theirs) != here[level]:
            print(
                f"WARNING [{level}]: this TREC DL grid {here[level]} does NOT match the MS MARCO run at\n"
                f"         {MSMARCO_COUNTERPART_CONFIG} ({theirs}). The two sets of curves will not be\n"
                f"         point-for-point comparable. Drop the {'ALPHA' if level == 'embed' else 'BETA'}_* "
                f"overrides to use the matching default."
            )


def evaluate_nfairr(run_path, fairr_metric, test_qids_int):
    retrievalresults_all = FaiRRMetricHelper().read_retrievalresults_from_runfile(str(run_path), cut_off=CANDIDATE_CUTOFF)
    retrievalresults = {q: v for q, v in retrievalresults_all.items() if q in test_qids_int}
    result = compute_nfairr_full_collection_background(fairr_metric, retrievalresults, NFAIRR_THRESHOLDS)
    return result["metrics_avg"]["NFaiRR"][K]


# ============================================================
# main
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preferred_device = get_best_device()  # sets TRANSFORMERLENS_ALLOW_MPS=1 if MPS is the pick, before Dot() loads the model
    print(f"Model: {MODEL_NAME}  (short name: {SHORT_NAME})")
    print(f"Dataset: {DATASET} -- {CFG['query_set']}")
    print(f"  qrels: {QRELS_PATH.name}  (relevant at grade >= {MIN_REL})")
    if DATASET == "trecdl":
        print(f"Seed: {SEED}  (selects the MS MARCO-derived vectors in vector_construction{SEED_SUFFIX or ''}; "
              "the TREC DL query set is fixed and seed-independent)")
        if TEST_QUERIES_SPLIT != "full":
            print(f"NOTE: TEST_QUERIES_SPLIT={TEST_QUERIES_SPLIT!r} is ignored under DATASET=trecdl -- all 30 "
                  "TREC DL queries are evaluation data and there is no validation/held-out split. "
                  f"run_config.json records test_queries_split={CFG['split_label']!r}.")
    else:
        print(f"Seed: {SEED}  (n5_seed{SEED})")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Preferred device: {preferred_device}")
    print(f"Levels: {RUN_LEVELS}")
    if "embed" in RUN_LEVELS:
        print(f"  embed: alpha grid ({len(ALPHA_VALUES)} points, {ALPHA_MIN} to {ALPHA_MAX} step {ALPHA_STEP}): {ALPHA_VALUES}")
    if "attn" in RUN_LEVELS:
        print(f"  attn:  beta grid ({len(BETA_VALUES)} points, {BETA_MIN} to {BETA_MAX} step {BETA_STEP}): {BETA_VALUES}")
        print(f"  attn:  ATTN_STEER_SCOPE={ATTN_STEER_SCOPE}, steering only documents with >=1 gendered term")
        if ATTN_STEER_SCOPE == "gendered_positions" and MODEL_KEY in CLS_POOLED_MODELS:
            print(f"  attn:  WARNING -- gendered_positions on CLS-pooled {MODEL_KEY} is an architectural no-op")
    check_grid_parity()
    if N_TEST_QUERIES_SUBSET:
        print(f"N_TEST_QUERIES_SUBSET={N_TEST_QUERIES_SUBSET} -- SMOKE TEST MODE, not the full {DATASET} run")

    construction_qids = load_construction_qids()

    queries = load_test_queries(TEST_QUERIES_PATH, construction_qids, limit=N_TEST_QUERIES_SUBSET)
    test_qids = set(queries)
    test_qids_int = {int(q) for q in test_qids}
    print(f"{len(queries)} test queries loaded ({len(construction_qids)} construction queries confirmed disjoint)")

    candidates = load_bm25_candidates(BM25_RUN_PATH, test_qids)
    n_candidates = sum(len(v) for v in candidates.values())
    print(f"{n_candidates} (qid, docid) candidate pairs across {len(candidates)} queries (cutoff={CANDIDATE_CUTOFF})")
    if DATASET == "trecdl":
        # TREC DL has no split file guaranteeing every evaluation query has a pool, so check it.
        # NOT a "100 candidates each" check: qid 855410 legitimately has only 5.
        missing = sorted(test_qids - set(candidates), key=int)
        assert not missing, f"TREC DL queries with no BM25 candidates: {missing}"

    needed_docids = {docid for docs in candidates.values() for docid, _rank in docs}
    doc_texts = fetch_doc_texts(COLLECTION_PATH, needed_docids)

    wordlist = load_wordlist_terms(WORDLIST_PATH)
    print(f"{len(wordlist)} wordlist terms for gendered-position matching")

    print(f"loading model + tokenizer: {MODEL_NAME}")
    model = Dot(MODEL_NAME, pooling_type=MODEL_POOL, sim_func_type=MODEL_SIM_FUNC)
    tl_model = model._model
    tokenizer = model.tokenizer
    device = model._device
    print(f"resolved device: {device}")

    # deterministic per-doc preprocessing (tokenization + gendered-position lookup), done ONCE --
    # not a "cached baseline embedding" shortcut, since no model forward pass happens here.
    print("tokenizing candidate docs and finding gendered positions (once, reused across the whole sweep)...")
    doc_cache = {}
    for docid, text in tqdm(doc_texts.items(), desc="preprocessing docs"):
        encoded = tokenizer(text, add_special_tokens=True, return_tensors="pt")
        idx_diff = find_gendered_token_positions(text, tokenizer, wordlist)
        doc_cache[docid] = (encoded["input_ids"].to(device), idx_diff)
    n_with_gendered = sum(1 for _, idx in doc_cache.values() if idx)
    print(f"{n_with_gendered}/{len(doc_cache)} candidate docs have >=1 gendered token position")

    # unsteered query embeddings -- computed once, reused for every level/direction/coefficient
    print("computing unsteered query embeddings (once, queries are never steered)...")
    query_embeds = {}
    for qid, text in tqdm(queries.items(), desc="query embeddings"):
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            query_embeds[qid] = model.forward(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))

    print(f"loading v2 neutrality scores for NFaiRR: {NEUTRALITY_SCORES_V2_PATH}")
    fairr_metric = load_fairr_metric(NEUTRALITY_SCORES_V2_PATH)

    results = []
    for level in RUN_LEVELS:
        vector_dir = EMBED_VECTOR_DIR if level == "embed" else ATTN_VECTOR_DIR
        grid = ALPHA_VALUES if level == "embed" else BETA_VALUES

        for direction in DIRECTIONS:
            direction_dir = vector_dir / direction
            if level == "embed":
                steering_data = load_embed_vector(direction_dir).to(device)
            else:
                steering_data = {k: v.to(device) for k, v in load_attn_per_head_vectors(direction_dir).items()}

            if direction == DIRECTIONS[0]:
                assert_steering_has_effect(model, tl_model, doc_cache, level, grid, steering_data)

            for coef in tqdm(grid, desc=f"{level}/{direction} sweep"):
                run_rows = []
                for qid, doc_list in candidates.items():
                    q_emb = query_embeds[qid]
                    scored = []
                    for docid, _rank in doc_list:
                        input_ids, idx_diff = doc_cache[docid]
                        attention_mask = torch.ones_like(input_ids)
                        doc_emb = compute_steered_doc_embedding(
                            model, tl_model, input_ids, attention_mask, idx_diff, level, coef, steering_data
                        )
                        score = model._score_func(q_emb, doc_emb).item()
                        scored.append((docid, score))
                    scored.sort(key=lambda x: -x[1])
                    for rank, (docid, score) in enumerate(scored, start=1):
                        run_rows.append((qid, docid, rank, score))

                run_dir = OUTPUT_DIR / level / direction / coef_folder(level, coef)
                run_path = run_dir / "run.trec"
                write_trec_run(run_path, run_rows)

                effectiveness = evaluate_effectiveness(run_path, test_qids)
                nfairr = evaluate_nfairr(run_path, fairr_metric, test_qids_int)
                results.append({
                    "model": SHORT_NAME, "level": level, "direction": direction, "coefficient": coef,
                    **effectiveness, "NFaiRR@10": nfairr,
                })

    csv_path = OUTPUT_DIR / "sweep_results.csv"
    write_sweep_results(csv_path, results, set(RUN_LEVELS))
    print(f"\nwrote {csv_path}")

    run_config = {
        "model": MODEL_NAME, "short_name": SHORT_NAME, "run_levels": RUN_LEVELS, "seed": SEED,
        "dataset": DATASET, "query_set": CFG["query_set"],
        "qrels": str(QRELS_PATH), "min_rel": MIN_REL,
        "alpha_min": ALPHA_MIN, "alpha_max": ALPHA_MAX, "alpha_step": ALPHA_STEP,
        "beta_min": BETA_MIN, "beta_max": BETA_MAX, "beta_step": BETA_STEP,
        "attn_steer_scope": ATTN_STEER_SCOPE,
        "test_queries_split": CFG["split_label"],
        "n_test_queries": len(queries), "n_test_queries_subset_requested": N_TEST_QUERIES_SUBSET,
        "candidate_cutoff": CANDIDATE_CUTOFF,
    }
    write_run_config(OUTPUT_DIR / "run_config.json", run_config, set(RUN_LEVELS))

    results.sort(key=lambda r: (r["level"], r["direction"], r["coefficient"]))
    header = f"\n{'model':<35} {'level':<6} {'dir':<4} {'coef':>8}" + "".join(f"{m:>10}" for m in EFFECTIVENESS_FIELDS) + f"{'NFaiRR@10':>11}"
    print(header)
    for r in results:
        metrics = "".join(f"{r[m]:>10.4f}" for m in EFFECTIVENESS_FIELDS)
        print(f"{r['model']:<35} {r['level']:<6} {r['direction']:<4} {r['coefficient']:>8.2f}{metrics}{r['NFaiRR@10']:>11.4f}")


if __name__ == "__main__":
    main()

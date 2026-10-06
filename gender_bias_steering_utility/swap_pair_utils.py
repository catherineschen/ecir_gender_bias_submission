"""Shared utilities for building M/F/N swap-pair steering vectors from
construction_pairs.jsonl doc-pairs, used by both the embedding-level script
(steering_vectors_embed_msmarco_fair.py) and the attention-level script
(steering_vectors_attn_msmarco_fair.py), and for applying+evaluating them
(apply_steering_and_evaluate.py).
"""
import os

import numpy as np
import torch


def get_best_device():
    """cuda > mps > cpu. transformer_lens gates MPS behind TRANSFORMERLENS_ALLOW_MPS=1
    by default (past correctness issues with hook-based interventions on MPS -- see
    https://github.com/TransformerLensOrg/TransformerLens/issues/1178); this sets that
    env var when MPS is selected. Verified numerically before enabling this: a
    hook_full_embed actadd intervention on this repo's tas_b model produced max abs
    diff ~1e-6 between CPU and MPS (float32 noise) for both the unsteered and steered
    output, with the steering effect size itself matching to 6 decimal places."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
        return "mps"
    return "cpu"


# ============================================================
# copied as-is from submission_code_gender_bias_emnlp/steering_embed.py
# ============================================================
def get_gendered_token_indices(doc_m, doc_f, doc_n, tokenizer):
    """Find token positions that differ across the three document variants."""
    tok_m = np.array(tokenizer.encode(doc_m))
    tok_f = np.array(tokenizer.encode(doc_f))
    tok_n = np.array(tokenizer.encode(doc_n))

    if not (tok_m.shape == tok_f.shape == tok_n.shape):
        return None, None, None, None

    diff_mf = tok_m != tok_f
    diff_mn = tok_m != tok_n
    diff_fn = tok_f != tok_n
    idx_diff = np.where(diff_mf | diff_mn | diff_fn)[0]

    return tok_m, tok_f, tok_n, idx_diff


def pooled_direction_per_query(diffs_by_query):
    """Per-query nested averaging: each query's own mean diff vector (mean over that
    query's doc-pairs), then the pooled steering vector = mean of the per-query means
    -- equal weight per query, regardless of how many doc-pairs a query contributed.

    diffs_by_query: {query_id: array of shape (n_docs_for_query, dim)}, one entry per
    query that contributed at least one doc-pair (empty/zero-doc queries excluded by
    the caller before calling this).
    """
    per_query_means = np.stack([diffs.mean(axis=0) for diffs in diffs_by_query.values()])
    return per_query_means.mean(axis=0)

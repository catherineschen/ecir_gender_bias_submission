"""NFaiRR with the full MS MARCO collection as the background set, instead of a per-query pool.

Per Rekabsaz et al.'s own methodology, IFaiRR (NFaiRR's normalizing "ideal ranking" term)
should be computed from the full collection's neutrality distribution, not a per-query
top-k retrieved pool -- using a top-100 BM25 pool as background inflates NFaiRR, because
it excludes the mostly-neutral bulk of the collection from the "ideal" ranking.

Since the background set is now identical for every query, IFaiRR is computed once per
threshold (not once per query, and never as an 8.8M-doc dict repeated per query) and that
same value is reused for every query's NFaiRR.

This is the default NFaiRR path for the BM25 baseline, the 5 dense-model baselines, and
every steered condition -- import compute_nfairr_full_collection_background from here
rather than reimplementing the full-collection IFaiRR logic in each evaluation script.
"""
import heapq
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FAIRNESS_CODE_DIR = REPO_ROOT / "data/FairnessRetrievalResults/adversarial_mitigation/fairness_measurement"
sys.path.insert(0, str(FAIRNESS_CODE_DIR))
from metrics_fairness import FaiRRMetric, FaiRRMetricHelper  # noqa: E402  (unmodified; only FaiRR/the numerator is reused from it)

DEFAULT_THRESHOLDS = [5, 10, 20, 50]


def load_fairr_metric(collection_neutrality_path):
    """Reuses FaiRRMetric's own file-parsing as-is. This FaiRRMetric version doesn't precompute
    anything from background_doc_set in __init__ (that happens lazily inside
    calc_FaiRR_retrievalresults), so an empty dict here is harmless -- we never call that
    method's own per-query IFaiRR path, only its FaiRR/numerator computation (see below)."""
    return FaiRRMetric(str(collection_neutrality_path), background_doc_set={})


def compute_ideal_fairr_full_collection(documents_neutrality, thresholds=DEFAULT_THRESHOLDS):
    """The one-time IFaiRR@threshold values: the neutrality-descending top-`threshold` docs from
    the whole collection, discounted by the same rank-based position bias FaiRR itself uses."""
    max_threshold = max(thresholds)
    top_neutrality = heapq.nlargest(max_threshold, documents_neutrality.values())  # avoids sorting all ~8.8M scores
    position_biases = [1 / math.log2(rank + 1) for rank in range(1, max_threshold + 1)]
    return {th: sum(n * b for n, b in zip(top_neutrality[:th], position_biases[:th])) for th in thresholds}


def compute_nfairr_full_collection_background(fairr_metric, retrievalresults, thresholds=DEFAULT_THRESHOLDS):
    """FaiRR (the numerator) is computed by calling FaiRRMetric.calc_FaiRR_retrievalresults()
    unmodified -- only NFaiRR's denominator (IFaiRR) is replaced, with the single full-collection
    value for each threshold standing in for what was previously a per-query background lookup.

    calc_FaiRR_retrievalresults() also computes its own (per-query-background) NFaiRR internally
    from fairr_metric.background_doc_set, using the same qryids as retrievalresults, or else it
    prints an "ERROR: query id ... does not exist in background document set" per query and hits
    a mean-of-empty-slice warning. That internal NFaiRR is discarded below in favor of the
    full-collection one, but populating background_doc_set (any non-empty per-qryid mapping does,
    so retrievalresults itself is reused) avoids those spurious prints/warnings for no extra cost --
    this only sets the instance attribute the class's own constructor would otherwise have set."""
    fairr_metric.background_doc_set = retrievalresults
    result = fairr_metric.calc_FaiRR_retrievalresults(retrievalresults, thresholds)  # FaiRR / FaiRR_perq: untouched
    ideal_fairr = compute_ideal_fairr_full_collection(fairr_metric.documents_neutrality, thresholds)

    nfairr_perq, nfairr_avg = {}, {}
    for th in thresholds:
        nfairr_perq[th] = {qid: fairr / ideal_fairr[th] for qid, fairr in result["metrics_perq"]["FaiRR"][th].items()}
        nfairr_avg[th] = sum(nfairr_perq[th].values()) / len(nfairr_perq[th])

    return {
        "metrics_avg": {"FaiRR": result["metrics_avg"]["FaiRR"], "NFaiRR": nfairr_avg},
        "metrics_perq": {"FaiRR": result["metrics_perq"]["FaiRR"], "NFaiRR": nfairr_perq},
        "ideal_fairr": ideal_fairr,  # single value per threshold now, not per-query -- exposed for logging/sanity checks
    }

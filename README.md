# Gender Sensitivity in Dense Retrieval: Mechanisms, Steering, and the Utility–Bias Tradeoff

Code and small result files to reproduce the paper's figures and tables. This repository
combines two experiment clusters:

- **Circuit analysis + pointwise steering** (`src/circuit_analysis/`, `src/pointwise_steering/`,
  parts of `figures/`) — mechanistic analysis of gender sensitivity in five dot-product
  bi-encoders on GREP-BiasIR, and pointwise ActAdd steering at the embedding and attention
  levels.
- **Utility–bias tradeoff** (`gender_bias_steering_utility/`) — extending the steering
  intervention to full retrieval on MS MARCO Fair and TREC DL19 Fair, measuring the
  utility/fairness tradeoff (MRR@10, nDCG@10, NFaiRR@10) across 5 random seeds.

## Setup

```bash
conda create -n gender_bias python=3.11
conda activate gender_bias
# Install a CUDA-matched torch for your machine first (or the CPU/MPS build if running locally):
#   https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

`mechir` (used for activation/path patching and ablation) is vendored in this repo at
`third_party/mechir/` rather than installed from PyPI, because the public release does not
yet include path-patching support. Add it to your `PYTHONPATH` before running anything under
`src/circuit_analysis/`, `src/pointwise_steering/`, or `gender_bias_steering_utility/`:

```bash
export PYTHONPATH="$(pwd)/third_party/mechir:$PYTHONPATH"
```

All commands below assume the repo root as the working directory.

## Data

Already included (small files):
- `data/grep_bias_ir_mechir_format.csv` — cleaned GREP-BiasIR dataset (see Appendix A below).
- `data/FairnessRetrievalResults/`, `data/GenderBias_IR/` — the specific small third-party
  files (wordlists, sample TSVs, the `DocumentNeutrality`/`FaiRRMetric` implementations) from
  Rekabsaz & Schedl's released code that our scripts import. Licenses are included; see those
  directories' `LICENSE` files for citation.
- `data/msmarco/` — small qrels files only.
- `gender_bias_steering_utility/{data,output,preprocess,experiments}/` — wordlists, BM25
  candidate runs, query splits, and small per-model sweep result files for all 5 seeds.
- `third_party/mechir/` — vendored subset of the MechIR library (Apache 2.0; see `NOTICE.md`).

**Not included** (large and/or regenerable — see `DOWNLOAD.md` in `data/` for how to get them):
- The full MS MARCO passage collection (`collection.tsv`, ~2.9GB).
- The PyTerrier BM25 index.
- Full-collection neutrality scores (`collection_neutralityscores*.tsv`, ~142MB each) —
  regenerate with `gender_bias_steering_utility/compute_baseline_v2.py`.
- Per-coefficient TREC run files under each model's `attn/`/`embed/` sweep directories, and
  `score_deltas.csv`/`per_query_deltas.csv` — regenerate with
  `gender_bias_steering_utility/apply_steering_and_evaluate.py`.

## Reproducing each figure/table

| Paper artifact | Command |
|---|---|
| Fig. 1 (activation patching, 2 example models) | `python figures/plot_activation_patching_paper_figure.py --cls-csv <patch_effect_df.csv for a CLS-pooled model> --mean-csv <... for a mean-pooled model> --save-dir out/ --versions a --orientations horizontal --components resid attn_out` (run `python src/circuit_analysis/experiment_patching.py --patch_type activation ...` per model first — see note below) |
| Fig. 1 (all 5 models, appendix) | same, looped over all 5 models via `src/circuit_analysis/experiment_patching.py --patch_type activation` |
| Fig. 2 (path patching) | `python src/circuit_analysis/experiment_patching.py --patch_type path --sender_attn_component z --receiver_attn_component v --direct_includes_mlps ...` |
| Table 2 (ablation score diffs) | `python src/circuit_analysis/compute_head_means.py --save_dir head_means/` → `python src/circuit_analysis/baseline.py --model_type bi --batch_size 32 --result_dir results_baseline/` → `python src/circuit_analysis/ablation_experiments.py --model_type bi --ablation_type mean --batch_size 32 --save_dir results_ablation/ --means_dir head_means/` → `python figures/plot_ablation_results.py --orig_csv results_baseline/... --abl_csv results_ablation/mean/...` |
| Sec. 4.2 (w/ vs. w/o neutral term) | same `plot_ablation_results.py` call, with/without `--filter_out_neutral` |
| Fig. 3 (attention by token category) | `python src/circuit_analysis/save_attention_patterns.py --model_type bi --save_dir dump/` → `python src/circuit_analysis/analyze_attention_patterns.py analyze -dd dump/` → `python figures/plot_attention_paper_fig.py --attn_results_dir dump/ --out_path fig3.pdf` |
| Fig. 4 (steering scatter) | `python src/pointwise_steering/steering_embed.py` and `steering_attn.py` (writes `results_steering_*_with_scores/`), then `python figures/plot_steering_scatter.py` |
| Fig. 5 (preference rate vs. α) | `python figures/make_paper_line_plots.py` (and `make_paper_stacked_bars.py` for the stacked-bar variant) |
| Fig. 6 (utility vs. NFaiRR@10 frontier, both datasets, 5-seed average) | `python gender_bias_steering_utility/camera_ready_figures.py --out-dir paper_figures/` (reads the 5 seed-suffixed `experiments/steering_heldout_final*` directories already in this repo) |
| Utility–bias tables/ARaB curves | same command; also see `gender_bias_steering_utility/compute_arab_from_runs.py` |


To rebuild the utility–bias pipeline from scratch (query splits, construction pairs, steering
vectors, sweep): see `slurm/run_seed_construction.sh` and `slurm/run_steering_sweep.sh`, which
document the full per-seed, per-model pipeline and are directly submittable on a SLURM cluster
after editing the `<<< EDIT` placeholders (conda env path, module name, email, HF cache
location).

## Methodology details

The sections below are adapted from the appendix of our companion paper on the circuit-analysis
and pointwise-steering results (the mechanistic findings this repository's
`src/circuit_analysis/` and `src/pointwise_steering/` code implements).

### A. GREP-BiasIR dataset details

The GREP-BiasIR dataset contains 708 documents total — 236 for each gendered variant (female,
male, neutral) — corresponding to 118 gender-neutral web search queries. Each document variant
contains identical content, differing only in gendered terms.

Circuit analysis requires aligned token positions across document variants, so we applied
several preprocessing steps to the raw dataset before producing
`data/grep_bias_ir_mechir_format.csv`

### B. Circuit analysis implementation details

We implement our experiments using the MechIR framework (vendored at `third_party/mechir/`),
adding modifications to support path patching. All code can be run on a single GPU.

**Patched components.** For activation patching, we patch two components at each layer: the
residual stream prior to the attention block, and the full attention block output. We patch at
individual token positions and average effects within each token category (CLS, gendered
tokens, query-matching tokens, all other document tokens, SEP).

For path patching, we begin by patching upstream attention head outputs to the final-layer
residual stream (post-transformer-block, pre-pooling) to identify the heads that most directly
affect the final relevance score. We then iteratively backtrack, patching upstream attention
head outputs to the value vectors of previously identified heads, to trace which earlier heads
feed into them and reconstruct the pathway carrying the signal. In all path-patching
experiments, patches are applied across all token positions simultaneously, with MLPs
recomputed.

**Patching metric.** To evaluate whether a patched component is important in the circuit, we use
the normalized performance-recovery metric of Polyakov et al. (2025), which measures the shift
in relevance score toward the contrastive input while down-weighting pairs with small baseline
differences. Let `s(X_i)` and `s(X_j)` denote the model's output score on the low-scoring and
high-scoring variants respectively, and `s(X_i^patched)` the score on `X_i` after patching in a
cached activation from `X_j`. The patching effect (PE) is:

```
PE = ( s(X_i^patched) - s(X_i) ) / sqrt( 1 + (s(X_j) - s(X_i))^2 )
```

0 indicates no effect; 1 indicates the patch fully accounts for the difference between `X_i`
and `X_j`.

### C. Baseline score differences

To verify that score differences exist between document variants, we plot, for each document,
the signed score differences `Δ_f-n = s_f - s_n` and `Δ_m-n = s_m - s_n` — points at the origin
indicate no effect of gender substitution. Three patterns are consistent across models: (1) a
cluster around the origin, where many documents score similarly across variants; (2) a skew
toward the lower-left quadrant, where gendered variants tend to score lower than neutral ones;
(3) `Δ_f-n` and `Δ_m-n` are correlated (0.83–0.89 across models) but not identical, with most
points above the `y=x` diagonal, indicating male variants tend to score higher than female
variants.

### D/E. Full activation and path patching results

We consider a component to be in the computational path if its patching effect is at least two
standard deviations from the mean effect across all heads, for a given gender comparison.
Patching to the final residual stream and to upstream attention heads was run for all 5 models;
the main text reports one representative model per pooling architecture (CLS-pooled vs.
mean-pooled), with the remaining models' results reproducible via the same commands above
looped over all 5 model names.

### F. Full ablation results

To confirm the causal role of the heads identified by path patching, we mean-ablate each
identified head and measure the resulting reduction in score differences between contrastive
pairs: the head's output is replaced with its mean activation, computed per head across all
token positions and all documents in the dataset. For the MvN and FvN comparisons, ablation
does not reduce score differences for the pairs with the largest score differences; these pairs
correspond to documents where the neutral term also appears in the query (see
`--filter_out_neutral` in `figures/plot_ablation_results.py`), after which the effect of
ablation is more uniform across the remaining samples.

### G. Full steering results

We compute steering vectors for all three pairwise gender comparisons (MvF, MvN, FvN). For each
comparison, the steering vector is the mean difference in activations between the two variants,
computed over the relevant document pairs. At the embedding level, steering is applied at the
token level, modifying only the gendered token's embedding. At the attention level, steering is
applied across all token positions at the identified final-layer attention heads.

In addition to score-difference visualizations (Fig. 4), we report **preference rates** — the
proportion of document pairs in which one variant scores higher than the other, across the
swept steering coefficients (Fig. 5) — as a complementary view of whether steering
systematically "flips" the preference between variants.

## Repository layout

```
data/                        cleaned GREP-BiasIR, small third-party data, external-data download notes
gender_bias_steering_utility/ utility-bias pipeline: annotation, query splits, steering vectors,
                              sweep, metrics, small per-seed/per-model results
src/circuit_analysis/        activation/path patching, ablation, attention-pattern analysis
src/pointwise_steering/      ActAdd steering vector construction (embedding + attention level)
figures/                     plotting scripts for every figure/table above
results/pointwise_steering/  small steering result files (JSON + small tensors) Figs. 4-5 read
slurm/                       SLURM submission scripts for the utility-bias sweep and seed construction
third_party/mechir/          vendored MechIR subset (Apache 2.0) with path-patching support
```

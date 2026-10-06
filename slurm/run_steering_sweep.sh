#!/bin/bash
#SBATCH --job-name=steering_test
#SBATCH --partition=gpu                    
#SBATCH --gres=gpu:1                       
#SBATCH --mem=32G
#SBATCH --time=12:00:00                    
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=your-email@example.com    # <<< EDIT: your address

# ============================================================
# Runs apply_steering_and_evaluate.py for one model per array task, each on its own
# GPU -- so all 5 models' sweeps run in parallel instead of sharing one GPU sequentially
# (which is what forced sequential/contended runs on a single-GPU machine).
#
# PREREQUISITES (none of these are created by this script):
#   1. This repo checked out on the cluster, with the same relative layout
#      (data/, gender_bias_steering_utility/) -- REPO_ROOT below assumes this.
#   2. The conda env with torch/transformer_lens/ir_measures/etc. installed per the repo
#      README -- install a CUDA-matched torch for your cluster's GPUs first, then the
#      rest of requirements.txt. mechir is vendored in this repo (third_party/mechir/,
#      not on PyPI) -- add it to PYTHONPATH (e.g. `export PYTHONPATH="$REPO_ROOT/third_party/mechir:$PYTHONPATH"`
#      below) rather than pip-installing it.
#   3. data/msmarco/collection.tsv, data/FairnessRetrievalResults/, and this repo's
#      gender_bias_steering_utility/experiments/vector_construction/ + experiments/baseline/
#      synced over (e.g. rsync/scp) -- these are large (collection.tsv is ~3GB) and
#      are inputs to this stage, not something it generates.
#   4. HF model weights reachable. Many SLURM compute nodes have no internet access.
#      If that's the case here, pre-download the 5 models on a LOGIN node first (one
#      AutoModel.from_pretrained call per model name is enough to populate the cache),
#      pointing HF_HOME at a location visible from compute nodes too, then uncomment
#      HF_HUB_OFFLINE=1 below so a compute node never tries to reach the network.
#
# USAGE -- submit with the sweep stage's parameters passed through via --export, e.g.:
#
#   # coarse validation pass: wide range, small query subset, to find the useful range
#   sbatch --export=ALL,TEST_QUERIES_SPLIT=validation,ALPHA_MIN=-50,ALPHA_MAX=50,ALPHA_STEP=5,OUTPUT_DIR_NAME=steering_validation_coarse \
#       run_steering_sweep.sh
#
#   # fine validation pass: range narrowed after inspecting the coarse pass's results
#   sbatch --export=ALL,TEST_QUERIES_SPLIT=validation,ALPHA_MIN=-10,ALPHA_MAX=10,ALPHA_STEP=0.5,OUTPUT_DIR_NAME=steering_validation_fine \
#       run_steering_sweep.sh
#
#   # final held-out pass: same range as the fine pass, held-out queries -- run ONCE,
#   # this is the number that gets reported. Never used for range/coefficient selection.
#   sbatch --export=ALL,TEST_QUERIES_SPLIT=heldout,ALPHA_MIN=-10,ALPHA_MAX=10,ALPHA_STEP=0.5,OUTPUT_DIR_NAME=steering_heldout_final \
#       run_steering_sweep.sh
#
# To run just one model instead of all 5, submit with e.g. --array=0 (tas_b only) --
# see MODELS below for the index mapping. Add --array=0-4%2 to cap concurrent array
# tasks at 2 if your allocation doesn't cover 5 simultaneous GPUs.
#
# ============================================================
# ATTENTION LEVEL
# ============================================================
# The examples above sweep the embedding level (RUN_LEVELS defaults to embed), which is
# what alpha controls. The attention level is selected with RUN_LEVELS=attn and swept over
# its OWN coefficient grid, BETA_MIN/BETA_MAX/BETA_STEP -- the two levels' vectors are on
# different scales (embedding vectors are raw mean-diffs with norm ~9-16, the per-head
# attention vectors are unit-normalized), so reusing alpha's range would land the
# attention sweep in the wrong part of the range. Run its own coarse pass first:
#
#   # coarse validation, attention level
#   sbatch --export=ALL,RUN_LEVELS=attn,TEST_QUERIES_SPLIT=validation,BETA_MIN=-20,BETA_MAX=20,BETA_STEP=2,OUTPUT_DIR_NAME=steering_validation_coarse \
#       run_steering_sweep.sh
#
#   # fine validation, attention level (range narrowed from the coarse pass)
#   sbatch --export=ALL,RUN_LEVELS=attn,TEST_QUERIES_SPLIT=validation,BETA_MIN=-5,BETA_MAX=5,BETA_STEP=0.25,OUTPUT_DIR_NAME=steering_validation_fine \
#       run_steering_sweep.sh
#
#   # final held-out pass, attention level -- run ONCE
#   sbatch --export=ALL,RUN_LEVELS=attn,TEST_QUERIES_SPLIT=heldout,BETA_MIN=-14,BETA_MAX=14,BETA_STEP=0.5,OUTPUT_DIR_NAME=steering_heldout_final \
#       run_steering_sweep.sh
#
# Attention runs can share an OUTPUT_DIR_NAME with the embedding runs: run files nest under
# <model>/attn/ vs <model>/embed/, and sweep_results.csv / run_config.json are merged rather
# than overwritten, so an attn run preserves the embed rows already there. RUN_LEVELS=embed,attn
# runs both levels in one job (each on its own grid) if you'd rather not run them separately.
#
# ATTN_STEER_SCOPE (default all_positions) controls which token positions inside a steered
# document the z-hook edits. Only documents containing >=1 gendered term are steered either
# way. ATTN_STEER_SCOPE=gendered_positions narrows it further, but is an architectural no-op
# for the three CLS-pooled models -- apply_steering_and_evaluate.py refuses to run a sweep
# that provably cannot have an effect, so that combination is for comparison only.
#
# ============================================================
# OTHER SEEDS
# ============================================================
# Everything above (and the USAGE examples) is unchanged and still targets seed 42 by
# default -- SEED defaults to 42 below, so submitting exactly as documented above still
# reproduces the existing seed-42 results byte-for-byte.
#
# To evaluate a different random seed, two things must happen, in order:
#
#   1. ONE-TIME PER SEED, not per model: build that seed's own query split, construction
#      pairs, and steering vectors first -- run_seed_construction.sh does this (submit it
#      once per new seed; it does not need --array, it's a single task):
#          sbatch --job-name=seed_construction_123 --export=ALL,SEED=123 run_seed_construction.sh
#      Wait for it to finish before step 2 -- apply_steering_and_evaluate.py will fail to
#      find that seed's vectors otherwise.
#
#   2. Then submit this script's normal coarse/fine/held-out sweep stages (see USAGE above)
#      with SEED added to --export, e.g.:
#          sbatch --job-name=steering_seed123 --export=ALL,SEED=123,TEST_QUERIES_SPLIT=validation,ALPHA_MIN=-50,ALPHA_MAX=50,ALPHA_STEP=5,OUTPUT_DIR_NAME=steering_validation_coarse \
#              run_steering_sweep.sh
#
# The #SBATCH --job-name line below can't reference $SEED itself (SBATCH directives are
# parsed literally, before the script runs as bash, so shell variables never expand in
# them) -- --job-name=... on the sbatch command line overrides it instead, as above. Doing
# so also makes --output/--error's %x expand to the seed-specific name, so job logs land
# in distinctly-named files per seed too, not just distinguishable squeue entries.
#
# Every seed's outputs land in their own sibling directories (preprocess/n5_seed123/,
# experiments/vector_construction_seed123/, experiments/steering_validation_coarse_seed123/,
# ...) -- nothing about seed 42's files is ever read from or written to when SEED != 42.
#
# ============================================================
# TREC DL 2019 (ZERO-SHOT EVALUATION)
# ============================================================
# DATASET=trecdl evaluates the SAME MS MARCO-derived steering vectors on TREC DL 2019 Fair, as a
# zero-shot transfer check. Nothing is constructed from TREC DL:
#
#   * The vectors still come from experiments/vector_construction{_seedN}/, built from the MS
#     MARCO construction queries. SEED still picks which ones -- it does NOT pick the query set,
#     which is fixed at all 30 TREC DL queries.
#   * ALL 30 queries are evaluation data. There is no validation/held-out split, and no
#     coefficient is ever selected on TREC DL. TEST_QUERIES_SPLIT is ignored (with a printed
#     note); run_config.json records test_queries_split=trecdl19_all.
#   * The coefficient grid DEFAULTS to the one the MS MARCO held-out sweep used (embed
#     -10..10 step 0.5, attn -14..14 step 0.5), so the curves are point-for-point comparable
#     without passing ALPHA_*/BETA_* at all. Overriding them prints a mismatch warning.
#   * Metrics use the TREC DL graded qrels with grade >= 2 counting as relevant for MRR@10 and
#     Recall@10; nDCG@10 uses the raw grades.
#
#   # embedding level, zero-shot -- note NO ALPHA_* is passed
#   sbatch --job-name=steering_trecdl19 \
#       --export=ALL,DATASET=trecdl,OUTPUT_DIR_NAME=steering_heldout_final \
#       run_steering_sweep.sh
#
#   # attention level, zero-shot
#   sbatch --job-name=steering_trecdl19_attn \
#       --export=ALL,DATASET=trecdl,RUN_LEVELS=attn,OUTPUT_DIR_NAME=steering_heldout_final \
#       run_steering_sweep.sh
#
# OUTPUT_DIR_NAME need NOT mention trecdl: the _trecdl19 suffix is added by the script, before any
# _seedN suffix, so the above land in experiments/steering_heldout_final_trecdl19/<model>/ and can
# never overwrite the MS MARCO rows in experiments/steering_heldout_final/<model>/. Reusing the MS
# MARCO run's OUTPUT_DIR_NAME as above is in fact what lets the script find its counterpart and
# check the two grids match.
# ============================================================

set -euo pipefail

# ---- cluster environment: EDIT for your site ----
# module load <your site's conda/mamba module>              # <<< EDIT: e.g. `module load miniforge3`
# source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh         # <<< EDIT: only needed if the module above sets this var
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /path/to/your/conda/envs/<env_name>           # <<< EDIT: path to the env set up per the repo README
# export HF_HOME=/path/to/shared/hf_cache  # <<< EDIT: see prerequisite 4 above
# export HF_HUB_OFFLINE=1                  # <<< uncomment once the HF cache is pre-populated

# ---- repo location: EDIT to where you checked this repo out on the cluster ----
# REPO_ROOT="${REPO_ROOT:-$HOME/git}"
# cd "$REPO_ROOT/gender_bias_steering_utility"
# mkdir -p logs
export PYTHONPATH="${REPO_ROOT:-$HOME/git}/third_party/mechir:${PYTHONPATH:-}"  # <<< EDIT if REPO_ROOT above is customized

# ---- model selection: one array task per model (index matches --array above) ----
MODELS=(
    tas_b
    multi-qa-distilbert
    multi-qa-minilm
    msmarco-bert-base
    msmarco-distilbert
    )
export MODEL_KEY="${MODELS[$SLURM_ARRAY_TASK_ID]}"

# ---- sweep parameters: override at submit time via --export (see USAGE above).
# These are just fallback defaults if you submit without --export. ----
export SEED="${SEED:-42}"                # <<< see "OTHER SEEDS" above -- run_seed_construction.sh first for SEED != 42
export DATASET="${DATASET:-msmarco}"     # <<< msmarco | trecdl -- see "TREC DL 2019" above
export RUN_LEVELS="${RUN_LEVELS:-embed}"
export ATTN_STEER_SCOPE="${ATTN_STEER_SCOPE:-all_positions}"
export OUTPUT_DIR_NAME="${OUTPUT_DIR_NAME:-steering}"
# export N_TEST_QUERIES_SUBSET=3           # <<< uncomment for a quick smoke test before a real submission

# The split and the grid fallbacks below are MS MARCO-only, and are deliberately NOT exported
# under DATASET=trecdl. TEST_QUERIES_SPLIT has no meaning there, and -- more importantly -- an
# exported ALPHA_*/BETA_* would override the dataset's own defaults inside
# apply_steering_and_evaluate.py, which for trecdl are precisely the MS MARCO held-out grid the
# zero-shot curves have to match. Passing ALPHA_*/BETA_* explicitly via --export still works for
# either dataset (trecdl then warns that the grids no longer correspond).
if [ "$DATASET" = "msmarco" ]; then
    export TEST_QUERIES_SPLIT="${TEST_QUERIES_SPLIT:-validation}"
    export ALPHA_MIN="${ALPHA_MIN:--50}"     # embedding level
    export ALPHA_MAX="${ALPHA_MAX:-50}"
    export ALPHA_STEP="${ALPHA_STEP:-5}"
    export BETA_MIN="${BETA_MIN:--20}"       # attention level -- separate grid, see "ATTENTION LEVEL" above
    export BETA_MAX="${BETA_MAX:-20}"
    export BETA_STEP="${BETA_STEP:-2}"
fi

echo "=== job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) ==="
echo "model: $MODEL_KEY  seed: $SEED  dataset: $DATASET"
echo "split: ${TEST_QUERIES_SPLIT:-n/a (trecdl: all queries)}  levels: $RUN_LEVELS"
echo "alpha: [${ALPHA_MIN:-dataset default}, ${ALPHA_MAX:-dataset default}] step ${ALPHA_STEP:-dataset default}"
echo "beta:  [${BETA_MIN:-dataset default}, ${BETA_MAX:-dataset default}] step ${BETA_STEP:-dataset default}"
echo "output dir name: $OUTPUT_DIR_NAME"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null || echo "nvidia-smi not available"
python -c "import torch; print('cuda available:', torch.cuda.is_available(), '| device count:', torch.cuda.device_count())"

python apply_steering_and_evaluate.py

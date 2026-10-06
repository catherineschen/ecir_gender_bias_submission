#!/bin/bash
#SBATCH --job-name=seed_construction
#SBATCH --partition=batch                  # <<< EDIT: your site's CPU (non-GPU) partition name -- no GPU needed, see note below
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=your-email@example.com    # <<< EDIT: your address


# ============================================================
# ONE-TIME PER SEED (not per model, no array): builds a new random seed's construction/
# test query split, the M/F/N construction pairs, and both levels' steering vectors --
# everything run_steering_sweep.sh's sweep stage needs to exist before it can evaluate
# that seed. Companion to run_steering_sweep.sh -- see its "OTHER SEEDS" section.
#
# No GPU requested -- steps 1-3 (query split, validation/held-out split, construction
# pairs) are pure CPU work (file I/O, spaCy POS tagging, tokenization only, no model
# forward pass). Step 4 (steering_vectors_embed_msmarco_fair.py) does run a model, but
# over a tiny amount of data (~500 docs x 5 models, one time), so CPU is fine and this
# doesn't need to compete with the sweep jobs for GPU allocation.
#
# RE-RUNNABLE: every step is skipped when its output is already complete, so this is safe
# to run against a seed that is only partly built -- which is the normal case for a seed
# whose embedding vectors were built before the attention step existed. Running it then
# builds ONLY the missing attention vectors and leaves the split, the construction pairs
# and the embedding vectors exactly as they were. The per-step skip conditions are at the
# bottom of this script; FORCE_REBUILD=1 overrides them for every step except the
# validation/held-out split, which is never regenerated (see the note there).
#
# Seed 42's own construction/vectors already exist (built before this script existed) and
# are not rebuilt by a normal run.
#
# PREREQUISITES: same as run_steering_sweep.sh's (repo synced, conda env, HF weights
# reachable) plus preprocess/annotated_candidates.tsv, preprocess/query_stats.tsv,
# preprocess/annotation_summary.json (annotate_gendered_candidates.py's outputs) synced
# over too -- split_queries_dedupe.py reads those, not raw collection data.
#
# USAGE:
#   sbatch --job-name=seed_construction_123 --export=ALL,SEED=123 run_seed_construction.sh
#   # then, once this completes: sbatch --job-name=steering_seed123 --export=ALL,SEED=123,... run_steering_sweep.sh
#
# --job-name on the sbatch command line (not the #SBATCH line below, which can't
# reference $SEED -- see run_steering_sweep.sh's "OTHER SEEDS" section) also makes
# --output/--error's %x expand to it, so each seed's logs land in distinctly-named files.
# ============================================================

set -euo pipefail

# ---- cluster environment: EDIT for your site (kept identical to run_steering_sweep.sh) ----
# module load <your site's conda/mamba module>              # <<< EDIT: e.g. `module load miniforge3`
# source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh         # <<< EDIT: only needed if the module above sets this var
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /path/to/your/conda/envs/<env_name>           # <<< EDIT: path to the env set up per the repo README
# export HF_HOME=/path/to/shared/hf_cache  # <<< EDIT: see run_steering_sweep.sh's prerequisite 4
# export HF_HUB_OFFLINE=1                  # <<< uncomment once the HF cache is pre-populated

# ---- repo location: EDIT to where you checked this repo out on the cluster ----
# REPO_ROOT="${REPO_ROOT:-$HOME/git}"
# cd "$REPO_ROOT/gender_bias_steering_utility"
# mkdir -p logs
export PYTHONPATH="${REPO_ROOT:-$HOME/git}/third_party/mechir:${PYTHONPATH:-}"  # <<< EDIT if REPO_ROOT above is customized

export SEED="${SEED:?SEED must be set, e.g. --export=ALL,SEED=123}"

# ---- re-runnable: each step is skipped if its output is already complete ----
# This matters for a seed that is PARTLY built -- e.g. the four seeds whose embedding
# vectors were built before the attention step existed. Without these guards, re-running
# to pick up the attention step dies at step 2: split_validation_heldout.py refuses by
# design to overwrite an existing validation/held-out split, and `set -e` turns that into
# an aborted job that never reaches step 4b.
#
# FORCE_REBUILD=1 re-runs steps 1, 3, 4 and 4b even when their outputs exist. It does NOT
# apply to step 2 -- that split is what every sweep for this seed was evaluated against,
# so regenerating it would silently invalidate those results. Delete the two split files
# by hand if you really mean to replace it.
# Locating the repo: under sbatch, SLURM copies this script to a spool directory on the
# compute node, so $BASH_SOURCE points at /var/spool/slurmd/job<ID>/slurm_script and NOT at
# the repo. SLURM_SUBMIT_DIR is the directory sbatch was invoked from, which is where the
# python calls below already resolve their relative paths from -- so it's the same
# directory either way, and the fallback covers running this script directly with bash.
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [ ! -f "$HERE/split_queries_dedupe.py" ]; then
    echo "ERROR: expected the repo scripts in '$HERE' but split_queries_dedupe.py is not there." >&2
    echo "       Submit from inside gender_bias_steering_utility/ (the python calls below are" >&2
    echo "       relative paths and need that as the working directory), or set SLURM_SUBMIT_DIR." >&2
    exit 1
fi
if [ "$SEED" = "42" ]; then SEED_SUFFIX=""; else SEED_SUFFIX="_seed$SEED"; fi
N5_DIR="$HERE/preprocess/n5_seed${SEED}"
PAIRS_DIR="$HERE/preprocess/vector_construction${SEED_SUFFIX}"
VEC_DIR="$HERE/experiments/vector_construction${SEED_SUFFIX}"
FORCE="${FORCE_REBUILD:-0}"
EXPECTED_VECTORS=15  # 5 models x 3 directions (mf/mn/fn)

count_vectors() {  # $1 = embed|attn, $2 = leaf filename written per model/direction
    find "$VEC_DIR/$1" -name "$2" 2>/dev/null | wc -l | tr -d ' '
}

echo "=== job ${SLURM_JOB_ID:-0} on $(hostname): building seed $SEED (CPU-only) ==="
echo "    suffix: '${SEED_SUFFIX}'  force_rebuild: ${FORCE}"

if [ "$FORCE" != "1" ] && [ -f "$N5_DIR/test_candidates.tsv" ]; then
    echo "--- 1/4: SKIP split_queries_dedupe.py ($N5_DIR/test_candidates.tsv exists) ---"
else
    echo "--- 1/4: split_queries_dedupe.py (construction/test query split) ---"
    python split_queries_dedupe.py
fi

# Deliberately ignores FORCE_REBUILD -- see the note above.
if [ -f "$N5_DIR/test_queries_heldout.tsv" ]; then
    echo "--- 2/4: SKIP split_validation_heldout.py (split exists; reusing it, never regenerating) ---"
else
    echo "--- 2/4: split_validation_heldout.py (validation/held-out split of the test queries) ---"
    python split_validation_heldout.py
fi

if [ "$FORCE" != "1" ] && [ -f "$PAIRS_DIR/construction_pairs.jsonl" ]; then
    echo "--- 3/4: SKIP build_construction_pairs.py ($PAIRS_DIR/construction_pairs.jsonl exists) ---"
else
    echo "--- 3/4: build_construction_pairs.py (M/F/N doc pairs) ---"
    python build_construction_pairs.py
fi

# Vector steps check for a COMPLETE set, not just a directory -- a job killed partway
# through leaves some models' vectors written and must re-run, not be treated as done.
if [ "$FORCE" != "1" ] && [ "$(count_vectors embed vector.pt)" = "$EXPECTED_VECTORS" ]; then
    echo "--- 4/4: SKIP steering_vectors_embed_msmarco_fair.py ($EXPECTED_VECTORS/$EXPECTED_VECTORS vectors present) ---"
else
    echo "--- 4/4: steering_vectors_embed_msmarco_fair.py (embedding-level vectors, all 5 models) ---"
    python steering_vectors_embed_msmarco_fair.py
fi

if [ "$FORCE" != "1" ] && [ "$(count_vectors attn per_head_vectors.json)" = "$EXPECTED_VECTORS" ]; then
    echo "--- 4b/4: SKIP steering_vectors_attn_msmarco_fair.py ($EXPECTED_VECTORS/$EXPECTED_VECTORS vectors present) ---"
else
    echo "--- 4b/4: steering_vectors_attn_msmarco_fair.py (attention-level vectors, all 5 models) ---"
    python steering_vectors_attn_msmarco_fair.py
fi

echo "=== seed $SEED construction complete ==="
echo "    embed vectors: $(count_vectors embed vector.pt)/$EXPECTED_VECTORS   attn vectors: $(count_vectors attn per_head_vectors.json)/$EXPECTED_VECTORS"

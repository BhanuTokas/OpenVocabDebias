#!/bin/bash
#SBATCH -J celeba_discovery_control
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Control for the CelebA discovery result: the same recipe and prompts as
# train_confirmed.sh DATASET=celeba (Young prompted, whitened repulse λ10 +
# grw_pseudo, 8-group eval), but the confirmed Male pair is replaced by an
# unrelated prompt pair ($IRRELEVANT).  A third pair alone enlarges the
# subspace and splits the pseudo-groups more finely; if WGA stays near the
# Young-only floor (0.484), the +0.30 of the discovered pair (0.780) needs the
# discovered concept.
#
# IRRELEVANT  control prompt pair (default prompts/control_irrelevant_celeba.json)
#
# Submit from the repo root (CELEBA_ROOT is the dir that CONTAINS celeba/):
#   mkdir -p logs
#   CELEBA_ROOT=/data/hkerner/Datasets/CelebA sbatch scripts/celeba_discovery_control.sh

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1              # keep ~/.local packages out of the env
export UV_PYTHON_PREFERENCE=only-managed  # uv's Python, not Sol's system one
export PYTHONUNBUFFERED=1             # live logs
cd "$SLURM_SUBMIT_DIR"                 # not dirname $0: SLURM runs a spooled copy
if [ ! -f discover_biases.py ]; then
    echo "Submit from the OpenVocabDebias repo root (discover_biases.py not found in $PWD)" >&2
    exit 1
fi
: "${CELEBA_ROOT:?set CELEBA_ROOT to the directory containing celeba/ when submitting}"
if [ ! -f "$CELEBA_ROOT/celeba/list_attr_celeba.txt" ]; then
    echo "CelebA not found under CELEBA_ROOT=$CELEBA_ROOT" >&2
    exit 1
fi
IRRELEVANT=${IRRELEVANT:-prompts/control_irrelevant_celeba.json}
if ! grep -q '"accept": true' "$IRRELEVANT"; then
    echo "No accepted prompt pairs in $IRRELEVANT" >&2
    exit 1
fi

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
CKPT_DIR=./checkpoints/discovery_controls/irrelevant
RESULTS_DIR=./results/discovery_controls/irrelevant/seed_${SEED}

RUN="uv run --locked python"
$RUN -c "import numpy, torch, sys; print(sys.executable); print('numpy', numpy.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
$RUN train.py --runs repulse_only --repulse whitened --lambda_repulse 10 \
    --group_reweight pseudo --no_lambda_schedules --extra_prompts "$IRRELEVANT" \
    --celeba_root "$CELEBA_ROOT" --target_attr Blond_Hair --concept_attr Male \
    --extra_concept_attrs Young --prompt_attrs Young \
    --epochs 10 --batch_size 128 --concept_subspace_k 9 --seeds "$SEED" \
    --num_workers "$WORKERS" --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR"
$RUN eval_checkpoints.py "$CKPT_DIR"/*/seed_"$SEED" --celeba_root "$CELEBA_ROOT" \
    --num_workers "$WORKERS" --out "$RESULTS_DIR/eval_summary.csv"

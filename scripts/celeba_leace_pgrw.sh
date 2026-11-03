#!/bin/bash
#SBATCH -J celeba_leace_pgrw
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# CelebA Blond_Hair / Male: same grid as scripts/wb_leace_pgrw.sh, one seed
# per array task.  k=9 (all library prompt pairs): in probe_clip_target.py the
# whitened eraser left a 0.83 gender probe at k=9 vs 0.91 at k=3.
#
# Submit from the repo root (CELEBA_ROOT is the dir that CONTAINS celeba/):
#   mkdir -p logs && CELEBA_ROOT=/data/hkerner/Datasets/CelebA sbatch scripts/celeba_leace_pgrw.sh

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1              # keep ~/.local packages out of the env
export UV_PYTHON_PREFERENCE=only-managed  # uv's Python, not Sol's system one
export PYTHONUNBUFFERED=1             # live logs
cd "$SLURM_SUBMIT_DIR"                 # not dirname $0: SLURM runs a spooled copy
if [ ! -f train.py ]; then
    echo "Submit from the OpenVocabDebias repo root (train.py not found in $PWD)" >&2
    exit 1
fi
: "${CELEBA_ROOT:?set CELEBA_ROOT to the directory containing celeba/ when submitting}"
if [ ! -f "$CELEBA_ROOT/celeba/list_attr_celeba.txt" ]; then
    echo "CelebA not found under CELEBA_ROOT=$CELEBA_ROOT" >&2
    exit 1
fi

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

RUN="uv run --locked python"
$RUN -c "import numpy, torch, sys; print(sys.executable); print('numpy', numpy.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

CKPT_DIR=./checkpoints/celeba_leace_pgrw
RESULTS_DIR=./results/celeba_leace_pgrw/seed_${SEED}
COMMON=(--celeba_root "$CELEBA_ROOT" --target_attr Blond_Hair --concept_attr Male
        --epochs 10 --batch_size 128 --concept_subspace_k 9 --seeds "$SEED"
        --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR")

$RUN train.py --runs erm "${COMMON[@]}"
$RUN train.py --runs align_only full --no_lambda_schedules "${COMMON[@]}"
$RUN train.py --runs align_only full --no_lambda_schedules --eraser leace "${COMMON[@]}"
$RUN train.py --runs erm --group_reweight pseudo "${COMMON[@]}"
$RUN train.py --runs erm --group_reweight oracle "${COMMON[@]}"

# Only this task's seed, into a per-seed file (array tasks run concurrently)
$RUN eval_checkpoints.py "$CKPT_DIR"/*/seed_"$SEED" --celeba_root "$CELEBA_ROOT" \
    --out "$RESULTS_DIR/eval_summary.csv"

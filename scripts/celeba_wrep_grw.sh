#!/bin/bash
#SBATCH -J celeba_wrep_grw
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# CelebA Blond_Hair / Male, one seed per array task (ResNet-18, k=9, 10 epochs):
#
# 1. Label-free selection for the finished celeba_leace_pgrw runs of this seed:
#    best_pseudo_wga.pt from their epoch_XX.pt (gives the fair label-free ERM).
#
# 2. The Waterbirds winner: whitened repulse on pseudo-group-balanced batches.
#       repulse_only_nosched_wrep_lrep{4,10}_grw_pseudo
#    lambda_repulse: the removable decorrelation signal per batch of 128 is 0.086
#    on CelebA vs 0.036 per batch of 64 on Waterbirds, so λ=4 matches the push of
#    Waterbirds' λ=10; λ=10 is the same nominal value.
#
# Submit from the repo root (CELEBA_ROOT is the dir that CONTAINS celeba/):
#   mkdir -p logs && CELEBA_ROOT=/data/hkerner/Datasets/CelebA sbatch scripts/celeba_wrep_grw.sh

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
WORKERS=${SLURM_CPUS_PER_TASK:-8}

RUN="uv run --locked python"
$RUN -c "import numpy, torch, sys; print(sys.executable); print('numpy', numpy.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ── 1. Label-free selection on the finished sweep (this seed only) ───────────
OLD_CKPT=./checkpoints/celeba_leace_pgrw
if compgen -G "$OLD_CKPT/*/seed_$SEED" > /dev/null; then
    $RUN eval_checkpoints.py "$OLD_CKPT"/*/seed_"$SEED" --pseudo_from_epochs \
        --ckpts best_pseudo_wga.pt --celeba_root "$CELEBA_ROOT" --num_workers "$WORKERS" \
        --out "./results/celeba_leace_pgrw/seed_${SEED}/eval_summary_pseudo.csv" \
        || echo "[warn] label-free selection on $OLD_CKPT failed; continuing with part 2" >&2
else
    echo "[skip] no $OLD_CKPT/*/seed_$SEED; no label-free selection for the old sweep" >&2
fi

# ── 2. Whitened repulse + pseudo-group reweighting ───────────────────────────
CKPT_DIR=./checkpoints/celeba_wrep_grw
RESULTS_DIR=./results/celeba_wrep_grw/seed_${SEED}
COMMON=(--celeba_root "$CELEBA_ROOT" --target_attr Blond_Hair --concept_attr Male
        --epochs 10 --batch_size 128 --concept_subspace_k 9 --seeds "$SEED"
        --num_workers "$WORKERS" --no_lambda_schedules
        --repulse whitened --group_reweight pseudo
        --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR")

$RUN train.py --runs repulse_only --lambda_repulse 4 "${COMMON[@]}"
$RUN train.py --runs repulse_only --lambda_repulse 10 "${COMMON[@]}"

$RUN eval_checkpoints.py "$CKPT_DIR"/*/seed_"$SEED" --celeba_root "$CELEBA_ROOT" \
    --num_workers "$WORKERS" --out "$RESULTS_DIR/eval_summary.csv"

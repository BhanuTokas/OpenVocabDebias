#!/bin/bash
#SBATCH -J celeba_multi_cka
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Multi-attribute CelebA (Blond_Hair; spurious Male AND Young, 8-group WGA),
# ResNet-18, 10 epochs, batch 128, k=9 per attribute, whitened repulse λ10 +
# grw_pseudo, zero-shot cutoff 0 (cutoff effects are in celeba_multi_cutoff.sh),
# one seed per array task.  CKA align toward the LEACE-erased CLIP embedding,
# eraser fitted on pseudo-group balanced train embeddings (see
# scripts/wb_cka_align.sh).  On CelebA erasure costs some target signal even
# with the balanced fit (balanced CLIP probe val WGA 0.83 vs 0.86 raw), so a
# smaller gain than on Waterbirds is expected.
#
#   1. wrep + efbal               effect of the balanced eraser on z alone
#   2. acka λ1 + wrep + efbal
#   3. acka λ3 + wrep + efbal
# Reference (B/32): wrep + grw_pseudo 0.827.
#
# ENC  encoder id (default OpenAI CLIP B/32)
#
# Submit from the repo root (CELEBA_ROOT is the dir that CONTAINS celeba/):
#   mkdir -p logs
#   CELEBA_ROOT=/data/hkerner/Datasets/CelebA sbatch scripts/celeba_multi_cka.sh

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
ENC=${ENC:-openai/clip-vit-base-patch32}

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
TAG=${ENC##*/}

RUN="uv run --locked python"
$RUN -c "import numpy, torch, sys; print(sys.executable); print('numpy', numpy.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

CKPT_DIR=./checkpoints/multi_cka_${TAG}
RESULTS_DIR=./results/multi_cka_${TAG}/seed_${SEED}
COMMON=(--clip_model "$ENC" --celeba_root "$CELEBA_ROOT" --target_attr Blond_Hair
        --concept_attr Male --extra_concept_attrs Young
        --epochs 10 --batch_size 128 --concept_subspace_k 9 --seeds "$SEED"
        --num_workers "$WORKERS" --no_lambda_schedules --group_reweight pseudo
        --repulse whitened --lambda_repulse 10 --eraser_fit balanced
        --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR")

$RUN train.py --runs repulse_only "${COMMON[@]}"
$RUN train.py --runs full --align cka --lambda_align 1 "${COMMON[@]}"
$RUN train.py --runs full --align cka --lambda_align 3 "${COMMON[@]}"

$RUN eval_checkpoints.py "$CKPT_DIR"/*/seed_"$SEED" --celeba_root "$CELEBA_ROOT" \
    --num_workers "$WORKERS" --out "$RESULTS_DIR/eval_summary.csv"

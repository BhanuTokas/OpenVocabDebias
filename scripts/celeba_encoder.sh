#!/bin/bash
#SBATCH -J celeba_encoder
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# CelebA (target Blond_Hair) with a different frozen image-text encoder in
# place of OpenAI CLIP B/32.  ResNet-18, 10 epochs, batch 128, k=9 per
# attribute, constant lambdas, whitened repulse λ10, one seed per array task.
# Zero-shot val accuracy (encoder_sweep.py), Male / Young / Male+Young:
#   B/32 0.993 / 0.842 / 0.837;  OpenCLIP H/14 0.993 / 0.868 / 0.863;
#   DataComp L/14 - / 0.865 / 0.858
#
# ENC    encoder id (required), e.g.
#          laion/CLIP-ViT-H-14-laion2B-s32B-b79K     (OpenCLIP LAION-2B)
#          laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
# MULTI  0 (default): spurious Male only.  B/32: ERM 0.591, grw_pseudo 0.756,
#                     wrep + grw_pseudo 0.844
#        1: spurious Male AND Young (joint code, 8-group WGA), plus the
#           propensity runs.  B/32: ERM 0.473, grw_pseudo 0.640,
#           grw_propensity 0.731, wrep + pseudo 0.827, wrep + propensity 0.816
#
# Submit from the repo root (CELEBA_ROOT is the dir that CONTAINS celeba/):
#   mkdir -p logs
#   export CELEBA_ROOT=/data/hkerner/Datasets/CelebA
#   ENC=laion/CLIP-ViT-H-14-laion2B-s32B-b79K sbatch scripts/celeba_encoder.sh
#   ENC=laion/CLIP-ViT-H-14-laion2B-s32B-b79K MULTI=1 sbatch scripts/celeba_encoder.sh
#   ENC=laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K MULTI=1 sbatch scripts/celeba_encoder.sh

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
: "${ENC:?set ENC to the encoder id when submitting}"
: "${CELEBA_ROOT:?set CELEBA_ROOT to the directory containing celeba/ when submitting}"
if [ ! -f "$CELEBA_ROOT/celeba/list_attr_celeba.txt" ]; then
    echo "CelebA not found under CELEBA_ROOT=$CELEBA_ROOT" >&2
    exit 1
fi
MULTI=${MULTI:-0}

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
TAG=${ENC##*/}
NAME=$([ "$MULTI" = 1 ] && echo multi || echo male)

RUN="uv run --locked python"
$RUN -c "import numpy, torch, sys; print(sys.executable); print('numpy', numpy.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

CKPT_DIR=./checkpoints/enc_${TAG}_${NAME}
RESULTS_DIR=./results/enc_${TAG}_${NAME}/seed_${SEED}
COMMON=(--clip_model "$ENC" --celeba_root "$CELEBA_ROOT" --target_attr Blond_Hair
        --concept_attr Male
        --epochs 10 --batch_size 128 --concept_subspace_k 9 --seeds "$SEED"
        --num_workers "$WORKERS" --no_lambda_schedules
        --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR")
if [ "$MULTI" = 1 ]; then
    COMMON+=(--extra_concept_attrs Young)
fi
WREP=(--runs repulse_only --repulse whitened --lambda_repulse 10)

$RUN train.py --runs erm "${COMMON[@]}"
$RUN train.py --runs erm --group_reweight pseudo "${COMMON[@]}"
$RUN train.py "${WREP[@]}" --group_reweight pseudo "${COMMON[@]}"
if [ "$MULTI" = 1 ]; then
    $RUN train.py --runs erm --group_reweight propensity "${COMMON[@]}"
    $RUN train.py "${WREP[@]}" --group_reweight propensity "${COMMON[@]}"
fi

$RUN eval_checkpoints.py "$CKPT_DIR"/*/seed_"$SEED" --celeba_root "$CELEBA_ROOT" \
    --num_workers "$WORKERS" --out "$RESULTS_DIR/eval_summary.csv"

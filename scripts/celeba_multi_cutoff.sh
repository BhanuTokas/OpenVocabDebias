#!/bin/bash
#SBATCH -J celeba_multi_cutoff
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Multi-attribute CelebA (Blond_Hair; spurious Male AND Young, 8-group WGA),
# ResNet-18, 10 epochs, batch 128, k=9 per attribute, λ10, one seed per task.
#
# 1. --zs_cutoff gmm: the zero-shot labels are cut where a 2-component Gaussian
#    mixture fitted on the unlabeled train scores has equal posteriors, instead
#    of at 0.  Young pseudo-labels at 0 miss 36-65% of older faces; the fitted
#    cutoff raises the worst (blond, sex, age) val cell from 0.40 to 0.56 (B/32).
#    Affects the training pseudo-groups and the val pseudo-groups behind
#    best_pseudo_wga.pt (so also the ERM and propensity rows).
#      B/32 at cutoff 0: ERM 0.473, grw_pseudo 0.640, grw_propensity 0.731,
#                        wrep+pseudo 0.827, wrep+propensity 0.816
#      H/14 at cutoff 0: 0.457 / 0.747 / 0.667 / 0.786 / 0.715
# 2. --repulse raw (cutoff 0): decorrelate from the centred raw text-subspace
#    scores instead of the whitened ones.  Same information, but the 13 raw
#    scores are correlated, so overlapping directions count more than once.
#    Compare with wrep+pseudo above.
#
# ENC  encoder id (default OpenAI CLIP B/32), e.g.
#        laion/CLIP-ViT-H-14-laion2B-s32B-b79K
#
# Submit from the repo root (CELEBA_ROOT is the dir that CONTAINS celeba/):
#   mkdir -p logs
#   export CELEBA_ROOT=/data/hkerner/Datasets/CelebA
#   sbatch scripts/celeba_multi_cutoff.sh
#   ENC=laion/CLIP-ViT-H-14-laion2B-s32B-b79K sbatch scripts/celeba_multi_cutoff.sh

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

CKPT_DIR=./checkpoints/multi_cutoff_${TAG}
RESULTS_DIR=./results/multi_cutoff_${TAG}/seed_${SEED}
COMMON=(--clip_model "$ENC" --celeba_root "$CELEBA_ROOT" --target_attr Blond_Hair
        --concept_attr Male --extra_concept_attrs Young
        --epochs 10 --batch_size 128 --concept_subspace_k 9 --seeds "$SEED"
        --num_workers "$WORKERS" --no_lambda_schedules
        --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR")
WREP=(--runs repulse_only --repulse whitened --lambda_repulse 10)
GMM=(--zs_cutoff gmm)

$RUN train.py --runs erm "${GMM[@]}" "${COMMON[@]}"
$RUN train.py --runs erm --group_reweight pseudo "${GMM[@]}" "${COMMON[@]}"
$RUN train.py --runs erm --group_reweight propensity "${GMM[@]}" "${COMMON[@]}"
$RUN train.py "${WREP[@]}" --group_reweight pseudo "${GMM[@]}" "${COMMON[@]}"
$RUN train.py "${WREP[@]}" --group_reweight propensity "${GMM[@]}" "${COMMON[@]}"
$RUN train.py --runs repulse_only --repulse raw --lambda_repulse 10 \
    --group_reweight pseudo "${COMMON[@]}"

$RUN eval_checkpoints.py "$CKPT_DIR"/*/seed_"$SEED" --celeba_root "$CELEBA_ROOT" \
    --num_workers "$WORKERS" --out "$RESULTS_DIR/eval_summary.csv"

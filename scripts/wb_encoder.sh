#!/bin/bash
#SBATCH -J wb_encoder
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Waterbirds with a different frozen image-text encoder in place of OpenAI
# CLIP B/32 (sentence prompts, k=3, constant lambdas, one seed per array task).
# The encoder supplies the concept subspace, the whitened eraser / concept
# scores, the training pseudo-groups and the val pseudo-groups behind
# best_pseudo_wga.pt.  Zero-shot place accuracy, val, sentence prompts
# (encoder_sweep.py): B/32 0.924, PE-Core B16 0.934, OpenCLIP H/14 0.926.
#
#   erm                      ERM; only the pseudo-label checkpoint selection changes
#   erm_grw_pseudo           B/32: R18 0.793, R50 0.870
#   whitened repulse + grw_pseudo   B/32: R18 λ10 0.859, R50 λ15 0.891
#
# ENC      encoder id (required), e.g.
#            hf-hub:timm/PE-Core-B-16                 (OpenCLIP backend)
#            laion/CLIP-ViT-H-14-laion2B-s32B-b79K    (OpenCLIP LAION-2B, HF port)
# BACKBONE resnet18 (default, λ10) or resnet50 (λ15; submit with --array=0-4 for 5 seeds)
#
# Submit from the repo root:
#   mkdir -p logs
#   ENC=hf-hub:timm/PE-Core-B-16 sbatch scripts/wb_encoder.sh
#   ENC=laion/CLIP-ViT-H-14-laion2B-s32B-b79K sbatch scripts/wb_encoder.sh
#   ENC=laion/CLIP-ViT-H-14-laion2B-s32B-b79K BACKBONE=resnet50 sbatch --array=0-4 --time=1-00:00:00 scripts/wb_encoder.sh

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1              # keep ~/.local packages out of the env
export UV_PYTHON_PREFERENCE=only-managed  # uv's Python, not Sol's system one
export PYTHONUNBUFFERED=1             # live logs
cd "$SLURM_SUBMIT_DIR"                 # not dirname $0: SLURM runs a spooled copy
if [ ! -f waterbirds_train.py ]; then
    echo "Submit from the OpenVocabDebias repo root (waterbirds_train.py not found in $PWD)" >&2
    exit 1
fi
: "${ENC:?set ENC to the encoder id when submitting}"
BACKBONE=${BACKBONE:-resnet18}
case "$BACKBONE" in
    resnet18) LAMBDA=10 ;;
    resnet50) LAMBDA=15 ;;
    *) echo "BACKBONE must be resnet18 or resnet50" >&2; exit 1 ;;
esac

SEEDS=(42 123 456 789 1011)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
TAG=${ENC##*/}

RUN="uv run --locked python"
$RUN -c "import numpy, torch, sys; print(sys.executable); print('numpy', numpy.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

CKPT_DIR=./checkpoints_waterbirds/enc_${TAG}_${BACKBONE}
RESULTS_DIR=./results_waterbirds/enc_${TAG}_${BACKBONE}/seed_${SEED}
COMMON=(--clip_model "$ENC" --backbone "$BACKBONE" --seeds "$SEED"
        --concept_subspace_k 3 --prompts sentences
        --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR")

$RUN waterbirds_train.py --runs erm "${COMMON[@]}"
$RUN waterbirds_train.py --runs erm --group_reweight pseudo "${COMMON[@]}"
$RUN waterbirds_train.py --runs repulse_only --no_lambda_schedules --repulse whitened \
    --group_reweight pseudo --lambda_repulse "$LAMBDA" "${COMMON[@]}"

$RUN eval_checkpoints.py "$CKPT_DIR"/*/seed_"$SEED" --out "$RESULTS_DIR/eval_summary.csv"

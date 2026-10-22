#!/bin/bash
#SBATCH -J wb_cka_align
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Waterbirds (ResNet-18, k=3, sentence prompts, 3 seeds, constant lambdas,
# grw_pseudo throughout): an align term that says what to keep, next to the
# decorrelation that says what to drop.
#
#   L_align = 1 − linear CKA( E(x), LEACE-erased CLIP embedding )   (--align cka)
#   no projection head; eraser fitted on pseudo-group balanced train
#   embeddings (--eraser_fit balanced).  On the biased train set bird type and
#   place agree 95% of the time, so the biased-fit eraser also removes bird
#   information: a balanced CLIP probe on the erased embedding drops to val WGA
#   0.63, vs 0.84 with the balanced fit (raw CLIP 0.81).
#
#   1. wrep λ10 + efbal                    effect of the balanced eraser on z alone
#   2. acka λ1 + wrep λ10 + efbal
#   3. acka λ3 + wrep λ10 + efbal
#   3b. acka λ10 + wrep λ10 + efbal       (1 − CKA starts at ~0.25 for ImageNet
#                                          R18, vs 10 × 0.12 for the repulse)
#   4. acka λ1 + wrep λ10, biased eraser   does the eraser fit matter for align?
#   5. acka λ1 alone + efbal               align without decorrelation
# Reference (sentences, grw_pseudo): wrep λ10 0.859, grw only 0.793.
#
# Submit from the repo root:   mkdir -p logs && sbatch scripts/wb_cka_align.sh

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

RUN="uv run --locked python"
$RUN -c "import numpy, torch, sys; print(sys.executable); print('numpy', numpy.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

CKPT_DIR=./checkpoints_waterbirds/cka_align
RESULTS_DIR=./results_waterbirds/cka_align
COMMON=(--seeds 42 123 456 --concept_subspace_k 3 --prompts sentences
        --no_lambda_schedules --group_reweight pseudo
        --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR")
WREP=(--repulse whitened --lambda_repulse 10)
CKA=(--runs full --align cka)

$RUN waterbirds_train.py --runs repulse_only "${WREP[@]}" --eraser_fit balanced "${COMMON[@]}"
$RUN waterbirds_train.py "${CKA[@]}" --lambda_align 1 "${WREP[@]}" --eraser_fit balanced "${COMMON[@]}"
$RUN waterbirds_train.py "${CKA[@]}" --lambda_align 3 "${WREP[@]}" --eraser_fit balanced "${COMMON[@]}"
$RUN waterbirds_train.py "${CKA[@]}" --lambda_align 10 "${WREP[@]}" --eraser_fit balanced "${COMMON[@]}"
$RUN waterbirds_train.py "${CKA[@]}" --lambda_align 1 "${WREP[@]}" "${COMMON[@]}"
$RUN waterbirds_train.py --runs align_only --align cka --lambda_align 1 --eraser_fit balanced "${COMMON[@]}"

$RUN eval_checkpoints.py "$CKPT_DIR" --out "$RESULTS_DIR/eval_summary.csv"

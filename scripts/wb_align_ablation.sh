#!/bin/bash
#SBATCH -J wb_align_ablation
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Align ablation on Waterbirds (ResNet-18, k=3, sentence prompts, 3 seeds,
# constant lambdas, all on CLIP pseudo-group-balanced batches).  Does the
# CLIP alignment loss (P(E(x)) -> V_I_perp) add anything once reweighting and the
# whitened repulse are in place?
#
#   reference (wb_sentences.sh): repulse_only ... wrep_lrep10_grw_pseudo   0.859 ± 0.019
#   align_only, whitened target                 align without repulse
#   align_only, orthogonal target
#   full, whitened target + whitened repulse    align added to the reference
#   full, orthogonal target + whitened repulse
#
# Submit from the repo root:   mkdir -p logs && sbatch scripts/wb_align_ablation.sh

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

CKPT_DIR=./checkpoints_waterbirds/align_ablation
RESULTS_DIR=./results_waterbirds/align_ablation
COMMON=(--seeds 42 123 456 --concept_subspace_k 3 --prompts sentences
        --no_lambda_schedules --group_reweight pseudo
        --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR")

$RUN waterbirds_train.py --runs align_only --eraser leace "${COMMON[@]}"
$RUN waterbirds_train.py --runs align_only "${COMMON[@]}"
$RUN waterbirds_train.py --runs full --eraser leace --repulse whitened --lambda_repulse 10 "${COMMON[@]}"
$RUN waterbirds_train.py --runs full --repulse whitened --lambda_repulse 10 "${COMMON[@]}"

$RUN eval_checkpoints.py "$CKPT_DIR" --out "$RESULTS_DIR/eval_summary.csv"

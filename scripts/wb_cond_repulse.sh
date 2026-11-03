#!/bin/bash
#SBATCH -J wb_cond_repulse
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Class-conditional whitened decorrelation WITHOUT group reweighting
# (--repulse whitened_cond): corr(E(x), CLIP concept scores | label).  Can one
# loss replace pseudo-group resampling?  Waterbirds, ResNet-18, k=3, sentence
# prompts, 3 seeds, constant lambdas.
#
# On natural (biased) train batches the CLIP place score also correlates -0.80
# with the label, so the marginal loss (0.050 signal per batch of 64) partly
# removes label information; the conditional one keeps only the within-class
# part (0.024).  On balanced batches the two coincide (0.037 each).  lambda 15
# matches the push of the reweighted lambda 10 run; 10 and 30 bracket it.
#
#   references (wb_sentences.sh):  erm 0.693, erm_grw_pseudo 0.793,
#                                  marginal wrep_lrep10 + grw_pseudo 0.859
#   (wb_wrep, words): marginal wrep_lrep10 without reweighting 0.549
#
# Submit from the repo root:   mkdir -p logs && sbatch scripts/wb_cond_repulse.sh

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

CKPT_DIR=./checkpoints_waterbirds/cond_repulse
RESULTS_DIR=./results_waterbirds/cond_repulse
COMMON=(--seeds 42 123 456 --concept_subspace_k 3 --prompts sentences
        --no_lambda_schedules --repulse whitened_cond
        --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR")

$RUN waterbirds_train.py --runs repulse_only --lambda_repulse 10 "${COMMON[@]}"
$RUN waterbirds_train.py --runs repulse_only --lambda_repulse 15 "${COMMON[@]}"
$RUN waterbirds_train.py --runs repulse_only --lambda_repulse 30 "${COMMON[@]}"

$RUN eval_checkpoints.py "$CKPT_DIR" --out "$RESULTS_DIR/eval_summary.csv"

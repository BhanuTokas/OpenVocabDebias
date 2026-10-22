#!/bin/bash
#SBATCH -J wb_raw_repulse
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Waterbirds (ResNet-18, k=3, sentence prompts, 3 seeds, λ10, grw_pseudo):
# decorrelate E(x) from the centred raw text-subspace scores (--repulse raw)
# instead of the whitened ones.  The whitened scores are the same 3 scores
# made uncorrelated with unit variance, so the two losses are identical for
# k=1 and differ only through the correlation between the 3 raw scores
# (0.10-0.39 on Waterbirds).  Compare with wrep λ10 + grw_pseudo: 0.859.
#
# (No cutoff run here: the fitted Waterbirds cutoff is +0.003, i.e. ≈ 0.)
#
# Submit from the repo root:   mkdir -p logs && sbatch scripts/wb_raw_repulse.sh

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

CKPT_DIR=./checkpoints_waterbirds/raw_repulse
RESULTS_DIR=./results_waterbirds/raw_repulse
COMMON=(--seeds 42 123 456 --concept_subspace_k 3 --prompts sentences
        --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR")

$RUN waterbirds_train.py --runs repulse_only --no_lambda_schedules --repulse raw \
    --group_reweight pseudo --lambda_repulse 10 "${COMMON[@]}"

$RUN eval_checkpoints.py "$CKPT_DIR" --out "$RESULTS_DIR/eval_summary.csv"

#!/bin/bash
#SBATCH -J wb_lambda_sweep
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Waterbirds (ResNet-18, k=3, 3 seeds, constant lambdas): lambda_repulse sweep for
# the best configuration so far, whitened repulse on pseudo-group-balanced batches.
# lambda 10 is already in checkpoints_waterbirds/wrep (WGA 0.835 ± 0.007); this
# adds 3 and 30 to show whether 10 sits on a plateau.
#
# Submit from the repo root:   mkdir -p logs && sbatch scripts/wb_lambda_sweep.sh

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

CKPT_DIR=./checkpoints_waterbirds/wrep_lsweep
RESULTS_DIR=./results_waterbirds/wrep_lsweep
COMMON=(--seeds 42 123 456 --concept_subspace_k 3 --no_lambda_schedules
        --repulse whitened --group_reweight pseudo
        --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR")

$RUN waterbirds_train.py --runs repulse_only --lambda_repulse 3 "${COMMON[@]}"
$RUN waterbirds_train.py --runs repulse_only --lambda_repulse 30 "${COMMON[@]}"

$RUN eval_checkpoints.py "$CKPT_DIR" --out "$RESULTS_DIR/eval_summary.csv"

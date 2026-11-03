#!/bin/bash
#SBATCH -J wb_r50
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Waterbirds with ResNet-50 (the standard backbone, so numbers compare with the
# literature), 5 seeds, k=3, word prompts, constant lambdas.
#
#   erm, erm_grw_pseudo, erm_grw_oracle
#   repulse_only_nosched_wrep_lrep15_grw_pseudo
#
# lambda_repulse 15: the removable decorrelation signal per batch is 0.024 with
# ResNet-50's 2048-dim features vs 0.036 with ResNet-18's 512, so 15 matches the
# push of ResNet-18's lambda 10.
#
# Submit from the repo root:   mkdir -p logs && sbatch scripts/wb_r50.sh

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

CKPT_DIR=./checkpoints_waterbirds/r50
RESULTS_DIR=./results_waterbirds/r50
COMMON=(--backbone resnet50 --seeds 42 123 456 789 1011 --concept_subspace_k 3
        --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR")

$RUN waterbirds_train.py --runs erm "${COMMON[@]}"
$RUN waterbirds_train.py --runs erm --group_reweight pseudo "${COMMON[@]}"
$RUN waterbirds_train.py --runs erm --group_reweight oracle "${COMMON[@]}"
$RUN waterbirds_train.py --runs repulse_only --no_lambda_schedules --repulse whitened \
    --group_reweight pseudo --lambda_repulse 15 "${COMMON[@]}"

$RUN eval_checkpoints.py "$CKPT_DIR" --out "$RESULTS_DIR/eval_summary.csv"

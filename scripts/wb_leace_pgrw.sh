#!/bin/bash
#SBATCH -J wb_leace_pgrw
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Waterbirds: whitened (LEACE) alignment target vs orthogonal, against a
# CLIP pseudo-group reweighting baseline.  All with constant lambdas.
#
#   erm                                    plain ERM
#   align_only_nosched, full_nosched       orthogonal V_I_perp (current method)
#   align_only_nosched_leace,
#   full_nosched_leace                     whitened V_I_perp (label-free)
#   erm_grw_pseudo                         group-balanced ERM, CLIP zero-shot groups (label-free)
#   erm_grw_oracle                         group-balanced ERM, true groups (reference only)
#
# Then re-evaluates every run with best.pt (label-free selection) and best_wga.pt.
#
# Submit from the repo root:   mkdir -p logs && sbatch scripts/wb_leace_pgrw.sh

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

CKPT_DIR=./checkpoints_waterbirds/leace_pgrw
RESULTS_DIR=./results_waterbirds/leace_pgrw
COMMON=(--seeds 42 123 456 --concept_subspace_k 3 --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR")

$RUN waterbirds_train.py --runs erm "${COMMON[@]}"
$RUN waterbirds_train.py --runs align_only full --no_lambda_schedules "${COMMON[@]}"
$RUN waterbirds_train.py --runs align_only full --no_lambda_schedules --eraser leace "${COMMON[@]}"
$RUN waterbirds_train.py --runs erm --group_reweight pseudo "${COMMON[@]}"
$RUN waterbirds_train.py --runs erm --group_reweight oracle "${COMMON[@]}"

$RUN eval_checkpoints.py "$CKPT_DIR" --out "$RESULTS_DIR/eval_summary.csv"

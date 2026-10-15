#!/bin/bash
#SBATCH -J wb_sched_control
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Waterbirds control for the repulse collapse: do the calibrated lambda
# schedules cause it, or does running align + repulse together?
#
#   erm, full, repulse_only          — current defaults (schedules on)
#   full_nosched, repulse_only_nosched — same, constant lambdas
#   sequential                       — repulse-only then align-only (no schedules)
#
# Then re-evaluates every run with best.pt (label-free selection) and
# best_wga.pt, probing both E(x) and P(E(x)).
#
# Submit from the repo root:   mkdir -p logs && sbatch scripts/wb_sched_control.sh

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONUNBUFFERED=1             # live logs
cd "$SLURM_SUBMIT_DIR"                 # not dirname $0: SLURM runs a spooled copy
if [ ! -f waterbirds_train.py ]; then
    echo "Submit from the OpenVocabDebias repo root (waterbirds_train.py not found in $PWD)" >&2
    exit 1
fi

# No pyproject.toml in this repo: run in a uv-managed env built from requirements.txt
RUN="uv run --no-project --with-requirements requirements.txt python"

CKPT_DIR=./checkpoints_waterbirds/sched_control
RESULTS_DIR=./results_waterbirds/sched_control
SEEDS="42 123 456"

$RUN waterbirds_train.py --runs erm full repulse_only sequential \
    --seeds $SEEDS --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR"

$RUN waterbirds_train.py --runs full repulse_only --no_lambda_schedules \
    --seeds $SEEDS --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR"

$RUN eval_checkpoints.py "$CKPT_DIR" --out "$RESULTS_DIR/eval_summary.csv"

# Lambda calibration values, to check the λ_repulse≈100 / λ_task≈0.01 hypothesis
grep -h "lambda_.*_0=\|lambda_.* warmup\]\|lambda_repulse decay\]" "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" || true

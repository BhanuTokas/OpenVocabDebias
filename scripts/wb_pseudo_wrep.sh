#!/bin/bash
#SBATCH -J wb_pseudo_wrep
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Waterbirds, two label-free additions (ResNet-18, k=3, constant lambdas):
#
# 1. Label-free checkpoint selection for the finished leace_pgrw sweep:
#    best_pseudo_wga.pt is picked from its epoch_XX.pt by worst-group val accuracy
#    over CLIP zero-shot pseudo-groups, then evaluated on test.
#
# 2. Whitened repulse (--repulse whitened): L_repulse becomes the squared batch
#    correlation between E(x) and the whitened CLIP concept scores.
#       repulse_only_nosched_wrep_lrep{10,30}         repulse only
#       full_nosched_leace_wrep_lrep10                whitened target + whitened repulse
#       repulse_only_nosched_wrep_lrep10_grw_pseudo   whitened repulse on pseudo-group-balanced batches
#    New runs save best_pseudo_wga.pt during training; all three selections are
#    evaluated at the end.
#
# Submit from the repo root:   mkdir -p logs && sbatch scripts/wb_pseudo_wrep.sh

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

# ── 1. Label-free selection on the finished sweep ────────────────────────────
OLD_CKPT=./checkpoints_waterbirds/leace_pgrw
# Independent of part 2, so a failure here is reported but does not stop training
if [ -d "$OLD_CKPT" ]; then
    $RUN eval_checkpoints.py "$OLD_CKPT" --pseudo_from_epochs --ckpts best_pseudo_wga.pt \
        --out ./results_waterbirds/leace_pgrw/eval_summary_pseudo.csv \
        || echo "[warn] label-free selection on $OLD_CKPT failed; continuing with part 2" >&2
else
    echo "[skip] $OLD_CKPT not found; no label-free selection for the old sweep" >&2
fi

# ── 2. Whitened repulse ──────────────────────────────────────────────────────
CKPT_DIR=./checkpoints_waterbirds/wrep
RESULTS_DIR=./results_waterbirds/wrep
# lambda_repulse: on pretrained ResNet-18 features a batch of 64 gives L ≈ 0.083
# against a chance floor of 3/64 ≈ 0.047, so the removable signal (~0.036) needs
# λ ≈ 10 to be comparable to the cross-entropy; 30 tests a stronger push.
COMMON=(--seeds 42 123 456 --concept_subspace_k 3 --no_lambda_schedules
        --repulse whitened --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR")

$RUN waterbirds_train.py --runs repulse_only --lambda_repulse 10 "${COMMON[@]}"
$RUN waterbirds_train.py --runs repulse_only --lambda_repulse 30 "${COMMON[@]}"
$RUN waterbirds_train.py --runs full --eraser leace --lambda_repulse 10 "${COMMON[@]}"
$RUN waterbirds_train.py --runs repulse_only --group_reweight pseudo --lambda_repulse 10 "${COMMON[@]}"

$RUN eval_checkpoints.py "$CKPT_DIR" --out "$RESULTS_DIR/eval_summary.csv"

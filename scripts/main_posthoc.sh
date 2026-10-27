#!/bin/bash
#SBATCH -J main_posthoc
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Main tables, post-hoc baselines (submitted by scripts/submit_main.sh after the
# ERM runs):
#   METHOD=last_layer  DFR and AFR on the ERM run of each seed (array over
#                      SEEDS), with true groups (reference) and CLIP
#                      pseudo-groups (label-free) -> results_<ROOT>/<TAG>/last_layer/
#   METHOD=probe       frozen-CLIP zero-shot / probe / balanced probe (one task,
#                      seed-independent)                -> results_<ROOT>/<DATASET>_clip_probe.csv
# DATASET, BACKBONE, SEEDS, ROOT, CELEBA_ROOT as in scripts/main_run.sh.

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1              # keep ~/.local packages out of the env
export UV_PYTHON_PREFERENCE=only-managed  # uv's Python, not Sol's system one
export PYTHONUNBUFFERED=1             # live logs
cd "$SLURM_SUBMIT_DIR"                 # not dirname $0: SLURM runs a spooled copy
if [ ! -f last_layer.py ]; then
    echo "Submit from the OpenVocabDebias repo root (last_layer.py not found in $PWD)" >&2
    exit 1
fi
: "${DATASET:?}" "${METHOD:?}"
read -ra SEEDS <<< "${SEEDS:-42 123 456 789 1011}"
WORKERS=${SLURM_CPUS_PER_TASK:-8}
ROOT=${ROOT:-main}
DATA_ARGS=()
if [ "$DATASET" = celeba ]; then
    : "${CELEBA_ROOT:?set CELEBA_ROOT to the directory containing celeba/}"
    DATA_ARGS=(--celeba_root "$CELEBA_ROOT")
fi
RUN="uv run --locked python"
$RUN -c "import numpy, torch, sys; print(sys.executable); print('numpy', numpy.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [ "$METHOD" = probe ]; then
    if [ "$DATASET" = wb ]; then
        $RUN clip_probe.py --dataset waterbirds --prompts sentences --num_workers "$WORKERS" \
            --out "results_${ROOT}/wb_clip_probe.csv"
    else
        $RUN clip_probe.py --dataset celeba "${DATA_ARGS[@]}" --concept_attr Male \
            --num_workers "$WORKERS" --out "results_${ROOT}/celeba_clip_probe.csv"
    fi
    exit 0
fi

: "${BACKBONE:?}"
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
TAG=${DATASET}_${BACKBONE}
RUN_DIR=$(ls -d "./checkpoints_${ROOT}/${TAG}/erm"/*/seed_"$SEED"/*/ | tail -1)
OUT=./results_${ROOT}/${TAG}/last_layer/seed_${SEED}.csv
rm -f "$OUT"
for m in dfr afr; do
    for g in true pseudo; do
        $RUN last_layer.py "$RUN_DIR" --method "$m" --groups "$g" --seed "$SEED" \
            "${DATA_ARGS[@]}" --num_workers "$WORKERS" --out "$OUT"
    done
done

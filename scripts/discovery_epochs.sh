#!/bin/bash
#SBATCH -J discovery_epochs
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Which model should discovery read: one epoch or fully trained?  Crosses the
# main ERM run's epoch_01.pt and best.pt with the train and val splits, reusing
# erm_ep1_train / erm_val of scripts/discovery_cf.sh if they exist.  No training.
# best.pt on train may have too few errors for a direction: that cell is then
# reported and skipped.
#
#   DATASET wb | celeba,  BACKBONE resnet18 | resnet50,  CELEBA_ROOT for celeba
#   SEED (42), ROOT of the main ERM runs (main)
# Outputs: results_cf/<tag>/erm_{ep1,full}_{train,val}.json, compared at the end.
# Correlations with the truth are only comparable within a split.
#
# Submit from the repo root:
#   mkdir -p logs
#   sbatch --export=ALL,DATASET=wb,BACKBONE=resnet18 scripts/discovery_epochs.sh

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1              # keep ~/.local packages out of the env
export UV_PYTHON_PREFERENCE=only-managed  # uv's Python, not Sol's system one
export PYTHONUNBUFFERED=1             # live logs
cd "$SLURM_SUBMIT_DIR"                 # not dirname $0: SLURM runs a spooled copy
if [ ! -f discover_biases.py ]; then
    echo "Submit from the OpenVocabDebias repo root (discover_biases.py not found in $PWD)" >&2
    exit 1
fi
: "${DATASET:?}" "${BACKBONE:?}"
SEED=${SEED:-42}
ROOT=${ROOT:-main}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
TAG=${DATASET}_${BACKBONE}
RES_BASE=./results_cf/${TAG}
RUN="uv run --locked python"
DATA_ARGS=()
if [ "$DATASET" = celeba ]; then
    : "${CELEBA_ROOT:?set CELEBA_ROOT to the directory containing celeba/}"
    DATA_ARGS=(--celeba_root "$CELEBA_ROOT")
fi
ERM=$(ls -d "./checkpoints_${ROOT}/${TAG}/erm"/*/seed_"$SEED"/*/ | tail -1)
mkdir -p "$RES_BASE"

# reuse the two cells discovery_cf.sh already produced
for pair in erm_ep1_train:erm_ep1_train erm_val:erm_full_val; do
    src=$RES_BASE/${pair%:*}.json; dst=$RES_BASE/${pair#*:}.json
    if [ -f "$src" ] && [ "$src" != "$dst" ]; then
        cp "$src" "$dst"; cp "${src%.json}_dirs.npz" "${dst%.json}_dirs.npz"
    fi
done

OUTS=()
for cell in ep1:epoch_01.pt:train ep1:epoch_01.pt:val full:best.pt:train full:best.pt:val; do
    IFS=: read -r name ckpt split <<< "$cell"
    out=$RES_BASE/erm_${name}_${split}.json
    if [ ! -f "$out" ]; then
        if [ ! -f "$ERM$ckpt" ]; then
            echo "no $ckpt in $ERM: skipping $name/$split" >&2
            continue
        fi
        if ! $RUN discover_biases.py "$ERM" --split "$split" --ckpt "$ckpt" --prompted none \
            --report_truth "${DATA_ARGS[@]}" --num_workers "$WORKERS" --out "$out"; then
            echo "$name/$split gave no direction (too few errors?): skipping" >&2
            continue
        fi
    fi
    OUTS+=("$out")
done
$RUN scripts/compare_discovery.py "${OUTS[@]}"

#!/bin/bash
#SBATCH -J discovery_vocab
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Does the naming depend on the vocabulary?  The same errors and directions (the
# main ERM run's epoch_01.pt on train), named from: the built-in list; the blind
# dataset-description lists (prompts/vocab/birds_run* or faces_run*, union of the
# 3 runs and each run alone); the blind generic lists (the same).  No training.
#
#   DATASET wb | celeba,  BACKBONE resnet18 | resnet50,  CELEBA_ROOT for celeba
#   SEED (42), ROOT of the main ERM runs (main)
# Outputs: results_vocab/<tag>/<list>.json, compared at the end of the log.
#
# Submit from the repo root:
#   mkdir -p logs
#   sbatch --export=ALL,DATASET=wb,BACKBONE=resnet18 scripts/discovery_vocab.sh

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
RES_BASE=./results_vocab/${TAG}
RUN="uv run --locked python"
if [ "$DATASET" = wb ]; then
    B=birds; DATA_ARGS=()
elif [ "$DATASET" = celeba ]; then
    : "${CELEBA_ROOT:?set CELEBA_ROOT to the directory containing celeba/}"
    B=faces; DATA_ARGS=(--celeba_root "$CELEBA_ROOT")
else
    echo "DATASET must be wb or celeba" >&2
    exit 1
fi
ERM=$(ls -d "./checkpoints_${ROOT}/${TAG}/erm"/*/seed_"$SEED"/*/ 2>/dev/null | tail -1)
if [ -z "$ERM" ] || [ ! -f "${ERM}epoch_01.pt" ]; then
    echo "no epoch_01.pt of the main ERM run (seed $SEED) for $TAG" >&2
    exit 1
fi
mkdir -p "$RES_BASE"
DISC=(--split train --ckpt epoch_01.pt --prompted none --report_truth
      "${DATA_ARGS[@]}" --num_workers "$WORKERS")

$RUN discover_biases.py "$ERM" "${DISC[@]}" --out "$RES_BASE/builtin.json"
OUTS=("$RES_BASE/builtin.json")
for L in "$B" generic; do
    FILES=()
    for R in 1 2 3; do
        F=prompts/vocab/${L}_run${R}.txt
        FILES+=("$F")
        $RUN discover_biases.py "$ERM" "${DISC[@]}" --vocab_file "$F" \
            --out "$RES_BASE/${L}_run${R}.json"
        OUTS+=("$RES_BASE/${L}_run${R}.json")
    done
    $RUN discover_biases.py "$ERM" "${DISC[@]}" --vocab_file "${FILES[@]}" \
        --out "$RES_BASE/${L}_union.json"
    OUTS+=("$RES_BASE/${L}_union.json")
done
$RUN scripts/compare_discovery.py "${OUTS[@]}"

#!/bin/bash
#SBATCH -J discovery_cf
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Discovery without the val split: errors on train from (a) the main ERM run's
# epoch_01.pt and (b) K-fold cross-fitting, where each train example is
# predicted by a 1-epoch ERM model that did not see it; compared with the usual
# val discovery of the same ERM run.  Submit from the repo root:
#
#   mkdir -p logs
#   J=$(sbatch --parsable --array=0-9 --export=ALL,STAGE=train,DATASET=wb,BACKBONE=resnet18 \
#         scripts/discovery_cf.sh)
#   sbatch --dependency=afterok:$J --time=04:00:00 \
#         --export=ALL,STAGE=discover,DATASET=wb,BACKBONE=resnet18 scripts/discovery_cf.sh
#
#   STAGE     train     one fold run per array task: task = repeat * K + fold
#                       (array 0..K*|REPEATS|-1)
#             discover  one task: discover_biases.py for every repeat (cross-fit),
#                       the main ERM run's epoch_01.pt on train, and its best.pt
#                       on val; then scripts/compare_discovery.py
#   DATASET wb | celeba,  BACKBONE resnet18 | resnet50,  CELEBA_ROOT for celeba
#   K (5), REPEATS (fold-split seeds, "0 1"), SEED (training seed, 42),
#   EPOCHS (1), ROOT of the main ERM runs (main)
# Outputs: checkpoints_cf/<tag>/ (epoch_01.pt only), results_cf/<tag>/*.json

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
: "${STAGE:?}" "${DATASET:?}" "${BACKBONE:?}"
K=${K:-5}
read -ra REPEATS <<< "${REPEATS:-0 1}"
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-1}
ROOT=${ROOT:-main}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
TAG=${DATASET}_${BACKBONE}
CKPT_BASE=./checkpoints_cf/${TAG}
RES_BASE=./results_cf/${TAG}
RUN="uv run --locked python"

if [ "$DATASET" = wb ]; then
    TRAIN=(waterbirds_train.py --prompts sentences --concept_subspace_k 3)
    DATA_ARGS=()
elif [ "$DATASET" = celeba ]; then
    : "${CELEBA_ROOT:?set CELEBA_ROOT to the directory containing celeba/}"
    TRAIN=(train.py --target_attr Blond_Hair --concept_attr Male --batch_size 128
           --concept_subspace_k 9 --num_workers "$WORKERS")
    DATA_ARGS=(--celeba_root "$CELEBA_ROOT")
else
    echo "DATASET must be wb or celeba" >&2
    exit 1
fi

fold_dir() {  # $1 = split seed, $2 = fold
    ls -d "$CKPT_BASE/s$1_f$2"/*/seed_"$SEED"/*/ 2>/dev/null | tail -1
}
DISC=(--prompted none --report_truth "${DATA_ARGS[@]}" --num_workers "$WORKERS")

if [ "$STAGE" = train ]; then
    T=${SLURM_ARRAY_TASK_ID:?run as an array}
    S=${REPEATS[$((T / K))]}
    F=$((T % K))
    $RUN "${TRAIN[@]}" "${DATA_ARGS[@]}" --runs erm --backbone "$BACKBONE" \
        --no_lambda_schedules --epochs "$EPOCHS" --seeds "$SEED" \
        --n_folds "$K" --fold "$F" --fold_seed "$S" \
        --checkpoint_dir "$CKPT_BASE/s${S}_f${F}" --results_dir "$RES_BASE/runs/s${S}_f${F}"
    d=$(fold_dir "$S" "$F")
    rm -f "$d"best*.pt   # 1 epoch: only epoch_01.pt is used
elif [ "$STAGE" = discover ]; then
    mkdir -p "$RES_BASE"
    OUTS=()
    for S in "${REPEATS[@]}"; do
        DIRS=()
        for ((F = 0; F < K; F++)); do
            d=$(fold_dir "$S" "$F")
            [ -n "$d" ] || { echo "missing fold run s${S}_f${F}" >&2; exit 1; }
            DIRS+=("$d")
        done
        $RUN discover_biases.py "${DIRS[@]}" --split train --ckpt "epoch_$(printf %02d "$EPOCHS").pt" \
            "${DISC[@]}" --out "$RES_BASE/cf_k${K}_s${S}.json"
        OUTS+=("$RES_BASE/cf_k${K}_s${S}.json")
    done
    ERM=$(ls -d "./checkpoints_${ROOT}/${TAG}/erm"/*/seed_"$SEED"/*/ 2>/dev/null | tail -1)
    if [ -n "$ERM" ] && [ -f "${ERM}epoch_01.pt" ]; then
        $RUN discover_biases.py "$ERM" --split train --ckpt epoch_01.pt "${DISC[@]}" \
            --out "$RES_BASE/erm_ep1_train.json"
        OUTS+=("$RES_BASE/erm_ep1_train.json")
    else
        echo "no epoch_01.pt of the main ERM run (seed $SEED): skipping the epoch-1 train version" >&2
    fi
    if [ -n "$ERM" ]; then
        $RUN discover_biases.py "$ERM" --split val --ckpt best.pt "${DISC[@]}" \
            --out "$RES_BASE/erm_val.json"
        OUTS+=("$RES_BASE/erm_val.json")
    fi
    $RUN scripts/compare_discovery.py "${OUTS[@]}"
else
    echo "STAGE must be train or discover" >&2
    exit 1
fi

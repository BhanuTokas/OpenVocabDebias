#!/bin/bash
#SBATCH -J discovery_schemes
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Which held-out errors should discovery read?  Every model trains on 0.8n train
# examples (repeats included) and is scored only on train examples it did not
# draw; val is not used at all.  Each scheme is read at epoch 1 and at the last
# epoch of the same ERM runs:
#
#   single_holdout      subsample r0                  1 model, 20% of train scored
#   single_bootstrap    bootstrap r0                  1 model, ~45%
#   repeated_holdout    subsample r0..r4              5 models, ~67%
#   kfold               folds 0..4 of one split       5 models, 100%
#   repeated_bootstrap  bootstrap r0..r4              5 models, ~95%
#
# plus the main ERM run's epoch_01.pt on all of train (the errors behind the
# current prompts; it has seen those examples).  Submit from the repo root:
#
#   mkdir -p logs
#   J=$(sbatch --parsable --array=0-14 --time=04:00:00 \
#         --export=ALL,STAGE=train,DATASET=wb,BACKBONE=resnet18 scripts/discovery_schemes.sh)
#   sbatch --dependency=afterok:$J --time=06:00:00 \
#         --export=ALL,STAGE=discover,DATASET=wb,BACKBONE=resnet18 scripts/discovery_schemes.sh
#
#   STAGE     train     one run per array task: 0-4 subsample, 5-9 folds,
#                       10-14 bootstrap
#             discover  one task: discover_biases.py per scheme and epoch, then
#                       scripts/compare_discovery.py per epoch
#   DATASET wb | celeba,  BACKBONE resnet18 | resnet50,  CELEBA_ROOT for celeba
#   SEED (training seed, 42), FRAC (0.8), EPOCHS (10), ROOT of the main ERM runs (main)
#   VOCABS    lists to name every cell with (default builtin): any of builtin,
#             described (prompts/vocab/ birds or faces, union of the runs),
#             generic (union of the runs).  The train stage does not use it, so
#             STAGE=discover can be rerun alone on finished runs.
# Outputs: checkpoints_schemes/<tag>/ (epoch_01.pt and the last epoch only),
#          results_schemes/<tag>/{ep1,last}_<scheme>.json (builtin),
#          results_schemes/<tag>/<vocab>/ (the other lists)

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
SEED=${SEED:-42}
FRAC=${FRAC:-0.8}
EPOCHS=${EPOCHS:-10}
ROOT=${ROOT:-main}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
TAG=${DATASET}_${BACKBONE}
CKPT_BASE=./checkpoints_schemes/${TAG}
RES_BASE=./results_schemes/${TAG}
LAST=$(printf "epoch_%02d.pt" "$EPOCHS")
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

run_dir() {  # $1 = run label
    ls -d "$CKPT_BASE/$1"/*/seed_"$SEED"/*/ 2>/dev/null | tail -1
}

if [ "$STAGE" = train ]; then
    T=${SLURM_ARRAY_TASK_ID:?run as an array}
    R=$((T % 5))
    case $((T / 5)) in
        0) LABEL=subsample_r$R; SPLIT=(--resample subsample --resample_frac "$FRAC" --resample_seed "$R") ;;
        1) LABEL=fold_$R;       SPLIT=(--n_folds 5 --fold "$R" --fold_seed 0) ;;
        2) LABEL=bootstrap_r$R; SPLIT=(--resample bootstrap --resample_frac "$FRAC" --resample_seed "$R") ;;
        *) echo "array tasks are 0-14" >&2; exit 1 ;;
    esac
    $RUN "${TRAIN[@]}" "${DATA_ARGS[@]}" --runs erm --backbone "$BACKBONE" \
        --no_lambda_schedules --epochs "$EPOCHS" --seeds "$SEED" "${SPLIT[@]}" \
        --checkpoint_dir "$CKPT_BASE/$LABEL" --results_dir "$RES_BASE/runs/$LABEL"
    d=$(run_dir "$LABEL")
    for f in "$d"*.pt; do
        b=$(basename "$f")
        [ "$b" = epoch_01.pt ] || [ "$b" = "$LAST" ] || rm -f "$f"
    done
elif [ "$STAGE" = discover ]; then
    mkdir -p "$RES_BASE"
    BASE=$RES_BASE
    [ "$DATASET" = wb ] && DESCRIBED=birds || DESCRIBED=faces
    DISC0=(--split train --prompted none --report_truth "${DATA_ARGS[@]}" --num_workers "$WORKERS"
           --clip_cache "$CKPT_BASE/clip_train.pt")
    dirs_of() {  # run labels -> their run dirs, or fail
        local l d
        for l in "$@"; do
            d=$(run_dir "$l")
            [ -n "$d" ] || { echo "missing run $l" >&2; return 1; }
            echo "$d"
        done
    }
    for VOCAB in ${VOCABS:-builtin}; do
    case $VOCAB in
        builtin)   RES_BASE=$BASE; DISC=("${DISC0[@]}") ;;
        described) RES_BASE=$BASE/$VOCAB; DISC=("${DISC0[@]}" --vocab_file prompts/vocab/${DESCRIBED}_run?.txt) ;;
        generic)   RES_BASE=$BASE/$VOCAB; DISC=("${DISC0[@]}" --vocab_file prompts/vocab/generic_run?.txt) ;;
        *)         echo "unknown vocabulary $VOCAB" >&2; exit 1 ;;
    esac
    mkdir -p "$RES_BASE"
    for stage in ep1:epoch_01.pt "last:$LAST"; do
        NAME=${stage%:*}; CKPT=${stage#*:}
        OUTS=()
        for cell in "single_holdout:subsample_r0" "single_bootstrap:bootstrap_r0" \
            "repeated_holdout:subsample_r0 subsample_r1 subsample_r2 subsample_r3 subsample_r4" \
            "kfold:fold_0 fold_1 fold_2 fold_3 fold_4" \
            "repeated_bootstrap:bootstrap_r0 bootstrap_r1 bootstrap_r2 bootstrap_r3 bootstrap_r4"; do
            SCHEME=${cell%%:*}
            read -ra LABELS <<< "${cell#*:}"
            mapfile -t DIRS < <(dirs_of "${LABELS[@]}")
            if [ "${#DIRS[@]}" -ne "${#LABELS[@]}" ]; then
                echo "$SCHEME is missing runs: stopping" >&2
                exit 1
            fi
            OUT=$RES_BASE/${NAME}_${SCHEME}.json
            if $RUN discover_biases.py "${DIRS[@]}" --ckpt "$CKPT" "${DISC[@]}" --out "$OUT"; then
                OUTS+=("$OUT")
            else
                echo "$NAME $SCHEME gave no direction (too few errors?)" >&2
            fi
        done
        if [ "$NAME" = ep1 ]; then
            ERM=$(ls -d "./checkpoints_${ROOT}/${TAG}/erm"/*/seed_"$SEED"/*/ 2>/dev/null | tail -1)
            if [ -n "$ERM" ] && [ -f "${ERM}epoch_01.pt" ]; then
                OUT=$RES_BASE/ep1_in_sample.json
                $RUN discover_biases.py "$ERM" --ckpt epoch_01.pt "${DISC[@]}" --out "$OUT"
                OUTS+=("$OUT")
            else
                echo "no epoch_01.pt of the main ERM run: skipping the in-sample reference" >&2
            fi
        fi
        echo "== $VOCAB $NAME"
        if [ "${#OUTS[@]}" -gt 0 ]; then
            $RUN scripts/compare_discovery.py "${OUTS[@]}"
        fi
    done
    done
else
    echo "STAGE must be train or discover" >&2
    exit 1
fi

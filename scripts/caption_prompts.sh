#!/bin/bash
#SBATCH -J caption_prompts
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Prompts named from the BLIP-2 caption vocabulary (scripts/caption_vocab.sh)
# instead of the built-in list, for training with scripts/train_confirmed.sh.
# Same errors as prompts/{wb,celeba}_ep1_train_seed42.json: the main ERM run's
# epoch_01.pt on all of train, nothing prompted removed, the class-0 direction
# accepted.  Captions name the target freely, so the target filter runs first
# (discover_biases.py --target_words ... --target_filter): phrases containing a
# class word are dropped, and so are phrases whose score follows the class
# more than the bias direction.  The dropped phrases are printed and kept in
# the file ("filtered").
#
#   DATASET wb | celeba,  BACKBONE (resnet18),  CELEBA_ROOT for celeba
#   SEED (42), ROOT of the main ERM runs (main)
#   TARGET_WORDS (wb: "bird waterbird landbird", celeba: "blond hair")
# Output: results_captions/<tag>/captions_<tag>_ep1_seed<SEED>.json
#
# Submit from the repo root, then train on it once it exists:
#   mkdir -p logs
#   J=$(sbatch --parsable --export=ALL,DATASET=wb scripts/caption_prompts.sh)
#   sbatch --dependency=afterok:$J --export=ALL,DATASET=wb,\
#       PROMPTS=results_captions/wb_resnet18/captions_wb_resnet18_ep1_seed42.json \
#       scripts/train_confirmed.sh

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
: "${DATASET:?set DATASET=wb or DATASET=celeba}"
BACKBONE=${BACKBONE:-resnet18}
SEED=${SEED:-42}
ROOT=${ROOT:-main}
TAG=${DATASET}_${BACKBONE}
DATA_ARGS=()
if [ "$DATASET" = wb ]; then
    WORDS=${TARGET_WORDS:-bird waterbird landbird}
elif [ "$DATASET" = celeba ]; then
    : "${CELEBA_ROOT:?set CELEBA_ROOT to the directory containing celeba/}"
    DATA_ARGS=(--celeba_root "$CELEBA_ROOT")
    WORDS=${TARGET_WORDS:-blond hair}
else
    echo "DATASET must be wb or celeba" >&2
    exit 1
fi
VOCAB=results_captions/${TAG}/vocab.txt
[ -f "$VOCAB" ] || { echo "no $VOCAB: run scripts/caption_vocab.sh first" >&2; exit 1; }
ERM=$(ls -d "./checkpoints_${ROOT}/${TAG}/erm"/*/seed_"$SEED"/*/ 2>/dev/null | tail -1)
if [ -z "$ERM" ] || [ ! -f "${ERM}epoch_01.pt" ]; then
    echo "no epoch_01.pt of the main ERM run (seed $SEED) for $TAG" >&2
    exit 1
fi
CACHE=()
[ -d "checkpoints_schemes/$TAG" ] && CACHE=(--clip_cache "checkpoints_schemes/$TAG/clip_train.pt")
read -ra WORDS_A <<< "$WORDS"
uv run --locked python discover_biases.py "$ERM" --ckpt epoch_01.pt --split train \
    --prompted none --report_truth "${DATA_ARGS[@]}" "${CACHE[@]}" \
    --num_workers "${SLURM_CPUS_PER_TASK:-8}" --vocab_file "$VOCAB" \
    --target_words "${WORDS_A[@]}" --target_filter --accept_class 0 \
    --out "results_captions/${TAG}/captions_${TAG}_ep1_seed${SEED}.json"

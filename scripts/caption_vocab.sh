#!/bin/bash
#SBATCH -J caption_vocab
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# A vocabulary from BLIP-2 captions of the main ERM run's epoch-1 train errors
# and as many correct train images (build_caption_vocab.py), for naming with
# discover_biases.py --vocab_file.  Then, e.g.:
#   sbatch --export=ALL,STAGE=discover,VOCABS=captions,DATASET=wb,BACKBONE=resnet18 \
#       scripts/discovery_schemes.sh
#
#   DATASET wb | celeba,  BACKBONE resnet18 | resnet50,  CELEBA_ROOT for celeba
#   SEED (42), ROOT of the main ERM runs (main), PER_CLASS (1000),
#   CAPTIONER (Salesforce/blip2-opt-2.7b; about 15 GB downloaded once to the HF cache)
# Output: results_captions/<tag>/{vocab.txt,captions.jsonl,b2t.json}
#
# Submit from the repo root:
#   mkdir -p logs
#   sbatch --export=ALL,DATASET=wb,BACKBONE=resnet18 scripts/caption_vocab.sh

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1              # keep ~/.local packages out of the env
export UV_PYTHON_PREFERENCE=only-managed  # uv's Python, not Sol's system one
export PYTHONUNBUFFERED=1             # live logs
cd "$SLURM_SUBMIT_DIR"                 # not dirname $0: SLURM runs a spooled copy
if [ ! -f build_caption_vocab.py ]; then
    echo "Submit from the OpenVocabDebias repo root (build_caption_vocab.py not found in $PWD)" >&2
    exit 1
fi
: "${DATASET:?}" "${BACKBONE:?}"
SEED=${SEED:-42}
ROOT=${ROOT:-main}
TAG=${DATASET}_${BACKBONE}
DATA_ARGS=()
if [ "$DATASET" = celeba ]; then
    : "${CELEBA_ROOT:?set CELEBA_ROOT to the directory containing celeba/}"
    DATA_ARGS=(--celeba_root "$CELEBA_ROOT")
fi
ERM=$(ls -d "./checkpoints_${ROOT}/${TAG}/erm"/*/seed_"$SEED"/*/ 2>/dev/null | tail -1)
if [ -z "$ERM" ] || [ ! -f "${ERM}epoch_01.pt" ]; then
    echo "no epoch_01.pt of the main ERM run (seed $SEED) for $TAG" >&2
    exit 1
fi
uv run --locked python build_caption_vocab.py "$ERM" --ckpt epoch_01.pt "${DATA_ARGS[@]}" \
    --captioner "${CAPTIONER:-Salesforce/blip2-opt-2.7b}" --per_class "${PER_CLASS:-1000}" \
    --num_workers "${SLURM_CPUS_PER_TASK:-8}" --out_dir "results_captions/${TAG}"

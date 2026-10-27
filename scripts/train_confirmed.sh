#!/bin/bash
#SBATCH -J train_confirmed
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Discovery experiment, stage 2: retrain with the discovered prompt pairs a user
# confirmed ("accept": true in a discover_biases.py proposals file), using the
# same recipe as everywhere else: whitened repulse λ10 + grw_pseudo, ResNet-18,
# one seed per array task.
#
#   DATASET=wb      no hand-written prompts at all (--prompts none): the
#                   confirmed pairs replace the place prompts.
#                   Reference: hand-written sentences 0.859, words 0.835.
#   DATASET=celeba  Young prompts + the confirmed pairs (e.g. the discovered
#                   Male pair), evaluated over all 8 (Blond, Male, Young) groups.
#                   Reference: Male + Young hand-prompted 0.827; Young only:
#                   scripts/celeba_young_only.sh.
#
# PROMPTS  the reviewed proposals file (required)
#
# Submit from the repo root:
#   mkdir -p logs
#   DATASET=wb PROMPTS=results/discovery/wb_erm_seed42.json sbatch scripts/train_confirmed.sh
#   DATASET=celeba PROMPTS=results/discovery/celeba_young_only_seed42.json \
#       CELEBA_ROOT=/data/hkerner/Datasets/CelebA sbatch scripts/train_confirmed.sh

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
: "${PROMPTS:?set PROMPTS to the reviewed proposals file}"
if ! grep -q '"accept": true' "$PROMPTS"; then
    echo "No accepted proposals in $PROMPTS: review it and set \"accept\": true first" >&2
    exit 1
fi

read -ra SEEDS <<< "${SEEDS:-42 123 456}"   # override, e.g. SEEDS="789 1011" with --array=0-1
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
STEM=$(basename "$PROMPTS" .json)
RECIPE=(--runs repulse_only --repulse whitened --lambda_repulse 10 --group_reweight pseudo
        --no_lambda_schedules --seeds "$SEED" --extra_prompts "$PROMPTS")

RUN="uv run --locked python"
$RUN -c "import numpy, torch, sys; print(sys.executable); print('numpy', numpy.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [ "$DATASET" = wb ]; then
    CKPT_DIR=./checkpoints_waterbirds/confirmed_${STEM}
    RESULTS_DIR=./results_waterbirds/confirmed_${STEM}/seed_${SEED}
    $RUN waterbirds_train.py "${RECIPE[@]}" --prompts none --concept_subspace_k 3 \
        --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR"
    $RUN eval_checkpoints.py "$CKPT_DIR"/*/seed_"$SEED" --out "$RESULTS_DIR/eval_summary.csv"
elif [ "$DATASET" = celeba ]; then
    : "${CELEBA_ROOT:?set CELEBA_ROOT to the directory containing celeba/ when submitting}"
    CKPT_DIR=./checkpoints/confirmed_${STEM}
    RESULTS_DIR=./results/confirmed_${STEM}/seed_${SEED}
    $RUN train.py "${RECIPE[@]}" --celeba_root "$CELEBA_ROOT" --target_attr Blond_Hair \
        --concept_attr Male --extra_concept_attrs Young --prompt_attrs Young \
        --epochs 10 --batch_size 128 --concept_subspace_k 9 --num_workers "$WORKERS" \
        --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR"
    $RUN eval_checkpoints.py "$CKPT_DIR"/*/seed_"$SEED" --celeba_root "$CELEBA_ROOT" \
        --num_workers "$WORKERS" --out "$RESULTS_DIR/eval_summary.csv"
else
    echo "DATASET must be wb or celeba" >&2
    exit 1
fi

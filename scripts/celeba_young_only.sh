#!/bin/bash
#SBATCH -J celeba_young_only
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Discovery experiment, stage 1 (CelebA, Blond_Hair): only Young is prompted,
# Male — the stronger bias — is left for discovery.  Evaluation is still over
# all 8 true (Blond, Male, Young) groups (--extra_concept_attrs Young), so the
# WGA here is the "Male unaddressed" lower bound.
#
#   train: whitened repulse λ10 + grw_pseudo, prompts for Young only
#          (--prompt_attrs Young); ResNet-18, 10 epochs, k=9, one seed per task
#   then:  discover_biases.py on the trained model's val errors, with the
#          Young prompts removed → results/discovery/celeba_young_only_seed<S>.json
#
# Reference: Male + Young prompted (same recipe) 0.827; Male unaddressed is
# expected to be much lower.  Locally, with a CLIP-probe stand-in for the model,
# discovery found Male ("a woman, a girl, a dress" vs "groom, a grandfather,
# an old man, a man, a boy"), and as prompts those labelled Male with worst
# cell 0.99.
#
# Next: review the proposals, set "accept": true, and run
#   DATASET=celeba PROMPTS=results/discovery/<file>.json sbatch scripts/train_confirmed.sh
#
# Submit from the repo root (CELEBA_ROOT is the dir that CONTAINS celeba/):
#   mkdir -p logs
#   CELEBA_ROOT=/data/hkerner/Datasets/CelebA sbatch scripts/celeba_young_only.sh

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
: "${CELEBA_ROOT:?set CELEBA_ROOT to the directory containing celeba/ when submitting}"
if [ ! -f "$CELEBA_ROOT/celeba/list_attr_celeba.txt" ]; then
    echo "CelebA not found under CELEBA_ROOT=$CELEBA_ROOT" >&2
    exit 1
fi

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
WORKERS=${SLURM_CPUS_PER_TASK:-8}

RUN="uv run --locked python"
$RUN -c "import numpy, torch, sys; print(sys.executable); print('numpy', numpy.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

CKPT_DIR=./checkpoints/discovery
RESULTS_DIR=./results/discovery/young_only/seed_${SEED}
mkdir -p results/discovery

$RUN train.py --runs repulse_only --repulse whitened --lambda_repulse 10 \
    --group_reweight pseudo --no_lambda_schedules \
    --celeba_root "$CELEBA_ROOT" --target_attr Blond_Hair --concept_attr Male \
    --extra_concept_attrs Young --prompt_attrs Young \
    --epochs 10 --batch_size 128 --concept_subspace_k 9 --seeds "$SEED" \
    --num_workers "$WORKERS" --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR"

RUN_DIR=$(ls -d "$CKPT_DIR"/*_pa-Young*/seed_"$SEED"/*/ | tail -1)
$RUN eval_checkpoints.py "$RUN_DIR" --celeba_root "$CELEBA_ROOT" \
    --num_workers "$WORKERS" --out "$RESULTS_DIR/eval_summary.csv"
$RUN discover_biases.py "$RUN_DIR" --ckpt best_pseudo_wga.pt --celeba_root "$CELEBA_ROOT" \
    --report_truth --num_workers "$WORKERS" \
    --out results/discovery/celeba_young_only_seed${SEED}.json

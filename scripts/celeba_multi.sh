#!/bin/bash
#SBATCH -J celeba_multi
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Multi-attribute CelebA: target Blond_Hair, spurious Male AND Young
# (--extra_concept_attrs Young).  The concept is the joint code Male + 2*Young,
# so groups, pseudo-labels and worst-group accuracy run over all 8
# (Blond_Hair, Male, Young) cells; per-attribute WGA and probes are reported
# too (attr0 = Male, attr1 = Young).  Smallest group: 463 train / 70 val / 62 test.
# Adding Eyeglasses or Wearing_Hat leaves groups with 0-10 val/test images.
#
# ResNet-18, 10 epochs, batch 128, k=9 per attribute (Young has 4 prompt pairs,
# so 13 concept scores), one seed per array task:
#   erm
#   erm_grw_pseudo        8 CLIP zero-shot pseudo-groups (val joint acc 0.84)
#   erm_grw_oracle        8 true groups (reference only)
#   erm_grw_propensity    w = 1/p(y|z) on the 13 whitened CLIP scores, no groups
#   whitened repulse λ10 + grw_pseudo / + grw_propensity
# lambda 10: the decorrelation signal per balanced batch is 0.084 with 13 scores
# vs 0.089 for Male alone, where λ10 gave the best CelebA result.
#
# Submit from the repo root (CELEBA_ROOT is the dir that CONTAINS celeba/):
#   mkdir -p logs && CELEBA_ROOT=/data/hkerner/Datasets/CelebA sbatch scripts/celeba_multi.sh

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1              # keep ~/.local packages out of the env
export UV_PYTHON_PREFERENCE=only-managed  # uv's Python, not Sol's system one
export PYTHONUNBUFFERED=1             # live logs
cd "$SLURM_SUBMIT_DIR"                 # not dirname $0: SLURM runs a spooled copy
if [ ! -f train.py ]; then
    echo "Submit from the OpenVocabDebias repo root (train.py not found in $PWD)" >&2
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

CKPT_DIR=./checkpoints/celeba_multi
RESULTS_DIR=./results/celeba_multi/seed_${SEED}
COMMON=(--celeba_root "$CELEBA_ROOT" --target_attr Blond_Hair --concept_attr Male
        --extra_concept_attrs Young
        --epochs 10 --batch_size 128 --concept_subspace_k 9 --seeds "$SEED"
        --num_workers "$WORKERS" --no_lambda_schedules
        --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR")
WREP=(--runs repulse_only --repulse whitened --lambda_repulse 10)

$RUN train.py --runs erm "${COMMON[@]}"
$RUN train.py --runs erm --group_reweight pseudo "${COMMON[@]}"
$RUN train.py --runs erm --group_reweight oracle "${COMMON[@]}"
$RUN train.py --runs erm --group_reweight propensity "${COMMON[@]}"
$RUN train.py "${WREP[@]}" --group_reweight pseudo "${COMMON[@]}"
$RUN train.py "${WREP[@]}" --group_reweight propensity "${COMMON[@]}"

$RUN eval_checkpoints.py "$CKPT_DIR"/*/seed_"$SEED" --celeba_root "$CELEBA_ROOT" \
    --num_workers "$WORKERS" --out "$RESULTS_DIR/eval_summary.csv"

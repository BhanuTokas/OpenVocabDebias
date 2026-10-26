#!/bin/bash
#SBATCH -J wb_core_promote
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Core-concept promotion on Waterbirds, fully discovered: the bias prompts come
# from discover_biases.py (forest... vs ocean..., confirmed) and the core
# direction from the remaining errors of the stage-1 model trained on them.
#
#   stage 1  the train_confirmed.sh DATASET=wb runs (WGA 0.836 ± 0.012), one per seed
#   core     discover_core.py on that seed's best_pseudo_wga.pt (val errors within
#            class x pseudo-place cells; pooled direction accepted)
#   stage 2  same recipe + --core_dirs, λ_promote ∈ {1, 3, 10}; task = λ x seed
#
# Locally (ERM R18, held-out test), the pooled core direction was background-free
# (corr with place 0.05) and named "a small bird, a songbird, perched" vs
# "a long neck, a long tail, long thin legs, webbed feet".
#
# PROMPTS  the confirmed bias proposals (default results/discovery/wb_erm_seed42.json)
# STAGE1   stage-1 checkpoint dir (default ./checkpoints_waterbirds/confirmed_<PROMPTS stem>)
#
# Submit from the repo root:   mkdir -p logs && sbatch scripts/wb_core_promote.sh

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1              # keep ~/.local packages out of the env
export UV_PYTHON_PREFERENCE=only-managed  # uv's Python, not Sol's system one
export PYTHONUNBUFFERED=1             # live logs
cd "$SLURM_SUBMIT_DIR"                 # not dirname $0: SLURM runs a spooled copy
if [ ! -f discover_core.py ]; then
    echo "Submit from the OpenVocabDebias repo root (discover_core.py not found in $PWD)" >&2
    exit 1
fi
PROMPTS=${PROMPTS:-results/discovery/wb_erm_seed42.json}
STEM=$(basename "$PROMPTS" .json)
STAGE1=${STAGE1:-./checkpoints_waterbirds/confirmed_${STEM}}

SEEDS=(42 123 456)
LAMBDAS=(1 3 10)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
LAM=${LAMBDAS[$((SLURM_ARRAY_TASK_ID / 3))]}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
RUN_DIR=$(ls -d "$STAGE1"/*/seed_"$SEED"/*/ | tail -1)
if [ ! -f "$RUN_DIR/best_pseudo_wga.pt" ]; then
    echo "No stage-1 best_pseudo_wga.pt under $RUN_DIR" >&2
    exit 1
fi
CORE=results/discovery/core_${STEM}_seed${SEED}_lpro${LAM}.json
CKPT_DIR=./checkpoints_waterbirds/core_promote_${STEM}
RESULTS_DIR=./results_waterbirds/core_promote_${STEM}/lpro${LAM}/seed_${SEED}
mkdir -p results/discovery

RUN="uv run --locked python"
$RUN -c "import numpy, torch, sys; print(sys.executable); print('numpy', numpy.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
$RUN discover_core.py "$RUN_DIR" --ckpt best_pseudo_wga.pt --vocab birds --report_truth \
    --num_workers "$WORKERS" --out "$CORE"
$RUN waterbirds_train.py --runs repulse_only --repulse whitened --lambda_repulse 10 \
    --group_reweight pseudo --no_lambda_schedules --prompts none --extra_prompts "$PROMPTS" \
    --concept_subspace_k 3 --core_dirs "$CORE" --lambda_promote "$LAM" --seeds "$SEED" \
    --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR"
$RUN eval_checkpoints.py "$CKPT_DIR"/*_lpro"$LAM"_*/seed_"$SEED" --out "$RESULTS_DIR/eval_summary.csv"

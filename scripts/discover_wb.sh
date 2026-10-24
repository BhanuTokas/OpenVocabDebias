#!/bin/bash
#SBATCH -J discover_wb
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Bias discovery on Waterbirds with nothing prompted: the val errors of an
# existing ERM ResNet-18 (best.pt, i.e. selected without any prompts) give
# within-class error directions in CLIP space, named with the scene /
# attribute vocabulary (discover_biases.py).  Locally, on a July ERM run, the
# class-0 direction came out as "a forest, a garden, a meadow, the woods, tree
# branches" vs "the ocean, the sea, a shore, a beach, a harbor" (corr with the
# true place 0.64); as prompts it labels place with val accuracy 0.89 and worst
# cell 0.80 (hand-written sentences 0.92 / 0.86, words 0.86 / 0.70).
#
# Output: results/discovery/wb_erm_seed<S>.json — review it, set "accept": true
# on the proposals that name a bias (and delete wrong phrases), then run
# scripts/train_confirmed.sh with that file.
#
# RUN_DIR  ERM run dir to analyse (default: the sentences ERM run, seed 42)
#
# Submit from the repo root:   mkdir -p logs && sbatch scripts/discover_wb.sh

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

RUN_DIR=${RUN_DIR:-$(ls -d ./checkpoints_waterbirds/sentences/erm_psentences/seed_42/*/ | tail -1)}
if [ ! -f "$RUN_DIR/best.pt" ]; then
    echo "No best.pt under RUN_DIR=$RUN_DIR" >&2
    exit 1
fi
mkdir -p results/discovery

uv run --locked python discover_biases.py "$RUN_DIR" --ckpt best.pt --prompted none \
    --report_truth --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
    --out results/discovery/wb_erm_seed42.json

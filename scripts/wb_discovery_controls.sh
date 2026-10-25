#!/bin/bash
#SBATCH -J wb_discovery_controls
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Controls for the Waterbirds discovery result (discovered prompts + whitened
# repulse λ10 + grw_pseudo: WGA 0.836 ± 0.012; ERM 0.693).  Task = control x seed:
#
#   tasks 0-2  grw_only    discovered prompts, grw_pseudo only (no repulse):
#                          how much of the gain is the discovered pseudo-groups.
#                          Hand-written sentences reference: 0.793.
#   tasks 3-5  irrelevant  same recipe as the discovered run, but with an
#                          unrelated prompt pair ($IRRELEVANT): if WGA falls
#                          toward ERM, the gain needs the right concept, not
#                          just the machinery running on any prompt.
#
# PROMPTS     the confirmed proposals file (default results/discovery/wb_erm_seed42.json)
# IRRELEVANT  control prompt pair (default prompts/control_irrelevant.json, checked in)
#
# Submit from the repo root:   mkdir -p logs && sbatch scripts/wb_discovery_controls.sh

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
PROMPTS=${PROMPTS:-results/discovery/wb_erm_seed42.json}
IRRELEVANT=${IRRELEVANT:-prompts/control_irrelevant.json}
for f in "$PROMPTS" "$IRRELEVANT"; do
    if ! grep -q '"accept": true' "$f"; then
        echo "No accepted prompt pairs in $f" >&2
        exit 1
    fi
done

SEEDS=(42 123 456)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
COMMON=(--prompts none --concept_subspace_k 3 --no_lambda_schedules --seeds "$SEED"
        --group_reweight pseudo)

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
    NAME=grw_only
    ARGS=(--runs erm --extra_prompts "$PROMPTS")
else
    NAME=irrelevant
    ARGS=(--runs repulse_only --repulse whitened --lambda_repulse 10 --extra_prompts "$IRRELEVANT")
fi
CKPT_DIR=./checkpoints_waterbirds/discovery_controls/$NAME
RESULTS_DIR=./results_waterbirds/discovery_controls/$NAME/seed_${SEED}

RUN="uv run --locked python"
$RUN -c "import numpy, torch, sys; print(sys.executable); print('numpy', numpy.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
$RUN waterbirds_train.py "${ARGS[@]}" "${COMMON[@]}" \
    --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR"
$RUN eval_checkpoints.py "$CKPT_DIR"/*/seed_"$SEED" --out "$RESULTS_DIR/eval_summary.csv"

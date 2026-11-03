#!/bin/bash
#SBATCH -J wb_propensity
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Continuous inverse-propensity reweighting (--group_reweight propensity):
# w = 1 / p(y | z), with p(y | z) a logistic model on the whitened CLIP concept
# scores z.  Label-free, no pseudo-groups or thresholds, works for any number
# of concepts.  Waterbirds, ResNet-18, k=3, sentence prompts, 3 seeds.
#
# On the train set it drops corr(y, z_place) from -0.80 to about -0.05 and puts
# 0.166 / 0.185 of the mass on the two minority groups (CLIP pseudo-groups:
# 0.150 / 0.178), effective sample size 15% (pseudo-groups 18%).
#
#   erm_psentences_grw_propensity                         cf. grw_pseudo 0.793
#   repulse_only_psentences_nosched_wrep_lrep10_grw_propensity   cf. 0.859
#
# Submit from the repo root:   mkdir -p logs && sbatch scripts/wb_propensity.sh

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1              # keep ~/.local packages out of the env
export UV_PYTHON_PREFERENCE=only-managed  # uv's Python, not Sol's system one
export PYTHONUNBUFFERED=1             # live logs
cd "$SLURM_SUBMIT_DIR"                 # not dirname $0: SLURM runs a spooled copy
if [ ! -f waterbirds_train.py ]; then
    echo "Submit from the OpenVocabDebias repo root (waterbirds_train.py not found in $PWD)" >&2
    exit 1
fi

RUN="uv run --locked python"
$RUN -c "import numpy, torch, sys; print(sys.executable); print('numpy', numpy.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

CKPT_DIR=./checkpoints_waterbirds/propensity
RESULTS_DIR=./results_waterbirds/propensity
COMMON=(--seeds 42 123 456 --concept_subspace_k 3 --prompts sentences
        --group_reweight propensity
        --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR")

$RUN waterbirds_train.py --runs erm "${COMMON[@]}"
$RUN waterbirds_train.py --runs repulse_only --no_lambda_schedules --repulse whitened \
    --lambda_repulse 10 "${COMMON[@]}"

$RUN eval_checkpoints.py "$CKPT_DIR" --out "$RESULTS_DIR/eval_summary.csv"

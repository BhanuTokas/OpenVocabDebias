#!/bin/bash
#SBATCH -J wb_sentences
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Waterbirds (ResNet-18, k=3, 3 seeds, constant lambdas) with sentence prompts
# ("a photo of a bird over the ocean" / "... in a bamboo forest") instead of the
# single words.  CLIP zero-shot place accuracy: val 0.924 vs 0.860, train 0.973
# vs 0.951.  The prompts drive the subspace, the whitened eraser, the training
# pseudo-groups and the val pseudo-groups behind best_pseudo_wga.pt.
#
#   erm_psentences                         ERM; only checkpoint selection changes
#   erm_psentences_grw_pseudo              cf. words 0.763 ± 0.030
#   repulse_only_psentences_nosched_wrep_lrep10_grw_pseudo   cf. words 0.835 ± 0.007
#
# Submit from the repo root:   mkdir -p logs && sbatch scripts/wb_sentences.sh

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

CKPT_DIR=./checkpoints_waterbirds/sentences
RESULTS_DIR=./results_waterbirds/sentences
COMMON=(--seeds 42 123 456 --concept_subspace_k 3 --prompts sentences
        --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR")

$RUN waterbirds_train.py --runs erm "${COMMON[@]}"
$RUN waterbirds_train.py --runs erm --group_reweight pseudo "${COMMON[@]}"
$RUN waterbirds_train.py --runs repulse_only --no_lambda_schedules --repulse whitened \
    --group_reweight pseudo --lambda_repulse 10 "${COMMON[@]}"

$RUN eval_checkpoints.py "$CKPT_DIR" --out "$RESULTS_DIR/eval_summary.csv"

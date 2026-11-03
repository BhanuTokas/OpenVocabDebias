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

# Core-concept promotion on Waterbirds.  The bias prompts are the discovered,
# confirmed ones (discover_biases.py: forest... vs ocean...); the promoted core
# concept depends on ARM:
#
#   discovered   discover_core.py on the stage-1 model of the same seed (the
#                train_confirmed.sh DATASET=wb runs, WGA 0.836 ± 0.012): pooled
#                mean(correct) - mean(errors) within class x pseudo-place cells
#   class_names  prompts/core_class_names.json ("waterbird" vs "landbird")
#   morphology   prompts/core_morphology.json (long-necked wading bird... vs
#                small songbird perched...)
#
# Every arm is scored residual to the bias subspace, where all three are
# background-free (B/32 test: corr with place 0.05 / -0.02 / -0.06; corr with y
# 0.43 / 0.55 / 0.43).  class_names is a 1-D bias-free CLIP teacher: the control
# that says whether discovery adds anything to promotion.
# Task = λ_promote ∈ {1, 3, 10} x seed; same recipe as stage 1 otherwise.
#
# ARM      discovered (default) | class_names | morphology
# PROMPTS  the confirmed bias proposals (default results/discovery/wb_erm_seed42.json)
# STAGE1   stage-1 checkpoint dir (default ./checkpoints_waterbirds/confirmed_<PROMPTS stem>)
#
# Submit from the repo root, one array per arm:
#   mkdir -p logs
#   for a in discovered class_names morphology; do ARM=$a sbatch scripts/wb_core_promote.sh; done

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
ARM=${ARM:-discovered}
PROMPTS=${PROMPTS:-results/discovery/wb_erm_seed42.json}
STEM=$(basename "$PROMPTS" .json)
STAGE1=${STAGE1:-./checkpoints_waterbirds/confirmed_${STEM}}

read -ra SEEDS <<< "${SEEDS:-42 123 456}"   # override with matching --array=0-(seeds x lambdas - 1)
read -ra LAMBDAS <<< "${LAMBDAS:-1 3 10}"
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))]}
LAM=${LAMBDAS[$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))]}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
CKPT_DIR=./checkpoints_waterbirds/core_promote_${STEM}/${ARM}
RESULTS_DIR=./results_waterbirds/core_promote_${STEM}/${ARM}/lpro${LAM}/seed_${SEED}

RUN="uv run --locked python"
$RUN -c "import numpy, torch, sys; print(sys.executable); print('numpy', numpy.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

case "$ARM" in
    discovered)
        RUN_DIR=$(ls -d "$STAGE1"/*/seed_"$SEED"/*/ | tail -1)
        if [ ! -f "$RUN_DIR/best_pseudo_wga.pt" ]; then
            echo "No stage-1 best_pseudo_wga.pt under $RUN_DIR" >&2
            exit 1
        fi
        mkdir -p results/discovery
        CORE=results/discovery/core_${STEM}_seed${SEED}_lpro${LAM}.json
        $RUN discover_core.py "$RUN_DIR" --ckpt best_pseudo_wga.pt --vocab birds --report_truth \
            --num_workers "$WORKERS" --out "$CORE"
        CORE_ARGS=(--core_dirs "$CORE")
        ;;
    class_names | morphology)
        CORE_ARGS=(--core_prompts "prompts/core_${ARM}.json")
        ;;
    *)
        echo "ARM must be discovered, class_names or morphology" >&2
        exit 1
        ;;
esac

$RUN waterbirds_train.py --runs repulse_only --repulse whitened --lambda_repulse 10 \
    --group_reweight pseudo --no_lambda_schedules --prompts none --extra_prompts "$PROMPTS" \
    --concept_subspace_k 3 "${CORE_ARGS[@]}" --lambda_promote "$LAM" --seeds "$SEED" \
    --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR"
$RUN eval_checkpoints.py "$CKPT_DIR"/*_lpro"$LAM"_*/seed_"$SEED" --out "$RESULTS_DIR/eval_summary.csv"

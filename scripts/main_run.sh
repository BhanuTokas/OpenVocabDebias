#!/bin/bash
#SBATCH -J main_run
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Main tables: one training run (METHOD x DATASET x BACKBONE) per array task,
# one seed per task (SEEDS, matching --array).  Submitted by
# scripts/submit_main.sh, which also sets --time and the stage-1 -> stage-2
# dependencies.  Every run is selected label-free (best_pseudo_wga.pt, CLIP
# zero-shot pseudo-groups of the same prompts) and evaluated on test.
#
#   DATASET   wb (sentences prompts, k=3) | celeba (Blond_Hair, Male, k=9,
#             10 epochs, batch 128)
#   BACKBONE  resnet18 | resnet50
#   METHOD    erm         ERM (also JTT stage 1 and the DFR / AFR features)
#             grw         CLIP pseudo-group reweighting (ablation)
#             ours        whitened decorrelation λ (LAMBDA) + pseudo-group reweighting
#             dro_pseudo  GroupDRO on CLIP pseudo-groups (B2T / ViG-Bias mitigation)
#             dro_oracle  GroupDRO on true groups (reference)
#             bam1        BAM stage 1, auxiliary coefficient AUX (2 epochs)
#             jtt         JTT stage 2: errors of the ERM run's epoch_01.pt, xUP
#             bam2        BAM stage 2: errors of the bam1 AUX run's epoch_01.pt, xUP
#   LAMBDA    ours: repulse weight (default 10)
#   AUX, UP   bam1 / bam2 / jtt hyperparameters
#   ROOT      output namespace (default main): checkpoints_<ROOT>/, results_<ROOT>/

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1              # keep ~/.local packages out of the env
export UV_PYTHON_PREFERENCE=only-managed  # uv's Python, not Sol's system one
export PYTHONUNBUFFERED=1             # live logs
cd "$SLURM_SUBMIT_DIR"                 # not dirname $0: SLURM runs a spooled copy
if [ ! -f error_set.py ]; then
    echo "Submit from the OpenVocabDebias repo root (error_set.py not found in $PWD)" >&2
    exit 1
fi
: "${DATASET:?}" "${BACKBONE:?}" "${METHOD:?}"
read -ra SEEDS <<< "${SEEDS:-42 123 456 789 1011}"
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
ROOT=${ROOT:-main}
TAG=${DATASET}_${BACKBONE}
CKPT_BASE=./checkpoints_${ROOT}/${TAG}
RES_BASE=./results_${ROOT}/${TAG}

if [ "$DATASET" = wb ]; then
    TRAIN=(waterbirds_train.py --prompts sentences --concept_subspace_k 3)
    DATA_ARGS=()
elif [ "$DATASET" = celeba ]; then
    : "${CELEBA_ROOT:?set CELEBA_ROOT to the directory containing celeba/}"
    TRAIN=(train.py --target_attr Blond_Hair --concept_attr Male --epochs 10 --batch_size 128
           --concept_subspace_k 9 --num_workers "$WORKERS")
    DATA_ARGS=(--celeba_root "$CELEBA_ROOT")
else
    echo "DATASET must be wb or celeba" >&2
    exit 1
fi

RUN="uv run --locked python"
stage1_dir() {  # $1 = method dir of the stage-1 run
    ls -d "$CKPT_BASE/$1"/*/seed_"$SEED"/*/ 2>/dev/null | tail -1
}
error_set() {   # $1 = stage-1 method dir, $2 = output npz
    local d
    d=$(stage1_dir "$1")
    if [ -z "$d" ] || [ ! -f "$d/epoch_01.pt" ]; then
        echo "No stage-1 epoch_01.pt under $CKPT_BASE/$1 for seed $SEED" >&2
        exit 1
    fi
    $RUN error_set.py "$d" --ckpt epoch_01.pt "${DATA_ARGS[@]}" --num_workers "$WORKERS" --out "$2"
}

ES_DIR=$RES_BASE/error_sets
mkdir -p "$ES_DIR"
case "$METHOD" in
    erm)        NAME=erm;        ARGS=(--runs erm) ;;
    grw)        NAME=grw;        ARGS=(--runs erm --group_reweight pseudo) ;;
    ours)       NAME=ours_l${LAMBDA:-10}
                ARGS=(--runs repulse_only --repulse whitened --lambda_repulse "${LAMBDA:-10}"
                      --group_reweight pseudo) ;;
    dro_pseudo) NAME=dro_pseudo; ARGS=(--runs erm --group_dro pseudo) ;;
    dro_oracle) NAME=dro_oracle; ARGS=(--runs erm --group_dro oracle) ;;
    bam1)       NAME=bam1_aux${AUX:?}; ARGS=(--runs erm --bam_aux "$AUX" --epochs 2) ;;
    jtt)        NAME=jtt_up${UP:?}
                ES=$ES_DIR/jtt_seed${SEED}_up${UP}.npz
                error_set erm "$ES"
                ARGS=(--runs erm --error_set "$ES" --upweight "$UP") ;;
    bam2)       NAME=bam2_aux${AUX:?}_up${UP:?}
                ES=$ES_DIR/bam_aux${AUX}_seed${SEED}_up${UP}.npz
                error_set "bam1_aux${AUX}" "$ES"
                ARGS=(--runs erm --error_set "$ES" --upweight "$UP") ;;
    *)          echo "unknown METHOD=$METHOD" >&2; exit 1 ;;
esac
CKPT_DIR=$CKPT_BASE/$NAME
RESULTS_DIR=$RES_BASE/$NAME/seed_${SEED}

$RUN -c "import numpy, torch, sys; print(sys.executable); print('numpy', numpy.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
$RUN "${TRAIN[@]}" "${DATA_ARGS[@]}" "${ARGS[@]}" --backbone "$BACKBONE" --no_lambda_schedules \
    --seeds "$SEED" --checkpoint_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR"
if [ "$METHOD" != bam1 ]; then
    $RUN eval_checkpoints.py "$CKPT_DIR"/*/seed_"$SEED" "${DATA_ARGS[@]}" \
        --out "$RESULTS_DIR/eval_summary.csv"
fi

# Keep the per-epoch val metrics (label-free hyperparameter selection reads
# val_pseudo_wga), then drop the epoch checkpoints (~1.4 GB per R18 run, ~3 GB
# per R50 run with optimizer state); erm and bam1 keep epoch_01.pt for stage 2.
for d in "$CKPT_DIR"/*/seed_"$SEED"/*/; do
    mkdir -p "$RESULTS_DIR"
    $RUN - "$d" "$RESULTS_DIR/val_history.json" <<'EOF'
import glob, json, sys, torch
d, out = sys.argv[1], sys.argv[2]
hist = {}
for f in sorted(glob.glob(f"{d}/epoch_*.pt")):
    s = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
    hist[s["epoch"]] = {k: s.get(k) for k in ("val_acc", "val_wga", "val_pseudo_wga")}
json.dump({"run_dir": d, "epochs": hist}, open(out, "w"), indent=1)
print(f"val history of {len(hist)} epochs -> {out}")
EOF
    for f in "$d"epoch_*.pt; do
        if [ "$(basename "$f")" = epoch_01.pt ] && { [ "$METHOD" = erm ] || [ "$METHOD" = bam1 ]; }; then
            continue
        fi
        rm -f "$f"
    done
done

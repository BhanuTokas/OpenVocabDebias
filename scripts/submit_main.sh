#!/bin/bash
# Submit the main-table runs (run on the login node from the repo root, NOT with
# sbatch):   PHASE=grid bash scripts/submit_main.sh
#
#   PHASE=grid    seeds 42 123 456: every method, with the JTT / BAM grids
#   PHASE=fix     seeds 42 123 456: only what changed after the first grid --
#                 CelebA GroupDRO (OOM fix), JTT (underfit stage 1, LR1 grid),
#                 BAM (SGD auxiliaries), GroupDRO with strong weight decay, and
#                 DFR / AFR on the ours and grw models.  Move the old BAM
#                 results aside first (same names), see the note printed below.
#   PHASE=bam     seeds 42 123 456: the BAM grid only, stage 1 now BAM_T epochs
#                 (default 3) with the error set at its last epoch (new names
#                 bam1_aux<A>_T<T> / bam2_aux<A>_T<T>_up<U>, no collision)
#   PHASE=extend  seeds 789 1011: every method at the chosen JTT / BAM values
#                 (JTT_LR_<tag>, JTT_UP_<tag>, BAM_AUX_<tag>, BAM_UP_<tag>, e.g.
#                 JTT_UP_wb_resnet18=50) picked label-free from the grid:
#                 eval "$(python scripts/summarize_main.py --select)"
#
# DATASETS (default "wb celeba"), BACKBONES (default "resnet18 resnet50"),
# CELEBA_ROOT (needed for celeba), ROOT (default main), DRY_RUN=1 to print only.
# Methods and outputs: scripts/main_run.sh and scripts/main_posthoc.sh.
set -euo pipefail
if [ ! -f scripts/main_run.sh ]; then
    echo "Run from the OpenVocabDebias repo root" >&2
    exit 1
fi
mkdir -p logs
PHASE=${PHASE:?set PHASE=grid, fix, bam or extend}
DATASETS=${DATASETS:-wb celeba}
BACKBONES=${BACKBONES:-resnet18 resnet50}
ROOT=${ROOT:-main}
if [ "$PHASE" = extend ]; then SEEDS="789 1011"; else SEEDS="42 123 456"; fi
if [ "$PHASE" = fix ] && compgen -G "checkpoints_${ROOT}/*/bam*" >/dev/null; then
    echo "BAM (now with SGD auxiliaries) reuses the names bam1_aux* / bam2_aux*: move the" >&2
    echo "old runs aside first, keeping their paths:" >&2
    echo "  for d in checkpoints_${ROOT}/*/bam* results_${ROOT}/*/bam*; do" >&2
    echo "    mkdir -p old_bam/\$(dirname \$d) && mv \$d old_bam/\$d; done" >&2
    exit 1
fi
N=$(wc -w <<< "$SEEDS")
ARRAY=0-$((N - 1))

sub() {  # sub <time> <dependency or -> VAR=value... -- script ; prints the job id
    local time=$1 dep=$2
    shift 2
    local exports="ALL,SEEDS=$SEEDS,ROOT=$ROOT"
    while [ "$1" != -- ]; do exports="$exports,$1"; shift; done
    shift
    local cmd=(sbatch --parsable --array="$ARRAY" --time="$time" --export="$exports")
    [ "$dep" != - ] && cmd+=(--dependency="afterok:$dep")
    cmd+=("$@")
    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "${cmd[*]}" >&2
        echo "DRY$RANDOM"
    else
        "${cmd[@]}"
    fi
}

for DATASET in $DATASETS; do
    if [ "$DATASET" = celeba ]; then
        : "${CELEBA_ROOT:?set CELEBA_ROOT for celeba}"
        JTT_UPS="20 50"; BAM_UPS="50 100"; WD=0.1
    else
        JTT_UPS="20 50 100"; BAM_UPS="50 140"; WD=1.0
    fi
    JTT_LRS="1e-5 1e-6"; BAM_AUXS="5 50"
    for BACKBONE in $BACKBONES; do
        TAG=${DATASET}_${BACKBONE}
        case $TAG in   # wall times: ERM-cost / with the CLIP loss
            wb_resnet18)     T_ERM=04:00:00; T_CLIP=06:00:00; LAMBDA=10 ;;
            wb_resnet50)     T_ERM=08:00:00; T_CLIP=12:00:00; LAMBDA=15 ;;
            celeba_resnet18) T_ERM=10:00:00; T_CLIP=16:00:00; LAMBDA=10 ;;
            celeba_resnet50) T_ERM=24:00:00; T_CLIP=48:00:00; LAMBDA=10 ;;
        esac
        if [ "$PHASE" = extend ]; then
            v=JTT_LR_$TAG; JTT_LRS=${!v:?set $v}
            v=JTT_UP_$TAG; JTT_UPS=${!v:?set $v}
            v=BAM_AUX_$TAG; BAM_AUXS=${!v:?set $v}
            v=BAM_UP_$TAG; BAM_UPS=${!v:?set $v}
        fi
        C=(DATASET=$DATASET BACKBONE=$BACKBONE CELEBA_ROOT=${CELEBA_ROOT:-})
        echo "== $TAG  $PHASE  seeds: $SEEDS"
        J_OURS=- ; J_GRW=-
        if [ "$PHASE" = bam ]; then
            for AUX in $BAM_AUXS; do
                J_B1=$(sub "$T_ERM" - "${C[@]}" METHOD=bam1 AUX=$AUX -- scripts/main_run.sh)
                for UP in $BAM_UPS; do
                    sub "$T_ERM" "$J_B1" "${C[@]}" METHOD=bam2 AUX=$AUX UP=$UP -- scripts/main_run.sh >/dev/null
                done
            done
            continue
        fi
        if [ "$PHASE" != fix ]; then
            J_ERM=$(sub "$T_ERM" - "${C[@]}" METHOD=erm -- scripts/main_run.sh)
            J_GRW=$(sub "$T_ERM" - "${C[@]}" METHOD=grw -- scripts/main_run.sh)
            J_OURS=$(sub "$T_CLIP" - "${C[@]}" METHOD=ours LAMBDA=$LAMBDA -- scripts/main_run.sh)
            sub 04:00:00 "$J_ERM" "${C[@]}" METHOD=last_layer -- scripts/main_posthoc.sh >/dev/null
        fi
        if [ "$PHASE" != fix ] || [ "$DATASET" = celeba ]; then
            sub "$T_ERM" - "${C[@]}" METHOD=dro_pseudo -- scripts/main_run.sh >/dev/null
            sub "$T_ERM" - "${C[@]}" METHOD=dro_oracle -- scripts/main_run.sh >/dev/null
        fi
        sub "$T_ERM" - "${C[@]}" METHOD=dro_pseudo_wd WD=$WD -- scripts/main_run.sh >/dev/null
        sub "$T_ERM" - "${C[@]}" METHOD=dro_oracle_wd WD=$WD -- scripts/main_run.sh >/dev/null
        for LR1 in $JTT_LRS; do
            J_J1=$(sub "$T_ERM" - "${C[@]}" METHOD=jtt1 LR1=$LR1 -- scripts/main_run.sh)
            for UP in $JTT_UPS; do
                sub "$T_ERM" "$J_J1" "${C[@]}" METHOD=jtt LR1=$LR1 UP=$UP -- scripts/main_run.sh >/dev/null
            done
        done
        for AUX in $BAM_AUXS; do
            J_B1=$(sub "$T_ERM" - "${C[@]}" METHOD=bam1 AUX=$AUX -- scripts/main_run.sh)
            for UP in $BAM_UPS; do
                sub "$T_ERM" "$J_B1" "${C[@]}" METHOD=bam2 AUX=$AUX UP=$UP -- scripts/main_run.sh >/dev/null
            done
        done
        # DFR / AFR on top of the ours and grw models: do their features help?
        sub 04:00:00 "$J_OURS" "${C[@]}" METHOD=last_layer SRC=ours_l$LAMBDA -- scripts/main_posthoc.sh >/dev/null
        sub 04:00:00 "$J_GRW" "${C[@]}" METHOD=last_layer SRC=grw -- scripts/main_posthoc.sh >/dev/null
    done
    if [ "$PHASE" = grid ]; then
        ARRAY=0-0 sub 04:00:00 - DATASET=$DATASET CELEBA_ROOT=${CELEBA_ROOT:-} METHOD=probe \
            -- scripts/main_posthoc.sh >/dev/null
    fi
done
echo "submitted ($PHASE)"

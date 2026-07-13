#!/bin/bash
#SBATCH --job-name=DOVE_BLONDE_GENDER_SUBSPACE_ABLATION
#SBATCH -G a100:1
#SBATCH -c 32
#SBATCH --mem 24G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 2-00:00:00   # time in d-hh:mm:ss

module purge
module load mamba/latest
source activate CCBM

cd ../

for K in 1 2 3 4 5 6 7 8 9
do
    echo "Running with concept_subspace_k=${K}"
    python train.py --celeba_root /data/hkerner/Datasets/CelebA/ \
    --checkpoint_dir "./checkpoints/ablation_subspace_gb/subspace_expanded_${K}/" \
    --results_dir ./results/subspace_expanded_gb/ \
    --concept_attr Male \
    --target_attr Blond_Hair \
    --batch_size 128 \
    --runs align_only \
    --concept_subspace_k "${K}"

done

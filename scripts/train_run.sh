#!/bin/bash
#SBATCH --job-name=DOVE_TRAIN
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

python train.py --runs erm full --seeds 42 123 456 --celeba_root /data/hkerner/Datasets/CelebA/

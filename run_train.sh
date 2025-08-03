#!/bin/bash
#SBATCH --job-name=RLaffinity_pdbbind_rna
#SBATCH --output=RLaffinity_pdbbind_rna_test.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=72:00:00

module purge
module load conda cuda/12.4.1
nvcc --version
conda activate RLaffinity

python train_model_pdbbind.py
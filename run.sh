#!/bin/bash

#SBATCH --job-name=regularized_run
#SBATCH --time=6:00:00
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

module load miniconda
conda activate cpsc552v2
python regularization.py
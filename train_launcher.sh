#!/bin/bash
#SBATCH --partition=education_gpu
#SBATCH --gpus 1
#SBATCH --time=20:00
#SBATCH --mem=45000
#SBATCH --output="output/%j_output.txt"
#SBATCH --error="error/%j_error.txt"

# ps -elf | grep python
# echo "Node: $(hostname)"

# export MASTER_PORT=$(shuf -i 49152-65535 -n 1)
# module load mpi/openmpi-4.1.5
# module load nccl/2.18.1-cuda11.8
# source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a/etc/profile.d/conda.sh
module load miniconda 
conda activate cpsc552v2

# Pass the max_num_atoms parameter to the Hydra configuration
python main.py
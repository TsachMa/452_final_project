#!/bin/bash
<<<<<<< HEAD
#SBATCH --partition=education_gpu
#SBATCH --gpus 1
#SBATCH --time=12:00:00
#SBATCH --mem=10000
=======
#SBATCH --gpus=p100:1
#SBATCH --partition=education_gpu
#SBATCH --time=10:00
>>>>>>> 6f68613 (minor changes)
#SBATCH --output="output/%j_output.txt"
#SBATCH --error="error/%j_error.txt"

# ps -elf | grep python
# echo "Node: $(hostname)"

# export MASTER_PORT=$(shuf -i 49152-65535 -n 1)
# module load mpi/openmpi-4.1.5
# module load nccl/2.18.1-cuda11.8
# source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a/etc/profile.d/conda.sh
module load miniconda 
<<<<<<< HEAD
module load cuDNN/8.7.0.84-CUDA-11.8.0
=======
>>>>>>> 6f68613 (minor changes)
conda activate cpsc552v2

# Pass the max_num_atoms parameter to the Hydra configuration
python main.py -m training.lr=0.0002,0.0004,0.001 hydra.job.name=aug_comp_lr_sweep model.composition_model=mlp 
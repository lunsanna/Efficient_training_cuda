#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --error=error.err

module load anaconda 
module load cuda

source activate test_env

srun python -u single_gpu.py >> output.out
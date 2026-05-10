#!/bin/bash
#SBATCH --job-name=resnet
#SBATCH --output=slurm_logs/resnet-%j.out
#SBATCH --error=slurm_logs/resnet-%j.err

#SBATCH --account=csso
#SBATCH --partition=a10
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=5

# >>> Load necessary modules <<<
module load anaconda

# >>> Activate your specific conda environment <<<
conda activate /scratch/gilbreth/shar1159/conda_envs/cynics

# >>> Navigate to the code directory <<<
cd /scratch/gilbreth/shar1159/cynics/mouse_dynamics || exit 1

python train_resnet.py
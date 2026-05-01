#!/bin/bash
#SBATCH -J trainVAE_%a           # Job name
#SBATCH -p gpu
#SBATCH -n 4 # Number of cpu cores
#SBATCH --gres=gpu:2
#SBATCH --mem=64G                    # Memory per node
#SBATCH -t 36:00:00                 # Time limit (hh:mm:ss)
#SBATCH -o /users/lhay/scratch/trainVAE_%j.out            # Standard output log
#SBATCH -e /users/lhay/scratch/trainVAE_%j.err            # Standard error log

echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"

# Activate the torch environment
source torch_env/bin/activate

# Run the training script
python3 -u trainVAE.py
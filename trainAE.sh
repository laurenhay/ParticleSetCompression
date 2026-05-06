#!/bin/bash
#SBATCH -J trainVAE_%a           # Job name
#SBATCH --array=0-3
#SBATCH -p gpu
#SBATCH -n 4 # Number of cpu cores
#SBATCH --gres=gpu:2
#SBATCH --mem=64G                    # Memory per node
#SBATCH -t 36:00:00                 # Time limit (hh:mm:ss)
#SBATCH -o /users/lhay/scratch/trainVAE_%A_%a.out         # Standard output log
#SBATCH -e /users/lhay/scratch/trainVAE_%A_%a.err         # Standard error log

echo "Starting job ${SLURM_ARRAY_TASK_ID:-single} on $HOSTNAME"

# Activate the torch environment
source torch_env/bin/activate

# Optional CLI override for single-run submissions:
#   sbatch trainVAE.sh pythia VAE
#   sbatch trainVAE.sh herwig PSAE
# If no args are provided, array task IDs map to:
#   0 -> pythia VAE
#   1 -> pythia PSAE
#   2 -> herwig VAE
#   3 -> herwig PSAE

if [[ -n "$1" && -n "$2" ]]; then
	GENERATOR="$1"
	MODEL="$2"
else
	GENERATORS=(pythia pythia herwig herwig)
	MODELS=(VAE PSAE VAE PSAE)
	TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

	if [[ "$TASK_ID" -lt 0 || "$TASK_ID" -ge "${#GENERATORS[@]}" ]]; then
		echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID. Expected 0-3."
		exit 1
	fi

	GENERATOR="${GENERATORS[$TASK_ID]}"
	MODEL="${MODELS[$TASK_ID]}"
fi

echo "Generator: ${GENERATOR}"
echo "Model: ${MODEL}"

# Run the training script
python3 -u trainVAE.py --generator "$GENERATOR" --model "$MODEL"
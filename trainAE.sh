#!/bin/bash
#SBATCH -J trainAE_%a           # Job name
#SBATCH --array=0-5
#SBATCH -p gpu
#SBATCH -n 4 # Number of cpu cores
#SBATCH --gres=gpu:2
#SBATCH --mem=64G                    # Memory per node
#SBATCH -t 36:00:00                 # Time limit (hh:mm:ss)
#SBATCH -o /users/lhay/scratch/trainAE_%A_%a.out         # Standard output log
#SBATCH -e /users/lhay/scratch/trainAE_%A_%a.err         # Standard error log

echo "Starting job ${SLURM_ARRAY_TASK_ID:-single} on $HOSTNAME"

# Activate the torch environment
source torch_env/bin/activate

# Optional CLI override for single-run submissions:
#   sbatch trainAE.sh pythia VAE MSE
#   sbatch trainAE.sh herwig PSAE SWD
# If no args are provided, array task IDs map to:
#   0 -> pythia VAE  MSE
#   1 -> pythia PSAE MSE
#   2 -> pythia PSAE SWD
#   3 -> herwig VAE  MSE
#   4 -> herwig PSAE MSE
#   5 -> herwig PSAE SWD

if [[ -n "$1" && -n "$2" && -n "$3" ]]; then
	GENERATOR="$1"
	MODEL="$2"
	LOSS="$3"
else
	GENERATORS=(pythia pythia pythia herwig herwig herwig)
	MODELS=(VAE    PSAE   PSAE   VAE    PSAE   PSAE  )
	LOSSES=(MSE    MSE    SWD    MSE    MSE    SWD   )
	TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

	if [[ "$TASK_ID" -lt 0 || "$TASK_ID" -ge "${#GENERATORS[@]}" ]]; then
		echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID. Expected 0-5."
		exit 1
	fi

	GENERATOR="${GENERATORS[$TASK_ID]}"
	MODEL="${MODELS[$TASK_ID]}"
	LOSS="${LOSSES[$TASK_ID]}"
fi

echo "Generator: ${GENERATOR}"
echo "Model: ${MODEL}"
echo "Loss: ${LOSS}"

# Run the training script
python3 -u trainAE.py --generator "$GENERATOR" --model "$MODEL" --loss "$LOSS"
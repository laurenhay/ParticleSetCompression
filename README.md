A particle-level autoencoder for compressing jet constituent data.

## Training with trainAE.py

Main training entry point: [trainAE.py](trainAE.py)

Supported CLI options:
- --generator / -g: pythia (default) or herwig
- --model / -m: VAE (KL + reconstruction loss) or PSAE (masked MSE loss)

Examples:

```bash
source torch_env/bin/activate

# Default run (pythia + VAE)
python trainAE.py

# Herwig with VAE
python trainAE.py --generator herwig --model VAE

# Pythia with PSAE
python trainAE.py --generator pythia --model PSAE
```

Outputs:
- Training/evaluation metrics are logged to Weights & Biases.
- Evaluation plots are written to [plots/](plots/).

## SLURM submission script

Current SLURM script in this repo: [trainAE.sh](trainAE.sh)

Note: this script launches [trainAE.py](trainAE.py). The script name is trainAE.sh, even though the comments and job names refer to VAE.

### Submit all 4 generator/model combinations

The script is configured as an array job with IDs 0-3, mapped as:
- 0 -> pythia + VAE
- 1 -> pythia + PSAE
- 2 -> herwig + VAE
- 3 -> herwig + PSAE

```bash
sbatch trainAE.sh
```

### Submit one specific configuration

You can override generator/model by passing two arguments:

```bash
sbatch trainAE.sh herwig PSAE
sbatch trainAE.sh pythia VAE
```

SLURM logs are written to:
- /users/lhay/scratch/trainAE_<array_job_id>_<task_id>.out
- /users/lhay/scratch/trainAE_<array_job_id>_<task_id>.err

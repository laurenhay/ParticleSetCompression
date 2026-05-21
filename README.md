A particle-level autoencoder for compressing jet constituent data.

## Training with trainAE.py

Main training entry point: [trainAE.py](trainAE.py)

Supported CLI options:
- --generator / -g: pythia (default) or herwig
- --model / -m: VAE (KL + reconstruction loss) or PSAE
- --loss / -l: loss function for PSAE — MSE (default) or SWD (Sliced Wasserstein Distance); ignored when using VAE

Examples:

```bash
source torch_env/bin/activate

# Default run (pythia + VAE)
python trainAE.py

# Herwig with VAE
python trainAE.py --generator herwig --model VAE

# Pythia with PSAE using masked MSE loss
python trainAE.py --generator pythia --model PSAE --loss MSE

# Pythia with PSAE using Sliced Wasserstein Distance loss
python trainAE.py --generator pythia --model PSAE --loss SWD
```

Outputs:
- Training/evaluation metrics are logged to Weights & Biases.
- Evaluation plots are written to [plots/](plots/).

### Loss functions

**VAE** uses ELBO: a reconstruction term (masked MSE over constituents) plus a KL-divergence regularisation term weighted by `beta`.

**PSAE** supports two loss options selectable via `--loss`:
- `MSE` (default): masked mean-squared error between input and reconstructed constituents.
- `SWD`: Sliced Wasserstein Distance between the true and reconstructed constituent point clouds, combined with a reconstruction term. SWD is permutation-invariant and better suited to unordered sets.

## SLURM submission script

Current SLURM script in this repo: [trainAE.sh](trainAE.sh)

Note: this script launches [trainAE.py](trainAE.py). The script name is trainAE.sh, even though the comments and job names refer to VAE.

### Submit all generator/model/loss combinations

The script is configured as an array job with IDs 0-5, mapped as:
- 0 -> pythia + VAE  + MSE
- 1 -> pythia + PSAE + MSE
- 2 -> pythia + PSAE + SWD
- 3 -> herwig + VAE  + MSE
- 4 -> herwig + PSAE + MSE
- 5 -> herwig + PSAE + SWD

```bash
sbatch trainAE.sh
```

### Submit one specific configuration

You can override generator, model, and loss by passing three arguments:

```bash
sbatch trainAE.sh herwig PSAE SWD
sbatch trainAE.sh pythia VAE MSE
```

SLURM logs are written to:
- /users/lhay/scratch/trainAE_<array_job_id>_<task_id>.out
- /users/lhay/scratch/trainAE_<array_job_id>_<task_id>.err

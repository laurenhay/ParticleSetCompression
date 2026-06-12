import os
import torch
from torch.utils.data import Dataset, DataLoader
from model_VAE import JetDeepSetVAE
from particleSetAE import ParticleSetAE
from particleloader import load
import numpy as np
import awkward as ak
import vector
import wandb
import argparse
from utils import *

from datetime import datetime

timestamp = datetime.now().strftime("%m%d_%H%M")  # e.g. "0521_1432"

vector.register_awkward()

#### start interactive session in terminal before any heavy lifting
## interact -n 12 -t 08:00:00 -m 64g
parser = argparse.ArgumentParser()
parser.add_argument("--generator", "-g", type = str, default = "pythia", required = False, help = "Which generator to use: options are herwig and pythia")
parser.add_argument("--model", "-m", type = str, default = "VAE", choices = ["VAE", "PSAE"], help = "Which model to use: VAE or Particle Set AE")
parser.add_argument("--loss", "-l", type = str, default = "MSE", choices = ["MSE", "SWD"], help = "Which loss function to use for PSAE -- mean square error or sliced wasserstein")
parser.add_argument("--N", "-n", type = int, default = 100000, help = "Number of jets to train w/")
parser.add_argument("--output-path", "-o", type = str,
    default = "/oscar/data/mleblan6/lhay/autoencoder/output/QG_jets_{generator}_{model_type}_{loss_label}_N{N}_noSoftmaskOnPT.npz",
    help = "Template path for the compressed output .npz file. May use {generator}, {model_type}, {loss_label}, and {N} placeholders, which are filled in based on the run's settings.")
args = parser.parse_args()

generator = args.generator
model_type = args.model
loss_type = args.loss

loss_label = "ELBO" if model_type == "VAE" else loss_type
config_label = f"Model: {model_type} | Loss: {loss_label}"

#### Check whether we've initiated any GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

##### check whether pythia dataset exists

if generator == "pythia":
    input_path = "/oscar/data/mleblan6/energyflow/datasets/QG_jets.npz"
else:
    generator = "herwig"
    input_path = "/oscar/data/mleblan6/energyflow/datasets/QG_jets_herwig_0.npz"
N = args.N
if not os.path.exists(input_path):
    dir = "work"
    jets, labels = load("qg_jets", N, cache_dir=dir, generator = generator)
else:
    with np.load(input_path) as input_data:
        jets, labels = input_data["X"][:N], input_data["y"][:N]

#### create mask of where zero padding is in DATASET
mask = (jets[:,:,0] > 0).astype(np.float32)
print(mask.shape)
print(mask[0])

#### make a class for easy input into torch training
class JetDataset(Dataset):

    def __init__(self, jets, mask):
        self.jets = torch.tensor(jets, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)

    def __len__(self):
        return len(self.jets)

    def __getitem__(self, idx):
        return self.jets[idx], self.mask[idx]

jets_input = jets[:, :, :3].copy()
jets_input[:, :, 0] = np.log1p(jets_input[:, :, 0])  # log(1+pt) to equalize scale with eta/phi
dataset = JetDataset(jets_input, mask)
feature_dim = dataset.jets.shape[-1]

batch_size = 512

### use torch loading
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=4
)
print("Feature dim (should be 3) ", feature_dim)
print("Nparticles ", len(mask[0]))

nconst = len(mask[0])

zdim = nconst*feature_dim
if model_type == "VAE":
    model = JetDeepSetVAE(
        din=feature_dim,
        nmax=nconst,
        zdim=zdim
    ).to(device)
else:
    model = ParticleSetAE(
        particle_dim=feature_dim,
        max_particles=nconst,
        z_dim=zdim
    ).to(device)

learning_rate = 1e-3
beta = 1e-3
epochs = 1000
early_stop_patience = 100
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=50, factor=0.5, min_lr=1e-6
)
# Start a new wandb run to track this script.
run = wandb.init(
    name = f"{generator}_{model_type}_{loss_label}_{timestamp}",
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="laurenhay-brown-university",
    # Set the wandb project where this run will be logged.
    project="compression_AE",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": learning_rate,
        "architecture": model_type,
        "dataset": generator,
        "epochs": epochs,
        "batch_size": batch_size,
        "beta": beta,
        "nmax": int(mask.shape[1]),
        "feature_dim": int(feature_dim),
    },
)


wandb.watch(model, log="all", log_freq=100)

best_loss = float("inf")
no_improve_count = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0
    running_swd = 0.0
    running_bce = 0.0

    for x, mask_batch in loader:
        x = x.to(device)
        mask_batch = mask_batch.to(device)

        optimizer.zero_grad()
        if model_type == "VAE": 
            xhat, mu, logv, z = model(x, mask_batch)
            loss, recon, kl = vae_loss(x, mask_batch, xhat, mu, logv, beta=beta)
            running_recon += recon.item()
            running_kl += kl.item()
        else:
            xhat, mask_logits, z = model(x, mask_batch)
            if loss_type == "MSE":
                loss, recon, bce = psae_mse_loss(xhat, mask_logits, x, mask_batch)
                running_recon += recon.item()
                running_bce += bce.item()   # reuse running_swd slot for bce
            elif loss_type == "SWD":
                loss, recon, swd, bce = psae_loss(xhat, mask_logits, x, mask_batch)
                running_recon += recon.item()
                running_swd += swd.item()
                running_bce += bce.item() 
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    epoch_loss = running_loss / len(loader)
    if model_type == "VAE":
        epoch_recon = running_recon / len(loader)
        epoch_kl = running_kl / len(loader)
        print(f"epoch {epoch:02d}  loss={epoch_loss:.6f}  recon={epoch_recon:.6f}  kl={epoch_kl:.6f}")
        run.log({
            "epoch": epoch,
            "train/loss": epoch_loss,
            "train/recon": epoch_recon,
            "train/kl": epoch_kl,
        })
    else:
        epoch_bce   = running_bce   / len(loader)
        if loss_type=="MSE":
            epoch_recon = running_recon / len(loader)
            print(f"epoch {epoch:02d}  loss={epoch_loss:.6f}  recon={epoch_recon:.6f}  bce={epoch_bce:.6f}")
            run.log({
                "epoch": epoch,
                "train/loss": epoch_loss,
                "train/recon": epoch_recon,
                "train/bce": epoch_bce,
            })
        elif loss_type=="SWD":
            epoch_recon = running_recon / len(loader)
            epoch_swd = running_swd / len(loader)
            print(f"epoch {epoch:02d}  loss={epoch_loss:.6f}  recon={epoch_recon:.6f}  swd={epoch_swd:.6f}  bce={epoch_bce:.6f}")
            run.log({
                "epoch": epoch,
                "train/loss": epoch_loss,
                "train/recon": epoch_recon,
                "train/swd": epoch_swd,
                "train/bce": epoch_bce,
            })

    scheduler.step(epoch_loss)
    run.log({"train/lr": optimizer.param_groups[0]["lr"]})

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        no_improve_count = 0
    else:
        no_improve_count += 1
        if no_improve_count >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            break

eval_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    drop_last=False
)


model.eval()

all_x = []
all_xhat = []
all_mask = []
all_mask_pred = []
all_mu = [] if model_type == "VAE" else None

with torch.no_grad():
    for x, mask_batch in eval_loader:
        x = x.to(device)
        mask_batch = mask_batch.to(device)

        if model_type == "VAE":
            xhat, mu, logv, z = model(x, mask_batch)
            all_mu.append(mu.cpu())
            all_mask_pred.append(mask_batch.cpu())  # VAE has no mask head, use true mask
        else:
            xhat, mask_logits, z = model(x, mask_batch)
            mask_pred = (torch.sigmoid(mask_logits) > 0.5).float()
            all_mask_pred.append(mask_pred.cpu())

        all_x.append(x.cpu())
        all_xhat.append(xhat.cpu())
        all_mask.append(mask_batch.cpu())           # always append true mask

all_x         = torch.cat(all_x,         dim=0).numpy()
all_xhat      = torch.cat(all_xhat,      dim=0).numpy()
all_mask      = torch.cat(all_mask,      dim=0).numpy()
all_mask_pred = torch.cat(all_mask_pred, dim=0).numpy()
if model_type == "VAE":
    all_mu = torch.cat(all_mu, dim=0).numpy()

compressed_output_path = args.output_path.format(generator=generator, model_type=model_type, loss_label=loss_label, N=N)
np.savez_compressed(
    compressed_output_path,
    X=all_xhat,
    y=np.asarray(labels),
)
print(f"Saved compressed particles to {compressed_output_path}")

def jets_from_constituents(p, mask):
    pt  = np.where(mask > 0, np.expm1(p[:, :, 0]), 0.0)  # invert log1p
    eta = np.where(mask > 0, p[:, :, 1], 0.0)
    phi = np.where(mask > 0, p[:, :, 2], 0.0)

    constituents = ak.zip(
        {
            "pt":   ak.Array(pt),
            "eta":  ak.Array(eta),
            "phi":  ak.Array(phi),
            "mass": ak.Array(np.zeros_like(pt)),  # massless assumption
        },
        with_name="Momentum4D",
    )
    jets = ak.sum(constituents, axis=1)

    return (
        ak.to_numpy(jets.pt),
        ak.to_numpy(jets.eta),
        ak.to_numpy(jets.phi),
        ak.to_numpy(jets.mass),   # non-zero from 4-vector sum even with massless inputs
    )

def jet_multiplicity(mask):
    return mask.sum(axis=1)

pt_true, eta_true, phi_true, mass_true = jets_from_constituents(all_x, all_mask)
mult_true = jet_multiplicity(all_mask)
pt_decoded, eta_decoded, phi_decoded, mass_decoded = jets_from_constituents(all_xhat, all_mask_pred)
mult_decoded = jet_multiplicity(all_mask_pred)


pt_ratio = pt_decoded / (pt_true + 1e-12)
eta_diff = eta_decoded - eta_true
phi_diff = np.arctan2(np.sin(phi_decoded - phi_true), np.cos(phi_decoded - phi_true))
mass_ratio = mass_decoded / (mass_true + 1e-12)

### plot after evaluation
import matplotlib.pyplot as plt
labels = np.array(labels).astype(int)
plot_dir = f"plots/{model_type}"
os.makedirs(plot_dir, exist_ok=True)

# constituent pt is stored in log1p space; invert for display in GeV
const_pt_true    = np.expm1(all_x[..., 0])
const_pt_decoded = np.expm1(all_xhat[..., 0])

def plot_constituent_hist(true_values, decoded_values, true_mask, decoded_mask, labels, bins, xlabel, title, filename, logy=False, ylim=None):
    qjet = (labels == 0)
    gjet = (labels != 0)

    fig, ax = plt.subplots()
    for vals, style, tag, cmask in [
        (true_values,    "solid",  "true",    true_mask),
        (decoded_values, "dashed", "decoded", decoded_mask),
    ]:
        qvals = vals[qjet][cmask[qjet].astype(bool)].reshape(-1)
        gvals = vals[gjet][cmask[gjet].astype(bool)].reshape(-1)
        ax.hist(qvals, bins=bins, histtype="step", linewidth=1.5, linestyle=style, label=f"quark {tag}", color="red")
        ax.hist(gvals, bins=bins, histtype="step", linewidth=1.5, linestyle=style, label=f"gluon {tag}", color="blue")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.set_title(f"{title}\n{config_label}", fontsize=10)
    ax.legend(fontsize=8)
    if logy:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, filename), dpi=200)
    return fig


def plot_jet_panel(true_vals, decoded_vals, response_vals, labels,
                   dist_bins, resp_bins, dist_xlabel, resp_xlabel, title, filename,
                   logy_dist=False):
    """Two-panel figure: true vs decoded distribution (top) + response/residual (bottom)."""
    qjet = (labels == 0)
    gjet = (labels != 0)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6, 7),
                                          gridspec_kw={"height_ratios": [2, 1]})
    ax_top.set_title(f"{title}\n{config_label}", fontsize=10)

    for vals, style, tag in [(true_vals, "solid", "true"), (decoded_vals, "dashed", "decoded")]:
        ax_top.hist(vals[qjet], bins=dist_bins, histtype="step", linewidth=1.5,
                    linestyle=style, label=f"quark {tag}", color="red")
        ax_top.hist(vals[gjet], bins=dist_bins, histtype="step", linewidth=1.5,
                    linestyle=style, label=f"gluon {tag}", color="blue")
    ax_top.set_xlabel(dist_xlabel)
    ax_top.set_ylabel("Counts")
    ax_top.legend(fontsize=8)
    if logy_dist:
        ax_top.set_yscale("log")

    ax_bot.hist(response_vals[qjet], bins=resp_bins, histtype="step", linewidth=1.5, label="quark", color="red")
    ax_bot.hist(response_vals[gjet], bins=resp_bins, histtype="step", linewidth=1.5, label="gluon", color="blue")
    ax_bot.set_xlabel(resp_xlabel)
    ax_bot.set_ylabel("Counts")
    ax_bot.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, filename), dpi=200)
    return fig


# ---- constituent distributions (true vs decoded) ----
const_pt_hist = plot_constituent_hist(
    const_pt_true, const_pt_decoded,
    all_mask, all_mask_pred,
    labels,
    bins=np.linspace(0, 500, 50),
    xlabel="Constituent $p_T$ [GeV]",
    title="Constituents: $p_T$ True vs Decoded",
    filename=f"constituent_pt_z{zdim}_{generator}_{loss_label}.png",
    logy=True,
)

const_eta_hist = plot_constituent_hist(
    all_x[..., 1], all_xhat[..., 1],
    all_mask, all_mask_pred,
    labels,
    bins=np.linspace(-5, 5, 60),
    xlabel="Constituent $\\eta$",
    title="Constituents: $\\eta$ True vs Decoded",
    filename=f"constituent_eta_z{zdim}_{generator}_{loss_label}.png",
)

const_phi_hist = plot_constituent_hist(
    all_x[..., 2], all_xhat[..., 2],
    all_mask, all_mask_pred,
    labels,
    bins=np.linspace(-0.5, 7, 64),
    xlabel="Constituent $\\phi$",
    title="Constituents: $\\phi$ True vs Decoded",
    filename=f"constituent_phi_z{zdim}_{generator}_{loss_label}.png",
)


jet_pt_hist = plot_jet_panel(
    pt_true, pt_decoded, pt_ratio, labels,
    dist_bins=np.linspace(0, max(np.percentile(np.concatenate([pt_true, pt_decoded]), 99.5), 1.0), 60),
    resp_bins=np.linspace(0, 2, 60),
    dist_xlabel="Jet $p_T$ [GeV]",
    resp_xlabel="$p_T^{\\mathrm{dec}} / p_T^{\\mathrm{true}}$",
    title="Jet $p_T$: True vs Decoded",
    filename=f"jet_pt_z{zdim}_{generator}_{loss_label}.png",
    logy_dist=True,
)

jet_eta_hist = plot_jet_panel(
    eta_true, eta_decoded, eta_diff, labels,
    dist_bins=np.linspace(
        min(np.percentile(np.concatenate([eta_true, eta_decoded]), 0.5), -0.1),
        max(np.percentile(np.concatenate([eta_true, eta_decoded]), 99.5), 0.1),
        60,
    ),
    resp_bins=np.linspace(-1, 1, 60),
    dist_xlabel="Jet $\\eta$",
    resp_xlabel="$\\eta^{\\mathrm{dec}} - \\eta^{\\mathrm{true}}$",
    title="Jet $\\eta$: True vs Decoded",
    filename=f"jet_eta_z{zdim}_{generator}_{loss_label}.png",
)

jet_phi_hist = plot_jet_panel(
    phi_true, phi_decoded, phi_diff, labels,
    dist_bins=np.linspace(-np.pi, np.pi, 64),
    resp_bins=np.linspace(-1, 1, 60),
    dist_xlabel="Jet $\\phi$",
    resp_xlabel="$\\phi^{\\mathrm{dec}} - \\phi^{\\mathrm{true}}$",
    title="Jet $\\phi$: True vs Decoded",
    filename=f"jet_phi_z{zdim}_{generator}_{loss_label}.png",
)

jet_mass_hist = plot_jet_panel(
    mass_true, mass_decoded, mass_ratio, labels,
    dist_bins=np.linspace(0, max(np.percentile(np.concatenate([mass_true, mass_decoded]), 99.5), 1.0), 60),
    resp_bins=np.linspace(0, 2, 60),
    dist_xlabel="Jet Mass [GeV]",
    resp_xlabel="$m^{\\mathrm{dec}} / m^{\\mathrm{true}}$",
    title="Jet Mass: True vs Decoded",
    filename=f"jet_mass_z{zdim}_{generator}_{loss_label}.png",
)

mult_diff = mult_decoded - mult_true
jet_mult_hist = plot_jet_panel(
    mult_true, mult_decoded, mult_diff, labels,
    dist_bins=np.arange(0, int(max(mult_true.max(), mult_decoded.max())) + 2) - 0.5,
    resp_bins=np.arange(int(mult_diff.min()) - 1, int(mult_diff.max()) + 2) - 0.5,
    dist_xlabel="Jet Multiplicity",
    resp_xlabel="Multiplicity$^{\\mathrm{dec}}$ $-$ Multiplicity$^{\\mathrm{true}}$",
    title="Jet Multiplicity: True vs Decoded",
    filename=f"jet_multiplicity_z{zdim}_{generator}_{loss_label}.png",
)

# Build stable image keys from figures created in this run only.
_run_figs = [
    (f"constituent_pt_z{zdim}_{generator}_{loss_label}.png",          const_pt_hist),
    (f"constituent_eta_z{zdim}_{generator}_{loss_label}.png",         const_eta_hist),
    (f"constituent_phi_z{zdim}_{generator}_{loss_label}.png",         const_phi_hist),
    (f"jet_pt_z{zdim}_{generator}_{loss_label}.png",                  jet_pt_hist),
    (f"jet_eta_z{zdim}_{generator}_{loss_label}.png",                 jet_eta_hist),
    (f"jet_phi_z{zdim}_{generator}_{loss_label}.png",                 jet_phi_hist),
    (f"jet_mass_z{zdim}_{generator}_{loss_label}.png",                jet_mass_hist),
    (f"jet_multiplicity_z{zdim}_{generator}_{loss_label}.png",        jet_mult_hist),
]
plot_images = {
    f"plots/{name.replace(f'_z{zdim}_{generator}_{loss_label}', '')}": wandb.Image(fig)
    for name, fig in _run_figs
}
print(plot_images)
run.log({
    "eval/pt_ratio_mean": float(np.mean(pt_ratio)),
    "eval/pt_ratio_std": float(np.std(pt_ratio)),
    "eval/eta_diff_mean": float(np.mean(eta_diff)),
    "eval/eta_diff_std": float(np.std(eta_diff)),
    "eval/phi_diff_mean": float(np.mean(phi_diff)),
    "eval/phi_diff_std": float(np.std(phi_diff)),
    "eval/mass_ratio_mean": float(np.mean(mass_ratio)),
    "eval/mass_ratio_std": float(np.std(mass_ratio)),
    "eval/mult_true_mean": float(np.mean(mult_true)),
    "eval/mult_decoded_mean": float(np.mean(mult_decoded)),
    **plot_images,
})

run.finish()

print(f"Saved plots to {plot_dir}/")
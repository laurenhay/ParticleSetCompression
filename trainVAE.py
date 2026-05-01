import os
import torch
from torch.utils.data import Dataset, DataLoader
from model_VAE import JetDeepSetVAE
from particleloader import load
import numpy as np
import wandb
import argparse

#### start interactive session in terminal before any heavy lifting
## interact -n 12 -t 08:00:00 -m 64g
parser = argparse.ArgumentParser()
parser.add_argument("--generator", "-g", type = str, default = "pythia", required = False, help = "Which generator to use: options are herwig and pythia")
args = parser.parse_args()

generator = args.generator

#### Check whether we've initiated any GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#### define our loss function

def vae_loss(x, mask, xhat, mu, logv, beta=1e-3):
    m = mask.unsqueeze(-1)
    recon = ((x - xhat)**2 * m).sum() / m.sum().clamp_min(1.0)

    kl = -0.5 * torch.sum(1 + logv - mu.pow(2) - logv.exp(), dim=1).mean()
    return recon + beta * kl, recon, kl

def masked_mse_loss(xhat, x, mask):
    # xhat, x: (B, N, D)
    # mask:    (B, N)
    diff2 = (xhat - x) ** 2
    diff2 = diff2 * mask.unsqueeze(-1)
    return diff2.sum() / mask.sum().clamp(min=1)

##### check whether pythia dataset exists
N = 1000
if generator == "pythia":
    input_path = "work/qg_jets/generator:pythia/with_bc:False/QG_jets.npz"
else:
    generator = "herwig"
    input_path = "work/qg_jets/generator:herwig/with_bc:False/QG_jets_herwig_0.npz"

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
    
dataset = JetDataset(jets[:, :, :3], mask)
feature_dim = dataset.jets.shape[-1]

### use torch loading
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    drop_last=False,
    num_workers=4
)
print("Feature dim (should be 3) ", feature_dim)
print("Nparticles ", len(mask[0]))

nconst = len(mask[0])
zdim = nconst*feature_dim
model = JetDeepSetVAE(
    din=feature_dim,
    nmax=nconst,
    zdim=zdim
).to(device)

learning_rate = 1e-3
beta = 1e-3
batch_size = 64
epochs = 20
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="laurenhay-brown-university",
    # Set the wandb project where this run will be logged.
    project="compression_VAE",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": learning_rate,
        "architecture": "VAE",
        "dataset": generator,
        "epochs": epochs,
        "batch_size": batch_size,
        "beta": beta,
        "nmax": int(mask.shape[1]),
        "feature_dim": int(feature_dim),
    },
)


wandb.watch(model, log="all", log_freq=100)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0

    for x, mask_batch in loader:
        x = x.to(device)
        mask_batch = mask_batch.to(device)

        optimizer.zero_grad()
        if model=VAE:
            xhat, mu, logv, z = model(x, mask_batch)
            loss, recon, kl = vae_loss(x, mask_batch, xhat, mu, logv, beta=beta)
            running_recon += recon.item()
            running_kl += kl.item()
        elif model=PSAE:
            xhat, mask_logits, z = model(x, mask_batch)
            loss = masked_mse_loss(xhat, x, mask_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    epoch_loss = running_loss / len(loader)
    if model=VAE:
        epoch_recon = running_recon / len(loader)
        epoch_kl = running_kl / len(loader)
        print(f"epoch {epoch:02d}  loss={epoch_loss:.6f}  recon={epoch_recon:.6f}  kl={epoch_kl:.6f}")
    print(f"epoch {epoch:02d}  loss={epoch_loss:.6f}")
    run.log({
        "epoch": epoch,
        "train/loss": epoch_loss,
        "train/recon": epoch_recon,
        "train/kl": epoch_kl,
    })


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
all_mu = []

with torch.no_grad():
    for x, mask_batch in eval_loader:
        x = x.to(device)
        mask_batch = mask_batch.to(device)

        xhat, mu, logv, z = model(x, mask_batch)

        all_x.append(x.cpu())
        all_xhat.append(xhat.cpu())
        all_mask.append(mask_batch.cpu())
        all_mu.append(mu.cpu())

all_x = torch.cat(all_x, dim=0).numpy()
all_xhat = torch.cat(all_xhat, dim=0).numpy()
all_mask = torch.cat(all_mask, dim=0).numpy()
all_mu = torch.cat(all_mu, dim=0).numpy()

def jets_from_constituents(p, mask):
    pt  = p[:, :, 0] * mask
    eta = p[:, :, 1]
    phi = p[:, :, 2]

    E  = np.sum(pt * np.cosh(eta) * mask, axis=1)
    px = np.sum(pt * np.cos(phi) * mask, axis=1)
    py = np.sum(pt * np.sin(phi) * mask, axis=1)
    pz = np.sum(pt * np.sinh(eta) * mask, axis=1)

    jet_pt = np.sqrt(px**2 + py**2)
    jet_phi = np.arctan2(py, px)

    pabs = np.sqrt(px**2 + py**2 + pz**2)
    jet_eta = 0.5 * np.log((pabs + pz + 1e-12) / (pabs - pz + 1e-12))

    return jet_pt, jet_eta, jet_phi

pt_true, eta_true, phi_true = jets_from_constituents(all_x, all_mask)
pt_reco, eta_reco, phi_reco = jets_from_constituents(all_xhat, all_mask)

pt_ratio = pt_reco / (pt_true + 1e-12)
eta_diff = eta_reco - eta_true
phi_diff = np.arctan2(np.sin(phi_reco - phi_true), np.cos(phi_reco - phi_true))

### plot after evaluation
import matplotlib.pyplot as plt
labels = np.array(labels).astype(int)
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

def plot_constituent_hist(values, constituent_mask, labels, bins, xlabel, title, filename, logy=False, ylim=None):
    qjet = (labels == 0)
    gjet = (labels != 0)

    qvals = values[qjet][constituent_mask[qjet].astype(bool)]
    gvals = values[gjet][constituent_mask[gjet].astype(bool)]

    plt.figure()
    plt.hist(
        qvals.reshape(-1),
        bins=bins,
        histtype="step",
        linewidth=1.5,
        label="quark const",
        color="red"
    )
    plt.hist(
        gvals.reshape(-1),
        bins=bins,
        histtype="step",
        linewidth=1.5,
        label="gluon const",
        color="blue"
    )

    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.title(title)
    plt.legend()
    if logy:
        plt.yscale("log")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=200)
    plt.close()
    

####plot original dataset
plot_constituent_hist(
    all_x[..., 0],
    all_mask,
    labels,
    bins=np.linspace(0, 500, 50),
    xlabel="Constituent $pT$ [GeV]",
    title="Original Constituents: $pT$",
    filename="original_constituent_pt.png",
    logy=True,
    ylim=(1, 1e5)
)

plot_constituent_hist(
    all_x[..., 1],
    all_mask,
    labels,
    bins=np.linspace(-5, 5, 60),
    xlabel="Constituent $\\eta$",
    title="Original Constituents: $\\eta$",
    filename="original_constituent_eta.png"
)

plot_constituent_hist(
    all_x[..., 2],
    all_mask,
    labels,
    bins=np.linspace(-1, 7, 64),
    xlabel="Constituent $\\phi$",
    title="Original Constituents: $\\phi$",
    filename="original_constituent_phi.png"
)
####plot decoded data
plot_constituent_hist(
    all_xhat[..., 0],
    all_mask,
    labels,
    bins=np.linspace(0, 500, 50),
    xlabel="Constituent $pT$ [GeV]",
    title="Decoded Constituents: $pT$",
    filename=f"vae_constituent_pt_z{zdim}.png",
    logy=True,
    ylim=(1, 1e5)
)

plot_constituent_hist(
    all_xhat[..., 1],
    all_mask,
    labels,
    bins=np.linspace(-5, 5, 60),
    xlabel="Constituent $\\eta$",
    title="Decoded Constituents: $\\eta$",
    filename=f"vae_constituent_eta_z{zdim}.png"
)

plot_constituent_hist(
    all_xhat[..., 2],
    all_mask,
    labels,
    bins=np.linspace(-1, 7, 64),
    xlabel="Contituent $\\phi$",
    title="Decoded Constituents: $\\phi$",
    filename=f"vae_constituent_phi_z{zdim}.png"
)

#### response plots
def plot_response_hist(values, labels, bins, xlabel, title, filename):
    qjet = (labels == 0)
    gjet = (labels != 0)

    plt.figure()
    plt.hist(
        values[qjet].reshape(-1),
        bins=bins,
        histtype="step",
        linewidth=1.5,
        label="quark jets",
        color="red"
    )
    plt.hist(
        values[gjet].reshape(-1),
        bins=bins,
        histtype="step",
        linewidth=1.5,
        label="gluon jets",
        color="blue"
    )

    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=200)
    plt.close()

plot_response_hist(
    pt_ratio,
    labels,
    bins=np.linspace(0, 2, 60),
    xlabel="$pT_{reco} / pT_{true}$",
    title="Jet Response: $pT$ Ratio",
    filename=f"response_pt_ratio_z{zdim}.png"
)

plot_response_hist(
    eta_diff,
    labels,
    bins=np.linspace(-1, 1, 60),
    xlabel="$\\eta_{reco} - \\eta_{true}$",
    title="Jet Response: $\\eta$ Residual",
    filename=f"response_eta_residual_z{zdim}.png"
)

plot_response_hist(
    phi_diff,
    labels,
    bins=np.linspace(-1, 1, 60),
    xlabel="$\\phi_{reco} - \\phi_{true}$",
    title="Jet Response: $\\phi$ Residual",
    filename=f"response_phi_residual_z{zdim}.png"
)

run.log({
    "eval/pt_ratio_mean": float(np.mean(pt_ratio)),
    "eval/pt_ratio_std": float(np.std(pt_ratio)),
    "eval/eta_diff_mean": float(np.mean(eta_diff)),
    "eval/eta_diff_std": float(np.std(eta_diff)),
    "eval/phi_diff_mean": float(np.mean(phi_diff)),
    "eval/phi_diff_std": float(np.std(phi_diff)),
})

for plot_name in sorted(os.listdir(plot_dir)):
    plot_path = os.path.join(plot_dir, plot_name)
    if os.path.isfile(plot_path):
        run.log({plot_name: wandb.Image(plot_path)})

run.finish()

print(f"Saved plots to {plot_dir}/")
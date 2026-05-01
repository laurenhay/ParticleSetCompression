import torch
import torch.nn as nn
import torch.nn.functional as F

def mlp(in_dim, hidden, out_dim, act=nn.ELU):
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)

class JetDeepSetVAE(nn.Module):
    """
    Jet = event, constituents = set elements.
    x:    (B, Nmax, Din)
    mask: (B, Nmax) 1 for real constituents, 0 for padding
    """
    def __init__(self, din=4, nmax=200, zdim=32, phi_hidden=(256,256,256), rho_hidden=(256,256), dec_hidden=(256,256)):
        super().__init__()
        self.din = din
        self.nmax = nmax
        self.zdim = zdim

        self.phi = mlp(din, phi_hidden, 2*zdim)         # per-particle -> (mu, logvar) per particle
        self.rho = mlp(zdim, rho_hidden, zdim)          # event embedding refinement (optional but helps)
        self.to_mu = nn.Linear(zdim, zdim)
        self.to_logvar = nn.Linear(zdim, zdim)

        self.dec = mlp(zdim, dec_hidden, nmax*din)      # z -> flattened constituents

    def encode(self, x, mask):
        B, N, D = x.shape
        h = self.phi(x.reshape(B*N, D)).reshape(B, N, 2*self.zdim)
        mu_p = h[:, :, :self.zdim]
        lv_p = h[:, :, self.zdim:]

        m = mask.to(x.dtype).unsqueeze(-1)  # (B,N,1)

        # simple DeepSets pooling (mean). This is the “sum over particles” idea.
        mu = (mu_p * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        lv = (lv_p * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

        # optional refinement
        emb = self.rho(mu)
        mu = self.to_mu(emb)
        lv = self.to_logvar(emb)
        return mu, lv

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        B = z.shape[0]
        out = self.dec(z).view(B, self.nmax, self.din)
        return out

    def forward(self, x, mask):
        mu, logvar = self.encode(x, mask)
        z = self.reparam(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z
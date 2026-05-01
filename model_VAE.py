import torch
import torch.nn as nn
import torch.nn.functional as F

class JetDeepSetVAE(nn.Module):
    def __init__(self, din=3, nmax=139, zdim=32):
        super().__init__()
        self.din = din
        self.nmax = nmax
        self.zdim = zdim

        self.phi = nn.Sequential(
            nn.Linear(din, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * zdim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(zdim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, nmax * din),
        )

    def encode(self, x, mask):
        B, N, D = x.shape
        h = self.phi(x.reshape(B * N, D)).reshape(B, N, 2 * self.zdim)

        mu_p = h[:, :, :self.zdim]
        logv_p = h[:, :, self.zdim:]

        m = mask.unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1.0)

        mu = (mu_p * m).sum(dim=1) / denom
        logv = (logv_p * m).sum(dim=1) / denom
        return mu, logv

    def reparam(self, mu, logv):
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        B = z.shape[0]
        out = self.decoder(z)
        return out.view(B, self.nmax, self.din)

    def forward(self, x, mask):
        mu, logv = self.encode(x, mask)
        z = self.reparam(mu, logv)
        xhat = self.decode(z)
        return xhat, mu, logv, z
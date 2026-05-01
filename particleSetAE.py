import torch
from torch import nn
import torch.nn.functional as F


class ParticleSetAE(nn.Module):
    """
    Variable-length particle-list autoencoder.

    Input:
        x:    (batch, max_particles, particle_features)
        mask: (batch, max_particles), True for real particles, False for padding

    Example particle_features:
        [pt, eta, phi, mass] or [z, y, phi, pid, charge, ...]
    """

    def __init__(
        self,
        particle_dim=4,
        phi_dim=128,
        z_dim=16,
        max_particles=128,
        hidden=200,
    ):
        super().__init__()

        self.max_particles = max_particles
        self.particle_dim = particle_dim
        self.z_dim = z_dim

        # Per-particle map Phi
        self.phi = nn.Sequential(
            nn.Linear(particle_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, phi_dim),
            nn.LeakyReLU(),
        )

        # Event-level map to latent bottleneck
        self.encoder_head = nn.Sequential(
            nn.Linear(phi_dim, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Linear(50, z_dim),
        )

        # Simple fixed-size decoder: reconstruct padded particle tensor
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 100),
            nn.LeakyReLU(),
            nn.Linear(100, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, max_particles * particle_dim),
        )

    def encode(self, x, mask):
        # x: (B, N, D), mask: (B, N)

        h = self.phi(x)  # (B, N, phi_dim)

        # zero out padded particles before pooling
        h = h * mask.unsqueeze(-1)

        # permutation-invariant pooling
        pooled = h.sum(dim=1)  # (B, phi_dim)

        z = self.encoder_head(pooled)
        return z

    def decode(self, z):
        out = self.decoder(z)
        out = out.view(-1, self.max_particles, self.particle_dim)
        return out

    def forward(self, x, mask):
        z = self.encode(x, mask)
        xhat = self.decode(z)
        return xhat, z
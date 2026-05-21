import torch
from torch import nn

class ParticleSetAE(nn.Module):
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

        self.phi = nn.Sequential(
            nn.Linear(particle_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, phi_dim),
            nn.LeakyReLU(),
        )

        self.encoder_head = nn.Sequential(
            nn.Linear(phi_dim, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Linear(50, z_dim),
        )

        # z_dim + 1 for the appended n_real fraction
        self.decoder_shared = nn.Sequential(
            nn.Linear(z_dim + 1, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 100),
            nn.LeakyReLU(),
            nn.Linear(100, hidden),
            nn.LeakyReLU(),
        )

        # head 1: particle features
        self.decoder_features = nn.Linear(hidden, max_particles * particle_dim)

        # head 2: per-slot occupancy logit (is this slot a real particle?)
        self.decoder_mask = nn.Linear(hidden, max_particles)

    def encode(self, x, mask):
        h = self.phi(x)
        h = h * mask.unsqueeze(-1)
        pooled = h.sum(dim=1)
        z = self.encoder_head(pooled)

        n_real = mask.sum(dim=1, keepdim=True) / self.max_particles  # (B, 1)
        z_aug = torch.cat([z, n_real], dim=-1)
        return z_aug

    def decode(self, z_aug):
        h = self.decoder_shared(z_aug)

        features = self.decoder_features(h)
        features = features.view(-1, self.max_particles, self.particle_dim)

        mask_logits = self.decoder_mask(h)   # (B, max_particles), raw logits
        return features, mask_logits

    def forward(self, x, mask):
        z_aug = self.encode(x, mask)
        xhat, mask_logits = self.decode(z_aug)
        return xhat, mask_logits, z_aug
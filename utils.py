import torch
from torch import distributions as dist
import torch.nn.functional as F

def masked_sliced_wasserstein(
    x: torch.Tensor,
    xhat: torch.Tensor,
    mask: torch.Tensor,
    num_projections: int = 200,
    p: float = 2.0,
) -> torch.Tensor:
    """
    Sliced Wasserstein Distance between true and reconstructed particle sets,
    respecting variable-length masking.

    Args:
        x:    (B, N, D) true particles
        xhat: (B, N, D) reconstructed particles
        mask: (B, N)    float, 1 = real particle, 0 = padding
        num_projections: number of random 1D projections
        p: Wasserstein order (1 or 2)

    Returns:
        Scalar SWD loss.
    """
    B, N, D = x.shape
    device = x.device

    # --- random unit projection vectors in R^D ---
    # shape: (D, num_projections)
    rand = torch.randn(D, num_projections, device=device)
    proj_matrix = rand / rand.norm(dim=0, keepdim=True)  # unit columns

    # --- zero out padding so it doesn't affect projections ---
    m = mask.unsqueeze(-1)            # (B, N, 1)
    x_real    = x    * m              # (B, N, D) — padded rows are 0
    xhat_real = xhat * m

    # --- project each particle onto all directions ---
    # (B, N, D) x (D, S) -> (B, N, S)
    x_proj    = torch.matmul(x_real,    proj_matrix)   # (B, N, S)
    xhat_proj = torch.matmul(xhat_real, proj_matrix)

    # --- sort along particle axis ---
    # Padded particles project to 0.0 and will cluster near the origin.
    # We push them to +inf before sorting so they land at the top and can be masked out.
    BIG = 1e9
    x_proj_fill    = x_proj    + BIG * (1 - m)   # (B, N, S)
    xhat_proj_fill = xhat_proj + BIG * (1 - m)

    x_sorted,    _ = torch.sort(x_proj_fill,    dim=1)   # (B, N, S)
    xhat_sorted, _ = torch.sort(xhat_proj_fill, dim=1)

    # --- build a sorted mask: 0s first, 1s last (matching sort order of BIG-filled values) ---
    mask_sorted, _ = torch.sort(mask, dim=1)              # (B, N) — 0s first, 1s last
    m_sorted = mask_sorted.unsqueeze(-1)                  # (B, N, 1)

    # --- Wasserstein distance: |sorted_x - sorted_xhat|^p, only over real particles ---
    diff = (x_sorted - xhat_sorted).abs().pow(p)          # (B, N, S)
    diff = diff * m_sorted                                 # zero out padded positions

    # normalize per event by number of real particles
    n_real = mask.sum(dim=1).clamp(min=1).view(B, 1, 1)  # (B, 1, 1)
    swd = (diff / n_real).sum(dim=1).mean()               # average over batch and projections

    return swd

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

def differentiable_1d_wasserstein(x, xhat, mask):
    """
    x, xhat: (B, N, D)
    mask: (B, N)
    Differentiable 1D Wasserstein per feature, averaged over batch.
    """
    B, N, D = x.shape
    # Zero out padding
    m = mask.unsqueeze(-1)  # (B, N, 1)
    x_real    = x    * m    # padded positions → 0
    xhat_real = xhat * m

    # Sort along particle axis — padding zeros go to front, real particles to end
    # But zeros will cluster; need to handle per-sample variable lengths carefully.
    # Easier: sort only real particles by using a large fill for padding
    BIG = 1e9
    x_fill    = x    * m + BIG * (1 - m)   # (B, N, D)
    xhat_fill = xhat * m + BIG * (1 - m)

    x_sorted,    _ = torch.sort(x_fill,    dim=1)  # (B, N, D)
    xhat_sorted, _ = torch.sort(xhat_fill, dim=1)

    # Only compare positions where both are real (mask sorted to end)
    mask_sorted, _ = torch.sort(mask, dim=1)        # (B, N) — 0s first, 1s last
    m_sorted = mask_sorted.unsqueeze(-1)            # (B, N, 1)

    diff = (x_sorted - xhat_sorted).abs() * m_sorted
    n_real = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # (B, 1)
    emd = (diff.sum(dim=1) / n_real).mean()              # scalar
    return emd

def mask_bce_loss(mask_logits, mask_true):
    # mask_true is float (0/1); BCEWithLogits handles the sigmoid internally
    return F.binary_cross_entropy_with_logits(mask_logits, mask_true)

def psae_loss(xhat, mask_logits, x, mask, swd_weight=1.0, mask_weight=1.0, num_projections=200):
    mse = masked_mse_loss(xhat, x, mask)
    swd = masked_sliced_wasserstein(x, xhat, mask, num_projections=num_projections)
    bce = mask_bce_loss(mask_logits, mask)
    loss = mse + swd_weight * swd + mask_weight * bce
    return loss, mse, swd, bce

def psae_mse_loss(xhat, mask_logits, x, mask, mask_weight=1.0):
    mse = masked_mse_loss(xhat, x, mask)
    bce = mask_bce_loss(mask_logits, mask)
    loss = mse + mask_weight * bce
    return loss, mse, bce
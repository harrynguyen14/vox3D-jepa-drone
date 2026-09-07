"""Voxel-JEPA: predict the next-frame voxel embedding in latent space.

No action is used (see idea.md section 3/7) — this is action-free spatial
representation learning on the ETH LiDAR sequence: given z_t, predict the
target encoder's embedding of frame t+1. The target encoder is an EMA copy
of the context encoder (standard JEPA/BYOL-style setup to avoid collapse).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sparse_encoder import SparseVoxelEncoder


class Predictor(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VoxelJEPA(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        patch_size: int = 8,
        grid_size: tuple[int, int, int] = (107, 107, 53),
        ema_momentum: float = 0.996,
        ema_momentum_final: float = 1.0,
    ):
        super().__init__()
        self.context_encoder = SparseVoxelEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            grid_size=grid_size,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = Predictor(embed_dim=embed_dim)
        self.ema_momentum_start = ema_momentum
        self.ema_momentum_final = ema_momentum_final

    @torch.no_grad()
    def update_target_encoder(self, progress: float = 1.0) -> None:
        """EMA-update target encoder from context encoder.

        progress: training progress in [0, 1]. Momentum is linearly
        scheduled start->final (I-JEPA/V-JEPA use 0.996->1.0) rather than
        held fixed: early on the target encoder needs to track the
        still-random context encoder closely (low momentum) so the
        predictor has a meaningful target to learn from, while late in
        training a near-frozen target (momentum->1.0) reduces target noise.
        """
        momentum = self.ema_momentum_start + (self.ema_momentum_final - self.ema_momentum_start) * progress
        for p_t, p_c in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.mul_(momentum).add_(p_c.data, alpha=1 - momentum)

    def forward(self, batch: dict, vicreg_weight: float = 0.0) -> dict:
        batch_size = batch["batch_size"]

        z_t = self.context_encoder(batch["coords_t"], batch["feats_t"], batch_size)
        z_pred = self.predictor(z_t)

        with torch.no_grad():
            z_t1_target = self.target_encoder(batch["coords_t1"], batch["feats_t1"], batch_size)

        loss = jepa_loss(z_pred, z_t1_target)
        if vicreg_weight > 0:
            # EMA + stop-gradient alone reduce collapse but don't forbid it
            # outright (the trivial constant-embedding solution still drives
            # cosine loss to 0). VICReg's variance/covariance terms are an
            # explicit second line of defense, applied to the context
            # encoder's own output z_t (not z_pred/z_t1_target, which are
            # already cosine-normalized above and would make a variance
            # penalty meaningless).
            loss = loss + vicreg_weight * vicreg_regularization(z_t)
        return {"loss": loss, "z_t": z_t, "z_pred": z_pred, "z_t1_target": z_t1_target}


def jepa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity, averaged over the batch (see idea.md section 8).

    F.normalize's default eps (1e-12) only guards the forward division; its
    backward divides by norm**3, which still overflows to Inf/NaN as norm
    approaches 0 (loss stays finite — only the *gradient* blows up, which is
    why this surfaces as "non-finite grad norm" rather than a NaN loss).
    Near-zero embeddings do occur here: a frame with very few/zero occupied
    voxels bottoms out through the encoder's nan_to_num'd padding path into
    a near-constant vector, which LayerNorm can't rescue. A much larger eps
    keeps the gradient bounded for those samples instead of only patching
    the forward value.
    """
    pred = F.normalize(pred, dim=-1, eps=1e-4)
    target = F.normalize(target, dim=-1, eps=1e-4)
    return (1 - (pred * target).sum(dim=-1)).mean()


def vicreg_regularization(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Variance + covariance terms from VICReg (Bardes et al. 2022), z: (B, D).

    variance_loss: hinge penalty pushing each dimension's per-batch std up
    to target_std — directly forbids every sample collapsing to the same
    vector, the failure mode EMA/stop-gradient only discourage.
    covariance_loss: pushes off-diagonal entries of the batch covariance
    matrix to 0 so different embedding dimensions carry independent
    information instead of duplicating one another.
    Needs B >= 2 (uses batch statistics); returns 0 otherwise.
    """
    if z.shape[0] < 2:
        return z.sum() * 0.0
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    variance_loss = torch.relu(target_std - std).mean()

    cov = (z.T @ z) / (z.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    covariance_loss = off_diag.pow(2).sum() / z.shape[1]

    return variance_loss + covariance_loss

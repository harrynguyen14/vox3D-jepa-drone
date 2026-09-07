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
        self.ema_momentum = ema_momentum

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        for p_t, p_c in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.mul_(self.ema_momentum).add_(p_c.data, alpha=1 - self.ema_momentum)

    def forward(self, batch: dict) -> dict:
        batch_size = batch["batch_size"]

        z_t = self.context_encoder(batch["coords_t"], batch["feats_t"], batch_size)
        z_pred = self.predictor(z_t)

        with torch.no_grad():
            z_t1_target = self.target_encoder(batch["coords_t1"], batch["feats_t1"], batch_size)

        loss = jepa_loss(z_pred, z_t1_target)
        return {"loss": loss, "z_t": z_t, "z_pred": z_pred, "z_t1_target": z_t1_target}


def jepa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity, averaged over the batch (see idea.md section 8)."""
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return (1 - (pred * target).sum(dim=-1)).mean()

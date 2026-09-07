"""Transformer-based voxel encoder for JEPA pretraining.

Replaces an earlier sparse-3D-CNN version. Follows the Point-JEPA / 3D-JEPA
design (see idea.md section 6): occupied voxels are grouped into fixed-size
cubic patches, each patch is embedded by a permutation-invariant mini-MLP
(PointNet-style), and a standard Transformer processes the patch tokens
with global attention. Attention over non-contiguous patches is a better
fit for JEPA's context/target-block prediction than a CNN's local
receptive field. Pure PyTorch — no sparse-conv library dependency.
"""
import torch
import torch.nn as nn


def _patch_id(coords: torch.Tensor, grid_size: tuple[int, int, int], patch_size: int) -> torch.Tensor:
    """Map each voxel to an integer id for the cubic patch it falls into."""
    nx, ny, nz = grid_size
    n_py = (ny + patch_size - 1) // patch_size
    n_pz = (nz + patch_size - 1) // patch_size
    px = coords[:, 1] // patch_size
    py = coords[:, 2] // patch_size
    pz = coords[:, 3] // patch_size
    return (coords[:, 0] * ((nx + patch_size - 1) // patch_size) + px) * n_py * n_pz + py * n_pz + pz


class PatchEmbed(nn.Module):
    """Permutation-invariant embedding of the voxels inside one patch."""

    def __init__(self, in_channels: int, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, feats: torch.Tensor, patch_idx: torch.Tensor, n_patches: int) -> torch.Tensor:
        """
        Args:
            feats: (N, C_in) voxel features
            patch_idx: (N,) long, patch id in [0, n_patches) for each voxel
            n_patches: total number of distinct patches in the batch

        Returns:
            (n_patches, embed_dim) max-pooled embedding per patch.
        """
        x = self.mlp(feats)  # (N, embed_dim)
        embed_dim = x.shape[1]
        pooled = torch.full((n_patches, embed_dim), float("-inf"), device=x.device, dtype=x.dtype)
        pooled = pooled.scatter_reduce(0, patch_idx.unsqueeze(1).expand(-1, embed_dim), x, reduce="amax")
        return torch.nan_to_num(pooled, neginf=0.0)


class SparseVoxelEncoder(nn.Module):
    """Voxel -> patch tokens -> Transformer -> pooled per-sample embedding."""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        patch_size: int = 8,
        grid_size: tuple[int, int, int] = (107, 107, 53),
        **_unused,  # accepts and ignores legacy `channels=` kwarg from older configs
    ):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(in_channels, embed_dim)
        # positional embedding from each patch's normalized centroid in grid space
        self.pos_mlp = nn.Sequential(nn.Linear(3, 64), nn.ReLU(inplace=True), nn.Linear(64, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, coords: torch.Tensor, feats: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Args:
            coords: (N, 4) int32 [batch_idx, x, y, z]
            feats: (N, C_in) float32
            batch_size: number of samples in the batch

        Returns:
            (batch_size, embed_dim) pooled embedding per sample.
        """
        if coords.shape[0] == 0:
            return torch.zeros(batch_size, self.embed_dim, device=feats.device)

        raw_patch_id = _patch_id(coords, self.grid_size, self.patch_size)
        unique_ids, patch_idx = torch.unique(raw_patch_id, return_inverse=True)
        n_patches = unique_ids.shape[0]

        patch_embeds = self.patch_embed(feats, patch_idx, n_patches)  # (P, embed_dim)

        # per-patch centroid (mean voxel coord) and owning sample index
        voxel_coords = coords[:, 1:].float()
        centroids = torch.zeros(n_patches, 3, device=feats.device)
        counts = torch.zeros(n_patches, device=feats.device)
        centroids.index_add_(0, patch_idx, voxel_coords)
        counts.index_add_(0, patch_idx, torch.ones_like(patch_idx, dtype=torch.float32))
        centroids = centroids / counts.clamp(min=1).unsqueeze(1)
        grid_norm = torch.tensor(self.grid_size, device=feats.device, dtype=torch.float32)
        tokens = patch_embeds + self.pos_mlp(centroids / grid_norm)

        # all voxels sharing a patch id come from the same sample by construction
        # (patch id already encodes batch_idx), so any member's batch_idx works
        patch_batch_idx = torch.zeros(n_patches, dtype=torch.long, device=feats.device)
        patch_batch_idx.scatter_(0, patch_idx, coords[:, 0].long())

        # Pad each sample's patch tokens to the batch max and run the
        # Transformer once over the whole batch — a per-sample Python loop
        # here would serialize what should be one batched GPU matmul.
        # "within_sample_pos" (segmented arange) is computed without a
        # Python loop: sort patches by sample, then subtract each sample's
        # first patch position from a running index.
        counts_per_sample = torch.bincount(patch_batch_idx, minlength=batch_size)
        max_tokens = max(int(counts_per_sample.max().item()), 1)

        sort_idx = torch.argsort(patch_batch_idx, stable=True)
        sorted_batch_idx = patch_batch_idx[sort_idx]
        sample_start = torch.zeros(batch_size, dtype=torch.long, device=tokens.device)
        sample_start[1:] = torch.cumsum(counts_per_sample, dim=0)[:-1]
        within_sample_pos = torch.arange(n_patches, device=tokens.device) - sample_start[sorted_batch_idx]

        padded = torch.zeros(batch_size, max_tokens, self.embed_dim, device=tokens.device, dtype=tokens.dtype)
        pad_mask = torch.ones(batch_size, max_tokens, dtype=torch.bool, device=tokens.device)
        padded[sorted_batch_idx, within_sample_pos] = tokens[sort_idx]
        pad_mask[sorted_batch_idx, within_sample_pos] = False
        # a fully-masked row (sample with zero patches) makes attention ill-defined
        pad_mask[counts_per_sample == 0, 0] = False

        # Run the Transformer in fp32 even under an outer AMP autocast, and
        # scrub NaN from its output before pooling. Two independent, known
        # PyTorch issues can make nn.TransformerEncoderLayer emit NaN for
        # PADDING rows specifically (real-token rows are unaffected):
        #   1. softmax(-inf, -inf, ..., -inf) = 0/0 = NaN. PyTorch's bool
        #      src_key_padding_mask is internally converted to -inf, and a
        #      row that ends up fully masked (or numerically close to it)
        #      produces NaN — this is independent of fp16/fp32
        #      (github.com/pytorch/pytorch/issues/64525, /issues/24816).
        #   2. fp16 attention scores can additionally underflow on heavily
        #      padded rows (this dataset's frames vary widely, ~5K-18K
        #      voxels/frame, so padding ratio varies a lot per batch).
        # Padding rows are never read past this point anyway (masked out by
        # `valid` below), so replacing their NaN with 0 is exact, not an
        # approximation — this is the documented community workaround, see
        # idea.md section 9i for the issue threads.
        with torch.autocast(device_type=padded.device.type, enabled=False):
            encoded = self.transformer(padded.float(), src_key_padding_mask=pad_mask)  # (B, max_tokens, embed_dim)
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0).to(tokens.dtype)
        valid = (~pad_mask).unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return self.norm(pooled)

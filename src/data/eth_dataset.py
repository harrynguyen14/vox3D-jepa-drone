"""PyTorch Dataset over preprocessed ETH point-cloud frames.

Reads .npy files produced by src/data/preprocess.py (one per /pointcloud
frame) and yields consecutive-frame pairs (t, t+1) for JEPA-style
pretraining: the encoder only ever sees point clouds, no action is used
here (see idea.md section 3 for why — Stage 1 is action-free spatial
representation learning).

Run preprocess.py once before using this dataset — reading directly from
.bag files per __getitem__ call re-scans the bag from the start each time
(O(N^2) per epoch) and doesn't parallelize safely across DataLoader workers.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.voxelize import VoxelConfig, points_to_sparse_voxels


@dataclass
class FramePair:
    scenario_dir: Path
    index_t: int
    index_t1: int


class ETHPointCloudPairDataset(Dataset):
    """Yields consecutive point-cloud-frame pairs voxelized for JEPA pretraining.

    Args:
        npy_dirs: one directory per scenario, each containing frames named
            "{index:06d}.npy" (as produced by preprocess.py).
        voxel_cfg: voxelization parameters.
    """

    def __init__(self, npy_dirs: list[Path], voxel_cfg: VoxelConfig | None = None):
        self.voxel_cfg = voxel_cfg or VoxelConfig()
        self._pairs: list[FramePair] = []
        self._index_dirs(npy_dirs)

    def _index_dirs(self, npy_dirs: list[Path]) -> None:
        for scenario_dir in npy_dirs:
            n_frames = len(list(scenario_dir.glob("*.npy")))
            for i in range(n_frames - 1):
                self._pairs.append(FramePair(scenario_dir, i, i + 1))

    def __len__(self) -> int:
        return len(self._pairs)

    @staticmethod
    def _load_frame(scenario_dir: Path, index: int) -> np.ndarray:
        return np.load(scenario_dir / f"{index:06d}.npy")

    def __getitem__(self, idx: int) -> dict:
        pair = self._pairs[idx]
        pts_t = self._load_frame(pair.scenario_dir, pair.index_t)
        pts_t1 = self._load_frame(pair.scenario_dir, pair.index_t1)

        coords_t, feats_t = points_to_sparse_voxels(pts_t, self.voxel_cfg)
        coords_t1, feats_t1 = points_to_sparse_voxels(pts_t1, self.voxel_cfg)

        return {
            "coords_t": torch.from_numpy(coords_t),
            "feats_t": torch.from_numpy(feats_t),
            "coords_t1": torch.from_numpy(coords_t1),
            "feats_t1": torch.from_numpy(feats_t1),
        }


def sparse_collate(batch: list[dict]) -> dict:
    """Stack a list of per-sample sparse voxel tensors into a batched sparse tensor.

    Coords use (N, 4) = [batch_idx, x, y, z]. This prepends the batch index
    for each sample before concatenating.
    """
    def _stack(key_coords: str, key_feats: str):
        coords_list, feats_list = [], []
        for b, sample in enumerate(batch):
            c = sample[key_coords]
            if c.shape[0] == 0:
                continue
            batch_col = torch.full((c.shape[0], 1), b, dtype=c.dtype)
            coords_list.append(torch.cat([batch_col, c], dim=1))
            feats_list.append(sample[key_feats])
        coords = torch.cat(coords_list, dim=0) if coords_list else torch.zeros((0, 4), dtype=torch.int32)
        feats = torch.cat(feats_list, dim=0) if feats_list else torch.zeros((0, 1), dtype=torch.float32)
        return coords, feats

    coords_t, feats_t = _stack("coords_t", "feats_t")
    coords_t1, feats_t1 = _stack("coords_t1", "feats_t1")
    return {
        "coords_t": coords_t,
        "feats_t": feats_t,
        "coords_t1": coords_t1,
        "feats_t1": feats_t1,
        "batch_size": len(batch),
    }

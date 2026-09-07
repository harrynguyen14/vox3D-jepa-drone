"""PyTorch Dataset over preprocessed, pre-voxelized ETH point-cloud frames.

Reads .npz files produced by src/data/preprocess.py (one per /pointcloud
frame, already voxelized into coords/feats) and yields consecutive-frame
pairs (t, t+1) for JEPA-style pretraining: the encoder only ever sees
point clouds, no action is used here (see idea.md section 3 for why —
Stage 1 is action-free spatial representation learning).

Run preprocess.py once before using this dataset. Voxelizing per
__getitem__ call was CPU-bound and starved the GPU (observed: 373% CPU,
~0% GPU on a Kaggle 2xT4 session) — preprocessing does it once instead.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class FramePair:
    scenario_dir: Path
    index_t: int
    index_t1: int


class ETHPointCloudPairDataset(Dataset):
    """Yields consecutive pre-voxelized frame pairs for JEPA pretraining.

    Args:
        npz_dirs: one directory per scenario, each containing frames named
            "{index:06d}.npz" with "coords"/"feats" arrays (as produced by
            preprocess.py).
    """

    def __init__(self, npz_dirs: list[Path]):
        self._pairs: list[FramePair] = []
        self._index_dirs(npz_dirs)

    def _index_dirs(self, npz_dirs: list[Path]) -> None:
        for scenario_dir in npz_dirs:
            n_frames = len(list(scenario_dir.glob("*.npz")))
            for i in range(n_frames - 1):
                self._pairs.append(FramePair(scenario_dir, i, i + 1))

    def __len__(self) -> int:
        return len(self._pairs)

    @staticmethod
    def _load_frame(scenario_dir: Path, index: int) -> tuple[np.ndarray, np.ndarray]:
        data = np.load(scenario_dir / f"{index:06d}.npz")
        return data["coords"], data["feats"]

    def __getitem__(self, idx: int) -> dict:
        pair = self._pairs[idx]
        coords_t, feats_t = self._load_frame(pair.scenario_dir, pair.index_t)
        coords_t1, feats_t1 = self._load_frame(pair.scenario_dir, pair.index_t1)

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

"""Convert a raw point cloud (N,3) into a sparse voxel representation.

Voxel grid is ego-centric: centered on the drone, axis-aligned to the
sensor frame (no rotation by heading — kept simple, matches how the
LiDAR point cloud is already expressed in body frame in the ETH bag).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class VoxelConfig:
    voxel_size: float = 0.15  # meters
    x_range: tuple[float, float] = (-8.0, 8.0)
    y_range: tuple[float, float] = (-8.0, 8.0)
    z_range: tuple[float, float] = (-4.0, 4.0)

    @property
    def grid_size(self) -> tuple[int, int, int]:
        nx = round((self.x_range[1] - self.x_range[0]) / self.voxel_size)
        ny = round((self.y_range[1] - self.y_range[0]) / self.voxel_size)
        nz = round((self.z_range[1] - self.z_range[0]) / self.voxel_size)
        return nx, ny, nz


def points_to_sparse_voxels(points: np.ndarray, cfg: VoxelConfig) -> tuple[np.ndarray, np.ndarray]:
    """Voxelize a point cloud into unique occupied voxel coordinates + features.

    Args:
        points: (N, 3) float array of xyz in the sensor/body frame.
        cfg: voxelization parameters.

    Returns:
        coords: (M, 3) int32 array of occupied voxel grid indices (x, y, z).
        feats: (M, 1) float32 array, occupancy count per voxel (density
            signal kept instead of collapsing to a binary flag).
    """
    x_min, x_max = cfg.x_range
    y_min, y_max = cfg.y_range
    z_min, z_max = cfg.z_range

    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min) & (points[:, 1] < y_max)
        & (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    pts = points[mask]
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int32), np.zeros((0, 1), dtype=np.float32)

    nx, ny, nz = cfg.grid_size
    ix = ((pts[:, 0] - x_min) / cfg.voxel_size).astype(np.int32).clip(0, nx - 1)
    iy = ((pts[:, 1] - y_min) / cfg.voxel_size).astype(np.int32).clip(0, ny - 1)
    iz = ((pts[:, 2] - z_min) / cfg.voxel_size).astype(np.int32).clip(0, nz - 1)
    voxel_idx = np.stack([ix, iy, iz], axis=1)

    coords, counts = np.unique(voxel_idx, axis=0, return_counts=True)
    feats = counts.astype(np.float32).reshape(-1, 1)
    return coords.astype(np.int32), feats

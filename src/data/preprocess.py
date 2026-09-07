"""One-time preprocessing: extract every /pointcloud frame from each ETH
.bag file into its own .npy file, so training doesn't re-scan the bag from
the start for every __getitem__ call (that was O(N^2) per epoch and doesn't
parallelize safely across DataLoader workers).

Run once per machine before training:

    python -m src.data.preprocess --dataset-dir /path/to/processed --out-dir /path/to/npy_cache
"""
import argparse
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from tqdm import tqdm

from src.data.myargs import POINTCLOUD_TOPIC


def _read_pointcloud_xyz(msg) -> np.ndarray:
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": ["<f4", "<f4", "<f4"],
            "offsets": [
                next(f.offset for f in msg.fields if f.name == "x"),
                next(f.offset for f in msg.fields if f.name == "y"),
                next(f.offset for f in msg.fields if f.name == "z"),
            ],
            "itemsize": msg.point_step,
        }
    )
    raw = np.frombuffer(bytes(msg.data), dtype=dtype, count=msg.width * msg.height)
    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
    return xyz[np.isfinite(xyz).all(axis=1)]


def extract_bag(bag_path: Path, out_dir: Path) -> int:
    scenario_dir = out_dir / bag_path.stem
    scenario_dir.mkdir(parents=True, exist_ok=True)

    n_frames = 0
    with AnyReader([bag_path]) as reader:
        connections = [c for c in reader.connections if c.topic == POINTCLOUD_TOPIC]
        for i, (connection, _timestamp, rawdata) in enumerate(
            tqdm(reader.messages(connections=connections), desc=bag_path.stem)
        ):
            msg = reader.deserialize(rawdata, connection.msgtype)
            xyz = _read_pointcloud_xyz(msg)
            np.save(scenario_dir / f"{i:06d}.npy", xyz)
            n_frames += 1
    return n_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["corridor_1", "corridor_2", "hg_1", "hg_2", "indoor", "ramp_1", "ramp_2", "stairs"],
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.scenarios:
        bag_path = dataset_dir / f"{name}.bag"
        if not bag_path.exists():
            print(f"skip {bag_path} (not found)")
            continue
        n = extract_bag(bag_path, out_dir)
        print(f"{name}: {n} frames -> {out_dir / name}")


if __name__ == "__main__":
    main()

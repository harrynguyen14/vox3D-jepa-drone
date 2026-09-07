"""Evaluate a trained Voxel-JEPA checkpoint on the full dataset.

Training only logs embedding_std/effective_rank/scenario_separation on the
last training batch (32 samples) every --log-every steps — fine as a live
health signal, too noisy/small a sample to judge the final checkpoint. This
runs the same metrics over every sample in the dataset once, no gradients.

    python -m src.eval_jepa --checkpoint checkpoints/voxel_jepa_final.pt \
        --npy-dir /path/to/npy_cache_v2
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.eth_dataset import ETHPointCloudPairDataset, sparse_collate
from src.data.myargs import parse_args as parse_train_args
from src.data.voxelize import VoxelConfig
from src.metrics import effective_rank, embedding_std, scenario_separation
from src.models.jepa import VoxelJEPA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--npy-dir", type=str, required=True)
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["corridor_1", "corridor_2", "hg_1", "hg_2", "indoor", "ramp_1", "ramp_2", "stairs"],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    # model shape must match the checkpoint being loaded — same defaults as
    # myargs.py so a default eval_jepa.py run matches a default train_jepa.py run
    default_train = parse_train_args([])
    parser.add_argument("--embed-dim", type=int, default=default_train.embed_dim)
    parser.add_argument("--depth", type=int, default=default_train.depth)
    parser.add_argument("--num-heads", type=int, default=default_train.num_heads)
    parser.add_argument("--mlp-ratio", type=float, default=default_train.mlp_ratio)
    parser.add_argument("--patch-size", type=int, default=default_train.patch_size)
    parser.add_argument("--voxel-size", type=float, default=default_train.voxel_size)
    parser.add_argument("--x-range", type=float, nargs=2, default=default_train.x_range)
    parser.add_argument("--y-range", type=float, nargs=2, default=default_train.y_range)
    parser.add_argument("--z-range", type=float, nargs=2, default=default_train.z_range)
    return parser.parse_args()


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    npz_root = Path(args.npy_dir)
    npz_dirs = [p for p in (npz_root / name for name in args.scenarios) if p.exists()]
    if not npz_dirs:
        raise FileNotFoundError(f"no preprocessed scenario folders found under {npz_root}")

    voxel_cfg = VoxelConfig(
        voxel_size=args.voxel_size,
        x_range=tuple(args.x_range),
        y_range=tuple(args.y_range),
        z_range=tuple(args.z_range),
    )
    dataset = ETHPointCloudPairDataset(npz_dirs)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=sparse_collate,
    )

    model = VoxelJEPA(
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        patch_size=args.patch_size,
        grid_size=voxel_cfg.grid_size,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.context_encoder.load_state_dict(state["context_encoder"])
    model.target_encoder.load_state_dict(state["target_encoder"])
    model.predictor.load_state_dict(state["predictor"])
    model.eval()

    all_z, all_scenarios, all_losses = [], [], []
    for batch in tqdm(dataloader, desc="eval"):
        batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        if batch["coords_t"].shape[0] == 0 or batch["coords_t1"].shape[0] == 0:
            continue
        out = model(batch)
        all_z.append(out["z_t"].cpu())
        all_scenarios.extend(batch["scenarios"])
        all_losses.append(out["loss"].item())

    z = torch.cat(all_z, dim=0)
    results = {
        "n_samples": z.shape[0],
        "mean_loss": sum(all_losses) / len(all_losses),
        "embedding_std": embedding_std(z),
        "effective_rank": effective_rank(z),
        "scenario_separation": scenario_separation(z, all_scenarios),
    }
    return results


def main() -> None:
    args = parse_args()
    results = evaluate(args)
    print("\n=== Voxel-JEPA checkpoint evaluation ===")
    print(f"checkpoint:          {args.checkpoint}")
    print(f"samples evaluated:   {results['n_samples']}")
    print(f"mean loss:           {results['mean_loss']:.4f}")
    print(f"embedding_std:       {results['embedding_std']:.4f}  (near 0 = collapsed)")
    print(f"effective_rank:      {results['effective_rank']:.2f}  (near 1 = collapsed)")
    print(f"scenario_separation: {results['scenario_separation']:.4f}  (>1 = scenes are distinguishable)")


if __name__ == "__main__":
    main()

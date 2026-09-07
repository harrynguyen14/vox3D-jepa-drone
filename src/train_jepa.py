"""Entry point: pretrain Voxel-JEPA (Stage 1) on the ETH dataset.

Run on the cloud GPU machine (see idea.md — this dev machine has no CUDA
and no spconv installed). All parameters are CLI args (see data/myargs.py):

    pip install -r requirements.txt
    python -m src.data.preprocess --dataset-dir /path/to/processed --out-dir /path/to/npy_cache
    python -m src.train_jepa --npy-dir /path/to/npy_cache --epochs 100

Speed notes (see idea.md section 9d for sources):
- AMP (autocast + GradScaler) halves memory and speeds up matmul-heavy
  Transformer layers on modern GPUs (works fine on Kaggle T4/P100).
- pin_memory + persistent_workers + prefetch_factor keep the GPU fed
  instead of stalling on .npy reads / voxelization each step.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.eth_dataset import ETHPointCloudPairDataset, sparse_collate
from src.data.myargs import parse_args
from src.data.voxelize import VoxelConfig
from src.models.jepa import VoxelJEPA


def build_dataloader(args: argparse.Namespace) -> tuple[DataLoader, tuple[int, int, int]]:
    npy_root = Path(args.npy_dir)
    npy_dirs = [npy_root / name for name in args.scenarios]
    npy_dirs = [p for p in npy_dirs if p.exists()]
    if not npy_dirs:
        raise FileNotFoundError(
            f"no preprocessed scenario folders found under {npy_root} "
            f"(run `python -m src.data.preprocess` first)"
        )

    voxel_cfg = VoxelConfig(
        voxel_size=args.voxel_size,
        x_range=tuple(args.x_range),
        y_range=tuple(args.y_range),
        z_range=tuple(args.z_range),
    )
    dataset = ETHPointCloudPairDataset(npy_dirs, voxel_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=sparse_collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    return dataloader, voxel_cfg.grid_size


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    dataloader, grid_size = build_dataloader(args)

    model = VoxelJEPA(
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        patch_size=args.patch_size,
        grid_size=grid_size,
        ema_momentum=args.ema_momentum,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.context_encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # predictor is trained jointly with the context encoder
    optimizer.add_param_group({"params": model.predictor.parameters()})
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"epoch {epoch}")
        for batch in pbar:
            batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            if batch["coords_t"].shape[0] == 0 or batch["coords_t1"].shape[0] == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                out = model(batch)
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.update_target_encoder()

            step += 1
            if step % args.log_every == 0:
                pbar.set_postfix(loss=loss.item())

        if (epoch + 1) % args.save_every_epoch == 0:
            torch.save(model.state_dict(), ckpt_dir / f"voxel_jepa_epoch{epoch+1}.pt")

    torch.save(model.state_dict(), ckpt_dir / "voxel_jepa_final.pt")


if __name__ == "__main__":
    train(parse_args())

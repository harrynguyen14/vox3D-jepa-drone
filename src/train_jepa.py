"""Entry point: pretrain Voxel-JEPA (Stage 1) on the ETH dataset.

Run on the cloud GPU machine (see idea.md — this dev machine has no CUDA
and no spconv installed). All parameters are CLI args (see data/myargs.py):

    pip install -r requirements.txt
    python -m src.data.preprocess --dataset-dir /path/to/processed --out-dir /path/to/npy_cache
    python -m src.train_jepa --npy-dir /path/to/npy_cache --epochs 100

Speed notes (see idea.md section 9d/9e for sources):
- AMP (autocast + GradScaler) halves memory and speeds up matmul-heavy
  Transformer layers on modern GPUs (works fine on Kaggle T4/P100).
- pin_memory + persistent_workers + prefetch_factor keep the GPU fed
  instead of stalling on .npz reads.
- DistributedDataParallel spawns one process per visible GPU (e.g. both
  T4s on a Kaggle session) — each process gets its own model replica and
  a DistributedSampler shard of the dataset; gradients are all-reduced
  automatically. Falls back to single-GPU/CPU when only one device (or
  none) is visible, so this still runs unmodified on this dev machine.
"""
import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.data.eth_dataset import ETHPointCloudPairDataset, sparse_collate
from src.data.myargs import parse_args
from src.data.voxelize import VoxelConfig
from src.metrics import effective_rank, embedding_std, scenario_separation
from src.models.jepa import VoxelJEPA


def build_dataset(args: argparse.Namespace) -> tuple[ETHPointCloudPairDataset, tuple[int, int, int]]:
    npz_root = Path(args.npy_dir)
    npz_dirs = [npz_root / name for name in args.scenarios]
    npz_dirs = [p for p in npz_dirs if p.exists()]
    if not npz_dirs:
        raise FileNotFoundError(
            f"no preprocessed scenario folders found under {npz_root} "
            f"(run `python -m src.data.preprocess` first)"
        )

    # grid_size only depends on voxel config, not on any point cloud —
    # voxelization itself already happened once in preprocess.py
    voxel_cfg = VoxelConfig(
        voxel_size=args.voxel_size,
        x_range=tuple(args.x_range),
        y_range=tuple(args.y_range),
        z_range=tuple(args.z_range),
    )
    dataset = ETHPointCloudPairDataset(npz_dirs)
    return dataset, voxel_cfg.grid_size


def build_dataloader(
    dataset: ETHPointCloudPairDataset, args: argparse.Namespace, rank: int | None, world_size: int
) -> DataLoader:
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=sparse_collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    ), sampler


def train_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    is_distributed = world_size > 1
    if is_distributed:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    is_main = rank == 0
    use_amp = args.amp and device.type == "cuda"

    dataset, grid_size = build_dataset(args)
    dataloader, sampler = build_dataloader(dataset, args, rank if is_distributed else None, world_size)

    model = VoxelJEPA(
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        patch_size=args.patch_size,
        grid_size=grid_size,
        ema_momentum=args.ema_momentum,
    ).to(device)

    # DDP wraps context_encoder + predictor (the params that receive
    # gradients); target_encoder stays a plain module updated by EMA only,
    # each rank's copy tracks its own (identical, since DDP keeps
    # context_encoder in sync across ranks) context_encoder locally.
    if is_distributed:
        model.context_encoder = DDP(model.context_encoder, device_ids=[rank])
        model.predictor = DDP(model.predictor, device_ids=[rank])

    def save_checkpoint(path: Path) -> None:
        # unwrap DDP so the saved state_dict matches plain VoxelJEPA (no
        # "module." prefix), so Stage 2 can load it without knowing DDP was used
        ctx = model.context_encoder.module if is_distributed else model.context_encoder
        pred = model.predictor.module if is_distributed else model.predictor
        state = {
            "context_encoder": ctx.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
            "predictor": pred.state_dict(),
        }
        torch.save(state, path)

    optimizer = torch.optim.AdamW(
        model.context_encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    optimizer.add_param_group({"params": model.predictor.parameters()})
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    ckpt_dir = Path(args.checkpoint_dir)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"epoch {epoch}", disable=not is_main)
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
            if is_main and step % args.log_every == 0:
                z_t = out["z_t"].detach()
                pbar.set_postfix(
                    loss=loss.item(),
                    std=embedding_std(z_t),
                    rank=effective_rank(z_t),
                    sep=scenario_separation(z_t, batch["scenarios"]),
                )

        if is_main and (epoch + 1) % args.save_every_epoch == 0:
            save_checkpoint(ckpt_dir / f"voxel_jepa_epoch{epoch+1}.pt")

    if is_main:
        save_checkpoint(ckpt_dir / "voxel_jepa_final.pt")

    if is_distributed:
        dist.destroy_process_group()


def train(args: argparse.Namespace) -> None:
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if world_size > 1:
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train_worker(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    train(parse_args())

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

Stability notes (see idea.md section 9g/9h): a NaN loss was observed after
6 stable epochs on Kaggle, likely AMP fp16 overflow. Beyond the isfinite
guard (section 9g), this adds:
- Gradient clipping (--grad-clip-norm, default 1.0): unscales AMP
  gradients before clipping, since the clip threshold is meaningless
  against the loss-scale factor otherwise.
- LR schedule: linear warmup (--warmup-steps) then cosine decay to
  --min-lr-ratio * --lr — a large initial LR hitting an under-warmed
  Transformer is a common source of early-training blowups.
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine_schedule(warmup_steps: int, total_steps: int, min_lr_ratio: float):
    """LR multiplier: linear warmup to 1.0, then cosine decay to min_lr_ratio."""
    import math

    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return fn


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

    # seed before model creation so weight init is identical across DDP
    # ranks (required — DDP assumes replicas start in sync), then re-seed
    # per-rank afterwards so things like dataloader worker shuffling don't
    # coincidentally line up across ranks
    set_seed(args.seed)

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
        set_seed(args.seed + rank)

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

    total_steps = args.epochs * len(dataloader)
    lr_schedule = warmup_cosine_schedule(args.warmup_steps, total_steps, args.min_lr_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    ckpt_dir = Path(args.checkpoint_dir)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    nan_streak = 0
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

            if not torch.isfinite(loss):
                # loss went NaN/Inf (e.g. AMP fp16 overflow) — GradScaler
                # only catches Inf/NaN *gradients*, not a bad loss going
                # into backward, so this has to be checked explicitly.
                # Skip the step instead of poisoning the model permanently.
                if is_main:
                    tqdm.write(f"[epoch {epoch} step {step}] non-finite loss ({loss.item()}), skipping batch")
                nan_streak += 1
                if nan_streak >= args.nan_streak_limit:
                    # this many *consecutive* skips means the model's own
                    # weights are already NaN (every batch now produces NaN
                    # regardless of input) — a single bad batch, in
                    # contrast, recovers on the next one. Continuing wastes
                    # GPU time producing nothing but more skipped batches.
                    if is_main:
                        tqdm.write(
                            f"[epoch {epoch} step {step}] {nan_streak} consecutive non-finite losses — "
                            f"model weights are almost certainly NaN, stopping. "
                            f"Re-run from the last good checkpoint with a lower --lr or smaller --grad-clip-norm."
                        )
                    if is_distributed:
                        dist.destroy_process_group()
                    return
                continue
            nan_streak = 0

            scaler.scale(loss).backward()
            all_params = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
            if args.grad_clip_norm > 0:
                # gradients must be unscaled before clipping, or the clip
                # threshold is meaningless against the AMP loss-scale factor
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.grad_clip_norm)
                if not torch.isfinite(grad_norm):
                    # An Inf gradient divided by clip_grad_norm_'s Inf-valued
                    # total_norm can come out as a finite-looking NaN/0, which
                    # then hides the overflow from GradScaler's own Inf check
                    # inside scaler.step() — so it applies a poisoned update
                    # instead of skipping it. Skip explicitly here instead of
                    # trusting scaler.step() to catch it after clipping.
                    if is_main:
                        tqdm.write(f"[epoch {epoch} step {step}] non-finite grad norm ({grad_norm.item()}), skipping batch")
                    # scaler.update() must run every step (even a skipped
                    # one) to reset the "already unscaled" internal flag —
                    # otherwise the next iteration's unscale_() raises
                    # "unscale_() has already been called... since the
                    # last update()".
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    nan_streak += 1
                    if nan_streak >= args.nan_streak_limit:
                        if is_main:
                            tqdm.write(f"[epoch {epoch} step {step}] {nan_streak} consecutive non-finite grads, stopping.")
                        if is_distributed:
                            dist.destroy_process_group()
                        return
                    continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.update_target_encoder()

            step += 1
            if is_main and step % args.log_every == 0:
                z_t = out["z_t"].detach()
                # tqdm sorts kwargs alphabetically and truncates long lines,
                # which silently dropped "std" off the end on narrow
                # terminals — pass an already-ordered dict instead, which
                # tqdm keeps in insertion order (std first: the most
                # reliable collapse signal, see idea.md section 9f)
                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "std": embedding_std(z_t),
                        "rank": effective_rank(z_t),
                        "sep": scenario_separation(z_t, batch["scenarios"]),
                        "lr": scheduler.get_last_lr()[0],
                    }
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

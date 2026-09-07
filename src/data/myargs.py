"""All CLI-configurable parameters for Stage 1 (Voxel-JEPA pretraining).

Single source of truth for data/model/train config — replaces
configs/jepa_stage1.yaml. Call parse_args() from a script's __main__.
"""
import argparse
import os


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--pointcloud-topic", type=str, default="/pointcloud")
    parser.add_argument("--npy-dir", type=str, default="D:/drone/npy_cache")
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["corridor_1", "corridor_2", "hg_1", "hg_2", "indoor", "ramp_1", "ramp_2", "stairs"],
    )
    parser.add_argument("--voxel-size", type=float, default=0.15)
    parser.add_argument("--x-range", type=float, nargs=2, default=[-8.0, 8.0])
    parser.add_argument("--y-range", type=float, nargs=2, default=[-8.0, 8.0])
    parser.add_argument("--z-range", type=float, nargs=2, default=[-4.0, 4.0])

    # model
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--ema-momentum", type=float, default=0.996, help="EMA momentum at step 0")
    parser.add_argument("--ema-momentum-final", type=float, default=1.0, help="EMA momentum at the final step (linear schedule, per I-JEPA/V-JEPA)")
    parser.add_argument("--vicreg-weight", type=float, default=0.025, help="weight of VICReg variance/covariance collapse regularizer, 0 disables it")

    # train
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4, help="weight decay at step 0")
    parser.add_argument("--weight-decay-final", type=float, default=1.0e-4, help="weight decay at the final step (linear schedule); set > --weight-decay to ramp up like I-JEPA/V-JEPA")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/jepa_stage1")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every-epoch", type=int, default=5)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="0 disables clipping")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01, help="cosine decay floor, as a fraction of --lr")
    parser.add_argument(
        "--nan-streak-limit",
        type=int,
        default=20,
        help="stop training after this many consecutive non-finite losses (model weights are likely NaN)",
    )
    parser.add_argument("--seed", type=int, default=42, help="for reproducibility, e.g. to re-debug a NaN run")

    return parser.parse_args(argv)


# default topic, importable without triggering argument parsing
POINTCLOUD_TOPIC = parse_args([]).pointcloud_topic

"""Run an exact-in-law matrix-free single-datum experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .lazy_gaussian import LazyGaussianLayerBank
from .lazy_training import initialize_lazy, lazy_trajectory
from .run import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--activation", default="identity")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.width, args.depth, args.steps) < 1:
        parser.error("width, depth, and steps must be positive")
    device, dtype = resolve_device(args.device), getattr(torch, args.dtype)
    w0, v, bank = initialize_lazy(args.width, args.depth, args.steps, args.seed,
                                  device, dtype)
    trajectory = lazy_trajectory(w0, v, bank, args.steps, args.learning_rate,
                                 activation=args.activation)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "trajectory.npz", prediction=trajectory)
    lazy_bytes = LazyGaussianLayerBank.storage_bytes(args.depth, args.width,
                                                     args.steps, dtype)
    dense_bytes = args.depth * args.width * args.width * dtype.itemsize
    metadata = {
        **vars(args), "output": str(args.output), "device": str(device),
        "torch_version": torch.__version__, "lazy_storage_bytes": lazy_bytes,
        "dense_storage_bytes": dense_bytes,
        "dense_to_lazy_ratio": dense_bytes / lazy_bytes,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

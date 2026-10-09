"""Run the dense synthetic width-depth grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import make_dataset
from .dense import train_dense


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment, data_config = config["experiment"], config["data"]
    device = resolve_device(experiment.get("device", "auto"))
    dtype = getattr(torch, experiment.get("dtype", "float32"))
    data = make_dataset(
        int(data_config["dimension"]), int(data_config["train_size"]),
        int(data_config["test_size"]), int(data_config["seed"]), dtype,
    )
    root = Path(experiment.get("output_dir", "results")) / experiment["name"]
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for activation in experiment["activations"]:
        for width in map(int, experiment["widths"]):
            for depth in map(int, experiment["depths"]):
                for seed in map(int, experiment["model_seeds"]):
                    output = root / activation / f"n_{width}" / f"L_{depth}" / f"seed_{seed}.npz"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    history = train_dense(
                        width, depth, activation, seed, data,
                        int(experiment["steps"]), float(experiment["learning_rate"]),
                        device, dtype,
                    )
                    np.savez_compressed(output, **history)
                    record = {
                        "activation": activation, "width": width, "depth": depth,
                        "seed": seed, "initial_train_loss": float(history["train_loss"][0]),
                        "final_train_loss": float(history["train_loss"][-1]),
                        "final_test_loss": float(history["test_loss"][-1]),
                    }
                    records.append(record)
                    print(json.dumps(record), flush=True)
    metadata = {
        "config": config, "torch_version": torch.__version__,
        "device": str(device), "cuda_version": torch.version.cuda,
        "runs": records,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                         encoding="utf-8")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)

"""Plot empirical prediction error across a completed width-depth grid."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_mean_prediction(directory: Path) -> np.ndarray:
    paths = sorted(directory.glob("seed_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no seed trajectories in {directory}")
    return np.mean([np.load(path)["test_prediction"] for path in paths], axis=0)


def plot(results: Path, output: Path) -> None:
    activations = sorted(path.name for path in results.iterdir() if path.is_dir())
    if not activations:
        raise FileNotFoundError(f"no activation directories in {results}")
    figure, axes = plt.subplots(1, len(activations), figsize=(5 * len(activations), 4),
                                squeeze=False, constrained_layout=True)
    for axis, activation in zip(axes[0], activations):
        base = results / activation
        widths = sorted(int(path.name.removeprefix("n_")) for path in base.glob("n_*"))
        depths = sorted(int(path.name.removeprefix("L_"))
                        for path in (base / f"n_{widths[0]}").glob("L_*"))
        reference = load_mean_prediction(base / f"n_{widths[-1]}" / f"L_{depths[-1]}")
        errors = np.empty((len(depths), len(widths)))
        for row, depth in enumerate(depths):
            for column, width in enumerate(widths):
                prediction = load_mean_prediction(base / f"n_{width}" / f"L_{depth}")
                errors[row, column] = np.sqrt(np.mean((prediction - reference) ** 2))
        image = axis.imshow(errors, origin="lower", aspect="auto")
        axis.set(title=activation, xlabel="width n", ylabel="depth L",
                 xticks=range(len(widths)), xticklabels=widths,
                 yticks=range(len(depths)), yticklabels=depths)
        figure.colorbar(image, ax=axis, label="trajectory RMS error to largest (n, L)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.results, args.output)

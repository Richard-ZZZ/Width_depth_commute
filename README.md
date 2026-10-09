# Width and Depth Limits Commute During Training in Linear Residual Networks

This repository contains the focused reproducibility code for the paper
**“Width and Depth Limits Commute During Training in Linear Residual Networks.”**

It implements finite-width, finite-depth residual networks

\[
y_0=W_0x, \qquad
y_l=y_{l-1}+L^{-1/2}W_l\phi(y_{l-1}), \qquad
f(x)=V^\top y_L.
\]

The endpoint matrices `W0` and `V` are frozen. Only the residual matrices
`W1,...,WL` are trained by full-batch vanilla gradient descent. The main
experiments use `identity` (linear ResNet) and `tanh` (nonlinear comparison).

## Install

Python 3.10 or newer is required. A CPU installation is enough for tests and
the smoke experiment.

```bash
conda env create -f environment.yml
conda activate width-depth-limits
```

Alternatively, install into an existing environment. Install the appropriate
PyTorch build for your CPU or CUDA system first, then run:

```bash
python -m pip install -e '.[test]'
```

## Quick verification

```bash
make test
make figures
```

The first command checks deterministic dense training and verifies the lazy
Gaussian implementation pathwise against explicitly materialized matrices.
The second runs a small CPU width-depth grid and writes `figures/smoke.png`.

## Synthetic width-depth experiment

Run the representative larger grid with:

```bash
python -m resnet_limits.run --config configs/paper_synthetic.yaml
python -m resnet_limits.plot \
  --results results/paper_synthetic \
  --output figures/paper_synthetic.png
```

Each configuration evaluates every requested width, depth, activation, and
model seed on one fixed synthetic regression dataset. It saves the complete
test-prediction trajectory and train/test losses. The plot averages prediction
trajectories over seeds and compares each `(n,L)` pair with the largest pair in
the same grid. This finite reference is an empirical diagnostic, not a proof.

Edit the YAML lists to match the compute budget available on your system.
`device: auto` selects CUDA when available and otherwise uses the CPU.

## Lazy Gaussian implementation

For a single training datum, each initially Gaussian residual matrix is
observed through only a few forward and transpose products. The module
`resnet_limits.lazy_gaussian` samples these products from their exact
conditional Gaussian law without materializing the matrices. For `T` gradient
steps, storage decreases from `O(L n^2)` to `O(T L n)`.

Run a matrix-free experiment with:

```bash
python -m resnet_limits.lazy_run \
  --width 4096 --depth 256 --steps 10 \
  --device auto --output results/lazy_n4096_L256
```

The output metadata reports both dense and lazy storage requirements. Lazy
runs are exact in distribution. A seed identifies the sequence of conditional
draws; it does not identify the same hypothetical dense matrices under a
different query order.

## Repository layout

```text
configs/                  experiment settings
resnet_limits/model.py    architecture and initialization
resnet_limits/dense.py    dense full-batch training
resnet_limits/run.py      synthetic grid runner
resnet_limits/plot.py     width-depth diagnostic figure
resnet_limits/lazy_*.py   exact-in-law matrix-free implementation
tests/                    focused correctness and reproducibility tests
.github/workflows/        CPU tests on Python 3.10--3.12
```

## Reproducibility notes

- Dataset and model randomness use explicit seeds.
- Results include the configuration, PyTorch version, device, and CUDA version.
- GPU floating-point results can vary slightly across hardware and software
  versions even when seeds match.
- The full-scale configuration can be expensive; validate the smoke workflow
  before increasing widths or depths.

## Citation

Please cite the accompanying paper. A machine-readable `CITATION.cff` will be
added with the public paper metadata before the archival release.

## License

Released under the [MIT License](LICENSE).

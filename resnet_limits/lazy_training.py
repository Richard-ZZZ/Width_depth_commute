"""Matrix-free single-datum training with lazy Gaussian operators."""
from __future__ import annotations

import math

import numpy as np
import torch

from .lazy_gaussian import LazyGaussianLayerBank
from .model import activate, activation_derivative


def initialize_lazy(width: int, depth: int, steps: int, seed: int,
                    device: torch.device, dtype: torch.dtype,
                    dense_reference: torch.Tensor | None = None):
    generator = torch.Generator(device=device).manual_seed(seed)
    w0 = torch.randn(width, generator=generator, device=device, dtype=dtype)
    v = torch.randn(width, generator=generator, device=device, dtype=dtype) / width
    bank = LazyGaussianLayerBank(
        depth, width, steps + 1, steps, device=device, dtype=dtype,
        generator=generator, dense_reference=dense_reference,
        assume_new_directions=True,
    )
    return w0, v, bank


@torch.no_grad()
def lazy_trajectory(w0: torch.Tensor, v: torch.Tensor, bank: LazyGaussianLayerBank,
                    steps: int, learning_rate: float, target: float = 1.0,
                    activation: str = "identity") -> np.ndarray:
    """Return f_0,...,f_steps under exact-in-law full-batch GD."""
    depth, width = bank.depth, bank.width
    scale = 1.0 / math.sqrt(depth)
    state_coordinates = w0.new_zeros(depth, steps, bank.max_forward_rank)
    adjoint_coordinates = w0.new_zeros(depth, steps, bank.max_transpose_rank)
    values: list[float] = []
    for step in range(steps + 1):
        state = w0
        derivatives = (w0.new_empty(depth, width)
                       if activation != "identity" and step < steps else None)
        for layer in range(depth):
            query = activate(state, activation)
            if derivatives is not None:
                derivatives[layer].copy_(activation_derivative(state, activation))
            base, coordinates = bank.forward(layer, query,
                                             assume_new_direction=(layer > 0))
            effective = base
            if step:
                dots = state_coordinates[layer, :step] @ coordinates
                update = bank.p_vector(layer, dots @ adjoint_coordinates[layer, :step])
                effective = effective - learning_rate * scale * update
            if step < steps:
                state_coordinates[layer, step].copy_(coordinates)
            state = state + scale * effective
        prediction = torch.dot(state, v)
        value = float(prediction)
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite prediction at step {step}")
        values.append(value)
        if step == steps:
            break
        if value == target:
            values.extend([value] * (steps - step))
            break
        adjoint = (prediction - target) * v
        for layer in range(depth - 1, -1, -1):
            base, coordinates = bank.transpose(layer, adjoint)
            effective = base
            if step:
                dots = adjoint_coordinates[layer, :step] @ coordinates
                update = bank.q_vector(layer, dots @ state_coordinates[layer, :step])
                effective = effective - learning_rate * scale * update
            adjoint_coordinates[layer, step].copy_(coordinates)
            residual_adjoint = scale * effective
            if derivatives is not None:
                residual_adjoint *= derivatives[layer]
            adjoint = adjoint + residual_adjoint
    return np.asarray(values)

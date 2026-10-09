"""Finite-width, finite-depth residual networks."""
from __future__ import annotations

import math

import torch
from torch import nn


ACTIVATIONS = ("identity", "tanh", "relu", "gelu", "silu")


def activate(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity":
        return x
    if name == "tanh":
        return torch.tanh(x)
    if name == "relu":
        return torch.relu(x)
    if name == "gelu":
        return torch.nn.functional.gelu(x)
    if name == "silu":
        return torch.nn.functional.silu(x)
    raise ValueError(f"unknown activation {name!r}; choose from {ACTIVATIONS}")


def activation_derivative(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity":
        return torch.ones_like(x)
    if name == "tanh":
        value = torch.tanh(x)
        return 1.0 - value.square()
    if name == "relu":
        return (x > 0).to(x.dtype)
    if name == "gelu":
        cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * x.square()) / math.sqrt(2.0 * math.pi)
        return cdf + x * pdf
    if name == "silu":
        sigmoid = torch.sigmoid(x)
        return sigmoid * (1.0 + x * (1.0 - sigmoid))
    raise ValueError(f"unknown activation {name!r}; choose from {ACTIVATIONS}")


def initialize(width: int, depth: int, input_dim: int, seed: int,
               *, device: torch.device, dtype: torch.dtype):
    """Initialize W0, residual matrices, and V with the paper's scaling."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    w0 = torch.randn(width, input_dim, generator=generator, dtype=dtype) / math.sqrt(input_dim)
    residual = [torch.randn(width, width, generator=generator, dtype=dtype) /
                math.sqrt(width) for _ in range(depth)]
    v = torch.randn(width, generator=generator, dtype=dtype) / width
    return w0.to(device), [weight.to(device) for weight in residual], v.to(device)


class ResidualNetwork(nn.Module):
    """y_l=y_(l-1)+L^-1/2 W_l phi(y_(l-1)); f(x)=V^T y_L."""

    def __init__(self, w0: torch.Tensor, residual: list[torch.Tensor],
                 v: torch.Tensor, activation: str = "identity") -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unsupported activation {activation!r}")
        self.w0 = nn.Parameter(w0.detach().clone(), requires_grad=False)
        self.residual = nn.ParameterList([
            nn.Parameter(weight.detach().clone()) for weight in residual
        ])
        self.v = nn.Parameter(v.detach().clone(), requires_grad=False)
        self.activation = activation
        self.scale = 1.0 / math.sqrt(len(residual))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x @ self.w0.T
        for weight in self.residual:
            state = state + self.scale * (activate(state, self.activation) @ weight.T)
        return state @ self.v

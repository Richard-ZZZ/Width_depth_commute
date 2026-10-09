"""Exact conditional sampling of Gaussian matrix-vector products.

The algorithm stores only the query/image subspaces observed during training.
Its memory is O(T L n), compared with O(L n^2) for dense residual matrices.
"""
from __future__ import annotations

import math

import torch


class LazyGaussianLayerBank:
    """Independent N(0, 1/width) operators sampled as they are queried."""

    def __init__(self, depth: int, width: int, max_forward_rank: int,
                 max_transpose_rank: int, *, device: torch.device,
                 dtype: torch.dtype, generator: torch.Generator | None = None,
                 dense_reference: torch.Tensor | None = None,
                 assume_new_directions: bool = False) -> None:
        if min(depth, width, max_forward_rank) < 1 or max_transpose_rank < 0:
            raise ValueError("invalid layer-bank dimensions")
        self.depth, self.width = depth, width
        self.max_forward_rank = max_forward_rank
        self.max_transpose_rank = max_transpose_rank
        self.device, self.dtype, self.generator = device, dtype, generator
        self.assume_new_directions = assume_new_directions
        self.gaussian_scale = 1.0 / math.sqrt(width)
        self.q = torch.empty(depth, max_forward_rank, width, device=device, dtype=dtype)
        self.aq = torch.empty_like(self.q)
        self.p = torch.empty(depth, max_transpose_rank, width, device=device, dtype=dtype)
        self.atp = torch.empty_like(self.p)
        self.q_rank = torch.zeros(depth, dtype=torch.int16)
        self.p_rank = torch.zeros(depth, dtype=torch.int16)
        if dense_reference is not None and tuple(dense_reference.shape) != (depth, width, width):
            raise ValueError("dense_reference must have shape (depth, width, width)")
        self.dense_reference = (dense_reference.to(device=device, dtype=dtype)
                                if dense_reference is not None else None)

    @staticmethod
    def storage_bytes(depth: int, width: int, steps: int, dtype: torch.dtype) -> int:
        return depth * width * (4 * steps + 2) * torch.empty((), dtype=dtype).element_size()

    @staticmethod
    def _orthogonal_residual(vector: torch.Tensor, basis: torch.Tensor):
        if basis.shape[0] == 0:
            return vector.new_empty(0), vector.clone()
        coefficients = basis @ vector
        residual = vector - coefficients @ basis
        correction = basis @ residual
        return coefficients + correction, residual - correction @ basis

    @staticmethod
    def _is_new(norm: torch.Tensor, vector: torch.Tensor) -> bool:
        threshold = 64 * torch.finfo(vector.dtype).eps * max(1.0, float(vector.norm()))
        return float(norm) > threshold

    def _normal(self) -> torch.Tensor:
        return torch.randn(self.width, device=self.device, dtype=self.dtype,
                           generator=self.generator) * self.gaussian_scale

    @torch.no_grad()
    def forward(self, layer: int, vector: torch.Tensor,
                assume_new_direction: bool | None = None):
        rank = int(self.q_rank[layer])
        coefficients, residual = self._orthogonal_residual(vector, self.q[layer, :rank])
        result = (coefficients @ self.aq[layer, :rank] if rank else torch.zeros_like(vector))
        norm = residual.norm()
        assume_new = self.assume_new_directions if assume_new_direction is None else assume_new_direction
        if assume_new or self._is_new(norm, vector):
            if rank >= self.max_forward_rank:
                raise RuntimeError(f"forward query rank exceeded at layer {layer}")
            direction = residual / norm
            if self.dense_reference is not None:
                image = self.dense_reference[layer] @ direction
            else:
                transpose_rank = int(self.p_rank[layer])
                basis = self.p[layer, :transpose_rank]
                if transpose_rank:
                    known = (self.atp[layer, :transpose_rank] @ direction) @ basis
                    fresh = self._normal()
                    image = known + fresh - (basis @ fresh) @ basis
                else:
                    image = self._normal()
            self.q[layer, rank].copy_(direction)
            self.aq[layer, rank].copy_(image)
            self.q_rank[layer] = rank + 1
            result = result + norm * image
            coefficients = torch.cat((coefficients, norm.reshape(1)))
        padded = vector.new_zeros(self.max_forward_rank)
        padded[:coefficients.numel()] = coefficients
        return result, padded

    @torch.no_grad()
    def transpose(self, layer: int, vector: torch.Tensor,
                  assume_new_direction: bool | None = None):
        rank = int(self.p_rank[layer])
        coefficients, residual = self._orthogonal_residual(vector, self.p[layer, :rank])
        result = (coefficients @ self.atp[layer, :rank] if rank else torch.zeros_like(vector))
        norm = residual.norm()
        assume_new = self.assume_new_directions if assume_new_direction is None else assume_new_direction
        if assume_new or self._is_new(norm, vector):
            if rank >= self.max_transpose_rank:
                raise RuntimeError(f"transpose query rank exceeded at layer {layer}")
            direction = residual / norm
            if self.dense_reference is not None:
                image = self.dense_reference[layer].T @ direction
            else:
                forward_rank = int(self.q_rank[layer])
                basis = self.q[layer, :forward_rank]
                if forward_rank:
                    known = (self.aq[layer, :forward_rank] @ direction) @ basis
                    fresh = self._normal()
                    image = known + fresh - (basis @ fresh) @ basis
                else:
                    image = self._normal()
            self.p[layer, rank].copy_(direction)
            self.atp[layer, rank].copy_(image)
            self.p_rank[layer] = rank + 1
            result = result + norm * image
            coefficients = torch.cat((coefficients, norm.reshape(1)))
        padded = vector.new_zeros(self.max_transpose_rank)
        padded[:coefficients.numel()] = coefficients
        return result, padded

    def q_vector(self, layer: int, coefficients: torch.Tensor) -> torch.Tensor:
        rank = int(self.q_rank[layer])
        return coefficients[:rank] @ self.q[layer, :rank]

    def p_vector(self, layer: int, coefficients: torch.Tensor) -> torch.Tensor:
        rank = int(self.p_rank[layer])
        return coefficients[:rank] @ self.p[layer, :rank]

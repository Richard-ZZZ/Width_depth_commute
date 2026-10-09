"""Deterministic synthetic regression data."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Dataset:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def make_dataset(dimension: int, train_size: int, test_size: int, seed: int,
                 dtype: torch.dtype = torch.float32) -> Dataset:
    generator = torch.Generator().manual_seed(seed)
    teacher = torch.randn(dimension, generator=generator, dtype=dtype)
    teacher /= teacher.norm()
    x_train = torch.randn(train_size, dimension, generator=generator, dtype=dtype)
    x_test = torch.randn(test_size, dimension, generator=generator, dtype=dtype)
    return Dataset(x_train, x_train @ teacher, x_test, x_test @ teacher)

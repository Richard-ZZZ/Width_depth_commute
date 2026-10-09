import math

import numpy as np
import torch

from resnet_limits.lazy_gaussian import LazyGaussianLayerBank
from resnet_limits.lazy_training import lazy_trajectory
from resnet_limits.model import ResidualNetwork


def dense_trajectory(w0, v, dense, steps, learning_rate, activation="identity"):
    model = ResidualNetwork(w0[:, None], list(dense), v, activation=activation)
    optimizer = torch.optim.SGD(model.residual, lr=learning_rate)
    x, target = torch.ones(1, 1, dtype=w0.dtype), torch.ones(1, dtype=w0.dtype)
    values = []
    for step in range(steps + 1):
        prediction = model(x)
        values.append(float(prediction.detach()))
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            (0.5 * (prediction - target).square().mean()).backward()
            optimizer.step()
    return np.asarray(values)


def test_forward_transpose_consistency_and_linearity():
    generator = torch.Generator().manual_seed(123)
    bank = LazyGaussianLayerBank(1, 12, 3, 3, device=torch.device("cpu"),
                                 dtype=torch.float64, generator=generator)
    x1 = torch.randn(12, generator=generator, dtype=torch.float64)
    x2 = torch.randn(12, generator=generator, dtype=torch.float64)
    y = torch.randn(12, generator=generator, dtype=torch.float64)
    ax1, _ = bank.forward(0, x1)
    ax2, _ = bank.forward(0, x2)
    aty, _ = bank.transpose(0, y)
    combination, _ = bank.forward(0, 2 * x1 - 0.5 * x2)
    torch.testing.assert_close(y @ ax1, x1 @ aty)
    torch.testing.assert_close(combination, 2 * ax1 - 0.5 * ax2)


def test_lazy_training_matches_materialized_matrices_pathwise():
    width, depth, steps = 9, 7, 3
    for activation in ("identity", "tanh"):
        generator = torch.Generator().manual_seed(91)
        w0 = torch.randn(width, generator=generator, dtype=torch.float64)
        v = torch.randn(width, generator=generator, dtype=torch.float64) / width
        dense = torch.randn(depth, width, width, generator=generator,
                            dtype=torch.float64) / math.sqrt(width)
        expected = dense_trajectory(w0, v, dense, steps, 1.0, activation)
        bank = LazyGaussianLayerBank(
            depth, width, steps + 1, steps, device=torch.device("cpu"),
            dtype=torch.float64, dense_reference=dense, assume_new_directions=True,
        )
        actual = lazy_trajectory(w0, v, bank, steps, 1.0,
                                 activation=activation)
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)


def test_lazy_storage_is_linear_in_width_and_depth():
    actual = LazyGaussianLayerBank.storage_bytes(3072, 61440, 10, torch.float32)
    assert actual == 3072 * 61440 * 42 * 4
    assert actual / 1e9 < 32

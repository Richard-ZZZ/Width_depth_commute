import numpy as np
import torch

from resnet_limits.data import make_dataset
from resnet_limits.dense import train_dense
from resnet_limits.model import ResidualNetwork, initialize


def test_only_residual_matrices_are_trainable():
    parameters = initialize(8, 3, 4, 0, device=torch.device("cpu"),
                            dtype=torch.float64)
    model = ResidualNetwork(*parameters)
    assert not model.w0.requires_grad
    assert not model.v.requires_grad
    assert all(weight.requires_grad for weight in model.residual)


def test_dense_run_is_exactly_reproducible():
    data = make_dataset(4, 8, 12, 7, torch.float64)
    arguments = (12, 3, "identity", 11, data, 2, 0.1,
                 torch.device("cpu"), torch.float64)
    first = train_dense(*arguments)
    second = train_dense(*arguments)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_linear_and_nonlinear_shapes():
    data = make_dataset(3, 7, 9, 0)
    for activation in ("identity", "tanh"):
        result = train_dense(8, 2, activation, 0, data, 2, 0.1,
                             torch.device("cpu"), torch.float32)
        assert result["train_loss"].shape == (3,)
        assert result["test_prediction"].shape == (3, 9)
        assert np.isfinite(result["test_prediction"]).all()

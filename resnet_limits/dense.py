"""Dense synthetic training utilities."""
from __future__ import annotations

import numpy as np
import torch

from .model import ResidualNetwork, initialize


def train_dense(width: int, depth: int, activation: str, model_seed: int,
                data, steps: int, learning_rate: float, device: torch.device,
                dtype: torch.dtype) -> dict[str, np.ndarray]:
    tensors = [value.to(device=device, dtype=dtype) for value in
               (data.x_train, data.y_train, data.x_test, data.y_test)]
    x_train, y_train, x_test, y_test = tensors
    model = ResidualNetwork(
        *initialize(width, depth, x_train.shape[1], model_seed,
                    device=device, dtype=dtype),
        activation=activation,
    )
    optimizer = torch.optim.SGD(model.residual, lr=learning_rate)
    history = {"train_loss": [], "test_loss": [], "test_prediction": []}
    for step in range(steps + 1):
        with torch.no_grad():
            train_prediction = model(x_train)
            test_prediction = model(x_test)
            history["train_loss"].append(0.5 * (train_prediction - y_train).square().mean().item())
            history["test_loss"].append(0.5 * (test_prediction - y_test).square().mean().item())
            history["test_prediction"].append(test_prediction.cpu().numpy())
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_train)
        loss = 0.5 * (prediction - y_train).square().mean()
        loss.backward()
        optimizer.step()
    return {key: np.asarray(value) for key, value in history.items()}

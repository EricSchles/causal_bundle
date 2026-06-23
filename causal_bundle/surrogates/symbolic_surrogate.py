# FILE: causalbundle/surrogates/symbolic_surrogate.py

import torch
import torch.nn as nn


class SymbolicSurrogate(nn.Module):
    """
    Differentiable approximation to the current symbolic model.
    """

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z)


def train_surrogate(
    surrogate,
    Z,
    symbolic_predictions,
    epochs=25,
    lr=1e-3,
):
    """
    Distill PySR into a differentiable surrogate.
    """

    optimizer = torch.optim.Adam(
        surrogate.parameters(),
        lr=lr,
    )

    Z_t = torch.tensor(Z, dtype=torch.float32)
    Y_t = torch.tensor(symbolic_predictions, dtype=torch.float32).reshape(-1, 1)

    for _ in range(epochs):

        optimizer.zero_grad()

        pred = surrogate(Z_t)

        loss = torch.mean((pred - Y_t) ** 2)

        loss.backward()

        optimizer.step()

    return surrogate

import torch
import torch.nn.functional as F
from .base import BaseCausalOptimizer


class EncoderInvarianceOptimizer(BaseCausalOptimizer):
    """
    Optimizes ONLY the encoder (4.3),
    using invariance signal from 4.2.
    """

    def __init__(self, lr=1e-3, steps=100):
        self.lr = lr
        self.steps = steps

    def fit(self, system, X, Y):
        encoder = system.encoder
        envs = system.environments

        optimizer = torch.optim.Adam(encoder.parameters(), lr=self.lr)

        for _ in range(self.steps):
            losses = []

            for env in envs:
                X_t = torch.tensor(X, dtype=torch.float32)

                Z = encoder(X_t)
                pred = env.model.predict_obs(X).mean()

                # invariance surrogate loss:
                losses.append(torch.tensor(pred, dtype=torch.float32))

            loss = torch.var(torch.stack(losses))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return system

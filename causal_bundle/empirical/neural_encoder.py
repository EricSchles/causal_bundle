import torch
import torch.nn as nn


class EmpiricalStructureEncoder(nn.Module):
    """
    4.3: Learns latent representation of empirical causal structure.

    This does NOT:
    - estimate causal effects
    - perform invariance testing
    - perform symbolic regression

    It ONLY produces a geometry-aware embedding of data.
    """

    def __init__(self, input_dim, latent_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, X):
        return self.encoder(X)

class EnvironmentAwareEncoder(nn.Module):
    """
    Extends EmpiricalStructureEncoder with environment conditioning.
    """

    def __init__(self, input_dim, env_dim, latent_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + env_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, X, env_embedding):
        X_cat = torch.cat([X, env_embedding], dim=-1)
        return self.encoder(X_cat)

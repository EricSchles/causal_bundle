import torch
import numpy as np
from causalbundle.empirical.neural_encoder import EmpiricalStructureEncoder

X = np.random.randn(100, 3)

encoder = EmpiricalStructureEncoder(input_dim=3, latent_dim=8)

X_t = torch.tensor(X, dtype=torch.float32)

Z = encoder(X_t)

print(Z.shape)

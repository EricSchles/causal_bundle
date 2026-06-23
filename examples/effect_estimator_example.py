import numpy as np
import torch
import torch.nn as nn

from causal_bundle.system.geometric_system import CausalGeometricSystem
from causal_bundle.bilevel.symbolic import SymbolicSection
from causal_bundle.bilevel.bundle import Bundle
from causal_bundle.bilevel.optimizer import BundleBilevelOptimizer

# Import your newly refactored structural treatment effect API items
from causal_bundle.treatment_effects.effect_estimator import EffectEstimator
from causal_bundle.treatment_effects.cate import CATEEstimator
from causal_bundle.treatment_effects.ate import ATEEstimator

# =====================================================================
# Mock Fitter Environment Requirements
# =====================================================================
class MockEnvModel:
    def predict_obs(self, X): return X[:, 0:1]

class EnvContainer:
    def __init__(self, name):
        self.name = name
        self.model = MockEnvModel()

class LinearBackend:
    def __init__(self): self.w = None
    def fit(self, Z, Y, env): self.w = np.linalg.pinv(Z) @ Y
    def predict(self, Z, env): return Z @ self.w if self.w is not None else np.zeros(len(Z))

# =====================================================================
# Generate Treatment Simulation Matrix (Index 0 is W, Index 1 is T)
# =====================================================================
np.random.seed(42)
N = 200
X_data = np.random.randn(N, 2) 

# Force binary treatment on index column 1
X_data[:, 1] = np.random.randint(0, 2, size=N)

# Structural Law: Y = 3.5 * T + 2.0 * W (True constant ATE should equal 3.5)
Y_data = (3.5 * X_data[:, 1:2]) + (2.0 * X_data[:, 0:1]) + (np.random.randn(N, 1) * 0.01)
env_idx = np.zeros((N, 1))
env_idx[N//2:] = 1  # Env 0 and Env 1 splits

# =====================================================================
# Assembly & Bilevel Fit Warmup
# =====================================================================
system = CausalGeometricSystem(
    environments=[EnvContainer("e0"), EnvContainer("e1")],
    encoder=nn.Sequential(nn.Linear(2, 2), nn.Tanh()),
    symbolic_backend=LinearBackend(),
    latent_dim=2, output_dim=1
)

optimizer = BundleBilevelOptimizer(
    bundle=Bundle(SymbolicSection(system.symbolic_backend)),
    outer_steps=3
)
optimized_system = optimizer.fit(system, X_data, Y_data, env_idx)

# =====================================================================
# Run Your Treatment Effect Estimation Pipeline
# =====================================================================
print("\n--- Invoking Causal Treatment Effect Estimation ---")

# Step 1: Initialize the primary base interventional effect engine targeting feature index 1 (T)
base_estimator = EffectEstimator(system=optimized_system, treatment_index=1)

# Step 2: Nest into CATE and ATE layer wrappers
cate_engine = CATEEstimator(effect_estimator=base_estimator)
ate_engine = ATEEstimator(cate_estimator=cate_engine)

# Evaluate global population treatment metrics
global_ate = ate_engine.estimate(X_data, t1=1, t0=0)
print(f"Discovered Global Population ATE: {global_ate:.4f} (Target Ground-Truth: 3.5000)")

# Check fiber-specific local environment treatment stability
env0_mask = (env_idx == 0).squeeze()
env1_mask = (env_idx == 1).squeeze()

ate_env0 = ate_engine.estimate(X_data[env0_mask], env_idx_np=env_idx[env0_mask])
ate_env1 = ate_engine.estimate(X_data[env1_mask], env_idx_np=env_idx[env1_mask])

print(f"Local Environment 0 Fiber ATE: {ate_env0:.4f}")
print(f"Local Environment 1 Fiber ATE: {ate_env1:.4f}")

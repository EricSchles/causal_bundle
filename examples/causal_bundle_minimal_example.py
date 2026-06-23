import numpy as np
import torch
import torch.nn as nn

# 1. Import your actual project API elements directly
from causal_bundle.optimizers import JointCausalOptimizer
from causal_bundle.system.geometric_system import CausalGeometricSystem

# =====================================================================
# 2. Setup Real/Stub Environment Models for System Orchestration
# =====================================================================
# Because CausalGeometricSystem requires environment objects with 
# .model.predict_obs() capabilities to compute invariance fields, 
# we specify a lightweight wrapper here.
class EmpiricalEnvironmentModel:
    def predict_obs(self, X):
        return X[:, 0:1] * 1.25

class EnvironmentContainer:
    def __init__(self, name):
        self.name = name
        self.model = EmpiricalEnvironmentModel()

environments = [EnvironmentContainer("env_0"), EnvironmentContainer("env_1")]


# =====================================================================
# 3. Define Concrete Step 6 Symbolic Backend Adapter
# =====================================================================
# If you are actively using PySR, you would swap out this OLS solver 
# for PySRRegressor() inside the Abstract class wrapper.
class ProjectSymbolicBackend:
    def __init__(self):
        self.weights = None

    def fit(self, Z: np.ndarray, Y: np.ndarray, env_idx: np.ndarray):
        # Emulates PySR fitting to latent layout maps
        self.weights = np.linalg.pinv(Z) @ Y

    def predict(self, Z: np.ndarray, env_idx: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return np.zeros(len(Z))
        return Z @ self.weights


# =====================================================================
# 4. Generate Causal Matrix Data 
# =====================================================================
np.random.seed(42)
torch.manual_seed(42)

N = 300
X_data = np.random.randn(N, 6)
env_idx = np.zeros((N, 1))
env_idx[N//2:] = 1  # 50/50 partition between env_0 and env_1

Y_data = np.zeros((N, 1))
# Invariant underlying mechanics mapped to dimension index 0 across both settings
Y_data[:N//2] = X_data[:N//2, 0:1] * 3.5 + np.random.randn(N//2, 1) * 0.05
Y_data[N//2:] = X_data[N//2:, 0:1] * 3.5 + np.random.randn(N//2, 1) * 0.40


# =====================================================================
# 5. Pipeline Initialization & Joint Optimization
# =====================================================================

# Construct a baseline geometric neural layer mapping observed nodes down to fibers
encoder_layer = nn.Sequential(
    nn.Linear(6, 3), 
    nn.Tanh()
)

# Instantiate the library system via your package module API
system = CausalGeometricSystem(
    environments=environments,
    encoder=encoder_layer,
    pysr_model=ProjectSymbolicBackend() # Fed directly into your orchestrated state slot
)

# Initialize the Step 7 corrected optimizer 
optimizer = JointCausalOptimizer(
    steps=8,
    lr=5e-3,
    lambda_irm=1.5,
    stability_enforcement="loss_scaling"
)

print("--- Starting System Optimization Routine ---")
print(f"Initial Stability Variance State: {system

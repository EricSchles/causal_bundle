import numpy as np
import torch
import torch.nn as nn

# 1. Import your actual project API elements directly from the package namespace
from causal_bundle.system.geometric_system import CausalGeometricSystem
from causal_bundle.optimizers.joint_optimizer import JointCausalOptimizer

# =====================================================================
# 2. Setup Real/Stub Environment Models for System Orchestration
# =====================================================================
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
class ProjectSymbolicBackend:
    def __init__(self):
        self.weights = None

    def fit(self, Z: np.ndarray, Y: np.ndarray, env_idx: np.ndarray):
        # Emulates PySR/OLS fitting to latent layout maps
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

# Construct a baseline geometric neural layer mapping 6 observed features down to 3 latent dimensions
LATENT_DIM = 3
OUTPUT_DIM = 1

encoder_layer = nn.Sequential(
    nn.Linear(6, LATENT_DIM), 
    nn.Tanh()
)

# Instantiate the library system matching your exact class signature parameters
system = CausalGeometricSystem(
    environments=environments,
    encoder=encoder_layer,
    symbolic_backend=ProjectSymbolicBackend(),  # Fixed parameter name
    latent_dim=LATENT_DIM,                      # Added parameter
    output_dim=OUTPUT_DIM                       # Added parameter
)

# Initialize the Step 7 corrected optimizer 
optimizer = JointCausalOptimizer(
    steps=8,
    lr=5e-3,
    lambda_irm=1.5,
    stability_enforcement="loss_scaling"
)

print("--- Starting System Optimization Routine ---")
print(f"Initial Stability Variance State: {system.compute_invariance_field(X_data)['variance']:.5f}")

# Trigger optimization via the formal framework API
optimized_system = optimizer.fit(system, X_data, Y_data, env_idx)

print("\n--- Optimization Complete ---")
print(f"Calculated Invariance Field Trend Log: {optimizer.stability_log}")

# Run the system's end-to-end evaluation pipeline pass
pipeline_results = optimized_system.run_full_pipeline(X_data, Y_data)
print(f"Final Execution Pipeline Flatness Score: {pipeline_results['invariance']['flatness_score']:.4f}")

import numpy as np
import torch
import torch.nn as nn

# =====================================================================
# 1. Pipeline Component Setup (Conforming to your API Architecture)
# =====================================================================
# Importing your updated core package components
from causal_bundle.system.geometric_system import CausalGeometricSystem

# Importing the newly structured bilevel components (using clean naming)
from causal_bundle.bilevel.symbolic import SymbolicSection
from causal_bundle.bilevel.bundle import Bundle
from causal_bundle.bilevel.optimizer import BundleBilevelOptimizer

# Light environment containers as required by CausalGeometricSystem
class EmpiricalEnvironmentModel:
    def predict_obs(self, X):
        return X[:, 0:1] * 1.5

class EnvironmentContainer:
    def __init__(self, name):
        self.name = name
        self.model = EmpiricalEnvironmentModel()

environments = [EnvironmentContainer("env_0"), EnvironmentContainer("env_1")]


# Concrete Step 6 Abstract Backend Implementation
class ProjectSymbolicBackend:
    def __init__(self):
        self.weights = None

    def fit(self, Z: np.ndarray, Y: np.ndarray, env_idx: np.ndarray):
        # OLS projection matrix calculation to emulate PySR equation fitting
        self.weights = np.linalg.pinv(Z) @ Y

    def predict(self, Z: np.ndarray, env_idx: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return np.zeros(len(Z))
        return Z @ self.weights


# =====================================================================
# 2. Generating Synthetic Multi-Environment Data
# =====================================================================
np.random.seed(42)
torch.manual_seed(42)

N = 300
X_data = np.random.randn(N, 6)
env_idx = np.zeros((N, 1))
env_idx[N//2:] = 1  # Splitting 50/50 between env_0 and env_1

Y_data = np.zeros((N, 1))
# Structural invariant constraint placed on feature index 0
Y_data[:N//2] = X_data[:N//2, 0:1] * 4.2 + np.random.randn(N//2, 1) * 0.05
Y_data[N//2:] = X_data[N//2:, 0:1] * 4.2 + np.random.randn(N//2, 1) * 0.35


# =====================================================================
# 3. Component Assembly & Execution Loop
# =====================================================================

LATENT_DIM = 3
OUTPUT_DIM = 1

# Latent space mapping neural encoder
encoder_network = nn.Sequential(
    nn.Linear(6, LATENT_DIM),
    nn.Tanh()
)

# Instantiate the library system coordinator
system = CausalGeometricSystem(
    environments=environments,
    encoder=encoder_network,
    symbolic_backend=ProjectSymbolicBackend(),
    latent_dim=LATENT_DIM,
    output_dim=OUTPUT_DIM
)

# Initialize the clean Bilevel Execution Hierarchy
lower_level_solver = SymbolicSection(
    symbolic_backend=system.symbolic_backend,
    lambda_irm=2.0
)

upper_level_evaluator = Bundle(
    symbolic=lower_level_solver,  # Connects the structural levels
    lambda_irm=2.0,
    lambda_complexity=0.05
)

# Initialize the new non-placeholder BundleBilevelOptimizer
bilevel_optimizer = BundleBilevelOptimizer(
    bundle=upper_level_evaluator,
    lr=1e-3,
    outer_steps=5,
    lambda_irm=2.0,
    lambda_complexity=0.05
)

print("--- Commencing Clean Bilevel Framework Optimization ---")
initial_variance = system.compute_invariance_field(X_data)['variance']
print(f"Initial Flatness Field Variance: {initial_variance:.5f}\n")

# Run the unified bilevel optimization fit loop
optimized_system = bilevel_optimizer.fit(system, X_data, Y_data, env_idx)

print("\n--- Optimization Complete ---")

# Run the standard forward assessment pipeline via the system manager
pipeline_results = optimized_system.run_full_pipeline(X_data, Y_data)
print(f"Final Execution Pipeline Flatness Score: {pipeline_results['invariance']['flatness_score']:.4f}")

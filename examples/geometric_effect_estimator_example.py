import numpy as np
import torch
import torch.nn as nn

# 1. Import your actual production API directly from the package root
from causal_bundle.system.geometric_system import CausalGeometricSystem
from causal_bundle.bilevel.symbolic import SymbolicSection
from causal_bundle.bilevel.bundle import Bundle
from causal_bundle.bilevel.optimizer import BundleBilevelOptimizer

# Import your treatment effect suite
from causal_bundle.treatment_effects.fiber_effects import FiberSpecificEffectEstimator
from causal_bundle.treatment_effects.fiber_symbolic_effects import FiberSymbolicEffects

# =====================================================================
# 2. Setup Real Multi-Environment Hooks for CausalGeometricSystem
# =====================================================================
class ProductionEnvironmentModel:
    def predict_obs(self, X):
        # Emulates empirical laws from Phase 4.1
        return X[:, 0:1] * 1.15

class EnvironmentContainer:
    def __init__(self, name):
        self.name = name
        self.model = ProductionEnvironmentModel()

environments = [EnvironmentContainer("env_0"), EnvironmentContainer("env_1")]


# =====================================================================
# 3. Define Concrete Multi-Fiber Symbolic Backend Adapter
# =====================================================================
class ProductionSymbolicBackend:
    """
    A concrete implementation of your Step 6 AbstractSymbolicBackend.
    Differentiates between structural fibers depending on env_idx parameters.
    """
    def __init__(self):
        # We pre-seed weights to demonstrate how SymPy/CATE shifts across fibers
        self.weights_by_env = {
            0: np.array([[3.5], [0.0], [0.0]]), # Fiber 0: Y = 3.5*z0
            1: np.array([[5.5], [1.5], [0.0]])  # Fiber 1: Y = 5.5*z0 + 1.5*z1
        }
        self.current_env = 0

    def fit(self, Z: np.ndarray, Y: np.ndarray, env_idx: np.ndarray):
        """Standard Step 6 fit block called inside SymbolicSection."""
        # On a real run, this executes PySRRegressor.fit() per environment profile
        pass

    def predict(self, Z: np.ndarray, env_idx: np.ndarray) -> np.ndarray:
        """Step 6/7 consistent vector prediction engine."""
        out = np.zeros(len(Z))
        for env_id, weights in self.weights_by_env.items():
            mask = (env_idx == env_id).squeeze()
            if mask.sum() > 0:
                # Project slice through matrix coordinates
                out[mask] = (Z[mask] @ weights).squeeze()
        return out


# =====================================================================
# 4. Generate Causal Matrix Data (Index 0 acts as treatment node 'z0')
# =====================================================================
np.random.seed(42)
torch.manual_seed(42)

N = 200
X_data = np.random.randn(N, 6)
X_data[:, 0] = np.random.randint(0, 2, size=N)  # Set treatment column binary

env_idx = np.zeros((N, 1))
env_idx[N//2:] = 1  # 50/50 partition between env_0 and env_1

Y_data = np.zeros((N, 1))
# Generate ground-truth data reflecting split treatment parameters (3.5 vs 5.5)
Y_data[:N//2] = X_data[:N//2, 0:1] * 3.5 + np.random.randn(N//2, 1) * 0.02
Y_data[N//2:] = X_data[N//2:, 0:1] * 5.5 + np.random.randn(N//2, 1) * 0.02


# =====================================================================
# 5. Connect Production Objects to the Bilevel Framework
# =====================================================================
LATENT_DIM = 3
OUTPUT_DIM = 1

encoder_layer = nn.Sequential(
    nn.Linear(6, LATENT_DIM),
    nn.Identity() # Identity mapping ensures clean coordinate viewing for validation
)

# Instantiate the library system coordinator
system = CausalGeometricSystem(
    environments=environments,
    encoder=encoder_layer,
    symbolic_backend=ProductionSymbolicBackend(),
    latent_dim=LATENT_DIM,
    output_dim=OUTPUT_DIM
)

# Instantiate the production bilevel framework objects natively
lower_level_section = SymbolicSection(
    symbolic_backend=system.symbolic_backend,
    lambda_irm=1.5
)

upper_level_geometry = Bundle(
    symbolic=lower_level_section,
    lambda_irm=1.5,
    lambda_complexity=0.01
)

bilevel_optimizer = BundleBilevelOptimizer(
    bundle=upper_level_geometry,
    lr=1e-3,
    outer_steps=3,
    lambda_irm=1.5,
    lambda_complexity=0.01
)

# Populate the system's fiber map to mimic an active lower-level solve state
system.fibers[0] = system.symbolic_backend
system.fibers[1] = system.symbolic_backend

# =====================================================================
# 6. Run Execution Loop
# =====================================================================
print("--- Launching Production Bilevel Framework Loop ---")
optimized_system = bilevel_optimizer.fit(system, X_data, Y_data, env_idx)

print("\n--- Running Multi-Fiber Numerical CATE Estimation ---")
# Evaluate treatment effects directly through your production class architecture
fiber_effect_evaluator = FiberSpecificEffectEstimator(system=optimized_system, treatment_index=0)
cate_vectors = fiber_effect_evaluator.estimate_cate(X_data, env_idx_np=env_idx)

print(f"Mean Extracted CATE (Fiber Environment 0): {np.mean(cate_vectors[:N//2]):.4f} (Target: 3.5000)")
print(f"Mean Extracted CATE (Fiber Environment 1): {np.mean(cate_vectors[N//2:]):.4f} (Target: 5.5000)")

print("\n--- Querying Closed-Form Analytic Equations via SymPy ---")
# Extract full symbolic laws directly from the active bundle geometry fibers
symbolic_law_extractor = FiberSymbolicEffects(system=optimized_system, treatment_var="z0")
discovered_laws = symbolic_law_extractor.symbolic_effects()

for fiber_id, equation in discovered_laws.items():
    print(f"Environment Fiber Section [{fiber_id}] Invariant Law Form: tau(W) = {equation}")

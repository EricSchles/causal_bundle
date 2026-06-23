import numpy as np
import torch

class SymbolicSection:
    """
    Lower-level optimization problem: s*(B_\theta)
    Discovers the optimal symbolic section given the current bundle coordinates.
    """
    def __init__(self, symbolic_backend, lambda_irm=1.0):
        self.symbolic_backend = symbolic_backend
        self.lambda_irm = lambda_irm

    def solve(self, system, Z_np, Y_np, env_idx_np):
        """Fits the symbolic expressions via the abstract backend abstraction layer."""
        # 1. Fit the decoupled black-box symbolic solver
        self.symbolic_backend.fit(Z_np, Y_np, env_idx_np)
        
        # 2. Collect predictions
        pred_np = self.symbolic_backend.predict(Z_np, env_idx_np)
        pred_loss = np.mean((Y_np - pred_np) ** 2)

        # 3. Compute empirical IRM penalty metrics across distinct environment matrices
        irm_penalty = self._compute_empirical_irm(pred_np, Y_np, env_idx_np)
        complexity = self._extract_complexity()

        return {
            "model": self.symbolic_backend,
            "pred_loss": pred_loss,
            "irm": irm_penalty,
            "complexity": complexity,
            "predictions": pred_np
        }

    def _compute_empirical_irm(self, pred_np, Y_np, env_idx_np):
        """Computes empirical IRM static penalty values for lower-level status reports."""
        unique_envs = np.unique(env_idx_np)
        irm_penalty = 0.0
        
        # We simulate a static gradient variance metric for evaluation feedback
        for env in unique_envs:
            mask = (env_idx_np == env).squeeze()
            if mask.sum() == 0:
                continue
            env_error = np.mean((Y_np[mask] - pred_np[mask]) ** 2)
            irm_penalty += env_error ** 2  # Simple variance proxy for numpy phase
            
        return irm_penalty

    def _extract_complexity(self):
        """Safely extracts equation complexity markers from the backend structure."""
        try:
            # Handles PySRRegressor dataframe metadata attributes if accessible
            if hasattr(self.symbolic_backend, "model") and hasattr(self.symbolic_backend.model, "equations_"):
                return float(self.symbolic_backend.model.equations_["complexity"].iloc[0])
        except Exception:
            pass
        return 1.0

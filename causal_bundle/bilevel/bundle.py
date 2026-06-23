import torch
import numpy as np
from .objective import bundle_objective

class Bundle:
    """
    Upper-level evaluation wrapper.
    Judges geometric setups by the quality of the invariant sections they admit.
    """
    def __init__(self, symbolic, lambda_irm=1.0, lambda_complexity=0.01):
        self.symbolic = symbolic
        self.lambda_irm = lambda_irm
        self.lambda_complexity = lambda_complexity

    def evaluate(self, system, X_np, Y_np, env_idx_np):
        """Evaluates total system performance by invoking the lower-level loop."""
        # Project data into learned coordinate channels cleanly using system capabilities
        Z_np = system.compute_latent_geometry(X_np)

        # Execute lower level optimization routine
        lower_result = self.symbolic.solve(system, Z_np, Y_np, env_idx_np)

        # Calculate final combined system objective performance scale
        total_obj = bundle_objective(
            pred_loss=lower_result["pred_loss"],
            irm_penalty=lower_result["irm"],
            complexity=lower_result["complexity"],
            lambda_irm=self.lambda_irm,
            lambda_complexity=self.lambda_complexity
        )

        return total_obj, lower_result

import numpy as np

class EffectEstimator:
    """
    Core causal interventional engine.
    Calculates counterfactual predictions by shifting target observed features 
    before projecting them into the bundle geometry.
    """
    def __init__(self, system, treatment_index: int):
        self.system = system
        self.treatment_index = treatment_index

    def potential_outcome(self, X_np, treatment_value, env_idx_np=None):
        """Calculates Y(t) by evaluating a counterfactual manipulation pass."""
        # Create counterfactual dataset copy
        X_cf = np.array(X_np, copy=True)
        X_cf[:, self.treatment_index] = treatment_value

        # Project modified observed features into the active geometric coordinate path
        # system.compute_latent_geometry handles safe torch/inference casting natively
        Z_np = self.system.compute_latent_geometry(X_cf)

        # Handle fallback if no environment matrix array is provided
        if env_idx_np is None:
            env_idx_np = np.zeros((len(X_np), 1))

        # Query the abstract symbolic section using our step 6/7 backend interface
        return self.system.symbolic_backend.predict(Z_np, env_idx_np)

import numpy as np

class FiberSpecificEffectEstimator:
    """
    Spatially localized treatment estimator.
    Leverages bundle topology mapping data to trace structural causal variation across environments.
    """
    def __init__(self, system, treatment_index: int):
        self.system = system
        self.treatment_index = treatment_index

    def estimate_cate(self, X_np, env_idx_np=None):
        """Computes localized potential outcomes using fiber assignment matching routines."""
        if env_idx_np is None:
            env_idx_np = np.zeros((len(X_np), 1))

        cate = np.zeros(len(X_np))
        unique_envs = np.unique(env_idx_np)

        for env_id in unique_envs:
            mask = (env_idx_np == env_id).squeeze()
            if mask.sum() == 0:
                continue

            # Route through system fibers collection using env tracking indices
            # Falls back to global fiber if localized mappings are not available
            fiber_model = self.system.fibers.get(env_id, self.system.fibers.get("global", None))
            
            if fiber_model is None:
                raise ValueError(f"No symbolic backend discovered for environment fiber: {env_id}")

            cate[mask] = self._compute_fiber_delta(fiber_model, X_np[mask], env_idx_np[mask])

        return cate

    def _compute_fiber_delta(self, fiber_model, X_sub, env_sub):
        X1 = np.array(X_sub, copy=True)
        X0 = np.array(X_sub, copy=True)
        X1[:, self.treatment_index] = 1
        X0[:, self.treatment_index] = 0

        # Run safe projection coordinates mapping loops
        Z1 = self.system.compute_latent_geometry(X1)
        Z0 = self.system.compute_latent_geometry(X0)

        y1 = fiber_model.predict(Z1, env_sub)
        y0 = fiber_model.predict(Z0, env_sub)
        return y1 - y0

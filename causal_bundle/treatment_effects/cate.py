import numpy as np

class CATEEstimator:
    """
    Conditional Average Treatment Effect (CATE) Estimator.
    Computes individual-level treatment effects: tau_i = Y_i(t_1) - Y_i(t_0)
    """
    def __init__(self, effect_estimator):
        self.effect_estimator = effect_estimator

    def estimate(self, X_np, t1=1, t0=0, env_idx_np=None):
        """Returns the treatment effect vector across the provided dataset layout."""
        y1 = self.effect_estimator.potential_outcome(X_np, treatment_value=t1, env_idx_np=env_idx_np)
        y0 = self.effect_estimator.potential_outcome(X_np, treatment_value=t0, env_idx_np=env_idx_np)
        
        return y1 - y0

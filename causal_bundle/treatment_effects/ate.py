import numpy as np

class ATEEstimator:
    """
    Average Treatment Effect (ATE) Estimator.
    Computes population-wide expected treatment effect: E[Y(t_1) - Y(t_0)]
    """
    def __init__(self, cate_estimator):
        self.cate_estimator = cate_estimator

    def estimate(self, X_np, t1=1, t0=0, env_idx_np=None):
        """Calculates the scalar expected effect over the data population."""
        cate_vectors = self.cate_estimator.estimate(X_np, t1=t1, t0=t0, env_idx_np=env_idx_np)
        
        return float(np.mean(cate_vectors))

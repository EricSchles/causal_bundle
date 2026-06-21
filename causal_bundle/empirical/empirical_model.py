# causalbundle/empirical/empirical_model.py

import numpy as np
from sklearn.linear_model import LinearRegression


class EmpiricalCausalModel:
    """
    Minimal empirical layer for causal constraints.

    Provides:
    - E[Y | X]
    - E[Y | do(X)] (placeholder / intervention override)
    - ATE estimation (simple version)
    """

    def __init__(self):
        self.models = {}
        self.data = None

    def fit_observational(self, X, Y):
        """
        Fit E[Y | X] using regression surrogate.
        """
        model = LinearRegression()
        model.fit(X, Y)
        self.models["obs"] = model
        return self

    def predict_obs(self, X):
        """
        E[Y | X]
        """
        return self.models["obs"].predict(X)

    def estimate_ate(self, X0, X1):
        """
        Simple plug-in ATE estimator:
        E[Y|X=1] - E[Y|X=0]
        """
        y1 = self.predict_obs(X1).mean()
        y0 = self.predict_obs(X0).mean()
        return y1 - y0

    def do_override(self, X, value):
        """
        Minimal intervention model:
        replaces X with constant value.
        """
        X_do = np.copy(X)
        X_do[:] = value
        return X_do

class EmpiricalCausalModel:
    # existing code unchanged ...

    def attach_encoder(self, encoder):
        """
        Optional 4.3 hook.
        Does NOT modify estimation logic.
        """
        self.encoder = encoder

    def encode(self, X):
        if hasattr(self, "encoder"):
            return self.encoder(torch.tensor(X, dtype=torch.float32)).detach().numpy()
        return X

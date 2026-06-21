import numpy as np


class CausalGeometricSystem:
    """
    4.4: Full integration of empirical + invariance + neural + symbolic layers.

    This does NOT redefine previous components.
    It ONLY orchestrates them.
    """

    def __init__(self, environments, encoder=None, pysr_model=None):
        self.environments = environments  # 4.2 objects
        self.encoder = encoder            # 4.3 object
        self.pysr_model = pysr_model      # symbolic regression engine

        self.fibers = {}  # graph/env -> symbolic model

    def compute_empirical_views(self, X):
        """
        Pull raw empirical structure from all environments.
        (Uses 4.1 models only)
        """

        return {
            env.name: env.model.predict_obs(X)
            for env in self.environments
        }

    def compute_invariance_field(self, X):
        """
        Measures geometric consistency across environments.
        """

        means = np.array([
            env.model.predict_obs(X).mean()
            for env in self.environments
        ])

        return {
            "mean_field": means,
            "variance": np.var(means),
            "flatness_score": 1.0 / (1e-8 + np.var(means))
        }

    def compute_latent_geometry(self, X):
        """
        Projects data into learned causal coordinate system.
        """

        if self.encoder is None:
            raise ValueError("Encoder not provided (4.3 required)")

        import torch

        X_t = torch.tensor(X, dtype=torch.float32)
        Z = self.encoder(X_t).detach().numpy()

        return Z

    def fit_symbolic_fibers(self, X, Y):
        """
        Fit symbolic mechanisms in latent space.
        """

        Z = self.compute_latent_geometry(X)

        if self.pysr_model is None:
            raise ValueError("PySR model not provided")

        model = self.pysr_model.fit(Z, Y)

        self.fibers["global"] = model

        return model

    def run_full_pipeline(self, X, Y):
        """
        Executes full causal geometric pipeline:

        1. Empirical estimation (4.1)
        2. Invariance field (4.2)
        3. Latent embedding (4.3)
        4. Symbolic regression (PySR)
        """

        empirical = self.compute_empirical_views(X)
        invariance = self.compute_invariance_field(X)
        Z = self.compute_latent_geometry(X)
        fiber = self.fit_symbolic_fibers(X, Y)

        return {
            "empirical": empirical,
            "invariance": invariance,
            "latent": Z,
            "fiber": fiber
        }

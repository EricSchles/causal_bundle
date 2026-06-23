import numpy as np
import torch

class CausalGeometricSystem:
    """
    4.4: Full integration of empirical + invariance + neural + symbolic layers.
    Orchestrates the components cleanly.
    """

    def __init__(self, environments, encoder=None, symbolic_backend=None, latent_dim=None, output_dim=None):
        self.environments = environments  
        self.encoder = encoder            
        self.symbolic_backend = symbolic_backend  # Step 6 Abstraction
        self.fibers = {}                  # graph/env -> symbolic model
        
        # Step 7 Proxy Hook
        self.surrogate = None
        if encoder is not None and latent_dim is not None and output_dim is not None:
            from causal_bundle.surrogates.symbolic_surrogate import SymbolicSurrogate
            self.surrogate = SymbolicSurrogate(input_dim=latent_dim)

    def compute_empirical_views(self, X):
        return {
            env.name: env.model.predict_obs(X)
            for env in self.environments
        }

    def compute_invariance_field(self, X):
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
        """Standard NumPy inference endpoint for the pipeline."""
        if self.encoder is None:
            raise ValueError("Encoder not provided (4.3 required)")

        self.encoder.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_t = torch.tensor(X, dtype=torch.float32)
            else:
                X_t = X.float()
            Z = self.encoder(X_t).cpu().numpy()
        return Z

    def fit_symbolic_fibers(self, X, Y, env_idx=None):
        """Fit symbolic mechanisms in latent space using abstract backend."""
        Z = self.compute_latent_geometry(X)

        if self.symbolic_backend is None:
            raise ValueError("Symbolic backend wrapper not provided")

        if env_idx is None:
            env_idx = np.zeros((len(X), 1))

        self.symbolic_backend.fit(Z, Y, env_idx)
        self.fibers["global"] = self.symbolic_backend

        return self.symbolic_backend

    def run_full_pipeline(self, X, Y):
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

import torch
import numpy as np
from .base import BaseCausalOptimizer


class PySRBackend:
    def __init__(self, **pysr_kwargs):
        from pysr import PySRRegressor
        self.model = PySRRegressor(**pysr_kwargs)
        
    def fit(self, Z: np.ndarray, Y: np.ndarray, env_idx: np.ndarray):
        # Global or local fiber mapping fallback
        self.model.fit(Z, Y)
        
    def predict(self, Z: np.ndarray, env_idx: np.ndarray) -> np.ndarray:
        return self.model.predict(Z)


def stability_lr_scale(stability, alpha=1.0):
    return 1.0 / (1.0 + alpha * stability)


class JointCausalOptimizer(BaseCausalOptimizer):

    def __init__(self, steps=10, lr=1e-3, lambda_irm=1.0, stability_enforcement="loss_scaling"):
        self.steps = steps
        self.lr = lr
        self.lambda_irm = lambda_irm
        self.stability_log = []
        self.stability_enforcement = stability_enforcement
        
        stability_enforcement_options = ["encoder_modulation", "loss_scaling", "none"]
        if stability_enforcement not in stability_enforcement_options:
            raise ValueError(f"stability enforcement must be one of: {','.join(stability_enforcement_options)}")

    def _symbolic_step(self, system, X_np, Y_np, env_idx_np):
        """Phase A: Update the true symbolic model based on the frozen geometry."""
        # Compute latent coordinates safely in eval/inference mode
        system.encoder.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_np, dtype=torch.float32)
            Z_np = system.compute_latent_geometry(X_t).cpu().numpy()

        # Update the decoupled symbolic backend
        system.symbolic_backend.fit(Z_np, Y_np, env_idx_np)
        system.fibers["global"] = system.symbolic_backend
        
        return Z_np, system

    def _build_surrogate(self, system, Z_np, env_idx_np):
        """Distill current black-box symbolic expressions into a differentiable neural proxy."""
        symbolic_predictions = system.symbolic_backend.predict(Z_np, env_idx_np)
        
        # Pull or instantiate the surrogate network via the system coordinator
        surrogate = system.surrogate 
        Z_t = torch.tensor(Z_np, dtype=torch.float32)
        pred_t = torch.tensor(symbolic_predictions, dtype=torch.float32)
        
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
        surrogate.train()
        
        # Fast inner optimization loop for proper distillation alignment
        for _ in range(150):
            optimizer.zero_grad()
            surr_pred = surrogate(Z_t)
            loss = torch.nn.functional.mse_loss(surr_pred, pred_t)
            loss.backward()
            optimizer.step()
            
        return surrogate

    def _compute_symmetric_loss(self, system, X_t, Y_t, env_idx_t, stability):
        """Calculates L_pred + lambda * L_IRM down through the surrogate proxy graph."""
        # Keep the graph connected directly back to the encoder weights!
        Z_t = system.compute_latent_geometry(X_t) 
        Y_hat = system.surrogate(Z_t)
        
        # 1. Base Prediction Loss
        pred_loss = torch.nn.functional.mse_loss(Y_hat, Y_t)
        
        # 2. True IRM Invariance Penalty across Fiber Environments
        unique_envs = torch.unique(env_idx_t)
        irm_penalty = torch.tensor(0.0, device=X_t.device)
        scale = torch.tensor(1.0, device=X_t.device, requires_grad=True)
        
        for env in unique_envs:
            mask = (env_idx_t == env).squeeze()
            if mask.sum() == 0:
                continue
            
            env_loss = torch.nn.functional.mse_loss(Y_hat[mask] * scale, Y_t[mask])
            env_grad = torch.autograd.grad(env_loss, scale, create_graph=True)[0]
            irm_penalty = irm_penalty + torch.sum(env_grad ** 2)
            
        total_loss = pred_loss + (self.lambda_irm * irm_penalty)
        
        # Apply stability adjustments if flagged
        if self.stability_enforcement == "loss_scaling":
            total_loss = total_loss * (1.0 + stability)
            
        return total_loss

    def _encoder_step(self, system, encoder_opt, X_np, Y_np, env_idx_np, stability):
        """Phase B: Optimize the geometric encoder layout using the joint proxy objective."""
        if encoder_opt is None:
            return

        # Modulate optimizer pace based on global environmental stability
        if self.stability_enforcement == "encoder_modulation":
            scale = stability_lr_scale(stability)
            for group in encoder_opt.param_groups:
                group["lr"] = self.lr * scale

        system.encoder.train()
        system.surrogate.eval() # Freeze proxy behaviors; we are shaping the latent spaces
        encoder_opt.zero_grad()
        
        # Load arrays directly into unified PyTorch tensors to keep gradients active
        X_t = torch.tensor(X_np, dtype=torch.float32)
        Y_t = torch.tensor(Y_np, dtype=torch.float32)
        env_idx_t = torch.tensor(env_idx_np, dtype=torch.long)
        
        total_loss = self._compute_symmetric_loss(system, X_t, Y_t, env_idx_t, stability)
        total_loss.backward()
        encoder_opt.step()

    def fit(self, system, X, Y, env_idx=None):
        """Fits the unified geometric system framework across alternating optimization blocks."""
        if env_idx is None:
            # Fallback to single environment if indexes are missing
            env_idx = np.zeros((len(X), 1))
            
        encoder_opt = None
        if system.encoder is not None:
            encoder_opt = torch.optim.Adam(system.encoder.parameters(), lr=self.lr)
            
        for step in range(self.steps):
            # Phase A: Fit raw data out to the black-box symbolic structures
            Z_np, system = self._symbolic_step(system, X, Y, env_idx)
            
            # Align the Neural Proxy Behavior to the new Symbolic Surface
            system.surrogate = self._build_surrogate(system, Z_np, env_idx)
            
            # Evaluate current geometric invariant conditions
            stability_field = system.compute_invariance_field(X)
            stability = stability_field["variance"] if "variance" in stability_field else 0.0
            self.stability_log.append(stability)
            
            # Phase B: Backprop through the Proxy to deform latent geometries symmetrically
            self._encoder_step(system, encoder_opt, X, Y, env_idx, stability)

        return system

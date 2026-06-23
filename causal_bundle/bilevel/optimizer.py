import copy
import torch
import numpy as np
from causal_bundle.surrogates.symbolic_surrogate import train_surrogate
from .objective import bundle_objective

class BundleBilevelOptimizer:
    """
    The True Bilevel Optimizer.
    Alternates between a symbolic lower-level solver and backpropagating through
    a distilled neural proxy to deform bundle geometries symmetrically.
    """
    def __init__(self, bundle, lr=1e-3, outer_steps=10, lambda_irm=1.0, lambda_complexity=0.01):
        self.bundle = bundle
        self.lr = lr
        self.outer_steps = outer_steps
        self.lambda_irm = lambda_irm
        self.lambda_complexity = lambda_complexity

    def fit(self, system, X_np, Y_np, env_idx_np=None):
        if env_idx_np is None:
            env_idx_np = np.zeros((len(X_np), 1))

        opt = torch.optim.Adam(system.encoder.parameters(), lr=self.lr)
        
        best_obj = float("inf")
        best_state = None

        for step in range(self.outer_steps):
            # 1. LOWER LEVEL SOLVE & UPPER LEVEL EVALUATION
            obj, result = self.bundle.evaluate(system, X_np, Y_np, env_idx_np)
            
            print(f"[Bilevel Step {step:02d}] Global Objective: {obj:.5f} | "
                  f"Pred Loss: {result['pred_loss']:.5f} | IRM: {result['irm']:.5f} | Complx: {result['complexity']:.1f}")

            # Keep track of the mathematically best bundle geometry
            if obj < best_obj:
                best_obj = obj
                best_state = copy.deepcopy(system.encoder.state_dict())

            # 2. SURROGATE SYNCHRONIZATION (Distillation Phase)
            # Fetch the frozen numpy coordinate grid to align proxy surface maps
            Z_np = system.compute_latent_geometry(X_np)
            system.surrogate = train_surrogate(
                surrogate=system.surrogate,
                Z=Z_np,
                symbolic_predictions=result["predictions"]
            )

            # 3. UPPER LEVEL GEOMETRY UPDATE (Differentiable Phase)
            system.encoder.train()
            system.surrogate.eval() # Treat the proxy section as a fixed surface map
            opt.zero_grad()

            # Execute active differentiable backprop pipeline
            surrogate_loss = self._compute_differentiable_upper_loss(system, X_np, Y_np, env_idx_np, result["complexity"])
            surrogate_loss.backward()
            opt.step()

        # Re-anchor back out to the lowest error layout recorded
        if best_state is not None:
            system.encoder.load_state_dict(best_state)

        return system

    def _compute_differentiable_upper_loss(self, system, X_np, Y_np, env_idx_np, structural_complexity):
        """Computes the complete differentiable loss chain straight back to the encoder."""
        X_t = torch.tensor(X_np, dtype=torch.float32)
        Y_t = torch.tensor(Y_np, dtype=torch.float32)
        env_t = torch.tensor(env_idx_np, dtype=torch.long)

        # Pure tensor forward mapping paths to retain graph connections
        Z_t = system.encoder(X_t)
        Y_hat = system.surrogate(Z_t)

        # 1. Differentiable Prediction Loss Component
        pred_loss_t = torch.nn.functional.mse_loss(Y_hat, Y_t)

        # 2. Differentiable IRM Invariance Penalty Component
        unique_envs = torch.unique(env_t)
        irm_penalty_t = torch.tensor(0.0, device=X_t.device)
        scale = torch.tensor(1.0, device=X_t.device, requires_grad=True)

        for env in unique_envs:
            mask = (env_t == env).squeeze()
            if mask.sum() == 0:
                continue
            env_loss = torch.nn.functional.mse_loss(Y_hat[mask] * scale, Y_t[mask])
            env_grad = torch.autograd.grad(env_loss, scale, create_graph=True)[0]
            irm_penalty_t = irm_penalty_t + torch.sum(env_grad ** 2)

        # 3. Combine with static complexity values derived from the lower-level solve step
        total_differentiable_loss = bundle_objective(
            pred_loss=pred_loss_t,
            irm_penalty=irm_penalty_t,
            complexity=torch.tensor(structural_complexity, device=X_t.device),
            lambda_irm=self.lambda_irm,
            lambda_complexity=self.lambda_complexity
        )

        return total_differentiable_loss

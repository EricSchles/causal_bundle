from .base import BaseCausalOptimizer
import torch
import numpy as np

def joint_objective(system, X, Y, Z, model, lambda_irm=1.0):
    """
    Single scalar objective used by BOTH encoder and PySR.
    """

    # symbolic predictions
    try:
        Y_hat = model.predict(Z)
    except:
        Y_hat = model.predict(Z.reshape(len(Z), -1))

    # prediction loss
    pred_loss = np.mean((Y - Y_hat) ** 2)

    # IRM penalty (Step 1)
    irm = system.invariance.irm_penalty(
        X=Z,
        Y_hat_fn=lambda X_: model.predict(X_)
    )

    return pred_loss + lambda_irm * irm

def encoder_coupling_loss(Z, Y_hat_list):
    """
    Encourages encoder to produce representations
    that make symbolic predictions stable across environments.
    """

    losses = []

    for Y_hat in Y_hat_list:
        y_t = torch.tensor(Y_hat, dtype=torch.float32)
        losses.append(y_t)

    losses = torch.stack(losses)

    return torch.var(losses)

def pysr_invariance_score(system, Z, Y):
    """
    Penalizes symbolic models that are unstable across environments.
    """

    base_model = system.pysr_model.fit(Z, Y)

    inv = system.compute_invariance_field(Z)["variance"]

    score = base_model.score(Z, Y) + inv

    return score, base_model

def stability_lr_scale(stability, alpha=1.0):
    """
    Converts stability signal into learning-rate multiplier.

    We use inverse scaling so:
    - high instability → small updates
    - low instability → normal updates
    """

    return 1.0 / (1.0 + alpha * stability)
    
class JointCausalOptimizer(BaseCausalOptimizer):

    def __init__(self, steps=10, lr=1e-3, lambda_irm=1.0, stability_enforcement=""):
        self.steps = steps
        self.lr = lr
        self.lambda_irm = lambda_irm
        self.stability_log = []
        self.stability_enforcement = stability_enforcement
        stability_enforcement_options = [
            "encoder_modulation",
            "loss_scaling"
        ]
        if stability_enforcement not in stability_enforcement_options:
            raise Exception(f"stability enforcement must be one of: {",".join(stability_enforcement_options)}")
            
        
    def fit(self, system, X, Y):

        encoder_opt = None
        if system.encoder is not None:
            encoder_opt = torch.optim.Adam(system.encoder.parameters(), lr=self.lr)
            
        for _ in range(self.steps):

            # =========================
            # STEP 1: latent encoding
            # =========================
            Z = system.compute_latent_geometry(X)

            # =========================
            # STEP 2: symbolic update (PySR with IRM bias)
            # =========================
            system.pysr_model.fit(Z, Y)
            model = system.pysr_model
            system.fibers["global"] = model

            # =========================
            # STEP 3: unified objective (CRITICAL FIX)
            # =========================
            total_loss = joint_objective(
                system=system,
                X=X,
                Y=Y,
                Z=Z,
                model=model,
                lambda_irm=self.lambda_irm
            )
            stability = system.compute_invariance_field(X)["variance"]
            if self.stability_enforcement == "encoder_modulation":
                scale = stability_lr_scale(stability)
            
            if encoder_opt is not None:

                if self.stability_enforcement == "encoder_modulation":
                    for group in encoder_opt.param_groups:
                        group["lr"] = self.lr * scale

                encoder_opt.zero_grad()

                Z_t = torch.tensor(Z, dtype=torch.float32)
                Y_hat = model.predict(Z)

                Y_hat_t = torch.tensor(Y_hat, dtype=torch.float32)
                Y_t = torch.tensor(Y, dtype=torch.float32)

                encoder_loss = torch.mean((Y_hat_t - Y_t) ** 2)

                if stability_enforcement == "loss_scaling":
                    encoder_loss = encoder_loss * (1.0 + stability)
                encoder_loss.backward()
                encoder_opt.step()

        return system

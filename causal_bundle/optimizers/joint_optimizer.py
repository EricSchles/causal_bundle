from .base import BaseCausalOptimizer
import torch
import numpy as np

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


class JointCausalOptimizer(BaseCausalOptimizer):
    def __init__(self, steps=10, lr=1e-3):
        self.steps = steps
        self.lr = lr

    def fit(self, system, X, Y):

        encoder_opt = None
        if system.encoder is not None:
            encoder_opt = torch.optim.Adam(system.encoder.parameters(), lr=self.lr)

        for _ in range(self.steps):

            # ==========================
            # STEP 1: latent geometry
            # ==========================
            Z = system.compute_latent_geometry(X)

            # ==========================
            # STEP 2: symbolic update (PySR biased by invariance)
            # ==========================
            score, model = pysr_invariance_score(system, Z, Y)
            system.fibers["global"] = model

            # ==========================
            # STEP 3: predictions from symbolic model
            # ==========================
            try:
                Y_hat = model.predict(Z)
            except:
                Y_hat = model.predict(Z.reshape(len(Z), -1))

            # ==========================
            # STEP 4: encoder coupling update
            # ==========================
            if encoder_opt is not None:

                Z_t = torch.tensor(Z, dtype=torch.float32)

                Y_hat_t = torch.tensor(Y_hat, dtype=torch.float32)

                loss = torch.var(Y_hat_t)

                encoder_opt.zero_grad()
                loss.backward()
                encoder_opt.step()

            # ==========================
            # STEP 5: optional stabilization
            # ==========================
            inv = system.compute_invariance_field(X)["variance"]

            # could be logged or used for annealing
            _ = score + inv

        return system

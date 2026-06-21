from .base import BaseCausalOptimizer
import numpy as np


from .base import BaseCausalOptimizer


class PySRInvarianceOptimizer(BaseCausalOptimizer):
    """
    PySR with invariance-aware model selection.

    Key upgrade:
    - symbolic models are scored by BOTH fit and IRM stability
    """

    def __init__(self, lambda_irm=1.0):
        self.lambda_irm = lambda_irm

    def score_model(self, system, model, Z, Y):
        """
        Combined objective:
        - predictive fit
        - IRM invariance penalty
        """

        # standard predictive error
        try:
            Y_pred = model.predict(Z)
        except:
            Y_pred = model.predict(Z.reshape(len(Z), -1))

        pred_loss = np.mean((Y - Y_pred) ** 2)

        # IRM penalty from Step 1
        irm_penalty = system.invariance.irm_penalty(
            X=Z,
            Y_hat_fn=lambda X: model.predict(X)
        )

        return pred_loss + self.lambda_irm * irm_penalty

    def fit(self, system, X, Y):

        Z = system.compute_latent_geometry(X)

        best_model = None
        best_score = float("inf")

        # NOTE: placeholder search loop (PySR internal would replace this)
        for _ in range(10):

            candidate_model = system.pysr_model.fit(Z, Y)

            score = self.score_model(system, candidate_model, Z, Y)

            if score < best_score:
                best_score = score
                best_model = candidate_model

        system.fibers["global"] = best_model

        return system

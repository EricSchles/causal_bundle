# causalbundle/empirical/invariance.py

import numpy as np
import torch

class InvarianceSystem:
    """
    4.2: Cross-environment structure over EmpiricalCausalModels.

    IMPORTANT:
    - Does NOT compute causal estimates (that's 4.1)
    - Only compares outputs across environments
    """

    def __init__(self, environments):
        self.environments = environments  # list[EmpiricalEnvironment]

    def compare_conditional_mean(self, X):
        """
        Collects E[Y|X] from each 4.1 model.

        Extension idea:
        - treats each EmpiricalCausalModel as a "section"
        - does NOT modify them
        """

        return {
            env.name: env.model.predict_obs(X)
            for env in self.environments
        }
    
    def compute_environment_risks(self, X, Y_hat_fn):
        """
        Computes empirical risk per environment.

        Y_hat_fn: function(Z or X) -> prediction
        """

        risks = {}

        for env in self.environments:
            X_env = X
            Y_pred = Y_hat_fn(X_env)

            # squared error surrogate risk
            risk = np.mean((env.model.predict_obs(X_env) - Y_pred) ** 2)

            risks[env.name] = risk

        return risks

    def irm_penalty(self, X, Y_hat_fn):
        """
        Measures invariance as gradient inconsistency proxy.

        We approximate IRM by:
        - comparing risks across environments
        """

        risks = self.compute_environment_risks(X, Y_hat_fn)

        risk_values = np.array(list(risks.values()))

        return np.var(risk_values)
    
    def invariant_mechanism_score(self, X, Y_hat_fn):
        """
        Now IRM-based instead of raw prediction variance.
        """

        penalty = self.irm_penalty(X, Y_hat_fn)

        return 1.0 / (1e-8 + penalty)

class EnvironmentAligner:
    """
    Optional utility for aligning empirical outputs.
    Does NOT affect causal estimation.
    """

    def align_means(self, X, environments):
        means = [
            env.model.predict_obs(X).mean()
            for env in environments
        ]

        global_mean = np.mean(means)

        return {
            "global_mean": global_mean,
            "environment_means": means
        }

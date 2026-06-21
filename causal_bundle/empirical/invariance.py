# causalbundle/empirical/invariance.py

import numpy as np


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
    
    def compute_variance_across_envs(self, X):
        """
        Measures disagreement between environments.

        Interpretation (GEOMETRIC):
        - low variance → flat section over environment space
        - high variance → curvature / non-invariance
        """

        means = np.array([
            env.model.predict_obs(X).mean()
            for env in self.environments
        ])

        return np.var(means)

    def invariant_mechanism_score(self, X):
        """
        Inverse of variance = crude invariance measure.

        This is NOT causal estimation.
        This is STRUCTURAL consistency across worlds.
        """

        return 1.0 / (1e-8 + self.compute_variance_across_envs(X))


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

class BaseCausalOptimizer:
    """
    Abstract optimizer over the causal geometric system.
    """

    def fit(self, system, X, Y):
        raise NotImplementedError

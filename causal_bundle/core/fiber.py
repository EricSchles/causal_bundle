# causalbundle/core/fiber.py

class SymbolicFiber:
    def __init__(self, graph, pysr_model=None, empirical_model=None):
        self.graph = graph
        self.pysr_model = pysr_model
        self.empirical_model = empirical_model
        self.mechanisms = {}

    def fit(self, node, X, y):
        """
        Fit symbolic model using PySR guided by empirical constraints.
        """
        if self.pysr_model is None:
            raise ValueError("PySR model required")

        # Step 1: fit empirical baseline
        self.empirical_model.fit_observational(X, y)

        # Step 2: symbolic regression
        model = self.pysr_model.fit(X, y)

        self.mechanisms[node] = model

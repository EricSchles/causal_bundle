# causalbundle/core/section.py

class Section:
    """
    A global assignment of mechanisms over a graph.
    """

    def __init__(self, fiber):
        self.fiber = fiber

    def intervene(self, node, value):
        """
        Implements do-operator at section level.
        """
        def modified_model(X):
            return value
        self.fiber.mechanisms[node] = modified_model

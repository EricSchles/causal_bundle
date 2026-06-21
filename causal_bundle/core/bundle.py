# causalbundle/core/bundle.py

class CausalBundle:
    def __init__(self):
        self.fibers = {}   # G -> fiber
        self.sections = {} # G -> section

    def add_fiber(self, graph, fiber):
        key = str(graph.graph.edges())
        self.fibers[key] = fiber

    def get_fiber(self, graph):
        return self.fibers[str(graph.graph.edges())]

    def set_section(self, graph, section):
        self.sections[str(graph.graph.edges())] = section

    def check_consistency(self):
        """
        Placeholder for gluing condition.
        """
        return True

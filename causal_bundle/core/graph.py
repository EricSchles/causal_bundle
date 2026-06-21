# causalbundle/core/graph.py

import networkx as nx

class CausalGraph:
    def __init__(self, edges=None):
        self.graph = nx.DiGraph()
        if edges:
            self.graph.add_edges_from(edges)

    def add_edge(self, u, v):
        self.graph.add_edge(u, v)

    def parents(self, node):
        return list(self.graph.predecessors(node))

    def children(self, node):
        return list(self.graph.successors(node))

    def copy(self):
        return CausalGraph(self.graph.edges())

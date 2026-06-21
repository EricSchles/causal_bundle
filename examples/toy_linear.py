# causalbundle/examples/toy_linear.py

import numpy as np
from causal_bundle.core.graph import CausalGraph
from causal_bundle.core.fiber import SymbolicFiber
from causal_bundle.symbolic.pysr_interface import PySRMechanismModel

# graph: X -> Y
G = CausalGraph(edges=[("X", "Y")])

# fake data
X = np.random.randn(100, 1)
Y = 2 * X + np.random.randn(100, 1) * 0.01

# fiber
model = PySRMechanismModel()
fiber = SymbolicFiber(G, pysr_model=model)

fiber.fit_mechanism("Y", X, Y)

print(fiber.mechanisms["Y"].model)

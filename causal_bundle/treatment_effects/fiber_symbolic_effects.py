from causal_bundle.symbolic.equation_extractor import EquationExtractor
from causal_bundle.treatment_effects.symbolic_derivative import SymbolicTreatmentEffect

class FiberSymbolicEffects:
    """
    Extracts explicit structural algebraic treatment formulas across all environments.
    """
    def __init__(self, system, treatment_var="z0"):
        self.system = system
        self.treatment_var = treatment_var

    def symbolic_effects(self):
        """Maps localized bundle assignments to analytic closed-form formulas."""
        effects = {}
        
        for fiber_id, model in self.system.fibers.items():
            # 1. Grab raw equation representation out of the targeted fiber
            raw_equation = EquationExtractor.best_equation(model)
            
            # 2. Extract explicit treatment algebra strings via SymPy wrappers
            derivative_engine = SymbolicTreatmentEffect(raw_equation, self.treatment_var)
            
            # 3. Collect formula structural expressions
            effects[fiber_id] = derivative_engine.treatment_effect_function()
            
        return effects

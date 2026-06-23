import sympy as sp

class SymbolicTreatmentEffect:
    """
    Analytic Causal Engine. Uses SymPy to extract treatment effects
    symbolically directly out of functional text laws.
    """
    def __init__(self, equation_str: str, treatment_var="z0"):
        """
        equation_str example: '3.5*z0 + 2.0*z1' (where z0 is treatment variable)
        """
        self.treatment_var = treatment_var
        # Clean string from possible numpy/bracket styles
        clean_str = equation_str.replace("[", "").replace("]", "")
        self.expr = sp.sympify(clean_str)
        self.T = sp.Symbol(treatment_var)

    def treatment_effect_function(self):
        """Analytic CATE law: f(T=1, x) - f(T=0, x)"""
        treated = self.expr.subs(self.T, 1)
        control = self.expr.subs(self.T, 0)
        return sp.simplify(treated - control)

    def derivative_effect(self):
        """Analytic dY/dT marginal equation form (for continuous designs)."""
        return sp.diff(self.expr, self.T)

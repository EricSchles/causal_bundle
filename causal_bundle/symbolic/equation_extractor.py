class EquationExtractor:
    """
    Utility handler to extract string equations from the symbolic backend wrappers.
    """
    @staticmethod
    def best_equation(symbolic_backend):
        # 1. Check if backend natively exposes a sympy equation generator method
        if hasattr(symbolic_backend, "sympy"):
            try:
                return str(symbolic_backend.sympy())
            except Exception:
                pass
                
        # 2. Check if it encapsulates an inner model instance (e.g., PySRRegressor)
        if hasattr(symbolic_backend, "model"):
            inner = symbolic_backend.model
            if hasattr(inner, "sympy"):
                try:
                    return str(inner.sympy())
                except Exception:
                    pass
            if hasattr(inner, "get_best"):
                try:
                    return str(inner.get_best())
                except Exception:
                    pass

        # 3. Fallback to basic state representation strings or default assignments
        if hasattr(symbolic_backend, "weights") and symbolic_backend.weights is not None:
            # Emulate simple poly weights string conversion
            terms = [f"{w:.3f}*z{i}" for i, w in enumerate(symbolic_backend.weights.flatten()) if abs(w) > 1e-5]
            return " + ".join(terms) if terms else "0.0"

        return "0.0"

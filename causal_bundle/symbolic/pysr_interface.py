# causalbundle/symbolic/pysr_interface.py

from pysr import PySRRegressor

class PySRMechanismModel:
    def __init__(self):
        self.model = PySRRegressor(
            niterations=50,
            binary_operators=["+", "-", "*", "/"]
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

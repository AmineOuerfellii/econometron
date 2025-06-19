from abc import ABC, abstractmethod

class nonLinear_RE(ABC):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def residual(self, coeffs, state, basis, policy_func):
        pass

    @abstractmethod
    def initial_guess(self, basis):
        pass


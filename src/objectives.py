import numpy as np
from abc import ABC, abstractmethod

from scipy.special import expit

class LossFunction(ABC):
    @abstractmethod
    def evaluate(self, r: np.ndarray, y: np.ndarray) -> float:
        """Returns the empirical risk L(r)/N."""
        pass

    @abstractmethod
    def gradient(self, r: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Returns the gradient vector \nabla L(r)/N."""
        pass

class Penalty(ABC):
    @abstractmethod
    def evaluate(self, w: np.ndarray) -> float:
        """Returns the penalization value R(w)."""
        pass

    @abstractmethod
    def gradient(self, w: np.ndarray) -> np.ndarray:
        """Returns the subgradient/gradient vector \nabla R(w)."""
        pass

class LogisticLoss(LossFunction):
    def evaluate(self, r: np.ndarray, y: np.ndarray) -> float:
        margins = y * r
        # Using numerically stable log-add-exp formulation
        return np.sum(np.logaddexp(0, -margins)) 

    def gradient(self, r: np.ndarray, y: np.ndarray) -> np.ndarray:
        margins = y * r
        probabilities = expit(-margins) # Equivalent to 1 / (1 + exp(margins))
        grad = - y * probabilities
        return grad

class RidgePenalty(Penalty):
    def evaluate(self, w: np.ndarray) -> float:
        return 0.5 * np.sum(w ** 2)

    def gradient(self, w: np.ndarray) -> np.ndarray:
        return w
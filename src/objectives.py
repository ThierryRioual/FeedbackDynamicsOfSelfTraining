import numpy as np
from abc import ABC, abstractmethod

from scipy.special import expit

class LossFunction(ABC):
    @abstractmethod
    def evaluate(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Returns the empirical risk L(w)."""
        pass

    @abstractmethod
    def gradient(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Returns the gradient vector \nabla L(w)."""
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
    def evaluate(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        n = X.shape[0]
        margins = y * (X @ w)
        # Using numerically stable log-add-exp formulation
        return np.sum(np.logaddexp(0, -margins)) / n

    def gradient(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        margins = y * (X @ w)
        probabilities = expit(-margins) # Equivalent to 1 / (1 + exp(margins))
        return -(1/n) * X.T @ (y * probabilities)

class RidgePenalty(Penalty):
    def evaluate(self, w: np.ndarray) -> float:
        return 0.5 * np.sum(w ** 2)

    def gradient(self, w: np.ndarray) -> np.ndarray:
        return w
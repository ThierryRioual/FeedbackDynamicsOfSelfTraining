import torch
from abc import ABC, abstractmethod

class LossFunction(ABC):
    @abstractmethod
    def evaluate(self, r: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Returns the empirical risk L(r)/N."""
        pass

    @abstractmethod
    def gradient(self, r: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Returns the gradient vector \nabla L(r)/N."""
        pass

class Penalty(ABC):
    @abstractmethod
    def evaluate(self, w: torch.Tensor) -> torch.Tensor:
        """Returns the penalization value R(w)."""
        pass

    @abstractmethod
    def gradient(self, w: torch.Tensor) -> torch.Tensor:
        """Returns the subgradient/gradient vector \nabla R(w)."""
        pass

class LogisticLoss(LossFunction):
    def evaluate(self, r: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        margins = y * r
        return torch.sum(torch.logaddexp(torch.zeros_like(margins), -margins)) 

    def gradient(self, r: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        margins = y * r
        probabilities = torch.sigmoid(-margins) 
        grad = -y * probabilities
        return grad

class RidgePenalty(Penalty):
    def evaluate(self, w: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(w ** 2)

    def gradient(self, w: torch.Tensor) -> torch.Tensor:
        return w

class SelectionFunction(ABC):
    """Abstract base class for pseudo-label selection masks."""
    
    @abstractmethod
    def __call__(self, preactivation: torch.Tensor, pos_margin: float, neg_margin: float) -> torch.Tensor:
        """Returns a mask with values in [0, 1] indicating selection."""
        pass

class HardSelection(SelectionFunction):
    """Standard hard thresholding used in empirical self-training."""
    
    def __call__(self, preactivation: torch.Tensor, pos_margin: float, neg_margin: float) -> torch.Tensor:
        return torch.where((preactivation >= pos_margin) | (preactivation <= neg_margin), 1.0, 0.0)

class LipschitzSelection(SelectionFunction):
    """
    Differentiable surrogate thresholding for theoretical State Evolution.
    Allows PyTorch Autograd to capture boundary probability mass.
    """
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def __call__(self, preactivation: torch.Tensor, pos_margin: float, neg_margin: float) -> torch.Tensor:
        if pos_margin == 0.0 and neg_margin == 0.0:
            return torch.ones_like(preactivation)
            
        mask_pos = torch.clamp((preactivation - (pos_margin - self.epsilon)) / (2 * self.epsilon), min=0.0, max=1.0)
        mask_neg = torch.clamp(((neg_margin + self.epsilon) - preactivation) / (2 * self.epsilon), min=0.0, max=1.0)
        
        return mask_pos + mask_neg
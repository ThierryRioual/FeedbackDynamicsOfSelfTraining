import math

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

def _validate_width(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return value

def _validate_hard_forward(value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError(
            "hard_forward must be a bool, "
            f"got {type(value).__name__}"
        )
    return value

class HardSelection(SelectionFunction):
    """Standard hard thresholding used in empirical self-training."""
    
    def __call__(self, preactivation: torch.Tensor, pos_margin: float, neg_margin: float) -> torch.Tensor:
        selected = (preactivation >= pos_margin) | (preactivation <= neg_margin)
        return selected.to(dtype=preactivation.dtype)

class LipschitzSelection(SelectionFunction):
    r"""Lipschitz approximation of hard confidence selection.

    The surrogate has half-width ``epsilon`` around each confidence boundary.
    By default, both the forward value and its derivative use this surrogate,
    matching the original Lipschitz selector. If ``hard_forward=True``, the
    forward value is instead exactly

    .. math::

        \mathbf 1\{r\in(-\infty,\kappa_-]\cup[\kappa_+,\infty)\}.

    In hard-forward mode, a straight-through construction still gives autograd
    the derivative of the same ``epsilon``-width surrogate. Thus there is one
    bandwidth and an independent choice of forward semantics.
    """

    def __init__(self, epsilon: float = 0.1, hard_forward: bool = False):
        self.epsilon = _validate_width(epsilon, "epsilon")
        self.hard_forward = _validate_hard_forward(hard_forward)

    def __call__(self, preactivation: torch.Tensor, pos_margin: float, neg_margin: float) -> torch.Tensor:
        if pos_margin == 0.0 and neg_margin == 0.0:
            return torch.ones_like(preactivation)

        selected = (preactivation >= pos_margin) | (preactivation <= neg_margin)
        hard = selected.to(dtype=preactivation.dtype)

        width = self.epsilon
        soft_pos = torch.clamp(
            (preactivation - (pos_margin - width)) / (2.0 * width),
            min=0.0,
            max=1.0,
        )
        soft_neg = torch.clamp(
            ((neg_margin + width) - preactivation) / (2.0 * width),
            min=0.0,
            max=1.0,
        )
        soft = soft_pos + soft_neg

        if self.hard_forward:
            # Exact hard values in the forward pass; Lipschitz derivatives in
            # the backward pass.
            return hard.detach() + (soft - soft.detach())
        return soft

class SmoothSelection(SelectionFunction):
    r"""Smooth approximation of hard confidence selection.

    The surrogate is a compactly supported, infinitely differentiable
    transition of half-width ``epsilon`` around each confidence boundary. It is
    built from the standard smooth step

        S(t) = exp(-1/t) / (exp(-1/t) + exp(-1/(1-t))).

    By default, both the forward value and its derivative use the smooth
    surrogate, matching the original selector. If ``hard_forward=True``, the
    forward value is the exact hard confidence mask while autograd continues to
    use the derivative of the same ``epsilon``-width surrogate.
    """

    def __init__(self, epsilon: float = 0.1, hard_forward: bool = False):
        self.epsilon = _validate_width(epsilon, "epsilon")
        self.hard_forward = _validate_hard_forward(hard_forward)

    @staticmethod
    def _smooth_step(x: torch.Tensor) -> torch.Tensor:
        """C-infinity step equal to zero/one outside the interval (0, 1)."""
        interior = (x > 0.0) & (x < 1.0)
        # Clamp before taking reciprocals so that inactive torch.where branches
        # remain finite (and hence have well-defined autograd derivatives).
        safe_x = x.clamp(min=torch.finfo(x.dtype).tiny, max=1.0)
        safe_one_minus_x = (1.0 - x).clamp(
            min=torch.finfo(x.dtype).tiny, max=1.0
        )
        log_left = -1.0 / safe_x
        log_right = -1.0 / safe_one_minus_x
        smooth = torch.sigmoid(log_left - log_right)
        return torch.where(x <= 0.0, 0.0, torch.where(interior, smooth, 1.0))

    def __call__(self, preactivation: torch.Tensor, pos_margin: float, neg_margin: float) -> torch.Tensor:
        if pos_margin == 0.0 and neg_margin == 0.0:
            return torch.ones_like(preactivation)

        selected = (preactivation >= pos_margin) | (preactivation <= neg_margin)
        hard = selected.to(dtype=preactivation.dtype)

        width = self.epsilon
        pos_coordinate = (
            preactivation - (pos_margin - width)
        ) / (2.0 * width)
        neg_coordinate = (
            (neg_margin + width) - preactivation
        ) / (2.0 * width)

        soft = self._smooth_step(pos_coordinate) + self._smooth_step(
            neg_coordinate
        )

        if self.hard_forward:
            # Exact hard values in the forward pass; C-infinity derivatives in
            # the backward pass.
            return hard.detach() + (soft - soft.detach())
        return soft

"""Explicit finite initialisation and theorem-external score helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from src.environment import QuenchedEnvironment


def sign_with_positive_tie(scores: torch.Tensor) -> torch.Tensor:
    """Return ``sign(scores)`` with the manuscript convention ``sign(0)=+1``."""

    return torch.where(scores >= 0, torch.ones_like(scores), -torch.ones_like(scores))


def compute_scores(X: torch.Tensor, b: float | torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute ``r=b 1 + Xw/sqrt(d)`` without mutating inputs."""

    if X.ndim != 2 or w.ndim != 1 or X.shape[1] != w.numel():
        raise ValueError("require X.shape == (n, d) and w.shape == (d,)")
    if X.device != w.device:
        raise ValueError("X and w must share a device")
    return X @ w / (w.numel() ** 0.5) + torch.as_tensor(b, dtype=X.dtype, device=X.device)


def compute_pseudo_labels_from_scores(scores: torch.Tensor) -> torch.Tensor:
    """Same-design/endogenous initialisation helper (theorem-external)."""

    return sign_with_positive_tie(scores)


@dataclass(frozen=True)
class SelfTrainingInitialization:
    """Full-index initial condition ``(b^0,w^0,Y^{init})``."""

    b_init: float | torch.Tensor
    w_init: torch.Tensor
    Y_init: torch.Tensor

    def for_environment(self, environment: QuenchedEnvironment) -> "SelfTrainingInitialization":
        w = torch.as_tensor(self.w_init, dtype=torch.float64, device=environment.mu.device).detach().clone()
        y_init = torch.as_tensor(self.Y_init, dtype=torch.float64, device=environment.mu.device).detach().clone()
        b = torch.as_tensor(self.b_init, dtype=torch.float64, device=environment.mu.device).detach().clone()
        if w.shape != (environment.d,):
            raise ValueError(f"w_init must have shape ({environment.d},)")
        if y_init.shape != (environment.n,):
            raise ValueError(f"Y_init must have shape ({environment.n},)")
        if b.numel() != 1 or not torch.isfinite(b).all() or not torch.isfinite(w).all():
            raise ValueError("b_init and w_init must be finite")
        if not torch.all((y_init == -1) | (y_init == 1)):
            raise ValueError("Y_init entries must belong to {-1,+1}")
        if torch.any(y_init[environment.I_L] != environment.Y[environment.I_L]):
            raise ValueError("Y_init must equal Y on labelled coordinates")
        return SelfTrainingInitialization(b.reshape(()), w, y_init)

    @classmethod
    def from_unlabeled_labels(
        cls,
        environment: QuenchedEnvironment,
        *,
        b_init: float | torch.Tensor,
        w_init: torch.Tensor,
        Y_init_unlabeled: torch.Tensor,
    ) -> "SelfTrainingInitialization":
        y_unl = torch.as_tensor(Y_init_unlabeled, dtype=torch.float64, device=environment.mu.device)
        if y_unl.shape != (environment.M,):
            raise ValueError(f"Y_init_unlabeled must have shape ({environment.M},)")
        y_init = environment.Y.clone()
        y_init[environment.I_U] = y_unl
        return cls(b_init, w_init, y_init).for_environment(environment)

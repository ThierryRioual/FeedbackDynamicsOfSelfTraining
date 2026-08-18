"""Shared pure mathematical primitives for finite GD and effective dynamics."""

from __future__ import annotations

import math
from typing import Optional

import torch

from src.initialization import sign_with_positive_tie


def pseudo_labels(t: int, scores: torch.Tensor, Y_init: torch.Tensor) -> torch.Tensor:
    if t < 0:
        raise ValueError("t must be nonnegative")
    if t == 0:
        if Y_init.shape != scores.shape:
            raise ValueError("Y_init and scores must have the same shape")
        return Y_init.to(dtype=scores.dtype, device=scores.device)
    return sign_with_positive_tie(scores)


def hard_selection(scores: torch.Tensor, kappa_minus: float, kappa_plus: float) -> torch.Tensor:
    if not kappa_minus < 0 < kappa_plus:
        raise ValueError("require kappa_minus < 0 < kappa_plus")
    return ((scores <= kappa_minus) | (scores >= kappa_plus)).to(dtype=scores.dtype)


def selection_rate(selection: torch.Tensor, Delta: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Empirical ``omega`` over unlabeled observations (zero if ``M=0``)."""

    if Delta is None:
        return selection.mean() if selection.numel() else selection.new_zeros(())
    unlabeled = Delta == 0
    M = int(unlabeled.sum().item())
    if M == 0:
        return selection.new_zeros(())
    return selection[unlabeled].sum() / M


def normalized_selection(selection: torch.Tensor, omega: torch.Tensor | float) -> torch.Tensor:
    rate = float(omega.detach().item()) if isinstance(omega, torch.Tensor) else float(omega)
    if not math.isfinite(rate):
        raise ValueError("omega must be finite")
    return selection / rate if rate > 0 else torch.zeros_like(selection)


def pseudo_residual(
    *,
    scores: torch.Tensor,
    Y: torch.Tensor,
    Delta: torch.Tensor,
    Yhat: torch.Tensor,
    selection: torch.Tensor,
    omega: torch.Tensor | float,
    pi: float,
    eta: float,
    rho: float,
    loss_function,
) -> torch.Tensor:
    """Finite/particle vector ``g=-eta`` times the frozen score gradient."""

    labelled = torch.zeros_like(scores) if rho == 0 else Delta / rho * loss_function.gradient(scores, Y)
    omega_value = omega.detach().item() if isinstance(omega, torch.Tensor) else float(omega)
    if pi == 0 or rho >= 1 or omega_value <= 0:
        return -eta * labelled
    unlabeled = ((1 - Delta) * pi / (1 - rho) * normalized_selection(selection, omega)
                 * loss_function.gradient(scores, Yhat))
    return -eta * (labelled + unlabeled)

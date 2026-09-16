import torch
torch.set_default_dtype(torch.float64)

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from src.objectives import (
    LossFunction, Penalty, LogisticLoss, RidgePenalty, 
    SelectionFunction, HardSelection, LipschitzSelection, SmoothSelection
)


@dataclass(frozen=True)
class DataConfig:
    """
    Population-level data generating process parameters.
    These define the macroscopic quantities that characterize the data distribution
    in the high-dimensional limit — the 'Laws of the Universe'.
    """
    scale: float                        # noise scale (σ)
    label_prior: float                  # class balance
    supervision_ratio: float            # probability a sample's label is observed ρ 
    data_to_dimension_ratio: float      # δ = n/d
    signal_law: Callable[[], float]   # distribution of signal vector entries

    def __post_init__(self):
        """Validates population-level parameters."""
        if not (0 < self.supervision_ratio <= 1):
            raise ValueError(
                f"supervision_ratio must be in (0, 1], got {self.supervision_ratio}"
            )
        if not (0 < self.label_prior < 1):
            raise ValueError(
                f"label_prior must be in (0, 1), got {self.label_prior}"
            )
        if self.data_to_dimension_ratio <= 0:
            raise ValueError(
                f"data_to_dimension_ratio must be positive, got {self.data_to_dimension_ratio}"
            )


class PseudoLabelSchedule(Protocol):
    """Experimental time-dependent pseudo-label weight.

    The manuscript model is represented by the absence of a schedule, i.e.
    ``pi_t == pi``.  This protocol deliberately keeps schedules outside the
    canonical parameter list while retaining the historical ramp facility.
    """

    def value(self, t: int, *, pi: float) -> float: ...


@dataclass(frozen=True)
class LinearRampSchedule:
    """Historical burn-in/linear-ramp extension (not part of the theorem)."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("require 0 <= start <= end for a linear ramp")

    def value(self, t: int, *, pi: float) -> float:
        if t < self.start:
            return 0.0
        if self.end == self.start or t >= self.end:
            return float(pi)
        return float(pi) * (t - self.start) / (self.end - self.start)


from dataclasses import dataclass, field
import typing

@dataclass(frozen=True)
class AlgorithmConfig:
    """
    Learning algorithm hyperparameters.
    """
    n_iterations: int          # T
    step_size: float           # \eta
    penalty_param: float       # \lambda
    pseudo_label_param: float  # \pi
    ramp_start: Optional[int] = None            # T_0
    ramp_end: Optional[int] = None           # T_1
    margin_threshold: Optional[float] = None  # \kappa
    positive_margin: Optional[float] = None
    negative_margin: Optional[float] = None
    include_bias: bool = True
    loss_function: LossFunction = field(default_factory=LogisticLoss) # \ell
    penalty_function: Penalty = field(default_factory=RidgePenalty)
    selection_function: SelectionFunction = field(default_factory=HardSelection) # Selection strategy
    experimental_schedule: Optional[PseudoLabelSchedule] = None
    initial_bias: float = 0.0
    bias_pseudo_label_param: Optional[float] = None  # \pi_b; None follows \pi^t
    
    normalized_threshold: bool = False  # Apply thresholds to r / tau (no sigma).
    normalize_unlabeled_loss: bool = True  # Divide unlabeled selection weights by omega.

    # Internal schedule stored as a list of floats
    pseudo_label_param_schedule_: list[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        """
        Precompute the pseudo-label parameter schedule as a list of floats.
        """
        if self.n_iterations < 0:
            raise ValueError("n_iterations must be nonnegative")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if self.penalty_param < 0:
            raise ValueError("penalty_param must be nonnegative")
        if self.pseudo_label_param < 0:
            raise ValueError("pseudo_label_param must be nonnegative")
        if not math.isfinite(float(self.initial_bias)):
            raise ValueError("initial_bias must be finite")
        if (
            self.bias_pseudo_label_param is not None
            and (
                not math.isfinite(float(self.bias_pseudo_label_param))
                or self.bias_pseudo_label_param < 0
            )
        ):
            raise ValueError("bias_pseudo_label_param must be finite and nonnegative")

        if self.margin_threshold is not None:
            if self.positive_margin is None:
                object.__setattr__(self, "positive_margin", self.margin_threshold)
            if self.negative_margin is None:
                object.__setattr__(self, "negative_margin", -self.margin_threshold)
        
        if self.positive_margin is None or self.negative_margin is None:
            raise ValueError(
                "must specify either margin_threshold, or both positive_margin and negative_margin"
            )
        if not (self.negative_margin < 0 < self.positive_margin):
            # The all-selected zero-threshold selector is retained as an
            # explicit theorem-external experimental variant.
            if not (self.negative_margin == 0.0 and self.positive_margin == 0.0):
                raise ValueError("require negative_margin < 0 < positive_margin")

        if self.experimental_schedule is not None:
            schedule = [
                float(self.experimental_schedule.value(t, pi=self.pseudo_label_param))
                for t in range(self.n_iterations)
            ]
        elif self.ramp_start is None and self.ramp_end is None:
            schedule = [self.pseudo_label_param] * self.n_iterations
        elif self.ramp_start is None or self.ramp_end is None:
            raise ValueError("ramp_start and ramp_end must be specified together")
        else:
            # Backward-compatible spelling of the explicitly experimental
            # schedule.  In particular (0, 0) is now genuinely no-burn.
            ramp = LinearRampSchedule(self.ramp_start, self.ramp_end)
            schedule = [ramp.value(t, pi=self.pseudo_label_param) for t in range(self.n_iterations)]

        # Necessary pattern to assign fields on frozen dataclasses
        object.__setattr__(self, "pseudo_label_param_schedule_", schedule)

    def selection_mask(self, scores: torch.Tensor, tau=None) -> torch.Tensor:
        """Evaluate the configured selector, optionally on ``r / tau``.

        The norm is frozen for score differentiation in state evolution.
        Old configurations without the optional field retain raw thresholds.
        """
        from src.objectives import selection_mask

        return selection_mask(
            scores, self.positive_margin, self.negative_margin,
            getattr(self, "normalized_threshold", False), tau, self.selection_function,
        )

    def get_pseudo_label_weight(self, t: int) -> float:
        r"""Returns the pseudo-label weight \pi^t at iteration t."""
        if t < 0 or t >= self.n_iterations:
            raise IndexError(f"iteration {t} is outside [0, {self.n_iterations})")
        return self.pseudo_label_param_schedule_[t]

    def get_bias_pseudo_label_weight(self, t: int) -> float:
        r"""Return the bias pseudo-label weight \pi_b^t at iteration ``t``.

        An unspecified bias-specific weight follows the ordinary pseudo-label
        schedule exactly, preserving the historical dynamics.  An explicitly
        configured value is fixed across iterations.
        """

        weight = self.get_pseudo_label_weight(t)
        if self.bias_pseudo_label_param is None:
            return weight
        return float(self.bias_pseudo_label_param)

    @property
    def is_canonical_fixed_pi(self) -> bool:
        """Whether this configuration is the fixed-``pi`` manuscript model."""

        return self.experimental_schedule is None and self.ramp_start is None and self.ramp_end is None

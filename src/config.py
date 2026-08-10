import torch
torch.set_default_dtype(torch.float64)

from dataclasses import dataclass, field
from typing import Callable, Optional

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
    ramp_start: int            # T_0
    ramp_end: int              # T_1
    margin_threshold: Optional[float] = None  # \kappa
    positive_margin: Optional[float] = None
    negative_margin: Optional[float] = None
    include_bias: bool = True
    loss_function: LossFunction = field(default_factory=LogisticLoss) # \ell
    penalty_function: Penalty = field(default_factory=RidgePenalty)
    selection_function: SelectionFunction = field(default_factory=HardSelection) # Selection strategy
    
    # Internal schedule stored as a list of floats
    pseudo_label_param_schedule_: list[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        """
        Precompute the pseudo-label parameter schedule as a list of floats.
        """
        if self.margin_threshold is not None:
            if self.positive_margin is None:
                object.__setattr__(self, "positive_margin", self.margin_threshold)
            if self.negative_margin is None:
                object.__setattr__(self, "negative_margin", -self.margin_threshold)
        
        assert self.positive_margin is not None and self.negative_margin is not None, \
            "Must specify either margin_threshold, or both positive_margin and negative_margin."

        schedule = []
        ramp_range = self.ramp_end - self.ramp_start
        
        for t in range(self.n_iterations):
            if t <= self.ramp_start:
                val = 0.0
            elif t >= self.ramp_end:
                val = self.pseudo_label_param
            else:
                val = self.pseudo_label_param * (t - self.ramp_start) / ramp_range
            schedule.append(float(val))

        # Necessary pattern to assign fields on frozen dataclasses
        object.__setattr__(self, "pseudo_label_param_schedule_", schedule)

    def get_pseudo_label_weight(self, t: int) -> float:
        """Returns the pseudo-label weight \pi^t at iteration t."""
        return self.pseudo_label_param_schedule_[t]

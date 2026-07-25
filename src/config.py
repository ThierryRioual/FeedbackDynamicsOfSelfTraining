import torch
import math

from dataclasses import dataclass, field
from typing import Optional, Callable

from src.objectives import LossFunction, Penalty, LogisticLoss, RidgePenalty

@dataclass(frozen=True)
class DataConfig:
    """
    Data Generating Process parameters.

    Note:
    - In the high-dimensional regime, the number of samples and the number of dimensions 
    depend on each other since they converge to a fixed ratio. 
    Therefore, we need to specify both in the data generating process parameters.
    """
    
    # Universal Parameters
    label_prior: float
    scale: float     
    
    # Finite Parameters (Empirical DGP)
    n_labeled: Optional[int] = None
    n_unlabeled: Optional[int] = None
    # Use default=None so the class can instantiate, then we overwrite it
    n_train: Optional[int] = field(default=None, init=False) 
    n_test: Optional[int] = None
    dimensions: Optional[int] = None
    signal_vector: Optional[torch.Tensor] = None
    
    # Asymptotic Parameters (State Evolution)
    labeled_data_ratio: Optional[float] = None
    data_to_dimension_ratio: Optional[float] = None
    signal_prior: Optional[Callable[[], float]] = None

    def __post_init__(self):
        """
        Validates the configuration and computes asymptotic limits 
        from finite sizes if they were not explicitly provided.
        """
        # If the user provided the finite sample sizes
        if self.n_labeled is not None and self.n_unlabeled is not None:
            n_train = self.n_labeled + self.n_unlabeled
            
            # The only way to assign variables in a frozen dataclass
            object.__setattr__(self, 'n_train', n_train)
            
            rho = self.n_labeled / n_train
            if self.labeled_data_ratio is None:
                object.__setattr__(self, 'labeled_data_ratio', rho)
            else:
                assert math.isclose(self.labeled_data_ratio, rho, rel_tol=1e-5), \
                    f"Ratio mismatch: theoretical={self.labeled_data_ratio}, empirical={rho}"

            if self.dimensions is not None:
                delta = n_train / self.dimensions
                if self.data_to_dimension_ratio is None:
                    object.__setattr__(self, 'data_to_dimension_ratio', delta)
                else:
                    assert math.isclose(self.data_to_dimension_ratio, delta, rel_tol=1e-5), \
                        f"Ratio mismatch: theoretical={self.data_to_dimension_ratio}, empirical={delta}"
                        
        # If they did not provide finite sample sizes, ensure they provided the ratios.
        elif self.labeled_data_ratio is None or self.data_to_dimension_ratio is None:
            raise ValueError(
                "DataConfig requires either finite sizes (n_labeled, n_unlabeled, dimensions) "
                "OR macroscopic ratios (labeled_data_ratio, data_to_dimension_ratio)."
            )

@dataclass(frozen=True)
class AlgorithmConfig:
    """
    Learning algorithm hyperparameters.
    """
    n_iterations: int # T
    margin_threshold: float # \kappa
    step_size: float # \eta
    penalty_param: float # \lambda
    pseudo_label_param: float # \pi
    ramp_start: int # T_0
    ramp_end: int # T_1
    include_bias: bool = True
    loss_function: LossFunction = field(default_factory=LogisticLoss) # \ell
    penalty_function: Penalty = field(default_factory=RidgePenalty)
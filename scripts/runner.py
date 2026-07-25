import torch
from dataclasses import dataclass, field, replace, asdict
from typing import Dict, List, Any, Type, Optional, Set

from src.config import AlgorithmConfig
from src.algorithms import SelfTrainedGradientDescent
from src.callbacks import TestEvaluatorCallback

from scripts.plotting import plot_experiment

@dataclass
class MonteCarloExperiment:
    """
    Orchestrates Monte Carlo sweeps, managing configuration state and result persistence.
    """
    data_gen: Any
    algorithm: Type
    base_config: AlgorithmConfig
    
    metrics: Set[str] = field(default_factory=lambda: {"test_error"})
    
    # State variables excluded from the constructor argument list
    results_: Optional[Dict[str, torch.Tensor]] = field(init=False, default=None)
    sweep_param_name_: Optional[str] = field(init=False, default=None)
    sweep_param_values_: Optional[List[Any]] = field(init=False, default=None)

    @property
    def d(self) -> int:
        return self.data_gen.cfg.dimensions

    @property
    def N(self) -> int:
        return self.data_gen.cfg.n_labeled

    @property
    def M(self) -> int:
        return self.data_gen.cfg.n_unlabeled

    @property
    def N_test(self) -> int:
        return self.data_gen.cfg.n_test

    @property
    def T(self) -> int:
        return self.base_config.n_iterations

    def run_sweep(
        self, 
        seeds: List[int],
        sweep_param_name: Optional[str] = None, 
        sweep_param_values: Optional[List[Any]] = None
    ) -> None:
        """Executes the Monte Carlo simulation and stores the result internally."""

        d = self.data_gen.cfg.dimensions  
        
        # Persist sweep configuration for the plotting module
        self.sweep_param_name_ = sweep_param_name
        self.sweep_param_values_ = sweep_param_values

        is_sweep = sweep_param_name is not None and sweep_param_values is not None
        values_to_test = sweep_param_values if is_sweep else [None]
        
        length = len(values_to_test)
        runs = len(seeds)
        
        # Initialize results as a dictionary of 3D PyTorch tensors
        res = {
            metric: torch.zeros((length, runs, self.T + 1), dtype=torch.float64) 
            for metric in self.metrics
        }

        for j, current_seed in enumerate(seeds):

            # Initialize the isolated PyTorch generator
            rng = torch.Generator().manual_seed(current_seed)
            self.data_gen.rng = rng

            X_lab, Y_lab, X_unl, Y_unl, X_test, Y_test = self.data_gen.sample()

            # Generate initial weights using PyTorch
            w0 = torch.randn(d, generator=rng, dtype=torch.float64)
            
            for i, val in enumerate(values_to_test):
                
                # The pythonic way to modify a frozen dataclass for a single run:
                if is_sweep:
                    assert sweep_param_name is not None 
                    # Creates a brand new config with just the sweep parameter altered
                    current_cfg = replace(self.base_config, **{sweep_param_name: val})
                else:
                    current_cfg = self.base_config
                
                callback = TestEvaluatorCallback(
                    X_lab=X_lab, Y_lab=Y_lab,
                    X_unl=X_unl, Y_unl=Y_unl,
                    X_test=X_test, Y_test=Y_test,
                    mu=self.data_gen._mu,
                    metrics=self.metrics
                )
                
                experiment = self.algorithm(cfg=current_cfg, callback=callback)
                
                # Use .clone() instead of .copy() for PyTorch tensors
                experiment.fit(X_lab, Y_lab, X_unl, initial_weights=w0.clone())
                
                for metric in self.metrics:
                    # Convert the list of python floats back into a 1D tensor and assign
                    res[metric][i, j, :] = torch.tensor(callback.history_[metric], dtype=torch.float64)

        self.results_ = res

    def plot_trajectories(self, metric: str = "test_error") -> None:
        """Delegates the visualization of stored results to the plotting module."""
        if self.results_ is None:
            raise ValueError("You must call run_sweep() before plotting.")
        if metric not in self.results_:
            raise KeyError(f"Metric '{metric}' was not tracked during the experiment.")
        
        plot_experiment(
            res=self.results_[metric],
            # Use asdict() so we don't have to rewrite the plotting script!
            base_params=asdict(self.base_config),
            d=self.d,
            N=self.N, 
            M=self.M, 
            N_test=self.N_test, 
            T=self.T,
            algorithm_name=self.algorithm.__name__,
            sweep_param_name=self.sweep_param_name_,
            sweep_param_values=self.sweep_param_values_,
            metric_name=metric
        )
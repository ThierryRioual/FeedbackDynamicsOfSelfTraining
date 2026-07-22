import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Type, Optional, Set

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
    N: int
    M: int
    N_test: int
    base_params: Dict[str, Any]

    T: int = field(init=False, default=None)
    
    metrics: Set[str] = field(default_factory=lambda: {"test_error"})
    
    # State variables excluded from the constructor argument list
    results_: Optional[Dict[str, np.ndarray]] = field(init=False, default=None)
    sweep_param_name_: Optional[str] = field(init=False, default=None)
    sweep_param_values_: Optional[List[Any]] = field(init=False, default=None)

    def __post_init__(self):
        """Sever the reference to the external dictionary to prevent state contamination."""
        # Copy to avoid mutating the caller's dict
        self.base_params = self.base_params.copy()
        self.T = self.base_params['n_iterations']

    def run_sweep(
        self, 
        seeds: List[int],
        sweep_param_name: Optional[str] = None, 
        sweep_param_values: Optional[List[Any]] = None
    ) -> Dict[str, np.ndarray]:
        """Executes the Monte Carlo simulation and stores the result internally."""

        d = self.data_gen.d  # Dimensionality of the feature space
        
        # Persist sweep configuration for the plotting module
        self.sweep_param_name_ = sweep_param_name
        self.sweep_param_values_ = sweep_param_values

        is_sweep = sweep_param_name is not None and sweep_param_values is not None
        values_to_test = sweep_param_values if is_sweep else [None]
        
        length = len(values_to_test)
        runs = len(seeds)
        res = {metric: np.zeros((length, runs, self.T + 1)) for metric in self.metrics}

        for j, current_seed in enumerate(seeds):

            rng = np.random.default_rng(current_seed)
            self.data_gen.rng = rng

            X_lab, Y_lab, X_unl, Y_unl, X_test, Y_test = self.data_gen.sample(self.N, self.M, self.N_test)

            w0 = rng.normal(0, 1 , size=d)
            
            for i, val in enumerate(values_to_test):
                current_params = self.base_params.copy()
                
                if is_sweep:
                    current_params[sweep_param_name] = val
                    
                current_params['n_iterations'] = self.T
                
                callback = TestEvaluatorCallback(
                    X_lab=X_lab, Y_lab=Y_lab,
                    X_unl=X_unl, Y_unl=Y_unl,
                    X_test=X_test, Y_test=Y_test,
                    metrics=self.metrics
                )
                
                experiment = self.algorithm(**current_params, callback=callback)
                experiment.fit(X_lab, Y_lab, X_unl, initial_weights=w0.copy())
                
                for metric in self.metrics:
                    res[metric][i, j, :] = callback.history_[metric]

        self.results_ = res
        return None

    def plot_trajectories(self, metric: str = "test_error") -> None:
        """Delegates the visualization of stored results to the plotting module."""
        if self.results_ is None:
            raise ValueError("You must call run_sweep() before plotting.")
        if metric not in self.results_:
            raise KeyError(f"Metric '{metric}' was not tracked during the experiment.")
        
        # Pass all necessary metadata to the stateless plotting function
        plot_experiment(
            res=self.results_[metric],
            base_params=self.base_params,
            N=self.N, 
            M=self.M, 
            N_test=self.N_test, 
            T=self.T,
            algorithm_name=self.algorithm.__name__,
            sweep_param_name=self.sweep_param_name_,
            sweep_param_values=self.sweep_param_values_,
            metric_name=metric
        )
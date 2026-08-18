"""Small Monte-Carlo runner for manuscript-faithful finite experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Literal, Optional, Set, Type

import torch

from src.algorithms import SelfTrainedGradientDescent
from src.callbacks import TestEvaluatorCallback
from src.initialization import SelfTrainingInitialization, compute_pseudo_labels_from_scores, compute_scores



RepetitionMode = Literal["fixed_environment", "resampled_environment"]
InitializationMode = Literal["exogenous_rademacher", "endogenous_scores"]


@dataclass
class MonteCarloExperiment:
    """Run finite repetitions with explicit environment/initialisation semantics."""

    data_gen: Any
    algorithm: Type
    base_config: Any
    metrics: Set[str] = field(default_factory=lambda: {"test_error"})
    stratified: bool = False
    repetition_mode: RepetitionMode = "resampled_environment"
    initialization_mode: InitializationMode = "exogenous_rademacher"

    results_: Optional[Dict[str, torch.Tensor]] = field(init=False, default=None)
    metadata_: Dict[str, Any] = field(init=False, default_factory=dict)
    sweep_param_name_: Optional[str] = field(init=False, default=None)
    sweep_param_values_: Optional[List[Any]] = field(init=False, default=None)

    @property
    def d(self) -> int: return self.data_gen.dimensions
    @property
    def N(self) -> int: return self.data_gen.n_labeled
    @property
    def M(self) -> int: return self.data_gen.n_unlabeled
    @property
    def N_test(self) -> int: return self.data_gen.n_test
    @property
    def T(self) -> int: return self.base_config.n_iterations

    def _tracked_metrics(self) -> Set[str]:
        tracked = set(self.metrics)
        if "population_error" in tracked:
            tracked.update({"bias_term", "weight_vector_norm", "weight_signal_alignment"})
        if "train_mean_residual" in tracked:
            tracked.update({"lab_mean_residual", "unl_mean_residual"})
        if "train_label_residual_alignment" in tracked:
            tracked.update({"lab_label_residual_alignment", "unl_label_residual_alignment"})
        return tracked

    def _initialization(self, env, X, rng: torch.Generator) -> SelfTrainingInitialization:
        w0 = torch.randn(env.d, generator=rng, dtype=torch.float64, device=env.mu.device)
        y_init = env.Y.clone()
        if env.M:
            if self.initialization_mode == "exogenous_rademacher":
                y_init[env.I_U] = torch.where(
                    torch.rand(env.M, generator=rng, device=env.mu.device) < .5,
                    1.0,
                    -1.0,
                )
            elif self.initialization_mode == "endogenous_scores":
                y_init[env.I_U] = compute_pseudo_labels_from_scores(compute_scores(X, 0.0, w0))[env.I_U]
            else:
                raise ValueError(f"unknown initialization_mode {self.initialization_mode}")
        return SelfTrainingInitialization(0.0, w0, y_init).for_environment(env)

    @staticmethod
    def _streams(seed: int) -> tuple[torch.Generator, torch.Generator, torch.Generator, torch.Generator]:
        """Spawn environment/design/init/test streams from one repetition seed."""

        master = torch.Generator().manual_seed(seed)
        child_seeds = torch.randint(0, 2**63 - 1, (4,), generator=master, dtype=torch.int64).tolist()
        return tuple(torch.Generator().manual_seed(int(child)) for child in child_seeds)

    def run_sweep(self, seeds: List[int], sweep_param_name: Optional[str] = None, sweep_param_values: Optional[List[Any]] = None) -> None:
        self.sweep_param_name_, self.sweep_param_values_ = sweep_param_name, sweep_param_values
        values = sweep_param_values if sweep_param_name is not None and sweep_param_values is not None else [None]
        tracked = self._tracked_metrics()
        res = {metric: torch.zeros((len(values), len(seeds), self.T + 1), dtype=torch.float64) for metric in tracked}
        fixed_env = None
        fixed_init = None
        if self.repetition_mode == "fixed_environment":
            env_rng, design_rng, init_rng, _ = self._streams(seeds[0])
            self.data_gen.rng = env_rng
            fixed_env = self.data_gen.sample_environment(stratified=self.stratified)
            X0, _ = self.data_gen.sample_design(fixed_env, generator=design_rng)
            fixed_init = self._initialization(fixed_env, X0, init_rng)
        elif self.repetition_mode != "resampled_environment":
            raise ValueError("repetition_mode must be fixed_environment or resampled_environment")

        for j, seed in enumerate(seeds):
            env_rng, design_rng, init_rng, test_rng = self._streams(seed)
            self.data_gen.rng = env_rng
            env = fixed_env if fixed_env is not None else self.data_gen.sample_environment(stratified=self.stratified)
            X, _ = self.data_gen.sample_design(env, generator=design_rng)
            X_test, Y_test = self.data_gen.sample_test(generator=test_rng)
            init = fixed_init if fixed_init is not None else self._initialization(env, X, init_rng)
            for i, value in enumerate(values):
                cfg = replace(self.base_config, **{sweep_param_name: value}) if value is not None else self.base_config
                callback = TestEvaluatorCallback(
                    X_lab=X[env.I_L], Y_lab=env.Y[env.I_L], X_unl=X[env.I_U], Y_unl=env.Y[env.I_U],
                    X_test=X_test, Y_test=Y_test, mu=env.mu, sigma=self.data_gen.cfg.scale,
                    p=self.data_gen.law.label_prior, metrics=set(tracked),
                )
                learner = self.algorithm(cfg=cfg, callback=callback).fit_full(X, env, init)
                for metric in tracked:
                    res[metric][i, j, :] = torch.tensor(callback.history_[metric], dtype=torch.float64)
        self.results_ = res
        self.metadata_ = {
            "repetition_mode": self.repetition_mode,
            "initialization_mode": self.initialization_mode,
            "theorem_compatible": self.initialization_mode == "exogenous_rademacher" and self.base_config.is_canonical_fixed_pi and self.base_config.include_bias,
            "experimental_extensions": [
                name for name, active in {
                    "endogenous_sign_r0": self.initialization_mode == "endogenous_scores",
                    "pseudo_label_schedule": not self.base_config.is_canonical_fixed_pi,
                    "no_bias": not self.base_config.include_bias,
                }.items() if active
            ],
        }

    def plot_trajectories(self, metric: str = "test_error") -> None:
        if self.results_ is None:
            raise ValueError("You must call run_sweep() before plotting.")
        if metric not in self.results_:
            raise KeyError(f"Metric '{metric}' was not tracked during the experiment.")
        from scripts.plotting import plot_experiment
        plot_experiment(res=self.results_[metric], base_params=asdict(self.base_config), d=self.d, N=self.N, M=self.M, N_test=self.N_test, T=self.T, algorithm_name=self.algorithm.__name__, sweep_param_name=self.sweep_param_name_, sweep_param_values=self.sweep_param_values_, metric_name=metric)

"""Shared, current-API utilities for the numerical experiment notebooks.

The helpers deliberately construct the finite and state-evolution experiments
from the same population laws but independent random draws.  They are for
notebook orchestration only: finite learning remains implemented exclusively
by :class:`src.algorithms.SelfTrainedGradientDescent`, and effective dynamics
by :class:`src.asymptotics.MacroscopicStateEvolution`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence
from unittest.mock import patch

from matplotlib import pyplot as plt
import numpy as np
import torch

from src.algorithms import SelfTrainedGradientDescent
from src.asymptotics import MacroscopicStateEvolution
from src.callbacks import TestEvaluatorCallback
from src.config import AlgorithmConfig, DataConfig
from src.dgp import IsotropicGaussian
from src.environment import (
    FourCellSampleTypeLaw,
    QuenchedEnvironment,
    state_evolution_sample_base_sampler,
    validate_finite_se_aspect_ratio,
)
from src.initialization import SelfTrainingInitialization, sign_with_positive_tie
from src.performance import bayes_parameters, population_error


DEFAULT_METRICS = {
    "population_error",
    "unl_usage",
    "weight_signal_alignment",
    "weight_vector_norm",
    "bias_term",
}


@dataclass
class ExperimentRun:
    """One finite/SE comparison and its immutable configuration."""

    name: str
    data_cfg: DataConfig
    algo_cfg: AlgorithmConfig
    environment: QuenchedEnvironment
    X: torch.Tensor
    X_test: torch.Tensor
    Y_test: torch.Tensor
    finite: SelfTrainedGradientDescent
    callback: TestEvaluatorCallback
    se: Optional[MacroscopicStateEvolution]
    signal_scale: float
    metadata: dict[str, Any]


def as_numpy(values: Iterable[Any]) -> np.ndarray:
    """Convert scalar tensor histories to a float NumPy array."""

    return np.asarray([float(torch.as_tensor(value)) for value in values], dtype=float)


def rademacher(
    size: int,
    generator: torch.Generator,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.where(
        torch.rand(size, generator=generator, dtype=dtype, device=device) < 0.5,
        torch.ones(size, dtype=dtype, device=device),
        -torch.ones(size, dtype=dtype, device=device),
    )


def exogenous_initial_labels(
    environment: QuenchedEnvironment, generator: torch.Generator
) -> torch.Tensor:
    """Return full ``Y_init`` with labelled coordinates fixed to ``Y``."""

    y_init = rademacher(environment.n, generator, device=environment.mu.device)
    y_init[environment.I_L] = environment.Y[environment.I_L]
    return y_init


def make_parameter_base_sampler(signal_std: float, initialization_correlation: float = 0.0):
    """Return the joint ``(mu_tilde,w_init_tilde)`` particle law.

    ``initialization_correlation`` is the coefficient in
    ``w_init=c mu/signal_std+sqrt(1-c^2) epsilon``.  Thus, asymptotically,
    ``E[mu*w_init]=c*signal_std`` and the initial weight variance remains one.
    """

    if not -1.0 <= initialization_correlation <= 1.0:
        raise ValueError("initialization_correlation must lie in [-1, 1]")

    def sampler(K: int, generator: torch.Generator, dtype: torch.dtype, device: torch.device):
        mu = signal_std * torch.randn(K, generator=generator, dtype=dtype, device=device)
        innovation = torch.randn(K, generator=generator, dtype=dtype, device=device)
        if signal_std == 0:
            w_init = innovation
        else:
            w_init = (
                initialization_correlation * mu / signal_std
                + np.sqrt(1.0 - initialization_correlation**2) * innovation
            )
        return mu, w_init

    return sampler


def make_algorithm_config(
    *,
    T: int,
    eta: float,
    penalty: float,
    pi: float,
    kappa: float,
    include_bias: bool = True,
    experimental_schedule=None,
) -> AlgorithmConfig:
    """Create a fixed-pi config unless an explicit experimental schedule is supplied."""

    return AlgorithmConfig(
        n_iterations=T,
        step_size=eta,
        penalty_param=penalty,
        pseudo_label_param=pi,
        margin_threshold=kappa,
        include_bias=include_bias,
        experimental_schedule=experimental_schedule,
    )


def run_experiment(
    *,
    name: str,
    d: int,
    delta: float,
    n_test: int,
    label_prior: float,
    rho: float,
    sigma: float,
    signal_std: float,
    algo_cfg: AlgorithmConfig,
    seed: int,
    K_w: Optional[int] = None,
    K_g: Optional[int] = None,
    run_state_evolution: bool = True,
    aspect_ratio_tolerance: float = 1e-12,
    initialization_correlation: float = 0.0,
    metadata: Optional[dict[str, Any]] = None,
) -> ExperimentRun:
    """Run finite GD and, optionally, its independent particle approximation.

    The finite initial labels are exogenous and the particle sampler uses the
    corresponding conditional law.  This is the canonical fixed-pi setup when
    ``algo_cfg.is_canonical_fixed_pi`` is true.
    """

    if d <= 0 or delta <= 0:
        raise ValueError("d and delta must be positive")
    n = int(round(delta * d))
    if n <= 0:
        raise ValueError("delta*d must yield at least one training observation")

    data_cfg = DataConfig(
        scale=sigma,
        label_prior=label_prior,
        supervision_ratio=rho,
        data_to_dimension_ratio=delta,
        # Explicit signal_vector below avoids the legacy global-RNG callable.
        signal_law=lambda: 0.0,
    )
    law = FourCellSampleTypeLaw.product(
        label_prior=label_prior, supervision_ratio=rho
    )
    finite_generator = torch.Generator().manual_seed(seed + 11)
    mu = signal_std * torch.randn(d, generator=finite_generator)
    initial_innovation = torch.randn(d, generator=finite_generator)
    if signal_std == 0:
        w_init = initial_innovation
    else:
        w_init = (
            initialization_correlation * mu / signal_std
            + np.sqrt(1.0 - initialization_correlation**2) * initial_innovation
        )
    dgp = IsotropicGaussian(
        cfg=data_cfg,
        n_train=n,
        n_test=n_test,
        dimensions=d,
        seed=seed + 23,
        signal_vector=mu,
        sample_type_law=law,
    )
    environment, X, X_test, Y_test = dgp.sample_full()
    validate_finite_se_aspect_ratio(
        environment, delta, tolerance=aspect_ratio_tolerance
    )
    init_generator = torch.Generator().manual_seed(seed + 31)
    initialization = SelfTrainingInitialization(
        b_init=0.0,
        w_init=w_init,
        Y_init=exogenous_initial_labels(environment, init_generator),
    )
    callback = TestEvaluatorCallback(
        X_lab=X[environment.I_L],
        Y_lab=environment.Y[environment.I_L],
        X_unl=X[environment.I_U],
        Y_unl=environment.Y[environment.I_U],
        X_test=X_test,
        Y_test=Y_test,
        mu=environment.mu,
        sigma=sigma,
        p=label_prior,
        metrics=DEFAULT_METRICS,
    )
    finite = SelfTrainedGradientDescent(cfg=algo_cfg, callback=callback)
    finite.fit_full(X, environment, initialization)

    se = None
    if run_state_evolution:
        if K_w is None or K_g is None:
            raise ValueError("K_w and K_g are required for state evolution")
        se = MacroscopicStateEvolution(
            data_cfg=data_cfg,
            algo_cfg=algo_cfg,
            mc_seed=seed + 47,
            K=None,
            K_w=K_w,
            K_g=K_g,
            parameter_base_sampler=make_parameter_base_sampler(
                signal_std, initialization_correlation
            ),
            sample_base_sampler=state_evolution_sample_base_sampler(law),
        )
        se.compute_trajectory()

    run_metadata = {
        "fixed_pi": algo_cfg.is_canonical_fixed_pi,
        "initialization": "exogenous independent Rademacher on unlabeled coordinates",
        "finite_environment": "one iid draw of the product/MCAR special case",
        "state_evolution": "independent particle approximation",
        "initialization_correlation": initialization_correlation,
    }
    if metadata:
        run_metadata.update(metadata)
    return ExperimentRun(
        name=name,
        data_cfg=data_cfg,
        algo_cfg=algo_cfg,
        environment=environment,
        X=X,
        X_test=X_test,
        Y_test=Y_test,
        finite=finite,
        callback=callback,
        se=se,
        signal_scale=float(torch.linalg.vector_norm(mu) / np.sqrt(d)),
        metadata=run_metadata,
    )


def finite_state_observables(run: ExperimentRun) -> dict[str, np.ndarray]:
    """State-indexed finite macroscopic trajectories (length T+1)."""

    callback = run.callback
    return {
        "error": np.asarray(callback.history_["population_error"]),
        "m": np.asarray(callback.history_["weight_signal_alignment"]),
        "tau": np.asarray(callback.history_["weight_vector_norm"]),
        "energy": np.asarray(callback.history_["weight_vector_norm"]) ** 2,
        "bias": np.asarray(callback.history_["bias_term"]),
    }


def finite_update_observables(run: ExperimentRun) -> dict[str, np.ndarray]:
    """Update-indexed finite statistics, including selected-label precision."""

    env = run.environment
    values = {key: [] for key in ("chi", "zeta", "omega", "accuracy", "correct_mass", "incorrect_mass")}
    for step in run.finite.update_records_:
        selected = (env.Delta == 0) & (step.selection > 0)
        selected_count = int(selected.sum().item())
        correct = selected & (step.pseudo_labels == env.Y)
        values["chi"].append(float(step.chi))
        values["zeta"].append(float(step.zeta))
        values["omega"].append(float(step.omega))
        values["accuracy"].append(float(correct.sum() / selected_count) if selected_count else np.nan)
        values["correct_mass"].append(float(correct.double().mean()))
        values["incorrect_mass"].append(float((selected & ~correct).double().mean()))
    return {key: np.asarray(value) for key, value in values.items()}


def state_evolution_state_observables(run: ExperimentRun) -> dict[str, np.ndarray]:
    """State-indexed particle trajectories."""

    if run.se is None:
        raise ValueError("this run does not contain state evolution")
    se = run.se
    tau = as_numpy(se.weight_norm)
    return {
        "error": as_numpy(se.error),
        "m": as_numpy(se.weight_signal_alignments),
        "tau": tau,
        "energy": tau**2,
        "bias": as_numpy(se.bias),
    }


def state_evolution_update_observables(run: ExperimentRun) -> dict[str, np.ndarray]:
    """Update-indexed particle statistics, conditional on confidence selection."""

    if run.se is None:
        raise ValueError("this run does not contain state evolution")
    se, T = run.se, run.algo_cfg.n_iterations
    values = {key: [] for key in ("chi", "zeta", "omega", "accuracy", "correct_mass", "incorrect_mass")}
    for t in range(T):
        r = se.preactivation[t]
        if r is None:
            raise RuntimeError("state-evolution trajectory is incomplete")
        yhat = se.initial_pseudo_label if t == 0 else sign_with_positive_tie(r)
        mask = se.algo_cfg.selection_function(r, se.algo_cfg.positive_margin, se.algo_cfg.negative_margin) > 0
        selected = (se.indicator == 0) & mask
        selected_count = int(selected.sum().item())
        correct = selected & (yhat == se.label)
        values["chi"].append(float(se.label_residual_alignments[t]))
        values["zeta"].append(float(se.mean_residual[t]))
        values["omega"].append(float(se.selection_rate[t]))
        values["accuracy"].append(float(correct.sum() / selected_count) if selected_count else np.nan)
        values["correct_mass"].append(float(correct.double().mean()))
        values["incorrect_mass"].append(float((selected & ~correct).double().mean()))
    return {key: np.asarray(value) for key, value in values.items()}


def class_conditional_update_observables(
    run: ExperimentRun, *, source: str = "state_evolution"
) -> dict[str, dict[str, np.ndarray]]:
    """Selection coverage and pseudo-label precision, conditional on class.

    Precision is ``NaN`` when no particles/observations of the requested class
    are selected at an update; treating that event as zero precision would
    confound lack of coverage with incorrect pseudo-labels.
    """

    if source not in {"finite", "state_evolution"}:
        raise ValueError("source must be 'finite' or 'state_evolution'")
    classes = {"positive": 1, "negative": -1}
    values = {
        name: {"coverage": [], "precision": [], "selected_mass": []}
        for name in classes
    }

    if source == "finite":
        env = run.environment
        for step in run.finite.update_records_:
            selected = (env.Delta == 0) & (step.selection > 0)
            for name, label in classes.items():
                class_mask = (env.Y == label) & (env.Delta == 0)
                selected_class = selected & class_mask
                count = int(selected_class.sum().item())
                class_count = int(class_mask.sum().item())
                correct = selected_class & (step.pseudo_labels == env.Y)
                values[name]["coverage"].append(count / class_count if class_count else np.nan)
                values[name]["precision"].append(float(correct.sum() / count) if count else np.nan)
                values[name]["selected_mass"].append(float(selected_class.double().mean()))
    else:
        if run.se is None:
            raise ValueError("this run does not contain state evolution")
        se = run.se
        for t in range(run.algo_cfg.n_iterations):
            r = se.preactivation[t]
            if r is None:
                raise RuntimeError("state-evolution trajectory is incomplete")
            yhat = se.initial_pseudo_label if t == 0 else sign_with_positive_tie(r)
            selected = (se.indicator == 0) & (
                se.algo_cfg.selection_function(
                    r, se.algo_cfg.positive_margin, se.algo_cfg.negative_margin
                ) > 0
            )
            for name, label in classes.items():
                class_mask = (se.label == label) & (se.indicator == 0)
                selected_class = selected & class_mask
                count = int(selected_class.sum().item())
                class_count = int(class_mask.sum().item())
                correct = selected_class & (yhat == se.label)
                values[name]["coverage"].append(count / class_count if class_count else np.nan)
                values[name]["precision"].append(float(correct.sum() / count) if count else np.nan)
                values[name]["selected_mass"].append(float(selected_class.double().mean()))
    return {
        name: {key: np.asarray(history) for key, history in metrics.items()}
        for name, metrics in values.items()
    }


def trajectory_diagnostics(
    run: ExperimentRun,
    *,
    source: str = "state_evolution",
    convergence_window: int = 10,
    collapse_bias_threshold: float = 3.0,
    collapse_alignment_threshold: float = 0.1,
) -> dict[str, float | int | bool]:
    """Return reproducible terminal, best-time, stability, and collapse diagnostics.

    ``collapsed`` is a transparent numerical flag, not a theorem-defined phase:
    it requires large terminal ``|b/tau|`` and small terminal ``|m/tau|``.
    """

    if source == "finite":
        state = finite_state_observables(run)
    elif source == "state_evolution":
        state = state_evolution_state_observables(run)
    else:
        raise ValueError("source must be 'finite' or 'state_evolution'")
    error = state["error"]
    normalized_bias = _normalised(state, "bias")
    normalized_alignment = _normalised(state, "m")
    window = max(1, min(convergence_window, error.size))
    late_error = error[-window:]
    return {
        "terminal_error": float(error[-1]),
        "minimum_error": float(np.nanmin(error)),
        "best_iteration": int(np.nanargmin(error)),
        "late_error_range": float(np.nanmax(late_error) - np.nanmin(late_error)),
        "terminal_normalized_bias": float(normalized_bias[-1]),
        "terminal_normalized_alignment": float(normalized_alignment[-1]),
        "terminal_weight_scale": float(state["tau"][-1]),
        "collapsed": bool(
            abs(normalized_bias[-1]) >= collapse_bias_threshold
            and abs(normalized_alignment[-1]) <= collapse_alignment_threshold
        ),
    }


_MACROSCOPIC_QUANTITIES = {
    "error": ("state", "error", "Population classification error"),
    "normalized_bias": ("state", "normalized_bias", r"Normalised intercept $b^t/\tau^t$"),
    "normalized_alignment": ("state", "normalized_alignment", r"Normalised alignment $m^t/\tau^t$"),
    "chi": ("update", "chi", r"Residual-label alignment $\chi^t$"),
    "tau": ("state", "tau", r"Weight scale $\tau^t$"),
    "omega": ("update", "omega", r"Unlabelled selection rate $\omega^t$"),
    "accuracy": ("update", "accuracy", r"Selected pseudo-label accuracy $A_{\rm PL}^t$"),
}

DEFAULT_MACROSCOPIC_QUANTITIES = (
    "error",
    "normalized_bias",
    "normalized_alignment",
    "chi",
    "tau",
    "omega",
)


def _normalised(values: dict[str, np.ndarray], key: str) -> np.ndarray:
    """Return a scale-normalised observable, retaining undefined zero scales."""

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(
            values[key], values["tau"], out=np.full_like(values[key], np.nan),
            where=values["tau"] != 0,
        )


def _bayes_macroscopic_quantities(run: ExperimentRun) -> dict[str, float]:
    """Return the Bayes-optimal horizontal-reference values for one run."""

    b_star, m_star, tau_star = bayes_parameters(
        run.signal_scale, run.data_cfg.scale, run.data_cfg.label_prior
    )
    return {
        "error": population_error(
            b_star, m_star, tau_star, run.data_cfg.scale, run.data_cfg.label_prior
        ),
        "normalized_bias": b_star / tau_star,
        "normalized_alignment": m_star / tau_star,
    }


def macroscopic_discrepancy(
    finite_runs: Sequence[ExperimentRun],
    state_evolution_runs: Sequence[ExperimentRun],
    *,
    quantities: Sequence[str] = DEFAULT_MACROSCOPIC_QUANTITIES,
) -> dict[str, dict[str, np.ndarray | float]]:
    """Summarize finite/particle disagreement across independent repetitions.

    For each requested observable ``M``, the returned ``max_gap`` is

    ``max_t |mean_r M_finite,r(t) - mean_s M_SE,s(t)|``.

    The pointwise standard errors of the two independent empirical means are
    returned separately.  They quantify finite-environment and particle Monte
    Carlo variation, respectively, and should not be conflated with a
    systematic state-evolution discrepancy.
    """

    invalid_quantities = set(quantities).difference(_MACROSCOPIC_QUANTITIES)
    if invalid_quantities:
        raise ValueError(f"unknown macroscopic quantities: {sorted(invalid_quantities)}")
    if not finite_runs or not state_evolution_runs:
        raise ValueError("finite_runs and state_evolution_runs must both be non-empty")

    def histories(runs: Sequence[ExperimentRun], source: str, quantity: str) -> np.ndarray:
        time_kind, value_key, _ = _MACROSCOPIC_QUANTITIES[quantity]
        values = []
        for run in runs:
            if source == "finite":
                observable = finite_state_observables(run) if time_kind == "state" else finite_update_observables(run)
            else:
                if run.se is None:
                    raise ValueError("each state-evolution run must contain a particle trajectory")
                observable = state_evolution_state_observables(run) if time_kind == "state" else state_evolution_update_observables(run)
            if value_key.startswith("normalized_"):
                normalised_key = {"normalized_bias": "bias", "normalized_alignment": "m"}[value_key]
                values.append(_normalised(observable, normalised_key))
            else:
                values.append(observable[value_key])
        return np.stack(values)

    def mean_and_se(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = values.mean(axis=0)
        if values.shape[0] == 1:
            return mean, np.full_like(mean, np.nan)
        return mean, values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])

    summary = {}
    for quantity in quantities:
        finite_mean, finite_se = mean_and_se(histories(finite_runs, "finite", quantity))
        se_mean, se_se = mean_and_se(histories(state_evolution_runs, "state_evolution", quantity))
        if finite_mean.shape != se_mean.shape:
            raise ValueError(f"incompatible trajectory lengths for {quantity}")
        gap = np.abs(finite_mean - se_mean)
        summary[quantity] = {
            "finite_mean": finite_mean,
            "finite_standard_error": finite_se,
            "state_evolution_mean": se_mean,
            "state_evolution_standard_error": se_se,
            "gap": gap,
            "max_gap": float(np.nanmax(gap)),
        }
    return summary


def plot_macroscopic_evolution(
    runs: ExperimentRun | Mapping[str, ExperimentRun],
    *,
    quantities: Sequence[str] = DEFAULT_MACROSCOPIC_QUANTITIES,
    sources: Sequence[str] = ("finite", "state_evolution"),
    include_bayes: bool = True,
    title: Optional[str] = None,
    ncols: Optional[int] = None,
    nrows: Optional[int] = None,
    figsize: Optional[tuple[float, float]] = None,
    show: bool = True,
):
    """Plot selected macroscopic trajectories from one or more experiments.

    The default six panels reproduce notebook 01: population error,
    normalised intercept and alignment, residual-label alignment, weight
    scale, and unlabelled selection rate.  ``runs`` may be a single
    :class:`ExperimentRun` or a labelled mapping for trajectory comparisons.
    ``sources`` selects finite GD and/or state evolution; the latter is skipped
    only when a run has no particle trajectory.  Bayes references are available
    for error and the two normalised state coordinates.
    """

    invalid_quantities = set(quantities).difference(_MACROSCOPIC_QUANTITIES)
    invalid_sources = set(sources).difference({"finite", "state_evolution"})
    if invalid_quantities:
        raise ValueError(f"unknown macroscopic quantities: {sorted(invalid_quantities)}")
    if invalid_sources:
        raise ValueError(f"unknown plot sources: {sorted(invalid_sources)}")
    if not quantities:
        raise ValueError("quantities must be non-empty")
    if not sources:
        raise ValueError("sources must be non-empty")

    if isinstance(runs, ExperimentRun):
        labelled_runs = {runs.name: runs}
        single_run = True
    else:
        labelled_runs = dict(runs)
        single_run = False
    if not labelled_runs:
        raise ValueError("runs must contain at least one experiment")

    n_panels = len(quantities)
    if ncols is None:
        ncols = min(3, n_panels)
    if nrows is None:
        nrows= int(np.ceil(n_panels / ncols))
    if figsize is None:
        figsize = (5.25 * ncols, 4.25 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, squeeze=False)

    for quantity, ax in zip(quantities, axes.flat):
        time_kind, value_key, panel_title = _MACROSCOPIC_QUANTITIES[quantity]
        for run_label, run in labelled_runs.items():
            for source in sources:
                if source == "finite":
                    values = finite_state_observables(run) if time_kind == "state" else finite_update_observables(run)
                    source_label = "finite GD"
                else:
                    if run.se is None:
                        continue
                    values = state_evolution_state_observables(run) if time_kind == "state" else state_evolution_update_observables(run)
                    source_label = "state evolution"
                if value_key.startswith("normalized_"):
                    normalised_key = {
                        "normalized_bias": "bias",
                        "normalized_alignment": "m",
                    }[value_key]
                    y = _normalised(values, normalised_key)
                else:
                    y = values[value_key]
                curve_label = source_label if single_run else f"{run_label} ({source_label})"
                ax.plot(np.arange(y.size), y, label=curve_label)

        if include_bayes and quantity in {"error", "normalized_bias", "normalized_alignment"}:
            references = [_bayes_macroscopic_quantities(run)[quantity] for run in labelled_runs.values()]
            for index, reference in enumerate(dict.fromkeys(references)):
                label = "Bayes" if single_run or len(references) == 1 else f"Bayes {index + 1}"
                ax.axhline(reference, color="red", linestyle="--", label=label)

        ax.set(title=panel_title, xlabel="iteration")
        ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.8)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.legend()

    for ax in axes.flat[n_panels:]:
        ax.remove()

    if title is None and single_run:
        run = next(iter(labelled_runs.values()))
        qualifier = (
            "theorem-external no-bias variant"
            if run.metadata.get("theorem_external", False)
            else "canonical fixed-pi model"
        )
        title = f"{run.name} ({qualifier})"
    if title:
        fig.suptitle(title, fontsize=15)
    if show:
        plt.show()
    return fig, axes


def mean_and_std(histories: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise mean and standard deviation for equal-length trajectories."""

    values = np.stack(histories)
    return values.mean(axis=0), values.std(axis=0, ddof=0)


def run_oracle_selected_label_counterfactual(run: ExperimentRun) -> tuple[SelfTrainedGradientDescent, TestEvaluatorCallback]:
    """Run a finite theorem-external oracle-label diagnostic.

    The finite GD implementation itself is unchanged.  During this isolated
    diagnostic, the pseudo-label primitive is replaced by the true full label
    vector, so selected unlabeled gradients use oracle labels at every update.
    It identifies the attainable benefit of selected additional samples and is
    not a realizable self-training algorithm or a theorem claim.
    """

    env, X = run.environment, run.X
    callback = TestEvaluatorCallback(
        X_lab=X[env.I_L],
        Y_lab=env.Y[env.I_L],
        X_unl=X[env.I_U],
        Y_unl=env.Y[env.I_U],
        X_test=run.X_test,
        Y_test=run.Y_test,
        mu=env.mu,
        sigma=run.data_cfg.scale,
        p=run.data_cfg.label_prior,
        metrics=DEFAULT_METRICS,
    )
    learner = SelfTrainedGradientDescent(cfg=run.algo_cfg, callback=callback)
    assert run.finite.initialization_ is not None

    def oracle_labels(_t: int, scores: torch.Tensor, _initial: torch.Tensor) -> torch.Tensor:
        return env.Y.to(dtype=scores.dtype, device=scores.device)

    with patch("src.algorithms.pseudo_labels", oracle_labels):
        learner.fit_full(X, env, run.finite.initialization_)
    return learner, callback

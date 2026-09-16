"""Shared, current-API utilities for the numerical experiment notebooks.

The helpers deliberately construct the finite and state-evolution experiments
from the same population laws but independent random draws.  They are for
notebook orchestration only: finite learning remains implemented exclusively
by :class:`src.algorithms.SelfTrainedGradientDescent`, and effective dynamics
by :class:`src.asymptotics.MacroscopicStateEvolution`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from unittest.mock import patch

from matplotlib import pyplot as plt
import numpy as np
import torch

from src.algorithms import SelfTrainedGradientDescent
import src.asymptotics as asymptotics_module
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
from src.primitives import pseudo_residual


DEFAULT_METRICS = {
    "population_error",
    "unl_usage",
    "weight_signal_alignment",
    "weight_vector_norm",
    "bias_term",
}


@dataclass
class ExperimentRun:
    """One finite/SE comparison, including best population-error summaries."""

    name: str
    data_cfg: DataConfig
    algo_cfg: AlgorithmConfig
    environment: Optional[QuenchedEnvironment]
    X: Optional[torch.Tensor]
    X_test: Optional[torch.Tensor]
    Y_test: Optional[torch.Tensor]
    finite: Optional[SelfTrainedGradientDescent]
    callback: Optional[TestEvaluatorCallback]
    se: Optional[MacroscopicStateEvolution]
    signal_scale: float
    finite_minimum_error: Optional[float]
    finite_minimum_error_iteration: Optional[int]
    state_evolution_minimum_error: Optional[float]
    state_evolution_minimum_error_iteration: Optional[int]
    metadata: dict[str, Any]
    error_diagnostics: dict[str, dict[str, np.ndarray]] = field(
        init=False, default_factory=dict
    )


def as_numpy(values: Iterable[Any]) -> np.ndarray:
    """Convert scalar tensor histories to a float NumPy array."""

    return np.asarray([float(torch.as_tensor(value)) for value in values], dtype=float)


def _normalized_population_error(
    normalized_bias: np.ndarray,
    normalized_alignment: np.ndarray,
    *,
    p: float,
    sigma: float,
) -> np.ndarray:
    """Evaluate population error at normalized state coordinates."""

    return np.asarray(
        [
            population_error(beta, align, 1.0, sigma, p)
            for beta, align in zip(normalized_bias, normalized_alignment)
        ],
        dtype=float,
    )


def compute_temporal_error_attribution(
    normalized_bias: Iterable[Any],
    normalized_alignment: Iterable[Any],
    *,
    p: float,
    sigma: float,
    error_evaluator: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> dict[str, np.ndarray]:
    """Return ordered and symmetric one-step error attributions.

    All arrays are indexed by transitions ``t -> t + 1``. By default, the
    helper reuses the normalized population-error evaluator; ``error_evaluator``
    is a vectorized test seam and does not affect production trajectories.

    The legacy ``alignment_contribution`` and ``bias_contribution`` keys
    remain aliases for the symmetric contributions.
    """

    normalized_bias = np.asarray(normalized_bias, dtype=float)
    normalized_alignment = np.asarray(normalized_alignment, dtype=float)
    if normalized_bias.ndim != 1 or normalized_alignment.ndim != 1:
        raise ValueError("normalized_bias and normalized_alignment must be one-dimensional")
    if normalized_bias.size != normalized_alignment.size:
        raise ValueError("normalized_bias and normalized_alignment must have equal lengths")

    def evaluate(bias: np.ndarray, alignment: np.ndarray) -> np.ndarray:
        values = (
            _normalized_population_error(bias, alignment, p=p, sigma=sigma)
            if error_evaluator is None
            else np.asarray(error_evaluator(bias, alignment), dtype=float)
        )
        if values.shape != bias.shape:
            raise ValueError("error_evaluator must return an array matching its inputs")
        return values

    error_old_old = evaluate(normalized_bias[:-1], normalized_alignment[:-1])
    error_old_new = evaluate(normalized_bias[:-1], normalized_alignment[1:])
    error_new_old = evaluate(normalized_bias[1:], normalized_alignment[:-1])
    error_new_new = evaluate(normalized_bias[1:], normalized_alignment[1:])
    alignment_contribution_old_bias = error_old_new - error_old_old
    alignment_contribution_new_bias = error_new_new - error_new_old
    bias_contribution_old_alignment = error_new_old - error_old_old
    bias_contribution_new_alignment = error_new_new - error_old_new
    interaction_contribution = (
        error_new_new - error_new_old - error_old_new + error_old_old
    )
    alignment_contribution_symmetric = 0.5 * (
        alignment_contribution_old_bias + alignment_contribution_new_bias
    )
    bias_contribution_symmetric = 0.5 * (
        bias_contribution_old_alignment + bias_contribution_new_alignment
    )
    error_increment = error_new_new - error_old_old

    def check_identity(name: str, lhs: np.ndarray, rhs: np.ndarray) -> None:
        finite = np.isfinite(lhs) & np.isfinite(rhs)
        if np.allclose(lhs[finite], rhs[finite], rtol=1e-12, atol=1e-14):
            return
        residual = lhs - rhs
        max_residual = float(np.max(np.abs(residual[finite])))
        raise RuntimeError(
            f"temporal error attribution identity failed for {name}: "
            f"max residual = {max_residual:.3e}"
        )

    check_identity(
        "alignment new-old interaction",
        alignment_contribution_new_bias - alignment_contribution_old_bias,
        interaction_contribution,
    )
    check_identity(
        "bias new-old interaction",
        bias_contribution_new_alignment - bias_contribution_old_alignment,
        interaction_contribution,
    )
    check_identity(
        "alignment-old plus bias-new",
        alignment_contribution_old_bias + bias_contribution_new_alignment,
        error_increment,
    )
    check_identity(
        "bias-old plus alignment-new",
        bias_contribution_old_alignment + alignment_contribution_new_bias,
        error_increment,
    )
    check_identity(
        "symmetric decomposition",
        alignment_contribution_symmetric + bias_contribution_symmetric,
        error_increment,
    )
    return {
        "error_old_old": error_old_old,
        "error_old_new": error_old_new,
        "error_new_old": error_new_old,
        "error_new_new": error_new_new,
        "error_increment": error_increment,
        "alignment_contribution_old_bias": alignment_contribution_old_bias,
        "alignment_contribution_new_bias": alignment_contribution_new_bias,
        "bias_contribution_old_alignment": bias_contribution_old_alignment,
        "bias_contribution_new_alignment": bias_contribution_new_alignment,
        "interaction_contribution": interaction_contribution,
        "alignment_contribution_symmetric": alignment_contribution_symmetric,
        "bias_contribution_symmetric": bias_contribution_symmetric,
        "alignment_contribution": alignment_contribution_symmetric,
        "bias_contribution": bias_contribution_symmetric,
    }


def compute_error_diagnostics(
    trajectory: Mapping[str, Iterable[Any]],
    *,
    p: float,
    sigma: float,
    s_mu: float,
    zero_tolerance: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Decompose completed-trajectory error using recorded observables only.

    Raw ``m``, ``bias``, and ``tau`` histories take precedence.  A trajectory
    without raw coordinates may instead provide ``normalized_alignment`` and
    ``normalized_bias``.

    For nonpositive alignment, the bias optimum is taken over extended real
    thresholds: constant majority prediction attains the infimum. At negative
    alignment with balanced classes, choose -inf among the two tied optima.
    ``zero_tolerance`` only restricts finite bias-tracking diagnostics near zero;
    positive-alignment error/regret calculations retain the finite optimum.
    """

    p = float(p)
    sigma = float(sigma)
    s_mu = float(s_mu)
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between zero and one")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not np.isfinite(s_mu) or s_mu < 0.0:
        raise ValueError("s_mu must be finite and nonnegative")

    def values(key: str) -> np.ndarray:
        result = np.asarray(trajectory[key], dtype=float)
        if result.ndim != 1:
            raise ValueError(f"trajectory['{key}'] must be one-dimensional")
        return result

    has_raw_coordinates = all(key in trajectory for key in ("m", "bias", "tau"))
    if has_raw_coordinates:
        alignment = values("m")
        bias = values("bias")
        weight_norm = values("tau")
        if not (alignment.size == bias.size == weight_norm.size):
            raise ValueError("m, bias, and tau must have equal lengths")
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized_alignment = np.divide(
                alignment,
                weight_norm,
                out=np.full_like(alignment, np.nan),
                where=weight_norm > 0.0,
            )
            normalized_bias = np.divide(
                bias,
                weight_norm,
                out=np.full_like(bias, np.nan),
                where=weight_norm > 0.0,
            )
    elif all(
        key in trajectory for key in ("normalized_alignment", "normalized_bias")
    ):
        normalized_alignment = values("normalized_alignment")
        normalized_bias = values("normalized_bias")
        if normalized_alignment.size != normalized_bias.size:
            raise ValueError(
                "normalized_alignment and normalized_bias must have equal lengths"
            )
    else:
        raise KeyError(
            "trajectory must contain either m, bias, tau or both normalized coordinates"
        )

    length = normalized_alignment.size
    if length == 0:
        raise ValueError("trajectory histories must be non-empty")

    population_error_from_coordinates = _normalized_population_error(
        normalized_bias, normalized_alignment, p=p, sigma=sigma
    )
    if zero_tolerance is None:
        zero_tolerance = float(np.sqrt(np.finfo(float).eps))
    else:
        zero_tolerance = float(zero_tolerance)
        if not np.isfinite(zero_tolerance) or zero_tolerance < 0.0:
            raise ValueError("zero_tolerance must be finite and nonnegative")

    def conditionally_optimal_bias(
        normalized_alignment_values: np.ndarray,
    ) -> np.ndarray:
        result = np.full(normalized_alignment_values.shape, np.nan, dtype=float)
        finite = np.isfinite(normalized_alignment_values)
        positive = finite & (normalized_alignment_values > 0.0)
        result[finite & ~positive] = -np.inf if p <= 0.5 else np.inf
        if p == 0.5:
            # At exactly zero alignment every finite bias has error 1/2.
            result[finite & (normalized_alignment_values >= 0.0)] = 0.0
        else:
            with np.errstate(over="ignore", divide="ignore"):
                result[positive] = (
                    (sigma**2 * np.log(p / (1.0 - p)) / 2.0)
                    / normalized_alignment_values[positive]
                )
        return result

    optimal_normalized_bias = conditionally_optimal_bias(normalized_alignment)
    # Infinite optimal thresholds define valid risks, but not finite distances
    # or increments. Keep those tracking quantities explicitly undefined.
    tracking_bias = np.where(
        np.isfinite(optimal_normalized_bias)
        & ((normalized_alignment > zero_tolerance)
           | ((p == 0.5) & (normalized_alignment >= 0.0))),
        optimal_normalized_bias,
        np.nan,
    )
    signed_bias_tracking_error = normalized_bias - tracking_bias
    normalized_bias_increment = np.diff(normalized_bias)
    optimal_normalized_bias_increment = np.diff(tracking_bias)
    signed_bias_tracking_error_increment = np.diff(signed_bias_tracking_error)
    tracking_increment_residual = signed_bias_tracking_error_increment - (
        normalized_bias_increment - optimal_normalized_bias_increment
    )
    finite_tracking = np.isfinite(tracking_increment_residual)
    # The two subtraction orders can differ by roundoff, especially when the
    # optimal bias is large near zero alignment. Scale by the endpoint values,
    # not the residual (whose reference value is zero) or cancelling increments.
    tracking_scale = np.maximum.reduce(
        [
            np.abs(normalized_bias[:-1]),
            np.abs(normalized_bias[1:]),
            np.abs(tracking_bias[:-1]),
            np.abs(tracking_bias[1:]),
        ]
    )
    #tracking_tolerance = 1e-14 + (16 * np.finfo(float).eps) * tracking_scale
    #if np.any(
    #    np.abs(tracking_increment_residual[finite_tracking])
    #    > tracking_tolerance[finite_tracking]
    #):
    #    raise RuntimeError(
    #        "signed bias-tracking increment identity failed: "
    #        f"max residual = {np.max(np.abs(tracking_increment_residual[finite_tracking])):.3e}"
    #    )
    conditionally_optimal_error = _normalized_population_error(
        optimal_normalized_bias, normalized_alignment, p=p, sigma=sigma
    )
    optimal_linear_bias = conditionally_optimal_bias(np.asarray([s_mu]))
    optimal_linear_error_value = _normalized_population_error(
        optimal_linear_bias, np.asarray([s_mu]), p=p, sigma=sigma
    )[0]
    optimal_linear_error = np.full(length, optimal_linear_error_value, dtype=float)
    alignment_regret = conditionally_optimal_error - optimal_linear_error
    bias_regret = population_error_from_coordinates - conditionally_optimal_error
    error_reconstructed = optimal_linear_error + alignment_regret + bias_regret

    recorded_error = None
    for error_key in ("error", "population_error"):
        if error_key in trajectory:
            recorded_error = values(error_key)
            if recorded_error.size != length:
                raise ValueError(
                    f"trajectory['{error_key}'] must match the state-history length"
                )
            break
    temporal_attribution = compute_temporal_error_attribution(
        normalized_bias, normalized_alignment, p=p, sigma=sigma
    )
    alignment_regret_increment = np.diff(alignment_regret)
    bias_regret_increment = np.diff(bias_regret)

    return {
        "normalized_alignment": normalized_alignment,
        "normalized_bias": normalized_bias,
        "optimal_normalized_bias": optimal_normalized_bias,
        "signed_bias_tracking_error": signed_bias_tracking_error,
        "normalized_bias_increment": normalized_bias_increment,
        "optimal_normalized_bias_increment": optimal_normalized_bias_increment,
        "signed_bias_tracking_error_increment": signed_bias_tracking_error_increment,
        "optimal_linear_error": optimal_linear_error,
        "alignment_regret": alignment_regret,
        "bias_regret": bias_regret,
        "error_reconstructed": error_reconstructed,
        **temporal_attribution,
        "alignment_regret_increment": alignment_regret_increment,
        "bias_regret_increment": bias_regret_increment,
    }


def plot_error_diagnostics(
    diagnostics: Mapping[str, Iterable[Any]],
    *,
    title: Optional[str] = None,
    show: bool = True,
    ylim: Optional[tuple[float, float]] = None
):
    """Plot population-error levels, regrets, and temporal attribution."""

    def as_1d(key: str) -> np.ndarray:
        values = np.asarray(diagnostics[key], dtype=float)
        if values.ndim != 1:
            raise ValueError(f"diagnostics['{key}'] must be one-dimensional")
        return values

    optimal_error = as_1d("optimal_linear_error")
    alignment_regret = as_1d("alignment_regret")
    bias_regret = as_1d("bias_regret")
    reconstructed_error = as_1d("error_reconstructed")
    alignment_contribution = as_1d("alignment_contribution")
    bias_contribution = as_1d("bias_contribution")
    error_increment = as_1d("error_increment")

    n_steps = reconstructed_error.size
    for name, values in {
        "optimal_linear_error": optimal_error,
        "alignment_regret": alignment_regret,
        "bias_regret": bias_regret,
    }.items():
        if values.size != n_steps:
            raise ValueError(
                f"diagnostics['{name}'] must have length {n_steps}, "
                f"got {values.size}"
            )
    n_transitions = max(n_steps - 1, 0)
    for name, values in {
        "alignment_contribution": alignment_contribution,
        "bias_contribution": bias_contribution,
        "error_increment": error_increment,
    }.items():
        if values.size != n_transitions:
            raise ValueError(
                f"diagnostics['{name}'] must have length {n_transitions}, "
                f"got {values.size}"
            )

    fig, axes = plt.subplots(
        1, 4, figsize=(18.0, 4.0), constrained_layout=True, squeeze=False
    )

    # Panel 1: exact additive decomposition of the population error.
    ax = axes.flat[0]
    time = np.arange(n_steps)
    ax.stackplot(
        time,
        optimal_error,
        alignment_regret,
        bias_regret,
        labels=(
            "optimal linear error",
            "alignment regret",
            "bias regret",
        ),
        colors=("#9ecae1", "#74c476", "#fd8d3c"),
        alpha=0.75,
    )
    ax.plot(
        time,
        reconstructed_error,
        color="black",
        linewidth=1.0,
        label="total population error",
        zorder=3,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set(
        title="Population-error decomposition",
        xlabel="iteration",
        ylabel="classification error",
    )
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.35)
    ax.legend(loc="best")
    

    # Panel 2: regret levels.
    ax = axes.flat[1]
    for key in ("alignment_regret", "bias_regret"):
        values = as_1d(key)
        ax.plot(np.arange(values.size), values, label=key)
    ax.set(title="Regret levels", xlabel="iteration")
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend()

    # Panel 3: one-step changes in the two regrets.
    ax = axes.flat[2]
    for key in ("alignment_regret_increment", "bias_regret_increment"):
        values = as_1d(key)
        ax.plot(np.arange(values.size), values, label=key)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.set(title="One-step regret changes", xlabel="iteration")
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.6)
    #ax.set_yscale('log')
    ax.legend()

    # Panel 4: exact symmetric attribution of each population-error change.
    ax = axes.flat[3]
    transition_index = np.arange(n_transitions)
    ax.plot(transition_index, alignment_contribution, label=r"alignment $A_t$")
    ax.plot(transition_index, bias_contribution, label=r"bias $B_t$")
    ax.plot(
        transition_index,
        error_increment,
        color="black",
        linewidth=1.2,
        label=r"total $A_t+B_t=\Delta\mathscr{E}_t$",
        zorder=3,
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    finite_error = np.isfinite(reconstructed_error)
    if n_transitions and np.any(finite_error):
        error_minimizing_iteration = int(np.nanargmin(reconstructed_error))
        marked_transition = min(error_minimizing_iteration, n_transitions - 1)
        transition_label = (
            rf"$t_{{\rm err}}={error_minimizing_iteration}$ "
            rf"(transition ${marked_transition}\to{marked_transition + 1}$)"
        )
        ax.axvline(
            marked_transition,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=transition_label,
        )
    ax.set(
        title="Symmetric temporal error attribution",
        xlabel=r"transition index $t$ ($t\to t+1$)",
        ylabel=r"population-error change",
    )
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend()

    if title:
        fig.suptitle(title, fontsize=15)
    if show:
        plt.show()

    return fig, axes


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

def get_s_mu(
    signal_mean: float,
    signal_std: float
) -> float:
    return (signal_std**2 + signal_mean**2)**0.5

def mahalanobis_separation(
        s_mu: float,
        sigma: float,
) -> float:
    return 2 * s_mu / sigma


def exogenous_initial_labels(
    environment: QuenchedEnvironment, generator: torch.Generator
) -> torch.Tensor:
    """Return full ``Y_init`` with labelled coordinates fixed to ``Y``."""

    y_init = rademacher(environment.n, generator, device=environment.mu.device)
    y_init[environment.I_L] = environment.Y[environment.I_L]
    return y_init


def make_parameter_base_sampler(signal_std: float, initialization_correlation: float = 0.0, signal_mean: float = 0.0):
    """Return the joint ``(mu_tilde,w_init_tilde)`` particle law.

    ``initialization_correlation`` is the coefficient in
    ``w_init=c mu/signal_std+sqrt(1-c^2) epsilon``.  Thus, asymptotically,
    ``E[mu*w_init]=c*signal_std`` and the initial weight variance remains one.
    """

    if not -1.0 <= initialization_correlation <= 1.0:
        raise ValueError("initialization_correlation must lie in [-1, 1]")

    def sampler(K: int, generator: torch.Generator, dtype: torch.dtype, device: torch.device):
        mu = signal_std * torch.randn(K, generator=generator, dtype=dtype, device=device) + signal_mean
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
    kappa: float = None,
    kappa_pos: float = None,
    kappa_neg: float = None,
    include_bias: bool = True,
    initial_bias: float = 0.0,
    bias_pseudo_label_param: Optional[float] = None,
    experimental_schedule = None,
    normalized_threshold: bool = False,
    normalize_unlabeled_loss: bool = True
) -> AlgorithmConfig:
    """Create a fixed-pi config unless an explicit experimental schedule is supplied.

    ``initial_bias`` is used whether the bias is trainable or fixed.  When
    ``bias_pseudo_label_param`` is omitted, the bias follows the ordinary
    pseudo-label weight (including an experimental schedule).
    """

    return AlgorithmConfig(
        n_iterations=T,
        step_size=eta,
        penalty_param=penalty,
        pseudo_label_param=pi,
        margin_threshold=kappa,
        positive_margin=kappa_pos,
        negative_margin=kappa_neg,
        include_bias=include_bias,
        initial_bias=initial_bias,
        bias_pseudo_label_param=bias_pseudo_label_param,
        experimental_schedule=experimental_schedule,
        normalized_threshold=normalized_threshold,
        normalize_unlabeled_loss=normalize_unlabeled_loss
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
    signal_mean: float = 0.0,
    K_w: Optional[int] = None,
    K_g: Optional[int] = None,
    run_finite: bool = True,
    run_state_evolution: bool = True,
    aspect_ratio_tolerance: float = 1e-12,
    initialization_correlation: float = 0.0,
    metadata: Optional[dict[str, Any]] = None,
) -> ExperimentRun:
    """Run finite GD and/or its independent particle approximation.

    The finite initial labels are exogenous and the particle sampler uses the
    corresponding conditional law.  This is the canonical fixed-pi setup when
    ``algo_cfg.is_canonical_fixed_pi`` is true.

    Set ``run_finite=False`` for state-evolution-only sweeps.  In that mode no
    finite design matrix or test set is allocated, and the finite fields of the
    returned :class:`ExperimentRun` are ``None``.
    """

    if d <= 0 or delta <= 0:
        raise ValueError("d and delta must be positive")
    if not run_finite and not run_state_evolution:
        raise ValueError("at least one of run_finite or run_state_evolution must be true")
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
    environment = X = X_test = Y_test = callback = finite = None
    mu = None
    if run_finite:
        finite_generator = torch.Generator().manual_seed(seed + 11)
        mu = signal_std * torch.randn(d, generator=finite_generator) + signal_mean
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
            b_init=algo_cfg.initial_bias,
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
                signal_std, initialization_correlation, signal_mean
            ),
            sample_base_sampler=state_evolution_sample_base_sampler(law),
        )
        se.compute_trajectory()

    if callback is None:
        finite_minimum_error = None
        finite_minimum_error_iteration = None
    else:
        finite_error = np.asarray(callback.history_["population_error"], dtype=float)
        finite_minimum_error_iteration = int(np.nanargmin(finite_error))
        finite_minimum_error = float(finite_error[finite_minimum_error_iteration])

    if se is None:
        state_evolution_minimum_error = None
        state_evolution_minimum_error_iteration = None
    else:
        state_evolution_error = np.asarray(se.error, dtype=float)
        state_evolution_minimum_error_iteration = int(np.nanargmin(state_evolution_error))
        state_evolution_minimum_error = float(
            state_evolution_error[state_evolution_minimum_error_iteration]
        )

    run_metadata = {
        "fixed_pi": algo_cfg.is_canonical_fixed_pi,
        "initialization": "exogenous independent Rademacher on unlabeled coordinates",
        "finite_environment": (
            "one iid draw of the product/MCAR special case"
            if run_finite
            else "not generated (state-evolution-only run)"
        ),
        "state_evolution": "independent particle approximation",
        "initialization_correlation": initialization_correlation,
    }
    if metadata:
        run_metadata.update(metadata)
    run = ExperimentRun(
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
        signal_scale=float(
            torch.linalg.vector_norm(mu) / np.sqrt(d)
            if mu is not None
            else torch.linalg.vector_norm(se.signal) / np.sqrt(se.K_w)
        ),
        finite_minimum_error=finite_minimum_error,
        finite_minimum_error_iteration=finite_minimum_error_iteration,
        state_evolution_minimum_error=state_evolution_minimum_error,
        state_evolution_minimum_error_iteration=state_evolution_minimum_error_iteration,
        metadata=run_metadata,
    )
    _attach_error_diagnostics(run)
    return run


def _finite_state_observables_from_callback(
    callback: TestEvaluatorCallback,
) -> dict[str, np.ndarray]:
    return {
        "error": np.asarray(callback.history_["population_error"]),
        "oracle_error": np.asarray(callback.history_["oracle_error"]),
        "m": np.asarray(callback.history_["weight_signal_alignment"]),
        "tau": np.asarray(callback.history_["weight_vector_norm"]),
        "energy": np.asarray(callback.history_["weight_vector_norm"]) ** 2,
        "bias": np.asarray(callback.history_["bias_term"]),
        "oracle_bias": np.asarray(callback.history_["oracle_bias"]),
    }


def finite_state_observables(run: ExperimentRun) -> dict[str, np.ndarray]:
    """State-indexed finite macroscopic trajectories (length T+1)."""

    callback = run.callback
    if callback is None:
        raise ValueError("this run does not contain a finite-gradient trajectory")
    return _finite_state_observables_from_callback(callback)


def finite_update_observables(run: ExperimentRun) -> dict[str, np.ndarray]:
    """Update-indexed finite statistics, including selected-label precision."""

    env, finite = run.environment, run.finite
    if env is None or finite is None:
        raise ValueError("this run does not contain a finite-gradient trajectory")
    values = {key: [] for key in ("chi", "zeta", "omega", "accuracy", "correct_mass", "incorrect_mass")}
    for step in finite.update_records_:
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


def _state_evolution_state_observables_from(
    se: MacroscopicStateEvolution,
) -> dict[str, np.ndarray]:
    tau = as_numpy(se.weight_norm)
    return {
        "error": as_numpy(se.error),
        "oracle_error": as_numpy(se.oracle_error),
        "m": as_numpy(se.weight_signal_alignments),
        "tau": tau,
        "energy": tau**2,
        "bias": as_numpy(se.bias),
        "oracle_bias": as_numpy(se.oracle_bias),
    }


def state_evolution_state_observables(run: ExperimentRun) -> dict[str, np.ndarray]:
    """State-indexed particle trajectories."""

    if run.se is None:
        raise ValueError("this run does not contain state evolution")
    return _state_evolution_state_observables_from(run.se)


def _attach_error_diagnostics(run: ExperimentRun) -> None:
    """Attach post-processing diagnostics after all requested trajectories exist."""

    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    p = run.data_cfg.label_prior
    sigma = run.data_cfg.scale
    if run.callback is not None:
        finite_diagnostics = compute_error_diagnostics(
            _finite_state_observables_from_callback(run.callback),
            p=p,
            sigma=sigma,
            s_mu=run.signal_scale,
        )
        run.callback.error_diagnostics = finite_diagnostics
        diagnostics["finite"] = finite_diagnostics
    if run.se is not None:
        state_evolution_diagnostics = compute_error_diagnostics(
            _state_evolution_state_observables_from(run.se),
            p=p,
            sigma=sigma,
            s_mu=run.signal_scale,
        )
        run.se.error_diagnostics = state_evolution_diagnostics
        diagnostics["state_evolution"] = state_evolution_diagnostics
    run.error_diagnostics = diagnostics


def plot_state_evolution_oracle_bias_error(
    run: ExperimentRun,
    *,
    ax=None,
    show: bool = True,
):
    """Compare actual-bias and oracle-calibrated finite/SE errors.

    This is a read-only diagnostic plot.  Within each source, both curves use
    the same weight trajectory; only the intercept used in the oracle error
    evaluation is replaced by its iteration-specific calibrated value.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    else:
        fig = ax.figure
    multiple_sources = run.callback is not None and run.se is not None
    if run.callback is not None:
        finite = finite_state_observables(run)
        iterations = np.arange(finite["error"].size)
        ax.plot(
            iterations,
            finite["error"],
            label=(
                r"finite GD: $\mathcal{E}(w^t,b^t)$"
                if multiple_sources
                else r"$\mathcal{E}(w^t,b^t)$"
            ),
        )
        ax.plot(
            iterations,
            finite["oracle_error"],
            label=(
                r"finite GD: $\mathcal{E}(w^t,b_{\rm oracle}^t)$"
                if multiple_sources
                else r"$\mathcal{E}(w^t,b_{\rm oracle}^t)$"
            ),
        )
    if run.se is not None:
        state_evolution = state_evolution_state_observables(run)
        iterations = np.arange(state_evolution["error"].size)
        ax.plot(
            iterations,
            state_evolution["error"],
            label=(
                r"state evolution: $\mathcal{E}(w^t,b^t)$"
                if multiple_sources
                else r"$\mathcal{E}(w^t,b^t)$"
            ),
        )
        ax.plot(
            iterations,
            state_evolution["oracle_error"],
            label=(
                r"state evolution: $\mathcal{E}(w^t,b_{\rm oracle}^t)$"
                if multiple_sources
                else r"$\mathcal{E}(w^t,b_{\rm oracle}^t)$"
            ),
        )
    if run.callback is None and run.se is None:
        raise ValueError("this run contains neither finite GD nor state evolution")
    ax.set(
        xlabel="iteration",
        ylabel="population classification error",
        title=f"{run.name}: oracle-calibrated bias diagnostic",
    )
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend()
    if show:
        plt.show()
    return fig, ax


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
        mask = se.selection_mask(r, t) > 0
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
        env, finite = run.environment, run.finite
        if env is None or finite is None:
            raise ValueError("this run does not contain a finite-gradient trajectory")
        for step in finite.update_records_:
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
                se.selection_mask(r, t) > 0
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


def compute_group_residual_diagnostics(
    labels: Iterable[Any],
    indicators: Iterable[Any],
    residuals: Iterable[Any],
) -> dict[str, float]:
    """Decompose one empirical residual vector by class and supervision cell."""

    labels = torch.as_tensor(labels)
    indicators = torch.as_tensor(indicators, device=labels.device)
    residuals = torch.as_tensor(residuals, device=labels.device)
    if labels.ndim != 1 or indicators.ndim != 1 or residuals.ndim != 1:
        raise ValueError("labels, indicators, and residuals must be one-dimensional")
    if not (labels.shape == indicators.shape == residuals.shape):
        raise ValueError("labels, indicators, and residuals must have equal shapes")
    if labels.numel() == 0:
        raise ValueError("group residual diagnostics require a non-empty sample")

    groups = {
        "plus_labeled": (1, 1),
        "minus_labeled": (-1, 1),
        "plus_unlabeled": (1, 0),
        "minus_unlabeled": (-1, 0),
    }
    result: dict[str, float] = {}
    for name, (label, indicator) in groups.items():
        group = (labels == label) & (indicators == indicator)
        count = int(group.sum().item())
        result[f"zeta_{name}"] = float(residuals[group].sum() / labels.numel())
        result[f"u_{name}"] = (
            float(residuals[group].mean()) if count else np.nan
        )

    result["zeta_reconstructed"] = sum(
        result[f"zeta_{name}"] for name in groups
    )
    result["chi_reconstructed"] = (
        result["zeta_plus_labeled"]
        + result["zeta_plus_unlabeled"]
        - result["zeta_minus_labeled"]
        - result["zeta_minus_unlabeled"]
    )
    result["zeta"] = float(residuals.mean())
    result["chi"] = float((labels * residuals).mean())
    result["zeta_reconstruction_error"] = (
        result["zeta_reconstructed"] - result["zeta"]
    )
    result["chi_reconstruction_error"] = (
        result["chi_reconstructed"] - result["chi"]
    )
    if not np.allclose(
        result["zeta_reconstructed"], result["zeta"], rtol=1e-12, atol=1e-14
    ):
        raise RuntimeError("class-conditional contributions do not reconstruct zeta")
    if not np.allclose(
        result["chi_reconstructed"], result["chi"], rtol=1e-12, atol=1e-14
    ):
        raise RuntimeError("class-conditional contributions do not reconstruct chi")
    return result


def compute_mechanism_diagnostics(
    run: ExperimentRun,
    *,
    source: str = "state_evolution",
    geometry_tolerance: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Return Task-A diagnostics from an already-computed trajectory.

    State quantities have length T+1 and update quantities have length T.
    True unlabeled labels are used only in this read-only post-processing.
    """

    if source not in {"finite", "state_evolution"}:
        raise ValueError("source must be 'finite' or 'state_evolution'")
    if not np.isfinite(geometry_tolerance) or geometry_tolerance < 0.0:
        raise ValueError("geometry_tolerance must be finite and nonnegative")

    error_diagnostics = run.error_diagnostics.get(source)
    if error_diagnostics is None:
        raise ValueError(f"this run does not contain {source} diagnostics")
    state = (
        finite_state_observables(run)
        if source == "finite"
        else state_evolution_state_observables(run)
    )
    signal_alignment_raw = np.asarray(state["m"], dtype=float)
    noise_scale = np.asarray(state["tau"], dtype=float)
    signal_scale = float(run.signal_scale)
    if signal_scale > 0.0:
        orthogonal_weight_energy = (
            noise_scale**2 - signal_alignment_raw**2 / signal_scale**2
        )
    else:
        orthogonal_weight_energy = np.full_like(noise_scale, np.nan)
    orthogonal_to_signal_ratio = np.full_like(orthogonal_weight_energy, np.nan)
    valid_alignment = (
        np.isfinite(orthogonal_weight_energy)
        & np.isfinite(signal_alignment_raw)
        & (signal_alignment_raw != 0.0)
    )
    orthogonal_to_signal_ratio[valid_alignment] = (
        orthogonal_weight_energy[valid_alignment]
        / signal_alignment_raw[valid_alignment] ** 2
    )
    geometry_reconstructed = np.full_like(noise_scale, np.nan)
    if signal_scale > 0.0:
        denominator = signal_scale**-2 + orthogonal_to_signal_ratio
        valid_denominator = np.isfinite(denominator) & (denominator > 0.0)
        geometry_reconstructed[valid_denominator] = 1.0 / denominator[
            valid_denominator
        ]
    geometry_identity_error = (
        np.asarray(error_diagnostics["normalized_alignment"], dtype=float) ** 2
        - geometry_reconstructed
    )
    finite_geometry = np.isfinite(geometry_identity_error)
    if not np.allclose(
        geometry_identity_error[finite_geometry], 0.0, rtol=1e-10, atol=1e-12
    ):
        raise RuntimeError(
            "weight-geometry identity failed: "
            f"max residual = {np.max(np.abs(geometry_identity_error[finite_geometry])):.3e}"
        )

    update_values: dict[str, list[float | int]] = {}

    def append(name: str, value: float | int) -> None:
        update_values.setdefault(name, []).append(value)

    def process_update(
        labels: torch.Tensor,
        indicators: torch.Tensor,
        residuals: torch.Tensor,
        scores: torch.Tensor,
        selection: torch.Tensor,
        pseudo_label_values: torch.Tensor,
        expected_zeta: float,
        expected_chi: float,
    ) -> None:
        group = compute_group_residual_diagnostics(labels, indicators, residuals)
        for name, value in group.items():
            append(name, value)
        if not np.allclose(group["zeta"], expected_zeta, rtol=1e-12, atol=1e-14):
            raise RuntimeError("class-conditional contributions do not match stored zeta")
        if not np.allclose(group["chi"], expected_chi, rtol=1e-12, atol=1e-14):
            raise RuntimeError("class-conditional contributions do not match stored chi")

        selected = (indicators == 0) & (selection > 0)
        sample_size = labels.numel()
        for class_name, class_label in (("plus", 1), ("minus", -1)):
            class_mask = (labels == class_label) & (indicators == 0)
            selected_class = selected & class_mask
            class_count = int(class_mask.sum().item())
            selected_count = int(selected_class.sum().item())
            error_count = int(
                (selected_class & (pseudo_label_values != labels)).sum().item()
            )
            append(f"unlabeled_count_true_{class_name}", class_count)
            append(f"selected_count_true_{class_name}", selected_count)
            append(
                f"selection_rate_true_{class_name}",
                selected_count / class_count if class_count else np.nan,
            )
            append(
                f"pseudo_label_error_count_true_{class_name}", error_count
            )
            append(
                f"pseudo_label_error_rate_true_{class_name}",
                error_count / selected_count if selected_count else np.nan,
            )
            append(
                f"selected_mass_true_{class_name}", selected_count / sample_size
            )
            append(
                f"selected_pseudolabel_plus_true_{class_name}",
                int((selected_class & (pseudo_label_values == 1)).sum().item()),
            )
            append(
                f"selected_pseudolabel_minus_true_{class_name}",
                int((selected_class & (pseudo_label_values == -1)).sum().item()),
            )
            append(
                f"mean_unlabeled_score_true_{class_name}",
                float(scores[class_mask].mean()) if class_count else np.nan,
            )
            append(
                f"mean_selected_score_true_{class_name}",
                float(scores[selected_class].mean()) if selected_count else np.nan,
            )
            append(
                f"mean_selected_residual_true_{class_name}",
                (
                    float(residuals[selected_class].mean())
                    if selected_count
                    else np.nan
                ),
            )
        append(
            "selected_pseudolabel_plus",
            int((selected & (pseudo_label_values == 1)).sum().item()),
        )
        append(
            "selected_pseudolabel_minus",
            int((selected & (pseudo_label_values == -1)).sum().item()),
        )

    if source == "finite":
        env, finite = run.environment, run.finite
        if env is None or finite is None:
            raise ValueError("this run does not contain a finite-gradient trajectory")
        for step in finite.update_records_:
            process_update(
                env.Y,
                env.Delta,
                step.g,
                step.scores,
                step.selection,
                step.pseudo_labels,
                float(step.g.mean()),
                float(step.chi),
            )
    else:
        se = run.se
        if se is None:
            raise ValueError("this run does not contain state evolution")
        for t in range(run.algo_cfg.n_iterations):
            residual = se.residual[t]
            score = se.preactivation[t]
            if residual is None or score is None:
                raise RuntimeError("state-evolution trajectory is incomplete")
            pseudo_label_values = (
                se.initial_pseudo_label
                if t == 0
                else sign_with_positive_tie(score)
            )
            selection = se.selection_mask(score, t)
            process_update(
                se.label,
                se.indicator,
                residual,
                score,
                selection,
                pseudo_label_values,
                float(se.mean_residual[t]),
                float(se.label_residual_alignments[t]),
            )

    result = {
        key: np.asarray(value)
        for key, value in update_values.items()
    }
    result.update(
        {
            "normalized_bias": np.asarray(error_diagnostics["normalized_bias"]),
            "normalized_alignment": np.asarray(
                error_diagnostics["normalized_alignment"]
            ),
            "optimal_normalized_bias": np.asarray(
                error_diagnostics["optimal_normalized_bias"]
            ),
            "signed_bias_tracking_error": np.asarray(
                error_diagnostics["signed_bias_tracking_error"]
            ),
            "normalized_bias_increment": np.asarray(
                error_diagnostics["normalized_bias_increment"]
            ),
            "optimal_normalized_bias_increment": np.asarray(
                error_diagnostics["optimal_normalized_bias_increment"]
            ),
            "signed_bias_tracking_error_increment": np.asarray(
                error_diagnostics["signed_bias_tracking_error_increment"]
            ),
            "alignment_contribution": np.asarray(
                error_diagnostics["alignment_contribution"]
            ),
            "bias_contribution": np.asarray(
                error_diagnostics["bias_contribution"]
            ),
            "error_increment": np.asarray(error_diagnostics["error_increment"]),
            "signal_alignment_raw": signal_alignment_raw,
            "noise_scale": noise_scale,
            "orthogonal_weight_energy": orthogonal_weight_energy,
            "orthogonal_to_signal_ratio": orthogonal_to_signal_ratio,
            "signal_alignment_increment": np.diff(signal_alignment_raw),
            "orthogonal_weight_energy_increment": np.diff(
                orthogonal_weight_energy
            ),
            "orthogonal_to_signal_ratio_increment": np.diff(
                orthogonal_to_signal_ratio
            ),
            "normalized_alignment_geometry_reconstruction_error": (
                geometry_identity_error
            ),
            "geometry_tolerance": np.asarray(geometry_tolerance),
        }
    )
    return result


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

    env, X, X_test, Y_test, finite = (
        run.environment,
        run.X,
        run.X_test,
        run.Y_test,
        run.finite,
    )
    if any(value is None for value in (env, X, X_test, Y_test, finite)):
        raise ValueError(
            "the finite oracle requires a run created with run_finite=True"
        )
    callback = TestEvaluatorCallback(
        X_lab=X[env.I_L],
        Y_lab=env.Y[env.I_L],
        X_unl=X[env.I_U],
        Y_unl=env.Y[env.I_U],
        X_test=X_test,
        Y_test=Y_test,
        mu=env.mu,
        sigma=run.data_cfg.scale,
        p=run.data_cfg.label_prior,
        metrics=DEFAULT_METRICS,
    )
    learner = SelfTrainedGradientDescent(cfg=run.algo_cfg, callback=callback)
    assert finite.initialization_ is not None

    def oracle_labels(_t: int, scores: torch.Tensor, _initial: torch.Tensor) -> torch.Tensor:
        return env.Y.to(dtype=scores.dtype, device=scores.device)

    with patch("src.algorithms.pseudo_labels", oracle_labels):
        learner.fit_full(X, env, finite.initialization_)
    callback.error_diagnostics = compute_error_diagnostics(
        _finite_state_observables_from_callback(callback),
        p=run.data_cfg.label_prior,
        sigma=run.data_cfg.scale,
        s_mu=run.signal_scale,
    )
    return learner, callback


def run_state_evolution_oracle_selected_label_counterfactual(
    run: ExperimentRun,
) -> MacroscopicStateEvolution:
    """Run the selected-label oracle counterfactual in state evolution.

    The returned trajectory has exactly the same particle base law, Gaussian
    innovations, confidence selection rule, and optimization configuration as
    ``run.se``.  Only the pseudo-label target is changed: selected unlabeled
    particles use their true label at every update.  Consequently this is the
    mean-field counterpart of :func:`run_oracle_selected_label_counterfactual`,
    and likewise a theorem-external diagnostic rather than a realizable
    self-training procedure.
    """

    source = run.se
    if source is None:
        raise ValueError("this run does not contain state evolution")
    if source.initial_pseudo_label is None:
        raise ValueError("state evolution must have an explicit initial pseudo-label law")
    initial_weight = source.weight[0]
    initial_bias = source.bias[0]
    if initial_weight is None or initial_bias is None:
        raise RuntimeError("state-evolution initial state is unavailable")

    def parameter_base_sampler(
        size: int, _generator: torch.Generator, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if size != source.K_w:
            raise RuntimeError("oracle parameter particle count does not match its source run")
        return (
            source.signal.to(dtype=dtype, device=device).detach().clone(),
            initial_weight.to(dtype=dtype, device=device).detach().clone(),
        )

    def sample_base_sampler(
        size: int, _generator: torch.Generator, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if size != source.K_g:
            raise RuntimeError("oracle sample particle count does not match its source run")
        return (
            source.label.to(dtype=dtype, device=device).detach().clone(),
            source.indicator.to(dtype=dtype, device=device).detach().clone(),
            source.initial_pseudo_label.to(dtype=dtype, device=device).detach().clone(),
        )

    oracle = MacroscopicStateEvolution(
        data_cfg=run.data_cfg,
        algo_cfg=run.algo_cfg,
        mc_seed=source.mc_seed,
        K=None,
        K_w=source.K_w,
        K_g=source.K_g,
        initial_bias=float(initial_bias),
        eps_rank=source.eps_rank,
        lstsq_rcond=source.lstsq_rcond,
        lstsq_driver=source.lstsq_driver,
        dtype=source.dtype,
        device=source.device,
        parameter_base_sampler=parameter_base_sampler,
        sample_base_sampler=sample_base_sampler,
    )

    def oracle_pseudo_residual(
        *,
        preactivation: torch.Tensor,
        label: torch.Tensor,
        indicator: torch.Tensor,
        selection_mask: torch.Tensor,
        selection_rate: float,
        coef: float,
        rho: float,
        eta: float,
        loss_function: Any,
        **_unused: Any,
    ) -> torch.Tensor:
        return pseudo_residual(
            scores=preactivation,
            Y=label,
            Delta=indicator.to(dtype=preactivation.dtype),
            Yhat=label,
            selection=selection_mask,
            omega=selection_rate,
            pi=coef,
            eta=eta,
            rho=rho,
            loss_function=loss_function,
        )

    with patch.object(
        asymptotics_module,
        "compute_abstract_pseudo_residual_from",
        oracle_pseudo_residual,
    ):
        oracle.compute_trajectory()
    oracle.error_diagnostics = compute_error_diagnostics(
        _state_evolution_state_observables_from(oracle),
        p=run.data_cfg.label_prior,
        sigma=run.data_cfg.scale,
        s_mu=run.signal_scale,
    )
    return oracle

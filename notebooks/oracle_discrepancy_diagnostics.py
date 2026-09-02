"""Finite-dimensional self-training/oracle discrepancy diagnostics.

This module is deliberately additive.  It consumes completed finite
trajectories produced by :class:`src.algorithms.SelfTrainedGradientDescent`
and never changes the optimizer.  The recursive bound computed here is a
post-hoc finite-sample diagnostic: its default selection-rate floor is the
minimum *observed* ST/oracle selection rate over the comparison horizon.
That floor is exact for the realised trajectory, but it is not an a priori
deterministic lower bound.

The implementation supports balanced labels with either a fixed or learned
bias, a symmetric hard selector, logistic loss, and ridge penalty.  When the
bias pseudo-label weight differs from the weight pseudo-label weight, its
residual discrepancy is propagated separately.  The constants are exact:

``Lip(ell') = 1/4``, ``||ell'||_infinity = 1``, and ``Lip(j') = 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Optional

from matplotlib import pyplot as plt
import numpy as np
from scipy.special import log_ndtr
import torch

from notebooks.experiment_helpers import (
    ExperimentRun,
    run_oracle_selected_label_counterfactual,
    run_state_evolution_oracle_selected_label_counterfactual,
)
from src.algorithms import SelfTrainedGradientDescent
from src.asymptotics import MacroscopicStateEvolution
from src.initialization import compute_scores
from src.objectives import HardSelection, LogisticLoss, RidgePenalty
from src.performance import population_error
from src.primitives import normalized_selection, pseudo_residual


def _ratio(rhs: float, lhs: float, *, tolerance: float = 1e-15) -> float:
    """Return a nonnegative slack factor, with explicit zero handling."""

    if abs(lhs) <= tolerance:
        return 1.0 if abs(rhs) <= tolerance else math.inf
    with np.errstate(over="ignore", invalid="ignore"):
        return rhs / lhs


def _as_float(value: Any) -> float:
    return float(torch.as_tensor(value).detach().cpu())


@dataclass(frozen=True)
class HardSelectorBound:
    """One finite-grid minimization of the hard-selector discrepancy bound."""

    discrepancy_radius: float
    actual_discrepancy: float
    minimized_bound: float
    effective_bound: float
    minimizing_h: float
    boundary_mass: float
    grid: np.ndarray
    values: np.ndarray


@dataclass
class OracleDiscrepancyDiagnostic:
    """Completed coupled trajectories and all finite discrepancy diagnostics."""

    st_learner: SelfTrainedGradientDescent
    oracle_learner: SelfTrainedGradientDescent
    supervised_learner: SelfTrainedGradientDescent
    state: dict[str, np.ndarray]
    update: dict[str, np.ndarray]
    constants: dict[str, Any]
    summary: dict[str, Any]
    hard_selector_bounds: tuple[HardSelectorBound, ...]
    notes: tuple[str, ...]


def _validate_supported_setting(run: ExperimentRun) -> None:
    if run.finite is None or run.environment is None or run.X is None:
        raise ValueError("the discrepancy diagnostic requires run_finite=True")
    cfg = run.algo_cfg
    if not math.isclose(run.data_cfg.label_prior, 0.5, abs_tol=1e-12):
        raise ValueError("the current diagnostic is restricted to balanced labels (p=1/2)")
    if not math.isfinite(float(cfg.initial_bias)):
        raise ValueError("initial_bias must be finite")
    if not cfg.is_canonical_fixed_pi:
        raise ValueError("the current diagnostic requires the fixed-pi manuscript model")
    if not isinstance(cfg.loss_function, LogisticLoss):
        raise NotImplementedError("exact constants are currently implemented only for LogisticLoss")
    if not isinstance(cfg.penalty_function, RidgePenalty):
        raise NotImplementedError("Lip(j') is currently exact only for RidgePenalty")
    if not isinstance(cfg.selection_function, HardSelection):
        raise NotImplementedError("the finite boundary-mass bound requires HardSelection")
    if not math.isclose(
        float(cfg.positive_margin), -float(cfg.negative_margin), rel_tol=0.0, abs_tol=1e-12
    ):
        raise NotImplementedError("the current hard-selector bound requires symmetric margins")
    if not 0.0 < run.environment.rho < 1.0:
        raise ValueError("the diagnostic requires both labelled and unlabeled observations")
    if run.data_cfg.scale <= 0.0:
        raise ValueError("sigma must be positive")


def _state_histories(learner: SelfTrainedGradientDescent) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return weight and bias states at times 0,...,T."""

    if learner.weights is None or learner.bias is None:
        raise RuntimeError("learner has not been fitted")
    weights = [value.detach().clone() for value in learner.weight_history_]
    biases = [record.bias.detach().clone() for record in learner.update_records_]
    biases.append(learner.bias.detach().clone())
    if len(weights) != len(biases):
        raise RuntimeError("finite state histories are inconsistent")
    return weights, biases


def hard_selector_bound(
    *,
    st_scores: torch.Tensor,
    oracle_scores: torch.Tensor,
    unlabeled_mask: torch.Tensor,
    kappa: float,
    discrepancy_radius: float,
    grid_size: int = 161,
) -> HardSelectorBound:
    r"""Minimize the explicit finite hard-selector inequality over ``h``.

    For every grid point ``h>0`` this evaluates exactly

    ``B_{t,n}(h) + x^2 / ((1-rho) h^2)``.

    The grid combines a scale-adaptive logarithmic grid, empirical boundary
    distances, and the paper's ``h ~ x^(2/3)`` scale.  ``effective_bound`` is
    ``min(1, minimized_bound)``; the extra cap is the exact trivial fact that
    an empirical disagreement rate lies in ``[0,1]``.  The uncapped requested
    quantity remains available as ``minimized_bound``.
    """

    st_scores = torch.as_tensor(st_scores, dtype=torch.float64)
    oracle_scores = torch.as_tensor(oracle_scores, dtype=torch.float64, device=st_scores.device)
    unlabeled_mask = torch.as_tensor(unlabeled_mask, dtype=torch.bool, device=st_scores.device)
    if st_scores.shape != oracle_scores.shape or st_scores.shape != unlabeled_mask.shape:
        raise ValueError("scores and unlabeled_mask must have the same one-dimensional shape")
    if st_scores.ndim != 1 or not unlabeled_mask.any():
        raise ValueError("at least one unlabeled score is required")
    x = float(discrepancy_radius)
    if not math.isfinite(x) or x < 0.0:
        raise ValueError("discrepancy_radius must be finite and nonnegative")
    if grid_size < 16:
        raise ValueError("grid_size must be at least 16")

    st_unlabeled = st_scores[unlabeled_mask]
    oracle_unlabeled = oracle_scores[unlabeled_mask]
    st_selection = torch.abs(st_unlabeled) >= float(kappa)
    oracle_selection = torch.abs(oracle_unlabeled) >= float(kappa)
    actual = float(torch.mean((st_selection != oracle_selection).to(torch.float64)))
    if x == 0.0:
        # Equal preactivations imply identical deterministic hard selections.
        return HardSelectorBound(
            discrepancy_radius=0.0,
            actual_discrepancy=actual,
            minimized_bound=0.0,
            effective_bound=0.0,
            minimizing_h=0.0,
            boundary_mass=0.0,
            grid=np.asarray([0.0]),
            values=np.asarray([0.0]),
        )

    distances = torch.abs(torch.abs(st_unlabeled) - float(kappa)).detach().cpu().numpy()
    score_scale = max(
        1.0,
        abs(float(kappa)),
        float(np.quantile(np.abs(st_unlabeled.detach().cpu().numpy()), 0.9)),
        x,
    )
    lower = max(np.finfo(float).tiny ** 0.25, score_scale * 1e-8)
    upper = max(score_scale * 1e2, x ** (2.0 / 3.0) * 1e2, lower * 10.0)
    logarithmic = np.geomspace(lower, upper, grid_size)
    positive_distances = distances[distances > 0.0]
    if positive_distances.size:
        quantile_count = min(grid_size, positive_distances.size)
        distance_candidates = np.quantile(
            positive_distances, np.linspace(0.0, 1.0, quantile_count)
        )
        # Values just below jumps of B(h) can be sharper than the jumps.
        distance_candidates = np.concatenate(
            (distance_candidates, np.nextafter(distance_candidates, 0.0))
        )
    else:
        distance_candidates = np.empty(0)
    theory_scale = np.asarray([x ** (2.0 / 3.0)])
    grid = np.unique(np.concatenate((logarithmic, distance_candidates, theory_scale)))
    grid = grid[np.isfinite(grid) & (grid > 0.0)]

    boundary_mass = np.asarray([(distances <= h).mean() for h in grid], dtype=float)
    # n(1-rho) equals the realised number of unlabeled observations.  E_r is
    # normalized by all n coordinates, hence the displayed Markov term retains
    # the factor 1/(1-rho).
    one_minus_rho = float(unlabeled_mask.to(torch.float64).mean())
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        values = boundary_mass + np.square(x / grid) / one_minus_rho
    minimizer = int(np.argmin(values))
    minimized = float(values[minimizer])
    return HardSelectorBound(
        discrepancy_radius=x,
        actual_discrepancy=actual,
        minimized_bound=minimized,
        effective_bound=min(1.0, minimized),
        minimizing_h=float(grid[minimizer]),
        boundary_mass=float(boundary_mass[minimizer]),
        grid=grid,
        values=values,
    )


def _normalized_selector_bound(
    xi_upper: float, *, one_minus_rho: float, omega_lower: float
) -> float:
    if omega_lower <= 0.0:
        return math.inf
    xi_upper = min(1.0, max(0.0, float(xi_upper)))
    return math.sqrt(one_minus_rho * xi_upper) / omega_lower + (
        math.sqrt(one_minus_rho) * xi_upper / (omega_lower**2)
    )


def _propagation_modulus(
    x: float,
    *,
    selector_bound: HardSelectorBound,
    eta: float,
    pi: float,
    rho: float,
    omega_lower: float,
    loss_gradient_lipschitz: float = 0.25,
    loss_gradient_bound: float = 1.0,
) -> float:
    """Explicit finite propagation modulus preceding the abstract constants."""

    if omega_lower <= 0.0:
        return math.inf
    selector_weight = _normalized_selector_bound(
        selector_bound.effective_bound,
        one_minus_rho=1.0 - rho,
        omega_lower=omega_lower,
    )
    return eta * loss_gradient_lipschitz / rho * x + eta * abs(pi) / (1.0 - rho) * (
        loss_gradient_lipschitz / omega_lower * x
        + loss_gradient_bound * selector_weight
    )


def _innovation_selection_floor(
    source: MacroscopicStateEvolution,
    oracle: MacroscopicStateEvolution,
    *,
    sigma: float,
    kappa: float,
    horizon: int,
) -> tuple[float, float, float]:
    """Return ``(v_min, omega_floor, log_omega_floor)`` through ``horizon``."""

    values: list[float] = []
    for process in (source, oracle):
        for value in process.forward_innovation_scale[: horizon + 1]:
            if value is None:
                return math.nan, math.nan, math.nan
            scalar = _as_float(value)
            if not math.isfinite(scalar) or scalar <= 0.0:
                return scalar, math.nan, math.nan
            values.append(scalar)
    v_min = min(values)
    # This is one half of 2 Phi(-kappa/(sigma v_min)), exactly as in the paper.
    standardized = -float(kappa) / (float(sigma) * v_min)
    log_floor = float(log_ndtr(standardized))
    return v_min, float(math.exp(log_floor)) if log_floor > math.log(np.finfo(float).tiny) else 0.0, log_floor


def _recursive_bound(
    *,
    horizon: int,
    omega_lower: float,
    st_records,
    oracle_records,
    unlabeled_mask: torch.Tensor,
    forcing: np.ndarray,
    bias_forcing: np.ndarray,
    a_d: float,
    b_d: float,
    c_w: float,
    eta: float,
    pi_schedule: np.ndarray,
    bias_pi_schedule: np.ndarray,
    include_bias: bool,
    rho: float,
    kappa: float,
    grid_size: int,
) -> np.ndarray:
    """Evaluate the data-dependent closed recursion through ``horizon``."""

    result = np.zeros(horizon + 1, dtype=float)
    A_d = max(1.0, a_d)
    maximum_safe_radius = 1e150
    for t in range(horizon):
        if not math.isfinite(result[t]) or result[t] > maximum_safe_radius / A_d:
            result[t + 1 :] = math.inf
            break
        x = A_d * result[t]
        selector_bound = hard_selector_bound(
            st_scores=st_records[t].scores,
            oracle_scores=oracle_records[t].scores,
            unlabeled_mask=unlabeled_mask,
            kappa=kappa,
            discrepancy_radius=x,
            grid_size=grid_size,
        )
        weight_psi = _propagation_modulus(
            x,
            selector_bound=selector_bound,
            eta=eta,
            pi=float(pi_schedule[t]),
            rho=rho,
            omega_lower=omega_lower,
        )
        weight_residual_bound = forcing[t] + weight_psi
        if include_bias:
            bias_psi = _propagation_modulus(
                x,
                selector_bound=selector_bound,
                eta=eta,
                pi=float(bias_pi_schedule[t]),
                rho=rho,
                omega_lower=omega_lower,
            )
            bias_residual_bound = bias_forcing[t] + bias_psi
        else:
            bias_residual_bound = 0.0
        with np.errstate(over="ignore", invalid="ignore"):
            result[t + 1] = (
                c_w * result[t]
                + b_d * weight_residual_bound
                + bias_residual_bound
            )
        if not math.isfinite(result[t + 1]):
            result[t + 1 :] = math.inf
            break
    return result


def run_finite_oracle_discrepancy_diagnostic(
    run: ExperimentRun,
    *,
    grid_size: int = 161,
    compute_state_evolution_lower_bound: bool = True,
    tolerance: float = 2e-10,
) -> OracleDiscrepancyDiagnostic:
    """Evaluate the paper's finite ST/oracle discrepancy chain on one run.

    The ST and selected-label oracle trajectories share ``(X,U,Y,Delta)`` and
    the complete initialization.  The oracle is then run independently by the
    repository's existing counterfactual helper.  A supervised trajectory is
    also restarted from that same state with ``pi=0`` solely to compute gains.
    """

    _validate_supported_setting(run)
    assert run.finite is not None and run.environment is not None and run.X is not None
    env, X, st = run.environment, run.X, run.finite
    cfg = run.algo_cfg
    oracle, _ = run_oracle_selected_label_counterfactual(run)
    if st.initialization_ is None:
        raise RuntimeError("the finite self-training initialization is unavailable")
    supervised_cfg = replace(
        cfg,
        pseudo_label_param=0.0,
        ramp_start=None,
        ramp_end=None,
        experimental_schedule=None,
        bias_pseudo_label_param=0.0,
    )
    supervised = SelfTrainedGradientDescent(supervised_cfg).fit_full(
        X, env, st.initialization_
    )

    st_weights, st_biases = _state_histories(st)
    oracle_weights, oracle_biases = _state_histories(oracle)
    supervised_weights, supervised_biases = _state_histories(supervised)
    T = cfg.n_iterations
    if not (
        len(st.update_records_) == len(oracle.update_records_) == T
        and len(st_weights) == len(oracle_weights) == T + 1
    ):
        raise RuntimeError("paired finite trajectories have inconsistent horizons")

    sigma, rho = float(run.data_cfg.scale), env.rho
    one_minus_rho = 1.0 - rho
    kappa = float(cfg.positive_margin)
    eta = float(cfg.step_size)
    pi_schedule = np.asarray([cfg.get_pseudo_label_weight(t) for t in range(T)])
    bias_pi_schedule = np.asarray(
        [cfg.get_bias_pseudo_label_weight(t) for t in range(T)]
    )
    unlabeled = env.Delta == 0
    sqrt_d, sqrt_n = math.sqrt(env.d), math.sqrt(env.n)
    mu_norm = _as_float(torch.linalg.vector_norm(env.mu) / sqrt_d)

    # The Gaussian noise matrix is reconstructed exactly from the stored design.
    U = (X - env.Y[:, None] * env.mu[None, :] / sqrt_d) / sigma
    operator_norm = _as_float(torch.linalg.matrix_norm(U, ord=2))
    a_d = mu_norm + sigma * operator_norm / sqrt_n
    b_d = mu_norm + sigma / math.sqrt(env.delta) * operator_norm / sqrt_d
    penalty_gradient_lipschitz = 1.0  # Ridge: j'(w)=w.
    c_w = 1.0 + eta * float(cfg.penalty_param) * penalty_gradient_lipschitz
    A_d = max(1.0, a_d)

    def state_quantities(weights: list[torch.Tensor], biases: list[torch.Tensor]):
        tau = np.asarray([_as_float(torch.linalg.vector_norm(w) / sqrt_d) for w in weights])
        m = np.asarray([_as_float(torch.dot(env.mu, w) / env.d) for w in weights])
        bias = np.asarray([_as_float(value) for value in biases])
        alignment = np.divide(m, tau, out=np.full_like(m, np.nan), where=tau > 0.0)
        normalized_bias = np.divide(
            bias, tau, out=np.full_like(bias, np.nan), where=tau > 0.0
        )
        error = np.asarray(
            [population_error(bb, mm, tt, sigma, 0.5) for bb, mm, tt in zip(bias, m, tau)]
        )
        return tau, m, bias, normalized_bias, alignment, error

    st_tau, st_m, st_bias, st_normalized_bias, st_alignment, st_error = state_quantities(
        st_weights, st_biases
    )
    (
        oracle_tau,
        oracle_m,
        oracle_bias,
        oracle_normalized_bias,
        oracle_alignment,
        oracle_error,
    ) = state_quantities(
        oracle_weights, oracle_biases
    )
    (
        supervised_tau,
        supervised_m,
        supervised_bias,
        supervised_normalized_bias,
        supervised_alignment,
        supervised_error,
    ) = state_quantities(
        supervised_weights, supervised_biases
    )

    E_w = np.asarray(
        [_as_float(torch.linalg.vector_norm(a - b) / sqrt_d) for a, b in zip(st_weights, oracle_weights)]
    )
    E_b = np.asarray([abs(_as_float(a - b)) for a, b in zip(st_biases, oracle_biases)])
    E_r = np.asarray(
        [
            _as_float(torch.linalg.vector_norm(a.scores - b.scores) / sqrt_n)
            for a, b in zip(st.update_records_, oracle.update_records_)
        ]
    )
    E_g = np.asarray(
        [
            _as_float(torch.linalg.vector_norm(a.g - b.g) / sqrt_n)
            for a, b in zip(st.update_records_, oracle.update_records_)
        ]
    )
    E_g_bias = np.asarray(
        [
            _as_float(torch.linalg.vector_norm(a.bias_residual - b.bias_residual) / sqrt_n)
            for a, b in zip(st.update_records_, oracle.update_records_)
        ]
    )
    omega_st = np.asarray([_as_float(record.omega) for record in st.update_records_])
    omega_oracle = np.asarray([_as_float(record.omega) for record in oracle.update_records_])

    forcing_norm = np.zeros(T)
    propagation_norm = np.zeros(T)
    forcing_formula = np.zeros(T)
    selected_error_rate = np.zeros(T)
    decomposition_error = np.zeros(T)
    forcing_identity_error = np.zeros(T)
    bias_forcing_norm = np.zeros(T)
    bias_propagation_norm = np.zeros(T)
    bias_forcing_formula = np.zeros(T)
    bias_decomposition_error = np.zeros(T)
    bias_forcing_identity_error = np.zeros(T)
    actual_selector_weight_discrepancy = np.zeros(T)
    selector_bounds: list[HardSelectorBound] = []
    for t, (st_step, oracle_step) in enumerate(zip(st.update_records_, oracle.update_records_)):
        oracle_at_st = pseudo_residual(
            scores=st_step.scores,
            Y=env.Y,
            Delta=env.Delta,
            Yhat=env.Y,
            selection=st_step.selection,
            omega=st_step.omega,
            pi=float(pi_schedule[t]),
            eta=eta,
            rho=rho,
            loss_function=cfg.loss_function,
        )
        e_pl = st_step.g - oracle_at_st
        e_prop = oracle_at_st - oracle_step.g
        bias_oracle_at_st = pseudo_residual(
            scores=st_step.scores,
            Y=env.Y,
            Delta=env.Delta,
            Yhat=env.Y,
            selection=st_step.selection,
            omega=st_step.omega,
            pi=float(bias_pi_schedule[t]),
            eta=eta,
            rho=rho,
            loss_function=cfg.loss_function,
        )
        e_pl_bias = st_step.bias_residual - bias_oracle_at_st
        e_prop_bias = bias_oracle_at_st - oracle_step.bias_residual
        forcing_norm[t] = _as_float(torch.linalg.vector_norm(e_pl) / sqrt_n)
        propagation_norm[t] = _as_float(torch.linalg.vector_norm(e_prop) / sqrt_n)
        decomposition_error[t] = _as_float(
            torch.linalg.vector_norm((st_step.g - oracle_step.g) - (e_pl + e_prop)) / sqrt_n
        )
        bias_forcing_norm[t] = _as_float(
            torch.linalg.vector_norm(e_pl_bias) / sqrt_n
        )
        bias_propagation_norm[t] = _as_float(
            torch.linalg.vector_norm(e_prop_bias) / sqrt_n
        )
        bias_decomposition_error[t] = _as_float(
            torch.linalg.vector_norm(
                (st_step.bias_residual - oracle_step.bias_residual)
                - (e_pl_bias + e_prop_bias)
            )
            / sqrt_n
        )
        selected = st_step.selection * (1.0 - env.Delta)
        selected_count = _as_float(selected.sum())
        if selected_count > 0.0:
            wrong = (st_step.pseudo_labels != env.Y).to(selected.dtype)
            selected_error_rate[t] = _as_float((selected * wrong).sum()) / selected_count
            forcing_formula[t] = abs(float(pi_schedule[t])) * eta * math.sqrt(
                selected_error_rate[t] / (one_minus_rho * omega_st[t])
            )
            bias_forcing_formula[t] = abs(float(bias_pi_schedule[t])) * eta * math.sqrt(
                selected_error_rate[t] / (one_minus_rho * omega_st[t])
            )
        forcing_identity_error[t] = abs(forcing_norm[t] - forcing_formula[t])
        bias_forcing_identity_error[t] = abs(
            bias_forcing_norm[t] - bias_forcing_formula[t]
        )
        st_normalized = normalized_selection(st_step.selection, st_step.omega)
        oracle_normalized = normalized_selection(oracle_step.selection, oracle_step.omega)
        actual_selector_weight_discrepancy[t] = _as_float(
            torch.linalg.vector_norm((1.0 - env.Delta) * (st_normalized - oracle_normalized))
            / sqrt_n
        )
        selector_bounds.append(
            hard_selector_bound(
                st_scores=st_step.scores,
                oracle_scores=oracle_step.scores,
                unlabeled_mask=unlabeled,
                kappa=kappa,
                discrepancy_radius=E_r[t],
                grid_size=grid_size,
            )
        )

    Xi = np.asarray([bound.actual_discrepancy for bound in selector_bounds])
    Xi_bound = np.asarray([bound.minimized_bound for bound in selector_bounds])
    Xi_effective_bound = np.asarray([bound.effective_bound for bound in selector_bounds])
    h_min = np.asarray([bound.minimizing_h for bound in selector_bounds])
    boundary_mass = np.asarray([bound.boundary_mass for bound in selector_bounds])
    Xi_ratio = np.asarray([_ratio(rhs, lhs) for rhs, lhs in zip(Xi_bound, Xi)])

    omega_empirical_prefix = np.minimum.accumulate(np.minimum(omega_st, omega_oracle))
    omega_empirical = float(omega_empirical_prefix[-1]) if T else math.nan

    oracle_se: Optional[MacroscopicStateEvolution] = None
    v_min_prefix = np.full(T + 1, np.nan)
    omega_theoretical_prefix = np.full(T + 1, np.nan)
    log_omega_theoretical_prefix = np.full(T + 1, np.nan)
    if compute_state_evolution_lower_bound and run.se is not None:
        oracle_se = run_state_evolution_oracle_selected_label_counterfactual(run)
        for horizon in range(T + 1):
            (
                v_min_prefix[horizon],
                omega_theoretical_prefix[horizon],
                log_omega_theoretical_prefix[horizon],
            ) = _innovation_selection_floor(
                run.se, oracle_se, sigma=sigma, kappa=kappa, horizon=horizon
            )
    omega_theoretical = float(omega_theoretical_prefix[T - 1]) if T else math.nan

    def direct_bounds(
        omega_lower: float,
        coefficient_schedule: np.ndarray,
        forcing_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        selector_weight = np.asarray(
            [
                _normalized_selector_bound(
                    xi, one_minus_rho=one_minus_rho, omega_lower=omega_lower
                )
                for xi in Xi_effective_bound
            ]
        )
        prop = np.asarray(
            [
                _propagation_modulus(
                    E_r[t],
                    selector_bound=selector_bounds[t],
                    eta=eta,
                    pi=float(coefficient_schedule[t]),
                    rho=rho,
                    omega_lower=omega_lower,
                )
                for t in range(T)
            ]
        )
        return selector_weight, prop, forcing_values + prop

    selector_weight_bound_emp, propagation_bound_emp, residual_bound_emp = direct_bounds(
        omega_empirical, pi_schedule, forcing_formula
    )
    (
        bias_selector_weight_bound_emp,
        bias_propagation_bound_emp,
        bias_residual_bound_emp,
    ) = direct_bounds(
        omega_empirical, bias_pi_schedule, bias_forcing_formula
    )
    if math.isfinite(omega_theoretical) and omega_theoretical > 0.0:
        selector_weight_bound_th, propagation_bound_th, residual_bound_th = direct_bounds(
            omega_theoretical, pi_schedule, forcing_formula
        )
        (
            bias_selector_weight_bound_th,
            bias_propagation_bound_th,
            bias_residual_bound_th,
        ) = direct_bounds(
            omega_theoretical, bias_pi_schedule, bias_forcing_formula
        )
    else:
        theoretical_fill = math.inf if math.isfinite(log_omega_theoretical_prefix[T - 1]) else math.nan
        selector_weight_bound_th = propagation_bound_th = residual_bound_th = np.full(T, theoretical_fill)
        bias_selector_weight_bound_th = bias_propagation_bound_th = bias_residual_bound_th = np.full(
            T, theoretical_fill
        )

    # For the main trajectory plot, use the full-horizon empirical floor.  For
    # each candidate comparison time, recompute the recursion with its sharper
    # prefix floor; the diagonal values below are therefore horizon-specific.
    recursive_full = _recursive_bound(
        horizon=T,
        omega_lower=omega_empirical,
        st_records=st.update_records_,
        oracle_records=oracle.update_records_,
        unlabeled_mask=unlabeled,
        forcing=forcing_formula,
        bias_forcing=bias_forcing_formula,
        a_d=a_d,
        b_d=b_d,
        c_w=c_w,
        eta=eta,
        pi_schedule=pi_schedule,
        bias_pi_schedule=bias_pi_schedule,
        include_bias=cfg.include_bias,
        rho=rho,
        kappa=kappa,
        grid_size=grid_size,
    )
    if T <= 100:
        recursive_by_horizon = np.zeros(T + 1)
        for horizon in range(1, T + 1):
            omega_floor = float(omega_empirical_prefix[horizon - 1])
            recursive_by_horizon[horizon] = _recursive_bound(
                horizon=horizon,
                omega_lower=omega_floor,
                st_records=st.update_records_,
                oracle_records=oracle.update_records_,
                unlabeled_mask=unlabeled,
                forcing=forcing_formula,
                bias_forcing=bias_forcing_formula,
                a_d=a_d,
                b_d=b_d,
                c_w=c_w,
                eta=eta,
                pi_schedule=pi_schedule,
                bias_pi_schedule=bias_pi_schedule,
                include_bias=cfg.include_bias,
                rho=rho,
                kappa=kappa,
                grid_size=grid_size,
            )[-1]
        recursive_horizon_mode = "prefix empirical omega floor"
    else:
        # Recomputing one complete recursion for every t0 is quadratic in T.
        # The full-horizon floor is no larger than any prefix floor, so the
        # already computed path remains a valid (possibly looser) bound for
        # every comparison time and makes long-horizon diagnostics linear in T.
        recursive_by_horizon = recursive_full.copy()
        recursive_horizon_mode = "full-horizon empirical omega floor (linear-time long-horizon mode)"

    forward_rhs = E_b[:T] + a_d * E_w[:T]
    bias_rhs = E_b[:T] + E_g_bias
    weight_rhs = c_w * E_w[:T] + b_d * E_g
    Z = E_b + E_w
    forward_ratio = np.asarray([_ratio(rhs, lhs) for rhs, lhs in zip(forward_rhs, E_r)])
    bias_ratio = np.asarray([_ratio(rhs, lhs) for rhs, lhs in zip(bias_rhs, E_b[1:])])
    weight_ratio = np.asarray([_ratio(rhs, lhs) for rhs, lhs in zip(weight_rhs, E_w[1:])])
    prop_ratio = np.asarray(
        [_ratio(rhs, lhs) for rhs, lhs in zip(propagation_bound_emp, propagation_norm)]
    )
    residual_ratio = np.asarray([_ratio(rhs, lhs) for rhs, lhs in zip(residual_bound_emp, E_g)])
    bias_prop_ratio = np.asarray(
        [_ratio(rhs, lhs) for rhs, lhs in zip(bias_propagation_bound_emp, bias_propagation_norm)]
    )
    bias_residual_ratio = np.asarray(
        [_ratio(rhs, lhs) for rhs, lhs in zip(bias_residual_bound_emp, E_g_bias)]
    )
    recursive_ratio = np.asarray([_ratio(rhs, lhs) for rhs, lhs in zip(recursive_full, Z)])

    tau_min_prefix = np.minimum.accumulate(np.minimum(st_tau, oracle_tau))
    alignment_difference = oracle_alignment - st_alignment
    alignment_abs = np.abs(alignment_difference)
    alignment_direct_bound = np.divide(
        2.0 * mu_norm * E_w,
        tau_min_prefix,
        out=np.full(T + 1, np.inf),
        where=tau_min_prefix > 0.0,
    )
    alignment_recursive_bound = np.divide(
        2.0 * mu_norm * recursive_full,
        tau_min_prefix,
        out=np.full(T + 1, np.inf),
        where=tau_min_prefix > 0.0,
    )
    alignment_direct_ratio = np.asarray(
        [_ratio(rhs, lhs) for rhs, lhs in zip(alignment_direct_bound, alignment_abs)]
    )
    alignment_recursive_ratio = np.asarray(
        [_ratio(rhs, lhs) for rhs, lhs in zip(alignment_recursive_bound, alignment_abs)]
    )

    normalized_bias_difference = oracle_normalized_bias - st_normalized_bias
    normalized_bias_abs = np.abs(normalized_bias_difference)
    normalized_bias_direct_bound = np.minimum(
        np.divide(E_b, oracle_tau, out=np.full(T + 1, np.inf), where=oracle_tau > 0.0)
        + np.divide(
            np.abs(st_bias) * E_w,
            st_tau * oracle_tau,
            out=np.full(T + 1, np.inf),
            where=(st_tau * oracle_tau) > 0.0,
        ),
        np.divide(E_b, st_tau, out=np.full(T + 1, np.inf), where=st_tau > 0.0)
        + np.divide(
            np.abs(oracle_bias) * E_w,
            st_tau * oracle_tau,
            out=np.full(T + 1, np.inf),
            where=(st_tau * oracle_tau) > 0.0,
        ),
    )
    normalized_bias_recursive_bound = np.minimum(
        np.divide(
            recursive_full,
            oracle_tau,
            out=np.full(T + 1, np.inf),
            where=oracle_tau > 0.0,
        )
        + np.divide(
            np.abs(st_bias) * recursive_full,
            st_tau * oracle_tau,
            out=np.full(T + 1, np.inf),
            where=(st_tau * oracle_tau) > 0.0,
        ),
        np.divide(
            recursive_full,
            st_tau,
            out=np.full(T + 1, np.inf),
            where=st_tau > 0.0,
        )
        + np.divide(
            np.abs(oracle_bias) * recursive_full,
            st_tau * oracle_tau,
            out=np.full(T + 1, np.inf),
            where=(st_tau * oracle_tau) > 0.0,
        ),
    )
    normalized_bias_direct_ratio = np.asarray(
        [_ratio(rhs, lhs) for rhs, lhs in zip(normalized_bias_direct_bound, normalized_bias_abs)]
    )
    normalized_bias_recursive_ratio = np.asarray(
        [_ratio(rhs, lhs) for rhs, lhs in zip(normalized_bias_recursive_bound, normalized_bias_abs)]
    )

    oracle_alignment_star = float(np.nanmax(oracle_alignment))
    oracle_regret = oracle_alignment_star - oracle_alignment
    recursive_alignment_by_horizon = np.divide(
        2.0 * mu_norm * recursive_by_horizon,
        tau_min_prefix,
        out=np.full(T + 1, np.inf),
        where=tau_min_prefix > 0.0,
    )
    if cfg.include_bias:
        recursive_bias_by_horizon = np.minimum(
            np.divide(
                recursive_by_horizon,
                oracle_tau,
                out=np.full(T + 1, np.inf),
                where=oracle_tau > 0.0,
            )
            + np.divide(
                np.abs(st_bias) * recursive_by_horizon,
                st_tau * oracle_tau,
                out=np.full(T + 1, np.inf),
                where=(st_tau * oracle_tau) > 0.0,
            ),
            np.divide(
                recursive_by_horizon,
                st_tau,
                out=np.full(T + 1, np.inf),
                where=st_tau > 0.0,
            )
            + np.divide(
                np.abs(oracle_bias) * recursive_by_horizon,
                st_tau * oracle_tau,
                out=np.full(T + 1, np.inf),
                where=(st_tau * oracle_tau) > 0.0,
            ),
        )
        oracle_error_regret = oracle_error - float(np.nanmin(oracle_error))
        population_lipschitz = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        gap_upper_by_time = oracle_error_regret + population_lipschitz * (
            recursive_alignment_by_horizon + recursive_bias_by_horizon
        )
        gap_upper_direct_coordinates = oracle_error_regret + population_lipschitz * (
            alignment_abs + normalized_bias_abs
        )
        gap_upper_direct_stability = oracle_error_regret + population_lipschitz * (
            alignment_direct_bound + normalized_bias_direct_bound
        )
        gap_bound_mode = "balanced learned-bias population-error Lipschitz bound"
    else:
        recursive_bias_by_horizon = np.zeros(T + 1)
        oracle_error_regret = oracle_error - float(np.nanmin(oracle_error))
        gap_upper_by_time = (
            oracle_regret + recursive_alignment_by_horizon
        ) / (sigma * math.sqrt(2.0 * math.pi))
        gap_upper_direct_coordinates = (
            oracle_regret + alignment_abs
        ) / (sigma * math.sqrt(2.0 * math.pi))
        gap_upper_direct_stability = (
            oracle_regret + alignment_direct_bound
        ) / (sigma * math.sqrt(2.0 * math.pi))
        gap_bound_mode = "balanced fixed-zero-bias alignment bound"
    best_comparison_time = int(np.nanargmin(gap_upper_by_time))
    oracle_gap_upper = float(gap_upper_by_time[best_comparison_time])

    best_st_error = float(np.nanmin(st_error))
    best_oracle_error = float(np.nanmin(oracle_error))
    best_supervised_error = float(np.nanmin(supervised_error))
    self_training_gain = best_supervised_error - best_st_error
    oracle_gain = best_supervised_error - best_oracle_error
    oracle_gap = best_st_error - best_oracle_error
    alignment_gap = float(np.nanmax(oracle_alignment) - np.nanmax(st_alignment))

    exact_checks = {
        "shared_weight_initialization": bool(torch.equal(st_weights[0], oracle_weights[0])),
        "shared_bias_initialization": bool(torch.equal(st_biases[0], oracle_biases[0])),
        "shared_initial_pseudo_labels": bool(
            st.initialization_ is not None
            and oracle.initialization_ is not None
            and torch.equal(st.initialization_.Y_init, oracle.initialization_.Y_init)
        ),
        "residual_decomposition": bool(np.all(decomposition_error <= tolerance)),
        "logistic_forcing_identity": bool(np.all(forcing_identity_error <= tolerance)),
        "bias_residual_decomposition": bool(
            np.all(bias_decomposition_error <= tolerance)
        ),
        "bias_logistic_forcing_identity": bool(
            np.all(bias_forcing_identity_error <= tolerance)
        ),
        "forward_inequality": bool(np.all(E_r <= forward_rhs + tolerance)),
        "bias_inequality": bool(np.all(E_b[1:] <= bias_rhs + tolerance)),
        "weight_inequality": bool(np.all(E_w[1:] <= weight_rhs + tolerance)),
        "selector_inequality": bool(np.all(Xi <= Xi_bound + tolerance)),
        "selector_weight_inequality_empirical": bool(
            np.all(actual_selector_weight_discrepancy <= selector_weight_bound_emp + tolerance)
        ),
        "propagation_inequality_empirical": bool(
            np.all(propagation_norm <= propagation_bound_emp + tolerance)
        ),
        "residual_inequality_empirical": bool(np.all(E_g <= residual_bound_emp + tolerance)),
        "bias_propagation_inequality_empirical": bool(
            np.all(bias_propagation_norm <= bias_propagation_bound_emp + tolerance)
        ),
        "bias_residual_inequality_empirical": bool(
            np.all(E_g_bias <= bias_residual_bound_emp + tolerance)
        ),
        "recursive_inequality_empirical": bool(np.all(Z <= recursive_full + tolerance)),
        "alignment_inequality_direct": bool(np.all(alignment_abs <= alignment_direct_bound + tolerance)),
        "alignment_inequality_recursive": bool(
            np.all(alignment_abs <= alignment_recursive_bound + tolerance)
        ),
        "normalized_bias_inequality_direct": bool(
            np.all(normalized_bias_abs <= normalized_bias_direct_bound + tolerance)
        ),
        "normalized_bias_inequality_recursive": bool(
            np.all(normalized_bias_abs <= normalized_bias_recursive_bound + tolerance)
        ),
    }

    stage_ratios: Mapping[str, np.ndarray] = {
        "forward": forward_ratio,
        "bias": bias_ratio,
        "weight": weight_ratio,
        "hard selector": Xi_ratio,
        "propagation": prop_ratio,
        "residual": residual_ratio,
        "bias propagation": bias_prop_ratio,
        "bias residual": bias_residual_ratio,
        "closed recursion": recursive_ratio,
        "alignment from E_w": alignment_direct_ratio,
        "alignment from recursion": alignment_recursive_ratio,
        "normalized bias from E_b,E_w": normalized_bias_direct_ratio,
        "normalized bias from recursion": normalized_bias_recursive_ratio,
    }
    stage_max_ratio = {
        name: float(np.nanmax(values)) if values.size else math.nan
        for name, values in stage_ratios.items()
    }
    finite_stage_ratios = {
        name: value for name, value in stage_max_ratio.items() if math.isfinite(value)
    }
    largest_loss_stage = (
        max(finite_stage_ratios, key=finite_stage_ratios.get)
        if finite_stage_ratios
        else "undefined because every relevant denominator is zero"
    )

    state = {
        "E_w": E_w,
        "E_b": E_b,
        "Z": Z,
        "st_weight_norm": st_tau,
        "oracle_weight_norm": oracle_tau,
        "supervised_weight_norm": supervised_tau,
        "st_alignment": st_alignment,
        "oracle_alignment": oracle_alignment,
        "supervised_alignment": supervised_alignment,
        "st_bias": st_bias,
        "oracle_bias": oracle_bias,
        "supervised_bias": supervised_bias,
        "st_normalized_bias": st_normalized_bias,
        "oracle_normalized_bias": oracle_normalized_bias,
        "supervised_normalized_bias": supervised_normalized_bias,
        "normalized_bias_discrepancy": normalized_bias_difference,
        "absolute_normalized_bias_discrepancy": normalized_bias_abs,
        "alignment_discrepancy": alignment_difference,
        "absolute_alignment_discrepancy": alignment_abs,
        "st_population_error": st_error,
        "oracle_population_error": oracle_error,
        "supervised_population_error": supervised_error,
        "recursive_bound": recursive_full,
        "recursive_bound_by_comparison_horizon": recursive_by_horizon,
        "tau_min_prefix": tau_min_prefix,
        "alignment_direct_bound": alignment_direct_bound,
        "alignment_recursive_bound": alignment_recursive_bound,
        "normalized_bias_direct_bound": normalized_bias_direct_bound,
        "normalized_bias_recursive_bound": normalized_bias_recursive_bound,
        "oracle_alignment_regret": oracle_regret,
        "oracle_error_regret": oracle_error_regret,
        "oracle_gap_upper_bound_by_time": gap_upper_by_time,
        "oracle_gap_upper_bound_direct_coordinates_by_time": gap_upper_direct_coordinates,
        "oracle_gap_upper_bound_direct_stability_by_time": gap_upper_direct_stability,
        "omega_empirical_prefix": np.concatenate(([math.nan], omega_empirical_prefix)),
        "v_min_prefix": v_min_prefix,
        "omega_theoretical_prefix": omega_theoretical_prefix,
        "log_omega_theoretical_prefix": log_omega_theoretical_prefix,
        "recursive_slack_ratio": recursive_ratio,
        "alignment_direct_slack_ratio": alignment_direct_ratio,
        "alignment_recursive_slack_ratio": alignment_recursive_ratio,
    }
    update = {
        "E_r": E_r,
        "E_g": E_g,
        "E_g_bias": E_g_bias,
        "omega_st": omega_st,
        "omega_oracle": omega_oracle,
        "Xi": Xi,
        "Xi_bound": Xi_bound,
        "Xi_effective_bound": Xi_effective_bound,
        "Xi_bound_ratio": Xi_ratio,
        "h_min": h_min,
        "boundary_mass_at_h_min": boundary_mass,
        "selected_pseudo_label_error_rate": selected_error_rate,
        "forcing_norm": forcing_norm,
        "forcing_formula": forcing_formula,
        "propagation_norm": propagation_norm,
        "actual_selector_weight_discrepancy": actual_selector_weight_discrepancy,
        "selector_weight_bound_empirical": selector_weight_bound_emp,
        "selector_weight_bound_theoretical": selector_weight_bound_th,
        "propagation_bound_empirical": propagation_bound_emp,
        "propagation_bound_theoretical": propagation_bound_th,
        "residual_bound_empirical": residual_bound_emp,
        "residual_bound_theoretical": residual_bound_th,
        "decomposition_error": decomposition_error,
        "forcing_identity_error": forcing_identity_error,
        "bias_forcing_norm": bias_forcing_norm,
        "bias_forcing_formula": bias_forcing_formula,
        "bias_propagation_norm": bias_propagation_norm,
        "bias_decomposition_error": bias_decomposition_error,
        "bias_forcing_identity_error": bias_forcing_identity_error,
        "bias_selector_weight_bound_empirical": bias_selector_weight_bound_emp,
        "bias_selector_weight_bound_theoretical": bias_selector_weight_bound_th,
        "bias_propagation_bound_empirical": bias_propagation_bound_emp,
        "bias_propagation_bound_theoretical": bias_propagation_bound_th,
        "bias_residual_bound_empirical": bias_residual_bound_emp,
        "bias_residual_bound_theoretical": bias_residual_bound_th,
        "forward_rhs": forward_rhs,
        "bias_rhs": bias_rhs,
        "weight_rhs": weight_rhs,
        "forward_slack_ratio": forward_ratio,
        "bias_slack_ratio": bias_ratio,
        "weight_slack_ratio": weight_ratio,
        "propagation_slack_ratio": prop_ratio,
        "residual_slack_ratio": residual_ratio,
        "bias_propagation_slack_ratio": bias_prop_ratio,
        "bias_residual_slack_ratio": bias_residual_ratio,
    }
    constants = {
        "n": env.n,
        "d": env.d,
        "delta": env.delta,
        "rho_empirical": rho,
        "sigma": sigma,
        "kappa": kappa,
        "mu_norm_d": mu_norm,
        "U_operator_norm_exact": operator_norm,
        "a_d": a_d,
        "b_d": b_d,
        "c_w": c_w,
        "A_d": A_d,
        "loss_gradient_lipschitz_exact": 0.25,
        "loss_gradient_bound_exact": 1.0,
        "penalty_gradient_lipschitz_exact": penalty_gradient_lipschitz,
        "omega_lower_empirical_full_horizon": omega_empirical,
        "v_min_state_evolution_full_horizon": (
            float(v_min_prefix[T - 1]) if T else math.nan
        ),
        "omega_lower_theoretical_full_horizon": omega_theoretical,
        "log_omega_lower_theoretical_full_horizon": (
            float(log_omega_theoretical_prefix[T - 1]) if T else math.nan
        ),
        "operator_norm_method": "exact torch.linalg.matrix_norm(U, ord=2)",
        "recursive_bound_kind": "post-hoc finite-dimensional bound using observed empirical omega floor",
        "recursive_horizon_mode": recursive_horizon_mode,
        "gap_bound_mode": gap_bound_mode,
        "include_bias": cfg.include_bias,
        "bias_pseudo_label_weight": (
            float(bias_pi_schedule[0]) if T else math.nan
        ),
    }
    positive_gap_ratio = oracle_gap_upper / oracle_gap if oracle_gap > 0.0 else math.nan
    summary = {
        "exact_checks": exact_checks,
        "all_finite_inequalities_satisfied": all(exact_checks.values()),
        "self_training_gain": self_training_gain,
        "oracle_gain": oracle_gain,
        "oracle_gap": oracle_gap,
        "alignment_gap": alignment_gap,
        "oracle_gap_upper_bound": oracle_gap_upper,
        "oracle_gap_upper_bound_direct_coordinates": float(
            np.nanmin(gap_upper_direct_coordinates)
        ),
        "oracle_gap_upper_bound_direct_stability": float(
            np.nanmin(gap_upper_direct_stability)
        ),
        "oracle_gap_upper_bound_ratio": positive_gap_ratio,
        "best_comparison_time": best_comparison_time,
        "bound_smaller_than_trivial_one": oracle_gap_upper < 1.0,
        "bound_certifies_self_training_gain": oracle_gap_upper < oracle_gain,
        "largest_loss_stage": largest_loss_stage,
        "stage_max_slack_ratio": stage_max_ratio,
        "st_and_oracle_separate_after_initialization": bool(
            T == 0 or torch.linalg.vector_norm(st_weights[-1] - oracle_weights[-1]) > tolerance
        ),
    }
    notes = (
        "The logistic constants 1/4 and 1 are exact for ell'(y,r)=-y/(1+exp(yr)).",
        "The ridge constant Lip(j')=1 is exact because j'(w)=w.",
        "The empirical omega floor is realised-data dependent and is not a deterministic theoretical lower bound.",
        "The state-evolution omega floor, when available, uses the fresh innovation scales and is an asymptotic high-probability device, not a deterministic finite-sample guarantee.",
        "When the theoretical omega floor underflows in float64, its logarithm is retained and the corresponding ordinary-scale propagation bound is reported as infinity.",
        "The recursive bound re-minimizes the explicit finite selector inequality at radius A_d*B_t; no abstract selector constants are assigned numerical values.",
        "When the bias residual differs from the weight residual, its forcing and propagation terms enter the Z recursion separately.",
        "For a learned bias, the oracle-gap certificate uses the balanced population error as a Lipschitz function of both normalized alignment and normalized bias.",
    )
    return OracleDiscrepancyDiagnostic(
        st_learner=st,
        oracle_learner=oracle,
        supervised_learner=supervised,
        state=state,
        update=update,
        constants=constants,
        summary=summary,
        hard_selector_bounds=tuple(selector_bounds),
        notes=notes,
    )


def plot_oracle_discrepancy_diagnostic(
    diagnostic: OracleDiscrepancyDiagnostic,
    *,
    title: Optional[str] = None,
    show: bool = True,
):
    """Create presentation-ready trajectory and cumulative-slack plots."""

    state, update = diagnostic.state, diagnostic.update
    state_time = np.arange(state["E_w"].size)
    update_time = np.arange(update["E_r"].size)
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(state_time, state["E_w"], label=r"$E_w^t$")
    ax.plot(state_time, state["E_b"], label=r"$E_b^t$")
    ax.plot(update_time, update["E_r"], label=r"$E_r^t$")
    ax.plot(update_time, update["E_g"], label=r"$E_g^t$")
    ax.set(title="Actual trajectory discrepancies", xlabel="iteration", ylabel="discrepancy")

    ax = axes[0, 1]
    ax.plot(update_time, update["forcing_formula"], label=r"forcing $\|e_{\rm PL}^t\|_n$")
    ax.plot(update_time, update["propagation_norm"], label=r"actual $\|e_{\rm prop}^t\|_n$")
    ax.plot(update_time, update["propagation_bound_empirical"], "--", label="propagation bound")
    ax.set(title="Forcing and propagation", xlabel="iteration", ylabel="residual norm")

    ax = axes[0, 2]
    ax.plot(state_time, state["Z"], label=r"actual $Z_t$")
    ax.plot(state_time, state["recursive_bound"], "--", label=r"post-hoc $\widehat B_t$")
    ax.set(title="Closed discrepancy recursion", xlabel="iteration", ylabel="bound")

    ax = axes[1, 0]
    ax.plot(state_time, state["absolute_alignment_discrepancy"], label="actual absolute gap")
    ax.plot(state_time, state["alignment_direct_bound"], "--", label=r"direct $E_w^t$ bound")
    ax.plot(state_time, state["alignment_recursive_bound"], ":", label="fully recursive bound")
    ax.set(title="Normalized-alignment discrepancy", xlabel="iteration", ylabel="alignment")

    ax = axes[1, 1]
    ax.plot(update_time, update["Xi"], label=r"actual $\Xi_t$")
    ax.plot(update_time, update["Xi_bound"], "--", label=r"$\inf_h\Xi_t^{\rm bd}(h)$")
    ax.set(title="Hard-selector discrepancy", xlabel="iteration", ylabel="rate")

    ax = axes[1, 2]
    slack_series = {
        "forward": update["forward_slack_ratio"],
        "weight": update["weight_slack_ratio"],
        "selector": update["Xi_bound_ratio"],
        "propagation": update["propagation_slack_ratio"],
        "recursion": state["recursive_slack_ratio"][1:],
        "alignment": state["alignment_recursive_slack_ratio"][1:],
    }
    for label, values in slack_series.items():
        finite = np.where(np.isfinite(values), values, np.nan)
        ax.plot(np.arange(finite.size), finite, label=label)
    ax.set(title="Cumulative slack factors", xlabel="iteration", ylabel="right side / left side")

    for ax in axes.flat:
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(fontsize=8)
        values = np.concatenate([np.asarray(line.get_ydata(), dtype=float) for line in ax.lines])
        positive = values[np.isfinite(values) & (values > 0.0)]
        if positive.size and positive.max() / positive.min() > 1e3:
            ax.set_yscale("symlog", linthresh=max(1e-14, positive.min()))
    if title:
        fig.suptitle(title, fontsize=15)
    if show:
        plt.show()
    return fig, axes


def format_oracle_discrepancy_report(diagnostic: OracleDiscrepancyDiagnostic) -> str:
    """Return a compact factual non-vacuity report for scripts/notebooks."""

    summary = diagnostic.summary
    lines = [
        "Finite ST/oracle discrepancy diagnostic",
        f"all intended finite inequalities satisfied: {summary['all_finite_inequalities_satisfied']}",
        f"actual oracle gap: {summary['oracle_gap']:.6g}",
        f"smallest oracle-gap upper bound: {summary['oracle_gap_upper_bound']:.6g}",
        f"minimizing comparison time: {summary['best_comparison_time']}",
        f"bound below trivial upper bound 1: {summary['bound_smaller_than_trivial_one']}",
        f"oracle gain: {summary['oracle_gain']:.6g}",
        f"bound certifies positive self-training gain: {summary['bound_certifies_self_training_gain']}",
        f"largest measured slack stage: {summary['largest_loss_stage']}",
    ]
    if summary["oracle_gap"] > 0.0:
        lines.append(
            f"upper-bound / positive oracle-gap ratio: {summary['oracle_gap_upper_bound_ratio']:.6g}"
        )
    else:
        lines.append("upper-bound / oracle-gap ratio: not reported because the oracle gap is nonpositive")
    return "\n".join(lines)

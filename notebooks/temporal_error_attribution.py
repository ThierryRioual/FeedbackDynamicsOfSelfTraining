"""Reusable finite-GD experiment and plotting helpers for notebook 13."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from scipy.special import logit

from notebooks.experiment_helpers import (
    ExperimentRun,
    compute_mechanism_diagnostics,
    make_algorithm_config,
    plot_error_diagnostics,
    run_experiment,
)


INTERACTION_RATIO_TOLERANCE = 1e-12
SIGN_TOLERANCE = 1e-12
MECHANISM_TOLERANCE = 1e-12


@dataclass(frozen=True)
class TemporalExperimentParameters:
    """Parameters for one finite-GD temporal-attribution experiment.

    Use ``kappa`` for a symmetric threshold.  For a future asymmetric run, set
    ``kappa=None`` and provide both signed boundaries ``kappa_neg < 0`` and
    ``kappa_pos > 0``.
    """

    name: str = "temporal error-attribution demo"
    seed: int = 49
    T: int = 100
    d: int = 2_000
    delta: float = 8.0
    rho: float = 0.01
    label_prior: float = 0.2
    sigma: float = 1.0
    signal_std: float = 1.0
    signal_mean: float = 0.0
    initialization_correlation: float = 0.0
    eta: float = 0.5
    penalty: float = 0.1
    pi: float = 2.0
    kappa: Optional[float] = float(logit(0.8))
    kappa_pos: Optional[float] = None
    kappa_neg: Optional[float] = None
    include_bias: bool = True
    initial_bias: float = 0.0
    bias_pseudo_label_param: Optional[float] = None
    normalized_threshold: bool = False  # Thresholds apply to r / tau when enabled.
    normalize_unlabeled_loss: bool = True

    def __post_init__(self) -> None:
        symmetric = self.kappa is not None
        asymmetric = self.kappa_pos is not None or self.kappa_neg is not None
        if symmetric and asymmetric:
            raise ValueError(
                "use either kappa or the pair (kappa_neg, kappa_pos), not both"
            )
        if not symmetric and (self.kappa_pos is None or self.kappa_neg is None):
            raise ValueError(
                "when kappa is None, both kappa_neg and kappa_pos are required"
            )


@dataclass
class TemporalPlotResult:
    """Numerical diagnostics and figure handles returned by the plot suite."""

    diagnostics: dict[str, np.ndarray]
    mechanism_diagnostics: dict[str, np.ndarray]
    figures: dict[str, Figure]
    interaction_ratio: np.ndarray


def run_temporal_experiment(
    parameters: TemporalExperimentParameters,
) -> ExperimentRun:
    """Run the finite gradient-descent trajectory used by the plot suite."""

    algorithm_config = make_algorithm_config(
        T=parameters.T,
        eta=parameters.eta,
        penalty=parameters.penalty,
        pi=parameters.pi,
        kappa=parameters.kappa,
        kappa_pos=parameters.kappa_pos,
        kappa_neg=parameters.kappa_neg,
        normalized_threshold=parameters.normalized_threshold,
        include_bias=parameters.include_bias,
        initial_bias=parameters.initial_bias,
        bias_pseudo_label_param=parameters.bias_pseudo_label_param,
        normalize_unlabeled_loss=parameters.normalize_unlabeled_loss
    )
    return run_experiment(
        name=parameters.name,
        d=parameters.d,
        delta=parameters.delta,
        n_test=0,
        label_prior=parameters.label_prior,
        rho=parameters.rho,
        sigma=parameters.sigma,
        signal_std=parameters.signal_std,
        signal_mean=parameters.signal_mean,
        initialization_correlation=parameters.initialization_correlation,
        algo_cfg=algorithm_config,
        seed=parameters.seed,
        run_state_evolution=False,
    )


def _temporal_turning_point(
    diagnostics: dict[str, np.ndarray],
) -> tuple[Optional[int], Optional[int]]:
    error = np.asarray(diagnostics["error_reconstructed"])
    if error.size < 2 or not np.any(np.isfinite(error)):
        return None, None
    state_index = int(np.nanargmin(error))
    return state_index, min(state_index, error.size - 2)


def _add_zero_line(ax: Any) -> None:
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)


def _add_error_marker(
    ax: Any,
    state_index: int,
    *,
    transition_axis: bool,
    n_transitions: int,
) -> int:
    location = min(state_index, n_transitions - 1) if transition_axis else state_index
    ax.axvline(location, color="red", linestyle="--", linewidth=1.0)
    return location


def _first_positive_transition(
    values: np.ndarray, tolerance: float = SIGN_TOLERANCE
) -> Optional[int]:
    indices = np.flatnonzero(np.isfinite(values) & (values > tolerance))
    return None if indices.size == 0 else int(indices[0])


def _first_tracking_crossing(
    values: np.ndarray, tolerance: float = MECHANISM_TOLERANCE
) -> Optional[int]:
    values = np.asarray(values)
    crossing = np.flatnonzero(
        np.isfinite(values[:-1])
        & np.isfinite(values[1:])
        & (
            ((values[:-1] < -tolerance) & (values[1:] > tolerance))
            | ((values[:-1] > tolerance) & (values[1:] < -tolerance))
        )
    )
    return None if crossing.size == 0 else int(crossing[0])


def _validate_diagnostics(diagnostics: dict[str, np.ndarray]) -> None:
    alignment = np.asarray(diagnostics["alignment_contribution"])
    bias = np.asarray(diagnostics["bias_contribution"])
    increment = np.asarray(diagnostics["error_increment"])
    np.testing.assert_allclose(alignment + bias, increment, atol=1e-14)

    interaction = np.asarray(diagnostics["interaction_contribution"])
    a_old = np.asarray(diagnostics["alignment_contribution_old_bias"])
    a_new = np.asarray(diagnostics["alignment_contribution_new_bias"])
    a_symmetric = np.asarray(diagnostics["alignment_contribution_symmetric"])
    b_old = np.asarray(diagnostics["bias_contribution_old_alignment"])
    b_new = np.asarray(diagnostics["bias_contribution_new_alignment"])
    b_symmetric = np.asarray(diagnostics["bias_contribution_symmetric"])
    np.testing.assert_allclose(a_new - a_old, interaction, atol=1e-14)
    np.testing.assert_allclose(b_new - b_old, interaction, atol=1e-14)
    np.testing.assert_allclose(a_old + b_new, increment, atol=1e-14)
    np.testing.assert_allclose(b_old + a_new, increment, atol=1e-14)
    np.testing.assert_allclose(a_symmetric + b_symmetric, increment, atol=1e-14)


def _plot_ordering_and_interaction(
    diagnostics: dict[str, np.ndarray], *, title: str
) -> tuple[Figure, np.ndarray]:
    transition = np.arange(np.asarray(diagnostics["error_increment"]).size)
    state_turning_point, marked_transition = _temporal_turning_point(diagnostics)
    a_old = np.asarray(diagnostics["alignment_contribution_old_bias"])
    a_new = np.asarray(diagnostics["alignment_contribution_new_bias"])
    a_symmetric = np.asarray(diagnostics["alignment_contribution_symmetric"])
    b_old = np.asarray(diagnostics["bias_contribution_old_alignment"])
    b_new = np.asarray(diagnostics["bias_contribution_new_alignment"])
    b_symmetric = np.asarray(diagnostics["bias_contribution_symmetric"])
    interaction = np.asarray(diagnostics["interaction_contribution"])
    increment = np.asarray(diagnostics["error_increment"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].plot(transition, a_old, "--", label=r"$A_t^{\mathrm{old}}$")
    axes[0, 0].plot(transition, a_new, ":", label=r"$A_t^{\mathrm{new}}$")
    axes[0, 0].plot(transition, a_symmetric, linewidth=2, label=r"$A_t$")
    axes[0, 0].set(title=r"Alignment terms $A_t$", ylabel=r"$\Delta \mathscr{E}$")
    axes[0, 0].legend()

    axes[0, 1].plot(transition, b_old, "--", label=r"$B_t^{\mathrm{old}}$")
    axes[0, 1].plot(transition, b_new, ":", label=r"$B_t^{\mathrm{new}}$")
    axes[0, 1].plot(transition, b_symmetric, linewidth=2, label=r"$B_t$")
    axes[0, 1].set(title=r"Bias terms $B_t$", ylabel=r"$\Delta \mathscr{E}$")
    axes[0, 1].legend()

    ax = axes[1, 0]
    ax.plot(transition, interaction, color="tab:purple", label=r"$I_t$")
    denominator = np.abs(a_symmetric) + np.abs(b_symmetric)
    ratio = np.full_like(denominator, np.nan)
    valid_ratio = denominator > INTERACTION_RATIO_TOLERANCE
    ratio[valid_ratio] = np.abs(interaction[valid_ratio]) / denominator[valid_ratio]
    ratio_axis = ax.twinx()
    ratio_axis.plot(
        transition, ratio, color="tab:gray", alpha=0.8, label=r"$R_t^{\mathrm{int}}$"
    )
    ax.set(
        title=r"Interaction $I_t$ and ordering sensitivity",
        xlabel=r"transition $t\to t+1$",
        ylabel=r"$I_t$",
    )
    ratio_axis.set_ylabel(r"$R_t^{\mathrm{int}}=|I_t|/(|A_t|+|B_t|)$")
    handles, labels = ax.get_legend_handles_labels()
    ratio_handles, ratio_labels = ratio_axis.get_legend_handles_labels()
    ax.legend(handles + ratio_handles, labels + ratio_labels, loc="best")

    ax = axes[1, 1]
    ax.plot(transition, increment, color="black", linewidth=2, label=r"$\Delta\mathscr{E}_t$")
    ax.plot(transition, a_old + b_new, "--", label=r"$A_t^{\mathrm{old}}+B_t^{\mathrm{new}}$")
    ax.plot(transition, b_old + a_new, ":", label=r"$B_t^{\mathrm{old}}+A_t^{\mathrm{new}}$")
    ax.plot(transition, a_symmetric + b_symmetric, "-.", label=r"$A_t+B_t$")
    ax.set(
        title=r"$\Delta\mathscr{E}_t=A_t+B_t$",
        xlabel=r"transition $t\to t+1$",
        ylabel=r"$\Delta\mathscr{E}_t$",
    )
    ax.legend()

    for ax in axes.flat:
        _add_zero_line(ax)
        if marked_transition is not None:
            ax.axvline(marked_transition, color="red", linestyle="--", linewidth=1.0)
        ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.8)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.6)
    if marked_transition is not None:
        fig.text(
            0.5,
            0.01,
            f"state t_err={state_turning_point}; marked transition "
            f"{marked_transition} to {marked_transition + 1}",
            ha="center",
        )
    fig.suptitle(title, fontsize=15)
    return fig, ratio


def _plot_bias_tracking(
    diagnostic: dict[str, np.ndarray], *, t_err: int, n_transitions: int, title: str
) -> Figure:
    state_index = np.arange(n_transitions + 1)
    transition_index = np.arange(n_transitions)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    axes[0].plot(state_index, diagnostic["normalized_bias"], label=r"$\breve b_t$")
    axes[0].plot(
        state_index,
        diagnostic["optimal_normalized_bias"],
        "--",
        label=r"$\breve b_t^{\star}$",
    )
    axes[0].set(title=r"$\breve b_t$ and $\breve b_t^{\star}$", xlabel=r"state $t$", ylabel=r"$\breve b_t$")
    axes[0].legend()
    axes[1].plot(state_index, diagnostic["signed_bias_tracking_error"], color="tab:purple")
    axes[1].set(title=r"$e_t=\breve b_t-\breve b_t^{\star}$", xlabel=r"state $t$", ylabel=r"$e_t$")
    _add_zero_line(axes[1])
    axes[2].plot(transition_index, diagnostic["normalized_bias_increment"], label=r"$\Delta\breve b_t$")
    axes[2].plot(transition_index, diagnostic["optimal_normalized_bias_increment"], "--", label=r"$\Delta\breve b_t^{\star}$")
    axes[2].plot(transition_index, diagnostic["signed_bias_tracking_error_increment"], ":", label=r"$\Delta e_t$")
    axes[2].set(title=r"$\Delta e_t=\Delta\breve b_t-\Delta\breve b_t^{\star}$", xlabel=r"transition $t\to t+1$", ylabel="one-step increment")
    _add_zero_line(axes[2])
    for index, ax in enumerate(axes):
        _add_error_marker(ax, t_err, transition_axis=index == 2, n_transitions=n_transitions)
        ax.grid(True, alpha=0.3)
    axes[2].legend()
    fig.suptitle(f"Figure A - Bias tracking: {title}")
    return fig


def _plot_residual_moments(
    diagnostic: dict[str, np.ndarray], *, t_err: int, n_transitions: int, title: str
) -> Figure:
    transition = np.arange(n_transitions)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for key, label in (
        ("zeta_plus_labeled", r"$\zeta_{+,L,t}$"),
        ("zeta_minus_labeled", r"$\zeta_{-,L,t}$"),
        ("zeta_plus_unlabeled", r"$\zeta_{+,U,t}$"),
        ("zeta_minus_unlabeled", r"$\zeta_{-,U,t}$"),
    ):
        axes[0].plot(transition, diagnostic[key], label=label)
    axes[0].set(title=r"Group contributions $\zeta_{y,\delta,t}$", xlabel=r"transition $t\to t+1$", ylabel=r"$\zeta_{y,\delta,t}$")
    axes[0].legend()
    axes[1].plot(transition, diagnostic["zeta"], label=r"$\zeta_t$")
    axes[1].plot(transition, diagnostic["chi"], label=r"$\chi_t$")
    axes[1].set(title=r"$\zeta_t=\mathbb{E}_n[g^t]$ and $\chi_t=\mathbb{E}_n[Yg^t]$", xlabel=r"transition $t\to t+1$", ylabel=r"$\zeta_t,\ \chi_t$")
    axes[1].legend()
    for ax in axes:
        _add_zero_line(ax)
        _add_error_marker(ax, t_err, transition_axis=True, n_transitions=n_transitions)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Figure B - Residual moments: {title}")
    return fig


def _plot_selection(
    diagnostic: dict[str, np.ndarray], *, t_err: int, n_transitions: int, title: str
) -> Figure:
    transition = np.arange(n_transitions)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    axes[0].plot(transition, diagnostic["selection_rate_true_plus"], label=r"$\omega_{+,t}$")
    axes[0].plot(transition, diagnostic["selection_rate_true_minus"], label=r"$\omega_{-,t}$")
    axes[0].set(title=r"Selection rates $\omega_{+,t},\ \omega_{-,t}$", xlabel=r"transition $t\to t+1$", ylabel=r"$\omega_{y,t}$")
    axes[1].plot(transition, diagnostic["pseudo_label_error_rate_true_plus"], label=r"$\epsilon_{+,t}$")
    axes[1].plot(transition, diagnostic["pseudo_label_error_rate_true_minus"], label=r"$\epsilon_{-,t}$")
    axes[1].set(title=r"Pseudo-label errors $\epsilon_{+,t},\ \epsilon_{-,t}$", xlabel=r"transition $t\to t+1$", ylabel=r"$\epsilon_{y,t}$")
    axes[2].plot(transition, diagnostic["selected_mass_true_plus"], label=r"$\mathbb{P}_n(Y=+1,\Delta=0,S_\kappa(r^t)=1)$")
    axes[2].plot(transition, diagnostic["selected_mass_true_minus"], label=r"$\mathbb{P}_n(Y=-1,\Delta=0,S_\kappa(r^t)=1)$")
    axes[2].set(title="Selected class masses", xlabel=r"transition $t\to t+1$", ylabel=r"$\mathbb{P}_n(Y=y,\Delta=0,S_\kappa(r^t)=1)$")
    for ax in axes:
        _add_error_marker(ax, t_err, transition_axis=True, n_transitions=n_transitions)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"Figure C - Selection and pseudo-label error: {title}")
    return fig


def _plot_weight_geometry(
    diagnostic: dict[str, np.ndarray], *, t_err: int, n_transitions: int, title: str
) -> Figure:
    state = np.arange(n_transitions + 1)
    transition = np.arange(n_transitions)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(state, diagnostic["signal_alignment_raw"], label=r"$m_t$")
    axes[0, 0].plot(state, diagnostic["noise_scale"], label=r"$\tau_t$")
    axes[0, 0].set(title=r"$m_t$ and $\tau_t$", xlabel=r"state $t$")
    axes[0, 0].legend()
    axes[0, 1].plot(state, diagnostic["orthogonal_weight_energy"], color="tab:green")
    axes[0, 1].set(title=r"$v_t=\tau_t^2-m_t^2/\bar s_\mu^2$", xlabel=r"state $t$", ylabel=r"$v_t$")
    _add_zero_line(axes[0, 1])
    axes[1, 0].plot(state, diagnostic["orthogonal_to_signal_ratio"], color="tab:purple")
    axes[1, 0].set(title=r"$q_{\perp,t}=v_t/m_t^2$ (symlog)", xlabel=r"state $t$", ylabel=r"$q_{\perp,t}$")
    axes[1, 0].set_yscale("symlog", linthresh=1e-2)
    axes[1, 1].plot(transition, diagnostic["orthogonal_weight_energy_increment"], label=r"$\Delta v_t$")
    axes[1, 1].plot(transition, diagnostic["orthogonal_to_signal_ratio_increment"], label=r"$\Delta q_{\perp,t}$")
    axes[1, 1].set(title=r"$\Delta v_t$ and $\Delta q_{\perp,t}$ (symlog)", xlabel=r"transition $t\to t+1$", ylabel="one-step increment")
    axes[1, 1].set_yscale("symlog", linthresh=1e-3)
    axes[1, 1].legend()
    _add_zero_line(axes[1, 1])
    for index, ax in enumerate(axes.flat):
        _add_error_marker(ax, t_err, transition_axis=index == 3, n_transitions=n_transitions)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Figure D - Weight geometry: {title}")
    return fig


def _plot_mechanism_summary(
    diagnostic: dict[str, np.ndarray], *, t_err: int, n_transitions: int, title: str
) -> Figure:
    transition = np.arange(n_transitions)
    state = np.arange(n_transitions + 1)
    selection_difference = diagnostic["selection_rate_true_plus"] - diagnostic["selection_rate_true_minus"]
    pseudo_label_error_difference = diagnostic["pseudo_label_error_rate_true_plus"] - diagnostic["pseudo_label_error_rate_true_minus"]
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), constrained_layout=True)
    series = (
        (state, diagnostic["signed_bias_tracking_error"], r"$e_t=\breve b_t-\breve b_t^{\star}$", True),
        (transition, diagnostic["zeta"], r"$\zeta_t=\mathbb{E}_n[g^t]$", False),
        (transition, diagnostic["chi"], r"$\chi_t=\mathbb{E}_n[Yg^t]$", False),
        (transition, selection_difference, r"$\omega_{+,t}-\omega_{-,t}$", False),
        (transition, pseudo_label_error_difference, r"$\epsilon_{+,t}-\epsilon_{-,t}$", False),
        (transition, diagnostic["alignment_contribution"], r"$A_t$", False),
        (transition, diagnostic["bias_contribution"], r"$B_t$", False),
        (transition, diagnostic["error_increment"], r"$\Delta\mathscr{E}_t=A_t+B_t$", False),
    )
    for ax, (x, values, panel_title, state_axis) in zip(axes.flat, series):
        ax.plot(x, values)
        ax.set(title=panel_title, xlabel=r"state $t$" if state_axis else r"transition $t\to t+1$")
        _add_zero_line(ax)
        _add_error_marker(ax, t_err, transition_axis=not state_axis, n_transitions=n_transitions)
        ax.grid(True, alpha=0.3)

    for ax, values, color, label in (
        (axes[2, 1], diagnostic["alignment_contribution"], "tab:purple", r"$t_{\mathrm{align}}$"),
        (axes[3, 0], diagnostic["bias_contribution"], "tab:orange", r"$t_{\mathrm{bias}}$"),
        (axes[3, 1], diagnostic["error_increment"], "tab:green", r"$t_{\mathrm{err}}$"),
    ):
        first_positive = _first_positive_transition(values)
        if first_positive is not None:
            ax.axvline(first_positive, color=color, linestyle=":", label=f"{label}={first_positive}")
            ax.legend()
    fig.suptitle(f"Figure E - U-shaped error mechanism: {title}")
    return fig


def _print_summary(
    run: ExperimentRun,
    diagnostics: dict[str, np.ndarray],
    mechanism: dict[str, np.ndarray],
    ratio: np.ndarray,
) -> None:
    increment = np.asarray(diagnostics["error_increment"])
    interaction = np.asarray(diagnostics["interaction_contribution"])
    t_err = int(np.nanargmin(diagnostics["error_reconstructed"]))
    t_near = min(t_err, increment.size - 1)
    selection_difference = mechanism["selection_rate_true_plus"] - mechanism["selection_rate_true_minus"]
    pseudo_label_error_difference = mechanism["pseudo_label_error_rate_true_plus"] - mechanism["pseudo_label_error_rate_true_minus"]
    print(f"finite minimum error: {run.finite_minimum_error:.4f} at state {run.finite_minimum_error_iteration}")
    print(f"max |A_t + B_t - Delta E_t|: {np.max(np.abs(diagnostics['alignment_contribution'] + diagnostics['bias_contribution'] - increment)):.2e}")
    print(f"maximum absolute interaction: {np.nanmax(np.abs(interaction)):.3e}")
    if np.any(np.isfinite(ratio)):
        print(f"maximum descriptive interaction ratio: {np.nanmax(ratio):.3e}")
    print(f"first signed tracking-error crossing transition: {_first_tracking_crossing(mechanism['signed_bias_tracking_error'])}")
    print(f"first positive bias-contribution transition: {_first_positive_transition(mechanism['bias_contribution'])}")
    print(f"first positive alignment-contribution transition: {_first_positive_transition(mechanism['alignment_contribution'])}")
    print(f"first positive total-error transition: {_first_positive_transition(increment)}")
    print(f"selection-rate difference near t_err: {selection_difference[t_near]:.3e}")
    print(f"pseudo-label-error-rate difference near t_err: {pseudo_label_error_difference[t_near]:.3e}")


def plot_temporal_experiment(
    run: ExperimentRun,
    *,
    attribution_ylim: Optional[tuple[float, float]] = (-0.02, 0.01),
    print_summary: bool = True,
    show: bool = True,
) -> TemporalPlotResult:
    """Plot the complete temporal-attribution suite for a finite-GD run.

    The function intentionally ignores state-evolution output.  Its return
    value retains both diagnostic arrays and figure handles for downstream
    comparisons across parameter settings.
    """

    if run.callback is None or "finite" not in run.error_diagnostics:
        raise ValueError("plot_temporal_experiment requires a finite-GD run")
    diagnostics = run.error_diagnostics["finite"]
    _validate_diagnostics(diagnostics)
    mechanism = compute_mechanism_diagnostics(
        run, source="finite", geometry_tolerance=MECHANISM_TOLERANCE
    )
    n_transitions = np.asarray(diagnostics["error_increment"]).size
    t_err = int(np.nanargmin(diagnostics["error_reconstructed"]))
    title = run.name

    attribution, _ = plot_error_diagnostics(
        diagnostics,
        title=f"Temporal attribution: {title}",
        show=False,
        ylim=attribution_ylim,
    )
    ordering, ratio = _plot_ordering_and_interaction(
        diagnostics, title=f"Ordering sensitivity: {title}"
    )
    figures = {
        "attribution": attribution,
        "ordering_interaction": ordering,
        "bias_tracking": _plot_bias_tracking(
            mechanism, t_err=t_err, n_transitions=n_transitions, title=title
        ),
        "residual_moments": _plot_residual_moments(
            mechanism, t_err=t_err, n_transitions=n_transitions, title=title
        ),
        "selection": _plot_selection(
            mechanism, t_err=t_err, n_transitions=n_transitions, title=title
        ),
        "weight_geometry": _plot_weight_geometry(
            mechanism, t_err=t_err, n_transitions=n_transitions, title=title
        ),
        "mechanism_summary": _plot_mechanism_summary(
            mechanism, t_err=t_err, n_transitions=n_transitions, title=title
        ),
    }
    if print_summary:
        _print_summary(run, diagnostics, mechanism, ratio)
    if show:
        plt.show()
    return TemporalPlotResult(diagnostics, mechanism, figures, ratio)

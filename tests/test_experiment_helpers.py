"""Tests for notebook experiment-run summaries."""

import numpy as np
import torch

from notebooks.experiment_helpers import (
    compute_error_diagnostics,
    finite_state_observables,
    make_algorithm_config,
    plot_error_diagnostics,
    plot_state_evolution_oracle_bias_error,
    run_experiment,
    run_oracle_selected_label_counterfactual,
    run_state_evolution_oracle_selected_label_counterfactual,
    state_evolution_state_observables,
)
from src.performance import population_error
from src.primitives import pseudo_residual


ERROR_DIAGNOSTIC_KEYS = {
    "normalized_alignment",
    "normalized_bias",
    "optimal_normalized_bias",
    "optimal_linear_error",
    "alignment_regret",
    "bias_regret",
    "error_reconstructed",
    "error_increment",
    "alignment_regret_increment",
    "bias_regret_increment",
}


def assert_error_decomposition_identities(diagnostic, error):
    np.testing.assert_allclose(diagnostic["error_reconstructed"], error)
    np.testing.assert_allclose(
        diagnostic["error_increment"],
        diagnostic["alignment_regret_increment"]
        + diagnostic["bias_regret_increment"],
        atol=1e-14,
    )


def test_balanced_zero_bias_error_diagnostics_reconstruct_and_decompose_error():
    sigma = 1.3
    trajectory = {
        "m": np.array([0.5, 0.9, 0.6]),
        "tau": np.array([1.0, 1.2, 1.1]),
        "bias": np.zeros(3),
    }
    normalized_alignment = trajectory["m"] / trajectory["tau"]
    trajectory["error"] = np.array(
        [population_error(0.0, value, 1.0, sigma, 0.5) for value in normalized_alignment]
    )

    diagnostic = compute_error_diagnostics(
        trajectory, p=0.5, sigma=sigma, s_mu=1.0
    )

    assert set(diagnostic) == ERROR_DIAGNOSTIC_KEYS
    np.testing.assert_allclose(
        diagnostic["normalized_alignment"], normalized_alignment
    )
    np.testing.assert_array_equal(diagnostic["optimal_normalized_bias"], 0.0)
    np.testing.assert_allclose(
        diagnostic["optimal_linear_error"],
        population_error(0.0, 1.0, 1.0, sigma, 0.5),
    )
    np.testing.assert_allclose(diagnostic["error_reconstructed"], trajectory["error"])
    np.testing.assert_allclose(diagnostic["bias_regret"], 0.0, atol=1e-15)
    np.testing.assert_allclose(diagnostic["bias_regret_increment"], 0.0, atol=1e-15)
    np.testing.assert_allclose(
        diagnostic["error_increment"],
        diagnostic["alignment_regret_increment"]
        + diagnostic["bias_regret_increment"],
        atol=1e-15,
    )
    fig, axes = plot_error_diagnostics(diagnostic, show=False)
    assert [len(ax.lines) for ax in axes.flat] == [1, 2, 3]
    fig.clear()


def test_error_diagnostics_normalized_fallback_and_imbalanced_zero_handling():
    diagnostic = compute_error_diagnostics(
        {
            "normalized_alignment": np.array([0.0, 1e-12, 0.5]),
            "normalized_bias": np.array([0.2, 0.2, 0.2]),
        },
        p=0.2,
        sigma=1.0,
        s_mu=1.0,
    )

    assert np.isnan(diagnostic["optimal_normalized_bias"][:2]).all()
    assert np.isfinite(diagnostic["optimal_normalized_bias"][2])
    assert np.isnan(diagnostic["alignment_regret"][:2]).all()
    assert np.isfinite(diagnostic["optimal_linear_error"]).all()
    np.testing.assert_array_equal(
        diagnostic["normalized_alignment"], np.array([0.0, 1e-12, 0.5])
    )

    balanced = compute_error_diagnostics(
        {
            "normalized_alignment": np.array([0.0]),
            "normalized_bias": np.array([0.0]),
        },
        p=0.5,
        sigma=1.0,
        s_mu=0.0,
    )
    np.testing.assert_array_equal(balanced["optimal_normalized_bias"], [0.0])


def test_run_experiment_records_finite_minimum_population_error():
    run = run_experiment(
        name="minimum-error summary",
        d=12,
        delta=2.0,
        n_test=20,
        label_prior=0.5,
        rho=0.5,
        sigma=1.0,
        signal_std=1.0,
        algo_cfg=make_algorithm_config(T=3, eta=0.1, penalty=0.1, pi=0.0, kappa=1.0),
        seed=17,
        run_state_evolution=False,
    )

    error = np.asarray(run.callback.history_["population_error"], dtype=float)
    assert run.finite_minimum_error == np.nanmin(error)
    assert run.finite_minimum_error_iteration == int(np.nanargmin(error))
    assert run.state_evolution_minimum_error is None
    assert run.state_evolution_minimum_error_iteration is None
    assert set(run.error_diagnostics) == {"finite"}
    assert set(run.error_diagnostics["finite"]) == ERROR_DIAGNOSTIC_KEYS
    assert run.callback.error_diagnostics is run.error_diagnostics["finite"]
    assert_error_decomposition_identities(run.error_diagnostics["finite"], error)


def test_notebook_config_parses_fixed_initial_bias_and_bias_pseudo_weight():
    config = make_algorithm_config(
        T=1,
        eta=0.1,
        penalty=0.1,
        pi=1.2,
        kappa=0.5,
        include_bias=False,
        initial_bias=0.4,
        bias_pseudo_label_param=0.0,
    )
    assert config.initial_bias == 0.4
    assert config.get_bias_pseudo_label_weight(0) == 0.0

    run = run_experiment(
        name="fixed nonzero bias",
        d=8,
        delta=2.0,
        n_test=10,
        label_prior=0.5,
        rho=0.5,
        sigma=1.0,
        signal_std=1.0,
        algo_cfg=config,
        seed=31,
        run_state_evolution=False,
    )
    torch.testing.assert_close(run.finite.bias, torch.tensor(0.4))
    torch.testing.assert_close(run.finite.update_records_[0].bias, torch.tensor(0.4))


def test_run_experiment_records_state_evolution_minimum_population_error():
    run = run_experiment(
        name="state-evolution minimum-error summary",
        d=12,
        delta=2.0,
        n_test=20,
        label_prior=0.5,
        rho=0.5,
        sigma=1.0,
        signal_std=1.0,
        algo_cfg=make_algorithm_config(T=3, eta=0.1, penalty=0.1, pi=0.0, kappa=1.0),
        seed=17,
        K_w=31,
        K_g=37,
    )

    error = np.asarray(run.se.error, dtype=float)
    assert run.state_evolution_minimum_error == np.nanmin(error)
    assert run.state_evolution_minimum_error_iteration == int(np.nanargmin(error))
    assert set(run.error_diagnostics) == {"finite", "state_evolution"}
    assert run.callback.error_diagnostics is run.error_diagnostics["finite"]
    assert run.se.error_diagnostics is run.error_diagnostics["state_evolution"]
    assert_error_decomposition_identities(
        run.error_diagnostics["finite"],
        np.asarray(run.callback.history_["population_error"]),
    )
    assert_error_decomposition_identities(
        run.error_diagnostics["state_evolution"], error
    )


def test_state_evolution_oracle_bias_observables_and_plot_are_available():
    run = run_experiment(
        name="oracle-bias diagnostic",
        d=8,
        delta=2.0,
        n_test=10,
        label_prior=0.2,
        rho=0.5,
        sigma=1.0,
        signal_std=1.0,
        algo_cfg=make_algorithm_config(T=2, eta=0.1, penalty=0.1, pi=0.0, kappa=1.0),
        seed=19,
        K_w=31,
        K_g=37,
    )

    finite = finite_state_observables(run)
    state_evolution = state_evolution_state_observables(run)
    np.testing.assert_allclose(
        finite["oracle_bias"], run.callback.history_["oracle_bias"], equal_nan=True
    )
    np.testing.assert_allclose(
        finite["oracle_error"], run.callback.history_["oracle_error"], equal_nan=True
    )
    np.testing.assert_allclose(
        state_evolution["oracle_bias"], run.se.oracle_bias, equal_nan=True
    )
    np.testing.assert_allclose(
        state_evolution["oracle_error"], run.se.oracle_error, equal_nan=True
    )

    fig, ax = plot_state_evolution_oracle_bias_error(run, show=False)
    assert len(ax.lines) == 4
    np.testing.assert_allclose(ax.lines[0].get_ydata(), finite["error"])
    np.testing.assert_allclose(
        ax.lines[1].get_ydata(), finite["oracle_error"], equal_nan=True
    )
    np.testing.assert_allclose(
        ax.lines[2].get_ydata(), state_evolution["error"]
    )
    np.testing.assert_allclose(
        ax.lines[3].get_ydata(), state_evolution["oracle_error"], equal_nan=True
    )
    fig.clear()


def test_finite_oracle_bias_diagnostic_is_nan_at_zero_alignment():
    run = run_experiment(
        name="zero finite alignment",
        d=8,
        delta=2.0,
        n_test=10,
        label_prior=0.2,
        rho=0.5,
        sigma=1.0,
        signal_std=0.0,
        algo_cfg=make_algorithm_config(T=2, eta=0.1, penalty=0.1, pi=0.0, kappa=1.0),
        seed=29,
        run_state_evolution=False,
    )

    finite = finite_state_observables(run)
    np.testing.assert_allclose(finite["m"], 0.0)
    assert np.isfinite(finite["error"]).all()
    assert np.isnan(finite["oracle_bias"]).all()
    assert np.isnan(finite["oracle_error"]).all()


def test_state_evolution_only_oracle_uses_true_selected_labels():
    config = make_algorithm_config(T=2, eta=0.1, penalty=0.1, pi=2.0, kappa=0.5)
    run = run_experiment(
        name="state-evolution-only oracle",
        d=8,
        delta=2.0,
        n_test=20,
        label_prior=0.5,
        rho=0.5,
        sigma=1.0,
        signal_std=1.0,
        algo_cfg=config,
        seed=23,
        K_w=17,
        K_g=19,
        run_finite=False,
    )

    assert run.finite is None
    assert run.callback is None
    oracle = run_state_evolution_oracle_selected_label_counterfactual(run)
    assert run.se is not None
    torch.testing.assert_close(oracle.signal, run.se.signal)
    torch.testing.assert_close(oracle.label, run.se.label)
    torch.testing.assert_close(oracle.indicator, run.se.indicator)

    scores = oracle.preactivation[0]
    selection = config.selection_function(
        scores, config.positive_margin, config.negative_margin
    )
    expected_residual = pseudo_residual(
        scores=scores,
        Y=oracle.label,
        Delta=oracle.indicator,
        Yhat=oracle.label,
        selection=selection,
        omega=oracle.selection_rate[0],
        pi=config.pseudo_label_param,
        eta=config.step_size,
        rho=run.data_cfg.supervision_ratio,
        loss_function=config.loss_function,
    )
    torch.testing.assert_close(oracle.residual[0], expected_residual)
    assert len(oracle.error) == config.n_iterations + 1
    assert set(oracle.error_diagnostics) == ERROR_DIAGNOSTIC_KEYS


def test_finite_selected_label_oracle_receives_postprocessing_diagnostics():
    config = make_algorithm_config(T=1, eta=0.1, penalty=0.1, pi=1.0, kappa=0.5)
    run = run_experiment(
        name="finite oracle source",
        d=8,
        delta=2.0,
        n_test=10,
        label_prior=0.5,
        rho=0.5,
        sigma=1.0,
        signal_std=1.0,
        algo_cfg=config,
        seed=37,
        run_state_evolution=False,
    )

    _, oracle_callback = run_oracle_selected_label_counterfactual(run)

    assert set(oracle_callback.error_diagnostics) == ERROR_DIAGNOSTIC_KEYS
    assert oracle_callback.error_diagnostics["normalized_alignment"].shape == (2,)
    assert oracle_callback.error_diagnostics["error_increment"].shape == (1,)

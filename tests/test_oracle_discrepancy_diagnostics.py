"""Focused tests for the additive finite ST/oracle bound diagnostic."""

import math

import numpy as np
import pytest
import torch

from notebooks.experiment_helpers import make_algorithm_config, run_experiment
from notebooks.oracle_discrepancy_diagnostics import (
    hard_selector_bound,
    run_finite_oracle_discrepancy_diagnostic,
)


@pytest.fixture(scope="module")
def paired_diagnostic():
    config = make_algorithm_config(
        T=4,
        eta=0.2,
        penalty=0.1,
        pi=2.0,
        kappa=0.5,
        include_bias=False,
    )
    run = run_experiment(
        name="paired discrepancy test",
        d=24,
        delta=4.0,
        n_test=32,
        label_prior=0.5,
        rho=0.25,
        sigma=1.0,
        signal_std=1.0,
        algo_cfg=config,
        seed=9182,
        run_state_evolution=False,
    )
    original_weights = [value.clone() for value in run.finite.weight_history_]
    diagnostic = run_finite_oracle_discrepancy_diagnostic(run, grid_size=41)
    return run, original_weights, diagnostic


def test_paired_trajectories_share_initialization_then_evolve_independently(paired_diagnostic):
    run, _, diagnostic = paired_diagnostic
    st = diagnostic.st_learner
    oracle = diagnostic.oracle_learner

    torch.testing.assert_close(st.weight_history_[0], oracle.weight_history_[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(st.update_records_[0].bias, oracle.update_records_[0].bias, rtol=0.0, atol=0.0)
    torch.testing.assert_close(st.initialization_.Y_init, oracle.initialization_.Y_init, rtol=0.0, atol=0.0)
    assert st.environment_ is run.environment
    assert oracle.environment_ is run.environment

    # The oracle is a complete trajectory: after the shared state at t=0 its
    # own true-label residual changes its iterate, and later scores/selections
    # are evaluated from that oracle iterate.
    assert diagnostic.summary["st_and_oracle_separate_after_initialization"]
    assert not torch.equal(st.weight_history_[-1], oracle.weight_history_[-1])
    oracle_scores_t1 = run.X @ oracle.weight_history_[1] / math.sqrt(run.environment.d)
    torch.testing.assert_close(oracle.update_records_[1].scores, oracle_scores_t1)


def test_residual_decomposition_and_logistic_forcing_identity_are_exact(paired_diagnostic):
    _, _, diagnostic = paired_diagnostic
    np.testing.assert_allclose(diagnostic.update["decomposition_error"], 0.0, atol=1e-14)
    np.testing.assert_allclose(
        diagnostic.update["forcing_norm"],
        diagnostic.update["forcing_formula"],
        rtol=1e-13,
        atol=1e-14,
    )
    assert diagnostic.summary["exact_checks"]["residual_decomposition"]
    assert diagnostic.summary["exact_checks"]["logistic_forcing_identity"]


def test_all_elementary_and_computable_bound_inequalities_hold(paired_diagnostic):
    _, _, diagnostic = paired_diagnostic
    checks = diagnostic.summary["exact_checks"]
    requested = {
        "forward_inequality",
        "bias_inequality",
        "weight_inequality",
        "selector_inequality",
        "selector_weight_inequality_empirical",
        "propagation_inequality_empirical",
        "residual_inequality_empirical",
        "recursive_inequality_empirical",
        "alignment_inequality_direct",
        "alignment_inequality_recursive",
    }
    assert requested <= checks.keys()
    assert all(checks[name] for name in requested)
    assert np.isfinite(diagnostic.summary["oracle_gap_upper_bound"])


def test_h_grid_contains_theoretical_scale_and_bounds_selector_discrepancy():
    st_scores = torch.tensor([-0.8, -0.45, 0.1, 0.55, 1.0], dtype=torch.float64)
    oracle_scores = torch.tensor([-0.4, -0.55, 0.1, 0.45, 0.7], dtype=torch.float64)
    radius = float(torch.linalg.vector_norm(st_scores - oracle_scores) / math.sqrt(5))
    result = hard_selector_bound(
        st_scores=st_scores,
        oracle_scores=oracle_scores,
        unlabeled_mask=torch.ones(5, dtype=torch.bool),
        kappa=0.5,
        discrepancy_radius=radius,
        grid_size=41,
    )

    assert np.any(np.isclose(result.grid, radius ** (2.0 / 3.0), rtol=1e-14, atol=0.0))
    assert result.actual_discrepancy <= result.minimized_bound + 1e-14
    assert result.effective_bound <= 1.0
    assert result.minimizing_h > 0.0


def test_diagnostic_is_additive_and_preserves_existing_finite_result(paired_diagnostic):
    run, original_weights, diagnostic = paired_diagnostic
    assert len(run.finite.weight_history_) == len(original_weights)
    for before, after in zip(original_weights, run.finite.weight_history_):
        torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
    assert diagnostic.st_learner is run.finite
    assert isinstance(diagnostic.state, dict)
    assert isinstance(diagnostic.update, dict)


def test_state_evolution_innovation_floor_is_recorded_when_available():
    config = make_algorithm_config(
        T=2,
        eta=0.2,
        penalty=0.1,
        pi=2.0,
        kappa=0.5,
        include_bias=False,
    )
    run = run_experiment(
        name="innovation-floor test",
        d=16,
        delta=4.0,
        n_test=20,
        label_prior=0.5,
        rho=0.25,
        sigma=1.0,
        signal_std=1.0,
        algo_cfg=config,
        seed=9182,
        K_w=31,
        K_g=37,
    )
    diagnostic = run_finite_oracle_discrepancy_diagnostic(run, grid_size=31)

    assert diagnostic.constants["v_min_state_evolution_full_horizon"] > 0.0
    omega = diagnostic.constants["omega_lower_theoretical_full_horizon"]
    assert 0.0 < omega < 0.5
    assert np.isfinite(diagnostic.update["propagation_bound_theoretical"]).all()


def test_learned_labelled_only_bias_uses_separate_bias_residual_bound():
    config = make_algorithm_config(
        T=4,
        eta=0.2,
        penalty=0.1,
        pi=2.0,
        kappa=0.5,
        include_bias=True,
        initial_bias=0.0,
        bias_pseudo_label_param=0.0,
    )
    run = run_experiment(
        name="learned labelled-only bias",
        d=24,
        delta=4.0,
        n_test=0,
        label_prior=0.5,
        rho=0.25,
        sigma=1.0,
        signal_std=1.0,
        algo_cfg=config,
        seed=9182,
        run_state_evolution=False,
    )
    diagnostic = run_finite_oracle_discrepancy_diagnostic(run, grid_size=41)

    assert diagnostic.constants["include_bias"]
    assert diagnostic.constants["bias_pseudo_label_weight"] == 0.0
    assert diagnostic.constants["gap_bound_mode"] == (
        "balanced learned-bias population-error Lipschitz bound"
    )
    np.testing.assert_allclose(diagnostic.update["bias_forcing_formula"], 0.0)
    np.testing.assert_allclose(diagnostic.update["bias_forcing_norm"], 0.0, atol=1e-14)
    assert np.any(np.abs(diagnostic.update["E_g"] - diagnostic.update["E_g_bias"]) > 1e-12)
    assert diagnostic.summary["exact_checks"]["bias_inequality"]
    assert diagnostic.summary["exact_checks"]["bias_residual_decomposition"]
    assert diagnostic.summary["exact_checks"]["bias_residual_inequality_empirical"]
    assert diagnostic.summary["exact_checks"]["normalized_bias_inequality_recursive"]
    assert np.isfinite(diagnostic.summary["oracle_gap_upper_bound"])

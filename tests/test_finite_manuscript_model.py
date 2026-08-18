"""Direct finite-dimensional checks for the manuscript-consistent core."""

import math

import pytest
import torch

from src.algorithms import SelfTrainedGradientDescent
from src.config import AlgorithmConfig, DataConfig, LinearRampSchedule
from src.dgp import IsotropicGaussian
from src.environment import FourCellSampleTypeLaw, QuenchedEnvironment
from src.initialization import SelfTrainingInitialization, compute_pseudo_labels_from_scores, compute_scores
from src.performance import bayes_parameters, population_error
from src.primitives import hard_selection, normalized_selection, pseudo_labels, selection_rate


def _cfg(T=2, pi=1.5):
    return AlgorithmConfig(T, .2, .3, pi, margin_threshold=.5)


def _problem():
    torch.manual_seed(7)
    d, n, sigma = 3, 5, .7
    mu = torch.tensor([1.0, -2.0, .5])
    Y = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0])
    Delta = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
    U = torch.randn(n, d)
    env = QuenchedEnvironment(mu, Y, Delta)
    X = Y[:, None] * mu / math.sqrt(d) + sigma * U
    init = SelfTrainingInitialization(.1, torch.tensor([.4, -.3, .2]), torch.tensor([1., 1., 1., -1., 1.]))
    return env, X, U, sigma, init


def test_environment_and_four_cell_law_are_joint_not_product():
    env = QuenchedEnvironment(torch.ones(2), torch.tensor([1., -1., 1., -1.]), torch.tensor([1., 0., 1., 0.]))
    assert (env.d, env.n, env.N, env.M, env.delta, env.rho) == (2, 4, 2, 2, 2.0, .5)
    torch.testing.assert_close(env.I_L, torch.tensor([0, 2]))
    law = FourCellSampleTypeLaw((.5, 0., 0., .5))
    Y, Delta = law.sample(200, generator=torch.Generator().manual_seed(2), dtype=torch.float64, device=torch.device("cpu"))
    assert torch.all(Y == (2 * Delta - 1))


def test_gaussian_design_is_conditional_on_environment_not_delta():
    cfg = DataConfig(.8, .4, .5, 2., lambda: 0.)
    mu = torch.tensor([1., -1.])
    first = QuenchedEnvironment(mu, torch.tensor([1., -1., 1.]), torch.tensor([1., 0., 1.]))
    second = QuenchedEnvironment(mu, first.Y, 1 - first.Delta)
    dgp = IsotropicGaussian(cfg, 3, 0, 2, signal_vector=mu)
    U = torch.tensor([[.2, -.1], [.3, .4], [-.5, .1]])
    X_first, _ = dgp.sample_design(first, noise=U)
    X_second, _ = dgp.sample_design(second, noise=U)
    torch.testing.assert_close(X_first, X_second)


def test_initialization_and_pseudo_label_timing_are_explicit():
    env, X, _, _, init = _problem()
    init.for_environment(env)
    with pytest.raises(ValueError, match="labelled"):
        SelfTrainingInitialization(0., torch.zeros(env.d), -init.Y_init).for_environment(env)
    scores = torch.tensor([0., -.1])
    torch.testing.assert_close(compute_pseudo_labels_from_scores(scores), torch.tensor([1., -1.]))
    torch.testing.assert_close(pseudo_labels(0, env.Y, init.Y_init), init.Y_init)
    torch.testing.assert_close(pseudo_labels(1, torch.tensor([-.2, 0.]), init.Y_init[:2]), torch.tensor([-1., 1.]))
    with pytest.raises(ValueError, match="initial_pseudo_labels"):
        SelfTrainedGradientDescent(_cfg(T=1)).fit(X[env.I_L], env.Y[env.I_L], X[env.I_U], initial_weights=init.w_init)


def test_selection_has_inclusive_boundaries_and_zero_contribution():
    scores = torch.tensor([-.5, .5, 0.])
    mask = hard_selection(scores, -.5, .5)
    torch.testing.assert_close(mask, torch.tensor([1., 1., 0.]))
    Delta = torch.tensor([0., 0., 1.])
    omega = selection_rate(mask, Delta)
    assert omega.item() == 1.
    torch.testing.assert_close(normalized_selection(torch.zeros_like(mask), 0.), torch.zeros_like(mask))


def test_fixed_pi_is_canonical_and_ramps_are_explicit_extensions():
    fixed = _cfg(T=3, pi=.8)
    assert fixed.is_canonical_fixed_pi
    assert [fixed.get_pseudo_label_weight(t) for t in range(3)] == [.8, .8, .8]
    ramped = AlgorithmConfig(3, .1, .0, .8, margin_threshold=.5, experimental_schedule=LinearRampSchedule(1, 2))
    assert not ramped.is_canonical_fixed_pi
    assert [ramped.get_pseudo_label_weight(t) for t in range(3)] == [0., 0., .8]


def test_finite_update_matches_objective_and_forward_backward_decomposition():
    env, X, U, sigma, init = _problem()
    cfg = _cfg(T=2)
    learner = SelfTrainedGradientDescent(cfg).fit_full(X, env, init)
    step = learner.update_records_[0]
    expected_yhat0 = init.Y_init
    torch.testing.assert_close(step.pseudo_labels, expected_yhat0)
    torch.testing.assert_close(learner.update_records_[1].pseudo_labels, compute_pseudo_labels_from_scores(learner.update_records_[1].scores))

    # Explicit finite objective gradient with frozen Yhat, S, omega.
    grad_lab = cfg.loss_function.gradient(step.scores, env.Y) * env.Delta / env.rho
    grad_unl = cfg.loss_function.gradient(step.scores, expected_yhat0) * (1-env.Delta) * cfg.pseudo_label_param / (1-env.rho) * step.selection / step.omega
    torch.testing.assert_close(step.g, -cfg.step_size * (grad_lab + grad_unl))
    next_w = step.weight - cfg.step_size * cfg.penalty_param * cfg.penalty_function.gradient(step.weight) + math.sqrt(env.d)/env.n * X.T @ step.g
    torch.testing.assert_close(learner.weight_history_[1], next_w)
    decomp = learner.finite_decomposition(X, U, 0, sigma=sigma)
    torch.testing.assert_close(decomp["r"], decomp["direct_score"], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(decomp["noise_update"], decomp["direct_update_noise"], atol=1e-12, rtol=1e-12)
    obs = learner.macroscopic_observables(0)
    torch.testing.assert_close(obs["chi"], step.chi)
    torch.testing.assert_close(obs["zeta"], step.zeta)


def test_bayes_benchmark_uses_actual_signal_scale():
    scale, sigma, p = 2., 1.3, .2
    b, m, tau = bayes_parameters(scale, sigma, p)
    assert m == scale**2 and tau == scale
    assert b == pytest.approx(sigma**2 / 2 * math.log(p / (1 - p)))
    assert population_error(b, m, tau, sigma, p) < population_error(0., m, tau, sigma, p)

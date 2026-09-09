"""Normalized confidence selection across finite and particle trajectories."""
from dataclasses import asdict

import pytest
import torch

from notebooks.experiment_helpers import (
    make_algorithm_config, run_experiment,
    run_oracle_selected_label_counterfactual,
    run_state_evolution_oracle_selected_label_counterfactual,
    class_conditional_update_observables,
)
from src.config import AlgorithmConfig
from src.objectives import HardSelection, LipschitzSelection, SmoothSelection


@pytest.mark.parametrize('selector', [HardSelection(), LipschitzSelection(), SmoothSelection()])
def test_disabled_selector_is_exact_and_old_config_loads(selector):
    cfg = AlgorithmConfig(2, .1, .1, .5, positive_margin=.8, negative_margin=-.4,
                          selection_function=selector)
    scores = torch.tensor([-2., -.4, -.3, 0., .7, .8, 2.])
    assert torch.equal(cfg.selection_mask(scores), selector(scores, .8, -.4))
    assert torch.equal(cfg.selection_mask(scores, float('nan')), selector(scores, .8, -.4))
    saved = asdict(cfg)
    saved.pop('normalized_threshold')
    saved.pop('pseudo_label_param_schedule_')
    assert not AlgorithmConfig(**saved).normalized_threshold


def test_scale_invariance_and_raw_threshold_behavior():
    X = torch.tensor([[1., 0.], [-1., 2.], [0., 0.], [2., 1.]])
    w, b = torch.tensor([.4, -.2]), .15
    scores = b + X @ w / 2**.5
    tau = torch.linalg.vector_norm(w) / 2**.5
    cfg = AlgorithmConfig(1, .1, 0., 1., positive_margin=.8, negative_margin=-.4,
                          normalized_threshold=True)
    mask = cfg.selection_mask(scores, tau)
    assert torch.equal(mask, ((scores/tau <= -.4) | (scores/tau >= .8)).double())
    for c in [.01, 3., 100.]:
        scaled_scores = c*b + X @ (c*w) / 2**.5
        assert torch.equal(mask, cfg.selection_mask(scaled_scores, c*tau))
    raw = AlgorithmConfig(1, .1, 0., 1., positive_margin=.8, negative_margin=-.4)
    assert not torch.equal(raw.selection_mask(scores), raw.selection_mask(3*scores))
    assert torch.equal(raw.selection_mask(3*scores), ((3*scores <= -.4) | (3*scores >= .8)).double())


@pytest.mark.parametrize('tau', [0., 1e-320])
def test_degenerate_norm_selects_nothing(tau):
    cfg = AlgorithmConfig(1, .1, 0., 1., margin_threshold=.5, normalized_threshold=True)
    result = cfg.selection_mask(torch.tensor([-1., 0., 1.]), tau)
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result) == 0


@pytest.mark.parametrize('normalized', [False, True])
def test_small_experiment_and_oracles(normalized):
    cfg = make_algorithm_config(T=2, eta=.1, penalty=.1, pi=.4,
                                kappa_pos=.7, kappa_neg=-.3,
                                **({"normalized_threshold": True} if normalized else {}))
    run = run_experiment(name='normalized threshold verification', d=24, delta=3.,
                         n_test=24, label_prior=.4, rho=.4, sigma=1.7,
                         signal_std=1., algo_cfg=cfg, seed=17, K_w=64, K_g=96)
    oracle, _ = run_oracle_selected_label_counterfactual(run)
    oracle_se = run_state_evolution_oracle_selected_label_counterfactual(run)
    for learner in [run.finite, oracle]:
        for step in learner.update_records_:
            q = step.scores / step.tau if normalized else step.scores
            expected = ((q <= -.3) | (q >= .7)).double() * (1-run.environment.Delta)
            assert torch.equal(step.selection, expected)
            assert torch.isfinite(step.g).all()
    for se in [run.se, oracle_se]:
        for t in range(2):
            r = se.preactivation[t]
            tau = se.compute_weight_norm(t)
            q = r/tau if normalized else r
            expected = ((q <= -.3) | (q >= .7)).double()
            assert torch.equal(se.selection_mask(r, t), expected)
            # Same scores and norm must give the finite configuration's mask.
            assert torch.equal(cfg.selection_mask(r, tau), expected)
            torch.testing.assert_close(se.selection_rate[t], ((1-se.indicator)*expected).mean()/(1-se.rho))
            assert torch.isfinite(se.residual[t]).all()
    for source in ['finite', 'state_evolution']:
        diagnostics = class_conditional_update_observables(run, source=source)
        assert diagnostics['positive']['coverage'].shape == (2,)


def test_normalized_surrogate_freezes_norm_for_score_derivatives():
    cfg = AlgorithmConfig(1, .1, 0., 1., margin_threshold=.5,
                          selection_function=LipschitzSelection(.1),
                          normalized_threshold=True)
    scores = torch.tensor([1.], requires_grad=True)
    tau = torch.tensor(2., requires_grad=True)
    mask = cfg.selection_mask(scores, tau)
    score_grad, norm_grad = torch.autograd.grad(mask.sum(), (scores, tau), allow_unused=True)
    torch.testing.assert_close(score_grad, torch.tensor([2.5]))
    assert norm_grad is None


def test_symmetric_temporal_runner_propagates_option():
    from notebooks.temporal_error_attribution import TemporalExperimentParameters, run_temporal_experiment
    parameters = TemporalExperimentParameters(T=2, d=24, delta=3., rho=.4,
                                              kappa=.5, normalized_threshold=True)
    run = run_temporal_experiment(parameters)
    assert run.algo_cfg.normalized_threshold
    for step in run.finite.update_records_:
        expected = (step.scores.abs() / step.tau >= .5).double() * (1-run.environment.Delta)
        assert torch.equal(step.selection, expected)

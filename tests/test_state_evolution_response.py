import math

import pytest
import torch

from src.asymptotics import MacroscopicStateEvolution
from src.config import AlgorithmConfig, DataConfig
from src.objectives import LipschitzSelection, LogisticLoss, SmoothSelection
from src.utils import compute_abstract_pseudo_residual_from


def _make_state_evolution(
    *,
    K=4,
    n_iterations=1,
    mc_seed=7,
    supervision_ratio=0.5,
    pseudo_label_param=1.0,
    selection_epsilon=0.03,
    include_bias=True,
    initial_bias=0.0,
    selection_function=None,
):
    data_cfg = DataConfig(
        scale=1.0,
        label_prior=0.5,
        supervision_ratio=supervision_ratio,
        data_to_dimension_ratio=2.0,
        signal_law=lambda: 1.0,
    )
    if selection_function is None:
        selection_function = LipschitzSelection(
            selection_epsilon,
            True,
        )
    algo_cfg = AlgorithmConfig(
        n_iterations=n_iterations,
        step_size=0.1,
        penalty_param=0.1,
        pseudo_label_param=pseudo_label_param,
        ramp_start=0,
        ramp_end=0,
        margin_threshold=math.log(4.0),
        include_bias=include_bias,
        selection_function=selection_function,
    )
    return MacroscopicStateEvolution(
        data_cfg=data_cfg,
        algo_cfg=algo_cfg,
        mc_seed=mc_seed,
        K=K,
        initial_bias=initial_bias,
    )


def test_selection_rate_is_detached_before_residual_reuse():
    se = _make_state_evolution(K=4)
    kappa = se.algo_cfg.positive_margin
    state = torch.tensor(
        [0.0, kappa - 0.01, kappa + 0.01, 0.0],
        requires_grad=True,
    )
    se.indicator = torch.tensor([1.0, 0.0, 0.0, 1.0])
    se.preactivation[0] = state

    selection_rate = se.compute_selection_rate(0)

    assert selection_rate.item() == 0.5
    assert not selection_rate.requires_grad
    assert selection_rate.grad_fn is None

    residual = se.compute_pseudo_residual_from(
        state,
        label=torch.ones_like(state),
        indicator=se.indicator,
        coef=1.0,
        selection_rate=selection_rate,
    )
    own_coordinate_gradient = torch.autograd.grad(
        residual[1], state, retain_graph=True
    )[0]
    off_diagonal = torch.cat(
        [own_coordinate_gradient[:1], own_coordinate_gradient[2:]]
    )
    torch.testing.assert_close(off_diagonal, torch.zeros_like(off_diagonal))


def test_selection_rate_is_conditioned_only_on_unlabeled_particles():
    se = _make_state_evolution(K=4)
    se.indicator = torch.tensor([1.0, 1.0, 0.0, 0.0])
    se.preactivation[0] = torch.tensor([2.0, 2.0, 2.0, 0.0])

    selection_rate = se.compute_selection_rate(0)

    # Three of four particles are selected, but only one of two unlabeled
    # particles is selected.
    assert selection_rate.item() == 0.5


def test_selection_rate_is_zero_when_no_unlabeled_particles_exist():
    se = _make_state_evolution(K=3, supervision_ratio=1.0)
    se.indicator = torch.ones(3)
    se.preactivation[0] = torch.tensor([2.0, 0.0, -2.0])

    selection_rate = se.compute_selection_rate(0)

    assert selection_rate.item() == 0.0
    assert torch.isfinite(selection_rate)
    assert not selection_rate.requires_grad


def test_residual_memory_matches_explicit_mean_diagonal_jacobian():
    se = _make_state_evolution(K=4)
    state = torch.tensor([-1.5, -0.5, 0.5, 1.5], requires_grad=True)
    residual = state.square() + 3.0 * state
    se.forward_noise[0] = state
    se.residual[0] = residual

    jacobian = torch.autograd.functional.jacobian(
        lambda value: value.square() + 3.0 * value,
        state,
    )
    expected = torch.diagonal(jacobian).mean()
    rng_state = torch.random.get_rng_state()

    actual = se.compute_residual_memory(0)

    torch.testing.assert_close(actual[0], expected)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert actual.dtype == state.dtype
    assert actual.device == state.device


def test_residual_memory_does_not_contract_with_all_ones_under_coupling():
    se = _make_state_evolution(K=3, mc_seed=7)
    state = torch.tensor([0.2, -0.3, 0.7], requires_grad=True)
    matrix = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [-3.0, 4.0, 5.0],
            [6.0, 0.0, 7.0],
        ]
    )
    residual = matrix @ state
    se.forward_noise[0] = state
    se.residual[0] = residual

    jacobian = torch.autograd.functional.jacobian(
        lambda value: matrix @ value,
        state,
    )
    mean_diagonal = torch.diagonal(jacobian).mean()
    full_jacobian_sum = jacobian.sum() / state.numel()

    actual = se.compute_residual_memory(0)[0]

    # This deterministic probe is exact for the chosen coupled Jacobian. The
    # previous all-ones VJP instead returned full_jacobian_sum.
    torch.testing.assert_close(actual, mean_diagonal)
    assert not torch.isclose(actual, full_jacobian_sum)


def test_zero_selection_has_zero_pseudo_labeled_contribution():
    preactivation = torch.tensor([0.2, -0.3], requires_grad=True)
    indicator = torch.zeros_like(preactivation)
    selection_mask = torch.zeros_like(preactivation)

    residual = compute_abstract_pseudo_residual_from(
        preactivation=preactivation,
        label=torch.ones_like(preactivation),
        indicator=indicator,
        selection_mask=selection_mask,
        selection_rate=preactivation.new_zeros(()),
        coef=5.0,
        rho=0.5,
        eta=0.1,
        loss_function=LogisticLoss(),
    )

    torch.testing.assert_close(residual, torch.zeros_like(residual))
    assert torch.isfinite(residual).all()


def test_resolvable_epsilon_gives_finite_boundary_response():
    kappa = math.log(4.0)
    state_values = torch.tensor(
        [
            -kappa - 0.015,
            -kappa + 0.015,
            kappa - 0.015,
            kappa + 0.015,
        ]
    )

    def evaluate(epsilon):
        se = _make_state_evolution(K=4, selection_epsilon=epsilon)
        state = state_values.clone().requires_grad_()
        se.indicator = torch.zeros(4)
        se.forward_noise[0] = state
        se.preactivation[0] = state
        selection_rate = se.compute_selection_rate(0)
        se.residual[0] = se.compute_pseudo_residual_from(
            state,
            label=torch.ones_like(state),
            indicator=se.indicator,
            coef=5.0,
            selection_rate=selection_rate,
        )
        return se.residual[0].detach(), se.compute_residual_memory(0)[0]

    unresolved_residual, unresolved_response = evaluate(1e-6)
    corrected_residual, corrected_response = evaluate(0.03)

    # The forward residual is unchanged because both selectors are exactly hard.
    torch.testing.assert_close(corrected_residual, unresolved_residual)
    assert torch.isfinite(corrected_response)
    assert 0.1 < corrected_response.abs() < 10.0
    assert not torch.isclose(corrected_response, unresolved_response)


def test_smooth_epsilon_gives_finite_boundary_response():
    kappa = math.log(4.0)
    state = torch.tensor(
        [
            -kappa - 0.015,
            -kappa + 0.015,
            kappa - 0.015,
            kappa + 0.015,
        ],
        requires_grad=True,
    )
    selector = SmoothSelection(
        0.03,
        True,
    )
    se = _make_state_evolution(K=4, selection_function=selector)
    se.indicator = torch.zeros(4)
    se.forward_noise[0] = state
    se.preactivation[0] = state
    selection_rate = se.compute_selection_rate(0)
    se.residual[0] = se.compute_pseudo_residual_from(
        state,
        label=torch.ones_like(state),
        indicator=se.indicator,
        coef=5.0,
        selection_rate=selection_rate,
    )

    response = se.compute_residual_memory(0)[0]

    assert torch.equal(
        selector(state, pos_margin=kappa, neg_margin=-kappa).detach(),
        torch.tensor([1.0, 0.0, 0.0, 1.0]),
    )
    assert torch.isfinite(response)
    assert 0.1 < response.abs() < 10.0


def test_labeled_only_trajectory_is_independent_of_selection_epsilon():
    narrow = _make_state_evolution(
        K=128,
        n_iterations=2,
        mc_seed=11,
        pseudo_label_param=0.0,
        selection_epsilon=1e-6,
    )
    narrow.compute_trajectory()
    wide = _make_state_evolution(
        K=128,
        n_iterations=2,
        mc_seed=11,
        pseudo_label_param=0.0,
        selection_epsilon=0.03,
    )
    wide.compute_trajectory()

    for narrow_value, wide_value in zip(narrow.bias, wide.bias):
        torch.testing.assert_close(narrow_value, wide_value, rtol=0.0, atol=0.0)
    for narrow_value, wide_value in zip(narrow.weight, wide.weight):
        torch.testing.assert_close(narrow_value, wide_value, rtol=0.0, atol=0.0)
    for narrow_value, wide_value in zip(
        narrow.residual_memory, wide.residual_memory
    ):
        torch.testing.assert_close(narrow_value, wide_value, rtol=0.0, atol=0.0)


def test_state_evolution_without_bias_keeps_zero_effective_intercept():
    se = _make_state_evolution(
        K=128,
        n_iterations=3,
        include_bias=False,
    )

    se.compute_trajectory()

    assert se.algo_cfg.include_bias is False
    assert not hasattr(se, "include_bias")
    for bias in se.bias:
        torch.testing.assert_close(bias, torch.zeros_like(bias))
        assert not bias.requires_grad

    expected_initial_preactivation = (
        se.weight_signal_alignments[0] * se.label + se.forward_noise[0]
    )
    torch.testing.assert_close(
        se.preactivation[0],
        expected_initial_preactivation,
    )


def test_state_evolution_with_bias_uses_effective_bias_recursion():
    se = _make_state_evolution(
        K=128,
        n_iterations=2,
        include_bias=True,
        initial_bias=0.7,
    )

    se.step(0)

    assert se.algo_cfg.include_bias is True
    assert se.bias[0].requires_grad
    torch.testing.assert_close(se.bias[0], torch.tensor(0.7))
    torch.testing.assert_close(
        se.bias[1],
        se.bias[0] + se.mean_residual[0],
    )
    expected_initial_preactivation = (
        se.bias[0]
        + se.weight_signal_alignments[0] * se.label
        + se.forward_noise[0]
    )
    torch.testing.assert_close(
        se.preactivation[0],
        expected_initial_preactivation,
    )


def test_algorithm_config_is_the_only_bias_configuration():
    with_bias = _make_state_evolution(include_bias=True)
    without_bias = _make_state_evolution(include_bias=False)

    assert with_bias.algo_cfg.include_bias is True
    assert without_bias.algo_cfg.include_bias is False
    assert not hasattr(with_bias, "include_bias")
    assert not hasattr(without_bias, "include_bias")


def test_state_evolution_rejects_nonzero_bias_when_bias_is_disabled():
    with pytest.raises(
        ValueError,
        match="initial_bias must be zero when include_bias=False",
    ):
        _make_state_evolution(
            include_bias=False,
            initial_bias=0.3,
        )


def test_state_evolution_accepts_none_initial_bias_when_bias_is_disabled():
    se = _make_state_evolution(
        include_bias=False,
        initial_bias=None,
    )

    torch.testing.assert_close(se.bias[0], torch.zeros_like(se.bias[0]))

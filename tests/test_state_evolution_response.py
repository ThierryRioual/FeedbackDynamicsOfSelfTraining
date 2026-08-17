import math
import random

import numpy as np
import pytest
import torch

from src.asymptotics import MacroscopicStateEvolution
from src.config import AlgorithmConfig, DataConfig
from src.objectives import HardSelection, LogisticLoss
from src.utils import compute_abstract_pseudo_residual_from


def _data_config(
    *,
    scale=1.0,
    label_prior=0.5,
    supervision_ratio=0.5,
    delta=2.0,
    signal_law=None,
):
    if signal_law is None:
        signal_law = lambda: 1.0
    return DataConfig(
        scale=scale,
        label_prior=label_prior,
        supervision_ratio=supervision_ratio,
        data_to_dimension_ratio=delta,
        signal_law=signal_law,
    )


def _algorithm_config(
    *,
    n_iterations=2,
    pseudo_label_param=0.0,
    include_bias=True,
    margin=1.0,
    loss_function=None,
):
    kwargs = {}
    if loss_function is not None:
        kwargs["loss_function"] = loss_function
    return AlgorithmConfig(
        n_iterations=n_iterations,
        step_size=0.1,
        penalty_param=0.1,
        pseudo_label_param=pseudo_label_param,
        ramp_start=0,
        ramp_end=0,
        margin_threshold=margin,
        include_bias=include_bias,
        selection_function=HardSelection(),
        **kwargs,
    )


def _make_state_evolution(
    *,
    K=None,
    K_w=32,
    K_g=41,
    n_iterations=2,
    mc_seed=7,
    scale=1.0,
    label_prior=0.5,
    supervision_ratio=0.5,
    delta=2.0,
    pseudo_label_param=0.0,
    include_bias=True,
    initial_bias=0.0,
    initial_weight=None,
    eps_rank=1e-12,
    signal_law=None,
    parameter_base_sampler=None,
    sample_base_sampler=None,
    loss_function=None,
    margin=1.0,
):
    data_cfg = _data_config(
        scale=scale,
        label_prior=label_prior,
        supervision_ratio=supervision_ratio,
        delta=delta,
        signal_law=signal_law,
    )
    algo_cfg = _algorithm_config(
        n_iterations=n_iterations,
        pseudo_label_param=pseudo_label_param,
        include_bias=include_bias,
        margin=margin,
        loss_function=loss_function,
    )
    particle_kwargs = {"K": K} if K is not None else {"K_w": K_w, "K_g": K_g}
    return MacroscopicStateEvolution(
        data_cfg=data_cfg,
        algo_cfg=algo_cfg,
        mc_seed=mc_seed,
        initial_bias=initial_bias,
        initial_weight=initial_weight,
        eps_rank=eps_rank,
        parameter_base_sampler=parameter_base_sampler,
        sample_base_sampler=sample_base_sampler,
        **particle_kwargs,
    )


def _zero_weight_sampler(size, generator, dtype, device):
    del generator
    signal = torch.linspace(-1.0, 1.0, size, dtype=dtype, device=device)
    return signal, torch.zeros(size, dtype=dtype, device=device)


def _dependent_parameter_sampler(size, generator, dtype, device):
    del generator
    signal = torch.linspace(-2.0, 2.0, size, dtype=dtype, device=device)
    return signal, 3.0 * signal - 2.0


def _dependent_sample_sampler(size, generator, dtype, device):
    del generator
    index = torch.arange(size, device=device)
    label = torch.where(index.remainder(2) == 0, 1.0, -1.0).to(dtype)
    indicator = (label > 0.0).to(dtype)
    initial_label = torch.where(indicator == 1.0, label, -label)
    return label, indicator, initial_label


def _initial_label_sample_sampler(size, generator, dtype, device):
    del generator
    index = torch.arange(size, device=device)
    label = torch.where(index.remainder(2) == 0, 1.0, -1.0).to(dtype)
    indicator = (index.remainder(2) == 0).to(dtype)
    # Labeled coordinates match Y.  Every unlabeled initial label is -1, which
    # differs from sign(0)=+1 in the deterministic first forward pass below.
    initial_label = label.clone()
    return label, indicator, initial_label


def _zero_signal_nonzero_weight_sampler(size, generator, dtype, device):
    del generator
    return (
        torch.zeros(size, dtype=dtype, device=device),
        torch.ones(size, dtype=dtype, device=device),
    )


class RecordingLogisticLoss(LogisticLoss):
    def __init__(self):
        self.targets = []

    def gradient(self, r, y):
        self.targets.append(y.detach().clone())
        return super().gradient(r, y)


def test_distinct_particle_populations_and_cross_labeled_innovation_shapes():
    se = _make_state_evolution(K_w=17, K_g=29, n_iterations=3)

    se.compute_trajectory()

    assert se.K is None
    assert se.signal.shape == (17,)
    assert se.label.shape == (29,)
    assert se.indicator.shape == (29,)
    assert len(se.sample_innovations) == se.T + 1
    assert len(se.parameter_innovations) == se.T + 1
    for t in range(se.T + 1):
        assert se.sample_innovations[t].shape == (29,)  # z_t^[g]
        assert se.forward_noise[t].shape == (29,)  # q^t
        assert se.preactivation[t].shape == (29,)
        assert se.residual[t].shape == (29,)
        assert se.parameter_innovations[t].shape == (17,)  # z_t^[w]
        assert se.backward_noise[t].shape == (17,)  # p^t
        assert se.weight[t].shape == (17,)

    # At t=0 there is no transported past.  These identities make the
    # cross-labeling operational rather than checking only tensor sizes.
    torch.testing.assert_close(
        se.forward_noise[0],
        se.forward_innovation_scale[0] * se.sample_innovations[0],
    )
    psi_g0 = se.residual_memory_coordinates[0]
    expected_p0 = (
        math.sqrt(se.delta)
        * (se.weight_basis.B[:, : psi_g0.numel()] @ psi_g0)
        + se.backward_innovation_scale[0] * se.parameter_innovations[0]
    )
    torch.testing.assert_close(se.backward_noise[0], expected_p0)


def test_zero_initialization_has_zero_q0_and_finite_first_step():
    se = _make_state_evolution(
        K_w=19,
        K_g=23,
        n_iterations=1,
        parameter_base_sampler=_zero_weight_sampler,
        label_prior=0.3,
    )

    se.step(0)

    torch.testing.assert_close(se.forward_noise[0], torch.zeros(23))
    assert se.forward_innovation_scale[0].item() == 0.0
    assert se.weight_rank[0] == 0
    assert se.weight_basis.B.shape == (19, 0)
    assert se.weight_basis.Theta.shape == (0, 1)
    assert math.isfinite(se.error[0])
    assert se.error[0] == pytest.approx(0.7)
    assert torch.isfinite(se.preactivation[0]).all()
    assert torch.isfinite(se.residual[0]).all()
    assert torch.isfinite(se.backward_noise[0]).all()
    assert torch.isfinite(se.weight[1]).all()


def test_full_trajectory_bases_are_orthonormal_and_reconstruct_when_untruncated():
    se = _make_state_evolution(
        K_w=64,
        K_g=83,
        n_iterations=3,
        mc_seed=19,
        eps_rank=1e-13,
    )

    se.compute_trajectory()

    assert not any(se.weight_rank_truncated)
    assert not any(se.residual_rank_truncated)
    assert se.weight_basis.rank == se.T + 1
    assert se.residual_basis.rank == se.T + 1

    weight_trajectory = torch.stack(se.weight, dim=1)
    residual_trajectory = torch.stack(se.residual, dim=1)
    torch.testing.assert_close(
        se.weight_basis.B @ se.weight_basis.Theta,
        weight_trajectory,
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(
        se.residual_basis.B @ se.residual_basis.Theta,
        residual_trajectory,
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(
        se.weight_basis.B.T @ se.weight_basis.B / se.K_w,
        torch.eye(se.weight_basis.rank),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        se.residual_basis.B.T @ se.residual_basis.B / se.K_g,
        torch.eye(se.residual_basis.rank),
        rtol=1e-12,
        atol=1e-12,
    )


def test_raw_memory_coefficients_are_lazy_diagnostics_of_psi_coordinates():
    se = _make_state_evolution(
        K_w=53,
        K_g=67,
        n_iterations=2,
        mc_seed=43,
        eps_rank=1e-13,
    )
    se.compute_trajectory()

    assert all(value is None for value in se.weight_memory)
    assert all(value is None for value in se.residual_memory)
    for t in range(se.T + 1):
        phi_w = se.compute_weight_memory(t)
        phi_g = se.compute_residual_memory(t)
        G_past = se.G_ring[:, :t]
        W_current = se.W_ring[:, : t + 1]
        psi_w = se.weight_memory_coordinates[t]
        psi_g = se.residual_memory_coordinates[t]
        B_g_past = se.B_g[:, : psi_w.numel()]
        B_w_current = se.B_w[:, : psi_g.numel()]

        torch.testing.assert_close(
            G_past @ phi_w,
            B_g_past @ psi_w,
            rtol=1e-10,
            atol=1e-11,
        )
        torch.testing.assert_close(
            W_current @ phi_g,
            B_w_current @ psi_g,
            rtol=1e-10,
            atol=1e-11,
        )


def test_complete_state_recursion_matches_direct_gram_reference():
    se = _make_state_evolution(
        K_w=47,
        K_g=59,
        n_iterations=2,
        mc_seed=61,
        eps_rank=1e-14,
        pseudo_label_param=0.0,
    )
    se.compute_trajectory()

    for t in range(se.T + 1):
        W = se.W_ring[:, : t + 1]
        G = se.G_ring[:, : t + 1]
        Q = se.Q_ring[:, : t + 1]
        P = se.P_ring[:, : t + 1]
        w = W[:, -1]
        g = G[:, -1]
        W_past = W[:, :-1]
        G_past = G[:, :-1]
        Q_past = Q[:, :-1]
        P_past = P[:, :-1]

        if t == 0:
            w_perp = w
            g_perp = g
            q_direct = torch.sqrt(torch.mean(w_perp.square())) * se.z_g[t]
            p_projection = torch.zeros(se.K_w, dtype=se.dtype)
        else:
            C_w_past = W_past.T @ W_past / se.K_w
            alpha_w = torch.linalg.pinv(C_w_past) @ (
                W_past.T @ w / se.K_w
            )
            w_perp = w - W_past @ alpha_w
            C_g_past = G_past.T @ G_past / se.K_g
            phi_w = torch.linalg.pinv(C_g_past) @ (
                P_past.T @ w_perp / se.K_w
            )
            q_direct = (
                Q_past @ alpha_w
                + (G_past @ phi_w) / math.sqrt(se.delta)
                + torch.sqrt(torch.mean(w_perp.square())) * se.z_g[t]
            )

            alpha_g = torch.linalg.pinv(C_g_past) @ (
                G_past.T @ g / se.K_g
            )
            g_perp = g - G_past @ alpha_g
            p_projection = P_past @ alpha_g

        C_w = W.T @ W / se.K_w
        phi_g = torch.linalg.pinv(C_w) @ (Q.T @ g_perp / se.K_g)
        p_direct = (
            p_projection
            + math.sqrt(se.delta) * (W @ phi_g)
            + torch.sqrt(torch.mean(g_perp.square())) * se.z_w[t]
        )

        torch.testing.assert_close(
            se.forward_noise[t], q_direct, rtol=1e-9, atol=1e-10
        )
        torch.testing.assert_close(
            se.backward_noise[t], p_direct, rtol=1e-9, atol=1e-10
        )


def test_aggressive_rank_truncation_is_explicit_and_finite():
    se = _make_state_evolution(
        K_w=23,
        K_g=31,
        n_iterations=3,
        mc_seed=73,
        eps_rank=2.0,
    )

    se.compute_trajectory()

    assert se.weight_rank == [0] * (se.T + 1)
    assert se.residual_rank == [0] * (se.T + 1)
    assert any(se.weight_rank_truncated)
    assert any(se.residual_rank_truncated)
    assert se.Q_w.shape == (se.K_g, 0)
    assert se.P_g.shape == (se.K_w, 0)
    for history in (
        se.weight,
        se.residual,
        se.forward_noise,
        se.backward_noise,
    ):
        assert all(torch.isfinite(value).all() for value in history)


def _state_tensors(se):
    for value in (
        se.signal,
        se.label,
        se.indicator,
        se.initial_pseudo_label,
        se.transported_forward_history,
        se.transported_backward_history,
        se.weight_basis.B,
        se.weight_basis.Theta,
        se.residual_basis.B,
        se.residual_basis.Theta,
    ):
        if isinstance(value, torch.Tensor):
            yield value

    history_names = (
        "bias",
        "weight",
        "preactivation",
        "residual",
        "forward_noise",
        "backward_noise",
        "sample_innovations",
        "parameter_innovations",
        "weight_memory",
        "residual_memory",
        "weight_projection_coordinates",
        "residual_projection_coordinates",
        "weight_coordinates",
        "residual_coordinates",
        "weight_memory_coordinates",
        "residual_memory_coordinates",
        "forward_innovation_scale",
        "backward_innovation_scale",
        "weight_signal_alignments",
        "label_residual_alignments",
        "mean_residual",
        "selection_rate",
        "decay",
        "weight_orthogonality_error",
        "residual_orthogonality_error",
        "weight_coordinate_singular_values",
        "residual_coordinate_singular_values",
        "weight_coordinate_condition_number",
        "residual_coordinate_condition_number",
    )
    for name in history_names:
        for value in getattr(se, name):
            if isinstance(value, torch.Tensor):
                yield value


def test_production_trajectory_is_graph_free(monkeypatch):
    def forbidden_autograd(*args, **kwargs):
        raise AssertionError("production state evolution called torch.autograd.grad")

    def forbidden_inverse(*args, **kwargs):
        raise AssertionError("production state evolution formed an explicit inverse")

    monkeypatch.setattr(torch.autograd, "grad", forbidden_autograd)
    monkeypatch.setattr(torch.linalg, "pinv", forbidden_inverse)
    monkeypatch.setattr(torch.linalg, "inv", forbidden_inverse)
    monkeypatch.setattr(torch, "inverse", forbidden_inverse)
    initial_weight = torch.randn(31, dtype=torch.float64, requires_grad=True)
    se = _make_state_evolution(
        K_w=31,
        K_g=37,
        n_iterations=3,
        initial_weight=initial_weight,
    )

    se.compute_trajectory()

    tensors = list(_state_tensors(se))
    assert tensors
    assert all(not tensor.requires_grad for tensor in tensors)
    assert all(tensor.grad_fn is None for tensor in tensors)


def _assert_reproducible_trajectory(left, right):
    for name in ("signal", "label", "indicator"):
        torch.testing.assert_close(
            getattr(left, name), getattr(right, name), rtol=0.0, atol=0.0
        )
    for name in ("weight", "residual", "forward_noise", "backward_noise"):
        for left_value, right_value in zip(getattr(left, name), getattr(right, name)):
            torch.testing.assert_close(left_value, right_value, rtol=0.0, atol=0.0)


def test_rng_is_reproducible_and_preserves_the_global_rng_state():
    def global_rng_signal_law():
        return torch.randn(()).item()

    torch.manual_seed(123456)
    global_state = torch.random.get_rng_state().clone()
    first = _make_state_evolution(
        K_w=27,
        K_g=35,
        n_iterations=2,
        mc_seed=101,
        signal_law=global_rng_signal_law,
    )
    first.compute_trajectory()
    assert torch.equal(torch.random.get_rng_state(), global_state)

    second = _make_state_evolution(
        K_w=27,
        K_g=35,
        n_iterations=2,
        mc_seed=101,
        signal_law=global_rng_signal_law,
    )
    second.compute_trajectory()
    assert torch.equal(torch.random.get_rng_state(), global_state)
    _assert_reproducible_trajectory(first, second)

    different = _make_state_evolution(
        K_w=27,
        K_g=35,
        n_iterations=2,
        mc_seed=102,
        signal_law=global_rng_signal_law,
    )
    different.compute_trajectory()
    assert not torch.equal(first.signal, different.signal)
    assert not torch.equal(first.weight[0], different.weight[0])
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_legacy_signal_law_isolates_python_and_numpy_global_rngs():
    def mixed_global_signal_law():
        return float(np.random.randn() + random.random())

    random.seed(654321)
    np.random.seed(123456)
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    first = _make_state_evolution(
        K_w=21,
        K_g=25,
        n_iterations=1,
        mc_seed=303,
        signal_law=mixed_global_signal_law,
    )
    assert random.getstate() == python_state
    restored_numpy_state = np.random.get_state()
    assert restored_numpy_state[0] == numpy_state[0]
    assert np.array_equal(restored_numpy_state[1], numpy_state[1])
    assert restored_numpy_state[2:] == numpy_state[2:]

    second = _make_state_evolution(
        K_w=21,
        K_g=25,
        n_iterations=1,
        mc_seed=303,
        signal_law=mixed_global_signal_law,
    )
    torch.testing.assert_close(first.signal, second.signal, rtol=0.0, atol=0.0)
    assert random.getstate() == python_state
    restored_numpy_state = np.random.get_state()
    assert restored_numpy_state[0] == numpy_state[0]
    assert np.array_equal(restored_numpy_state[1], numpy_state[1])
    assert restored_numpy_state[2:] == numpy_state[2:]

def test_joint_base_samplers_preserve_deliberate_dependence():
    se = _make_state_evolution(
        K_w=12,
        K_g=14,
        parameter_base_sampler=_dependent_parameter_sampler,
        sample_base_sampler=_dependent_sample_sampler,
    )

    torch.testing.assert_close(se.weight[0], 3.0 * se.signal - 2.0)
    torch.testing.assert_close(se.indicator, (se.label > 0.0).to(se.dtype))
    expected_initial_label = torch.where(
        se.indicator == 1.0, se.label, -se.label
    )
    torch.testing.assert_close(se.initial_pseudo_label, expected_initial_label)
    assert torch.equal(
        se.initial_pseudo_label[se.indicator == 1.0],
        se.label[se.indicator == 1.0],
    )


def test_active_pi0_requires_an_explicit_joint_sample_sampler():
    algo_cfg = _algorithm_config(n_iterations=2, pseudo_label_param=1.0)
    algo_cfg.pseudo_label_param_schedule_[0] = 1.0

    with pytest.raises(ValueError, match="explicit sample_base_sampler"):
        MacroscopicStateEvolution(
            data_cfg=_data_config(supervision_ratio=0.5),
            algo_cfg=algo_cfg,
            K_w=13,
            K_g=17,
        )


def test_forward_pass_uses_y_init_at_t0_and_sign_of_r_later():
    loss = RecordingLogisticLoss()
    algo_cfg = _algorithm_config(
        n_iterations=2,
        pseudo_label_param=1.0,
        margin=0.0,
        loss_function=loss,
    )
    algo_cfg.pseudo_label_param_schedule_[0] = 1.0
    se = MacroscopicStateEvolution(
        data_cfg=_data_config(supervision_ratio=0.5),
        algo_cfg=algo_cfg,
        K_w=18,
        K_g=20,
        parameter_base_sampler=_zero_signal_nonzero_weight_sampler,
        sample_base_sampler=_initial_label_sample_sampler,
    )
    se.sample_innovations[0].zero_()

    se.forward_pass(0)

    sign_at_zero = torch.ones_like(se.preactivation[0])
    differs_from_sign = (
        (se.indicator == 0.0)
        & (se.initial_pseudo_label != sign_at_zero)
    )
    assert torch.any(differs_from_sign)
    assert len(loss.targets) == 2
    torch.testing.assert_close(loss.targets[1], se.initial_pseudo_label)

    se.step(0)
    se.forward_pass(1)

    assert len(loss.targets) == 4
    sign_at_t1 = torch.where(se.preactivation[1] >= 0.0, 1.0, -1.0)
    torch.testing.assert_close(loss.targets[3], sign_at_t1)


def test_sigma_scales_both_preactivation_and_weight_update():
    sigma = 2.75
    delta = 3.25
    se = _make_state_evolution(
        K_w=29,
        K_g=37,
        n_iterations=1,
        scale=sigma,
        delta=delta,
        initial_bias=0.3,
    )

    se.step(0)

    expected_r0 = (
        se.bias[0]
        + se.weight_signal_alignments[0] * se.label
        + sigma * se.forward_noise[0]
    )
    torch.testing.assert_close(se.preactivation[0], expected_r0)
    unscaled_r0 = (
        se.bias[0]
        + se.weight_signal_alignments[0] * se.label
        + se.forward_noise[0]
    )
    assert not torch.allclose(se.preactivation[0], unscaled_r0)

    expected_w1 = (
        se.weight[0]
        + se.decay[0]
        + se.label_residual_alignments[0] * se.signal
        + (sigma / math.sqrt(delta)) * se.backward_noise[0]
    )
    torch.testing.assert_close(se.weight[1], expected_w1)
    unscaled_w1 = (
        se.weight[0]
        + se.decay[0]
        + se.label_residual_alignments[0] * se.signal
        + se.backward_noise[0] / math.sqrt(delta)
    )
    assert not torch.allclose(se.weight[1], unscaled_w1)


def test_selection_rate_uses_the_prescribed_population_denominator():
    se = _make_state_evolution(
        K=8,
        supervision_ratio=0.25,
        margin=1.0,
    )
    se.indicator = torch.tensor(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    se.preactivation[0] = torch.tensor(
        [2.0, 0.0, 2.0, 0.0, -2.0, 0.0, 0.0, 2.0]
    )

    selection_rate = se.compute_selection_rate(0)

    # Two selected unlabeled particles among K=8, divided by 1-rho=3/4.
    assert selection_rate.item() == pytest.approx((2.0 / 8.0) / 0.75)
    # This is deliberately not the empirical conditional fraction 2/5.
    assert selection_rate.item() != pytest.approx(2.0 / 5.0)
    assert not selection_rate.requires_grad
    assert selection_rate.grad_fn is None


def test_selection_rate_is_zero_when_the_population_is_fully_supervised():
    se = _make_state_evolution(K=7, supervision_ratio=1.0)
    se.indicator = torch.ones(7)
    se.preactivation[0] = torch.linspace(-2.0, 2.0, 7)

    selection_rate = se.compute_selection_rate(0)

    torch.testing.assert_close(selection_rate, torch.zeros_like(selection_rate))
    assert torch.isfinite(selection_rate)


def test_zero_selection_has_zero_pseudo_labeled_contribution():
    preactivation = torch.tensor([0.2, -0.3])
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


def test_state_evolution_without_bias_keeps_zero_effective_intercept():
    se = _make_state_evolution(
        K_w=47,
        K_g=53,
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
        se.weight_signal_alignments[0] * se.label
        + se.sigma * se.forward_noise[0]
    )
    torch.testing.assert_close(se.preactivation[0], expected_initial_preactivation)


def test_state_evolution_with_bias_uses_effective_bias_recursion():
    se = _make_state_evolution(
        K_w=41,
        K_g=49,
        n_iterations=2,
        include_bias=True,
        initial_bias=0.7,
    )

    se.step(0)

    assert se.algo_cfg.include_bias is True
    assert not se.bias[0].requires_grad
    torch.testing.assert_close(se.bias[0], torch.tensor(0.7))
    torch.testing.assert_close(se.bias[1], se.bias[0] + se.mean_residual[0])
    expected_initial_preactivation = (
        se.bias[0]
        + se.weight_signal_alignments[0] * se.label
        + se.sigma * se.forward_noise[0]
    )
    torch.testing.assert_close(se.preactivation[0], expected_initial_preactivation)


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
        _make_state_evolution(include_bias=False, initial_bias=0.3)


def test_state_evolution_accepts_none_initial_bias_when_bias_is_disabled():
    se = _make_state_evolution(include_bias=False, initial_bias=None)

    torch.testing.assert_close(se.bias[0], torch.zeros_like(se.bias[0]))

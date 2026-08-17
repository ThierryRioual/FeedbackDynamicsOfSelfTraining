import pytest
import torch

from src.asymptotics import MacroscopicStateEvolution
from src.config import AlgorithmConfig, DataConfig


def _state_evolution(n_iterations=3):
    data_cfg = DataConfig(
        scale=1.0,
        label_prior=0.4,
        supervision_ratio=0.3,
        data_to_dimension_ratio=2.0,
        signal_law=lambda: 1.0,
    )
    algo_cfg = AlgorithmConfig(
        n_iterations=n_iterations,
        margin_threshold=1.0,
        step_size=0.1,
        penalty_param=0.1,
        pseudo_label_param=0.0,
        ramp_start=0,
        ramp_end=0,
        include_bias=True,
    )
    return MacroscopicStateEvolution(
        data_cfg=data_cfg,
        algo_cfg=algo_cfg,
        K_w=24,
        K_g=31,
        mc_seed=17,
    )


def test_state_evolution_enforces_sequential_step_access():
    se = _state_evolution(n_iterations=3)

    assert se._current_t == 0
    with pytest.raises(RuntimeError, match="expected t=0"):
        se.step(1)

    se.step(0)
    assert se._current_t == 1
    with pytest.raises(RuntimeError, match="expected t=1"):
        se.step(0)
    with pytest.raises(RuntimeError, match="expected t=1"):
        se.step(2)

    se.step(1)
    se.step(2)
    assert se._current_t == se.T
    with pytest.raises(IndexError, match="Maximum number"):
        se.step(se.T)
    with pytest.raises(IndexError, match="terminal backward pass"):
        se.backward_pass(se.T)


def test_compute_trajectory_completes_the_terminal_effective_process():
    se = _state_evolution(n_iterations=3)

    # Exercise continuation from a partially computed trajectory.
    se.step(0)
    se.compute_trajectory()

    assert se._current_t == se.T
    assert len(se.weight) == se.T + 1
    assert len(se.bias) == se.T + 1
    assert len(se.preactivation) == se.T + 1
    assert len(se.residual) == se.T + 1
    assert len(se.forward_noise) == se.T + 1
    assert len(se.backward_noise) == se.T + 1
    assert all(value is not None for value in se.weight)
    assert all(value is not None for value in se.bias)
    assert all(value is not None for value in se.preactivation)
    assert all(value is not None for value in se.residual)
    assert all(value is not None for value in se.forward_noise)
    assert all(value is not None for value in se.backward_noise)
    assert all(value is not None for value in se.error)
    assert se.weight_basis.n_columns == se.T + 1
    assert se.residual_basis.n_columns == se.T + 1
    assert torch.isfinite(se.forward_noise[se.T]).all()
    assert torch.isfinite(se.backward_noise[se.T]).all()
    assert torch.isfinite(se.preactivation[se.T]).all()
    assert torch.isfinite(se.residual[se.T]).all()

    # The terminal half-step is diagnostic and does not allocate w^(T+1) or
    # advance the update counter.  Calling compute_trajectory again is a no-op.
    terminal_q = se.forward_noise[se.T].clone()
    terminal_p = se.backward_noise[se.T].clone()
    se.compute_trajectory()
    assert se._current_t == se.T
    torch.testing.assert_close(se.forward_noise[se.T], terminal_q)
    torch.testing.assert_close(se.backward_noise[se.T], terminal_p)


def test_noise_accessor_does_not_apply_the_parameter_update_early():
    se = _state_evolution(n_iterations=2)

    se.forward_pass(0)
    p0 = se.compute_backward_noise(0)

    assert p0.shape == (se.K_w,)
    assert se.weight[1] is None
    assert se.bias[1] is None
    assert se._current_t == 0

    se.step(0)
    assert se.weight[1] is not None
    assert se.bias[1] is not None
    assert se._current_t == 1

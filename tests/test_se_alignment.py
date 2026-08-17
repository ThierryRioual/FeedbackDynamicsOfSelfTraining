"""Low-cost finite/effective-process smoke tests.

These tests do not assert finite-sample agreement with state evolution.  The
finite data and Monte Carlo particles are different populations, so a precise
comparison would require a dedicated convergence experiment with uncertainty
quantification.  Here we check only matched initialization definitions and that
each implementation completes its first supervised update with ``pi_0=0``.
"""

import torch

from src.algorithms import SelfTrainedGradientDescent
from src.asymptotics import MacroscopicStateEvolution
from src.config import AlgorithmConfig, DataConfig
from src.dgp import IsotropicGaussian


def _algorithm_config(pseudo_label_param):
    return AlgorithmConfig(
        n_iterations=1,
        margin_threshold=1.0,
        step_size=0.1,
        penalty_param=0.1,
        pseudo_label_param=pseudo_label_param,
        ramp_start=0,
        ramp_end=0,
        include_bias=True,
    )


def _matched_setup():
    dimensions = 32
    n_train = 96
    signal = torch.linspace(-1.0, 1.0, dimensions, dtype=torch.float64)
    initial_weight = torch.linspace(
        1.0, -0.5, dimensions, dtype=torch.float64
    )
    initial_bias = 0.2
    data_cfg = DataConfig(
        scale=1.2,
        label_prior=0.4,
        supervision_ratio=0.25,
        data_to_dimension_ratio=n_train / dimensions,
        signal_law=lambda: 0.0,
    )
    return (
        dimensions,
        n_train,
        signal,
        initial_weight,
        initial_bias,
        data_cfg,
    )


def _fixed_parameter_sampler(signal, initial_weight):
    def sampler(size, generator, dtype, device):
        del generator
        assert size == signal.numel()
        return (
            signal.to(dtype=dtype, device=device),
            initial_weight.to(dtype=dtype, device=device),
        )

    return sampler


def test_matched_initialization_has_identical_macroscopic_definitions():
    (
        dimensions,
        _,
        signal,
        initial_weight,
        initial_bias,
        data_cfg,
    ) = _matched_setup()
    se = MacroscopicStateEvolution(
        data_cfg=data_cfg,
        algo_cfg=_algorithm_config(pseudo_label_param=3.0),
        K_w=dimensions,
        K_g=47,
        initial_bias=initial_bias,
        parameter_base_sampler=_fixed_parameter_sampler(signal, initial_weight),
        mc_seed=23,
    )

    finite_alignment = torch.mean(signal * initial_weight)
    finite_norm = torch.sqrt(torch.mean(initial_weight.square())).item()

    torch.testing.assert_close(se.signal, signal)
    torch.testing.assert_close(se.weight[0], initial_weight)
    torch.testing.assert_close(se.bias[0], torch.tensor(initial_bias))
    torch.testing.assert_close(
        se.compute_weight_signal_alignment(0), finite_alignment
    )
    assert se.compute_weight_norm(0) == finite_norm


def test_finite_and_effective_process_complete_one_supervised_smoke_step():
    (
        dimensions,
        n_train,
        signal,
        initial_weight,
        initial_bias,
        data_cfg,
    ) = _matched_setup()
    self_training_cfg = _algorithm_config(pseudo_label_param=3.0)
    supervised_cfg = _algorithm_config(pseudo_label_param=0.0)
    assert self_training_cfg.get_pseudo_label_weight(0) == 0.0

    dgp = IsotropicGaussian(
        cfg=data_cfg,
        n_train=n_train,
        n_test=0,
        dimensions=dimensions,
        seed=29,
        signal_vector=signal,
    )
    X_lab, Y_lab, X_unl, _, _, _ = dgp.sample(stratified=True)

    finite_self_training = SelfTrainedGradientDescent(cfg=self_training_cfg)
    finite_supervised = SelfTrainedGradientDescent(cfg=supervised_cfg)
    finite_self_training.fit(
        X_lab,
        Y_lab,
        X_unl,
        initial_bias=initial_bias,
        initial_weights=initial_weight.clone(),
    )
    finite_supervised.fit(
        X_lab,
        Y_lab,
        X_unl,
        initial_bias=initial_bias,
        initial_weights=initial_weight.clone(),
    )

    # pi_0=0 makes the finite first update exactly supervised, irrespective of
    # the configured later pseudo-label weight.
    torch.testing.assert_close(
        finite_self_training.weights,
        finite_supervised.weights,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        torch.as_tensor(finite_self_training.bias),
        torch.as_tensor(finite_supervised.bias),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.isfinite(finite_self_training.weights).all()
    assert torch.isfinite(torch.as_tensor(finite_self_training.bias))

    state_evolution = MacroscopicStateEvolution(
        data_cfg=data_cfg,
        algo_cfg=self_training_cfg,
        K_w=dimensions,
        K_g=53,
        initial_bias=initial_bias,
        parameter_base_sampler=_fixed_parameter_sampler(signal, initial_weight),
        mc_seed=31,
    )
    state_evolution.step(0)

    # This is a separate Monte Carlo smoke check, not an assertion that its
    # realized first iterate equals the finite-dimensional one.
    assert torch.isfinite(state_evolution.weight[1]).all()
    assert torch.isfinite(state_evolution.bias[1])
    assert torch.isfinite(state_evolution.preactivation[0]).all()
    assert torch.isfinite(state_evolution.residual[0]).all()
    assert state_evolution._current_t == 1

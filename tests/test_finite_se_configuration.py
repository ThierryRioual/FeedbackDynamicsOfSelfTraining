"""A matching finite/SE configuration smoke test, without a false finite-K equality claim."""

import torch

from src.algorithms import SelfTrainedGradientDescent
from src.asymptotics import MacroscopicStateEvolution
from src.config import AlgorithmConfig, DataConfig
from src.dgp import IsotropicGaussian
from src.environment import (
    FourCellSampleTypeLaw,
    state_evolution_sample_base_sampler,
    validate_finite_se_aspect_ratio,
)
from src.initialization import SelfTrainingInitialization


def test_finite_and_se_share_a_fixed_pi_joint_law_configuration():
    d, n = 20, 80
    law = FourCellSampleTypeLaw((.15, .25, .05, .55))
    data = DataConfig(scale=.8, label_prior=law.label_prior, supervision_ratio=law.supervision_ratio, data_to_dimension_ratio=n/d, signal_law=lambda: .3)
    cfg = AlgorithmConfig(n_iterations=1, step_size=.1, penalty_param=.1, pseudo_label_param=.7, margin_threshold=.5)
    dgp = IsotropicGaussian(data, n, 0, d, signal_vector=torch.full((d,), .3), sample_type_law=law, seed=4)
    env = dgp.sample_environment()
    validate_finite_se_aspect_ratio(env, data.data_to_dimension_ratio, tolerance=0.)
    X, _ = dgp.sample_design(env)
    y_init = env.Y.clone()
    y_init[env.I_U] = 1.
    finite = SelfTrainedGradientDescent(cfg).fit_full(X, env, SelfTrainingInitialization(0., torch.zeros(d), y_init))
    assert torch.isfinite(finite.weights).all()
    se = MacroscopicStateEvolution(data, cfg, K_w=37, K_g=41, sample_base_sampler=state_evolution_sample_base_sampler(law), mc_seed=4)
    se.compute_trajectory()
    assert torch.isfinite(se.weight[1]).all()
    assert len(se.theorem_trajectory.G) == 1

import warnings

import numpy as np
import torch

from src.algorithms import SelfTrainedGradientDescent
from src.config import AlgorithmConfig
from src.objectives import LogisticLoss, RidgePenalty


def _algorithm_config(n_iterations, ramp_start, ramp_end, step_size):
    return AlgorithmConfig(
        n_iterations=n_iterations,
        margin_threshold=1.0,
        step_size=step_size,
        penalty_param=0.01,
        pseudo_label_param=0.1,
        ramp_start=ramp_start,
        ramp_end=ramp_end,
        loss_function=LogisticLoss(),
        penalty_function=RidgePenalty(),
    )


def test_self_training_does_not_emit_runtime_warnings():
    rng = np.random.default_rng(0)
    X_lab = rng.normal(size=(40, 3))
    y_lab = np.where(X_lab[:, 0] + X_lab[:, 1] > 0, 1, -1)
    X_unl = rng.normal(size=(40, 3))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        learner = SelfTrainedGradientDescent(
            cfg=_algorithm_config(
                n_iterations=20,
                ramp_start=5,
                ramp_end=15,
                step_size=0.5,
            )
        )
        X_lab = torch.as_tensor(X_lab)
        y_lab = torch.as_tensor(y_lab)
        X_unl = torch.as_tensor(X_unl)
        learner.fit(X_lab, y_lab, X_unl)
        fields = learner.compute_preactivation(X_unl)

    assert torch.isfinite(fields).all()
    assert torch.isfinite(learner.weights).all()


def test_field_is_safe_for_extreme_weights():
    learner = SelfTrainedGradientDescent(
        cfg=_algorithm_config(
            n_iterations=5,
            ramp_start=1,
            ramp_end=4,
            step_size=0.1,
        )
    )

    fields = learner.compute_preactivation(
        torch.tensor([[1.0, -2.0], [0.5, 1.5]]),
        bias=1e300,
        weights=torch.tensor([1e300, -1e300]),
    )

    assert torch.isfinite(fields).all()

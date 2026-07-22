import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algorithms import SelfTrainedGradientDescent
from objectives import LogisticLoss, RidgePenalty


def test_self_training_does_not_emit_runtime_warnings():
    rng = np.random.default_rng(0)
    X_lab = rng.normal(size=(40, 3))
    y_lab = np.where(X_lab[:, 0] + X_lab[:, 1] > 0, 1, -1)
    X_unl = rng.normal(size=(40, 3))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        learner = SelfTrainedGradientDescent(
            n_iterations=20,
            margin_threshold=1.0,
            step_size=0.5,
            penalty_param=0.01,
            pseudo_label_param=0.1,
            ramp_start=5,
            ramp_end=15,
            loss_function=LogisticLoss(),
            penalty_function=RidgePenalty(),
        )
        learner.fit(X_lab, y_lab, X_unl)
        scores = learner.score(X_unl)

    assert np.isfinite(scores).all()
    assert np.isfinite(learner.weights).all()


def test_score_is_safe_for_extreme_weights():
    learner = SelfTrainedGradientDescent(
        n_iterations=5,
        margin_threshold=1.0,
        step_size=0.1,
        penalty_param=0.01,
        pseudo_label_param=0.1,
        ramp_start=1,
        ramp_end=4,
        loss_function=LogisticLoss(),
        penalty_function=RidgePenalty(),
    )

    old_err_settings = np.seterr(all="raise")
    try:
        scores = learner.score(
            np.array([[1.0, -2.0], [0.5, 1.5]]),
            weights=np.array([1e300, -1e300, 1e300]),
        )
    finally:
        np.seterr(**old_err_settings)

    assert np.isfinite(scores).all()

import numpy as np
from dataclasses import dataclass, field
from typing import List

from algorithms import SelfTrainedGradientDescent


@dataclass
class TestEvaluatorCallback:
    """
    Evaluates the model on a fixed test set at every iteration.
    """

    X_lab: np.ndarray
    Y_lab: np.ndarray
    X_unl: np.ndarray
    Y_unl: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray

    lab_error_history: List[float] = field(default_factory=list)
    unl_error_history: List[float] = field(default_factory=list)
    test_error_history: List[float] = field(default_factory=list)
    unl_usage_history: List[float] = field(default_factory=list)
    unl_flipping_rate_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.X_test = np.asarray(self.X_test)
        self.Y_test = np.asarray(self.Y_test)

    def __call__(self, learner: SelfTrainedGradientDescent):
        """
        This method is executed by the trainer at the end of each step.
        """
        # Evaluate on labeled, unlabeled, and test sets
        preds_lab = learner.predict(self.X_lab)
        lab_error = np.mean(preds_lab != self.Y_lab)
        self.lab_error_history.append(lab_error)

        preds_unl = learner.predict(self.X_unl)
        unl_error = np.mean(preds_unl != self.Y_unl)
        self.unl_error_history.append(unl_error)

        preds_test = learner.predict(self.X_test)
        test_error = np.mean(preds_test != self.Y_test)
        self.test_error_history.append(test_error)

        scores_unl = learner.score(self.X_unl)
        unl_usage = np.sum(np.abs(scores_unl) >= learner.margin_threshold) / len(self.Y_unl)
        self.unl_usage_history.append(unl_usage)

        if learner.prev_scores_ is not None:
            unl_flipping_rate = np.mean(preds_unl != np.where(learner.prev_scores_ >= 0, 1, -1))
            self.unl_flipping_rate_history.append(unl_flipping_rate)

        return 
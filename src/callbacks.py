import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Literal

from src.algorithms import SelfTrainedGradientDescent

# Define the allowed metrics for strict static type checking
MetricName = Literal[
    "lab_error", 
    "unl_error", 
    "test_error", 
    "unl_usage", 
    "unl_flipping_rate"
]

@dataclass
class TestEvaluatorCallback:
    """
    Evaluates the model dynamically based on requested tracking metrics.
    Possible metrics include:
    - "lab_error": Error on labeled data
    - "unl_error": Error on unlabeled data (requires Y_unl)
    - "test_error": Error on test data (requires X_test and Y_test)
    - "unl_usage": Fraction of unlabeled samples used for pseudo-labeling
    - "unl_flipping_rate": Rate of label flipping in unlabeled data across iterations
    """
    # Data arrays are optional; you only need to provide the ones you intend to track
    X_lab: Optional[np.ndarray] = None
    Y_lab: Optional[np.ndarray] = None
    X_unl: Optional[np.ndarray] = None
    Y_unl: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    Y_test: Optional[np.ndarray] = None

    # Use a set for O(1) runtime lookups
    metrics: Set[MetricName] = field(
        default_factory=lambda: {"test_error"} 
    )

    # Condense all tracking into a single dictionary
    history_: Dict[str, List[float]] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Initialize history dictionary and validate data dependencies."""
        for metric in self.metrics:
            self.history_[metric] = []

        # Fail fast: ensure the required data was provided for the requested metrics
        if "lab_error" in self.metrics:
            assert self.X_lab is not None and self.Y_lab is not None, "X_lab and Y_lab required for lab_error."
        if {"unl_error", "unl_usage", "unl_flipping_rate"} & self.metrics:
            assert self.X_unl is not None, "X_unl required for unlabeled metrics."
        if "unl_error" in self.metrics or "unl_usage" in self.metrics:
            assert self.Y_unl is not None, "Y_unl required for unl_error or unl_usage."
        if "test_error" in self.metrics:
            assert self.X_test is not None and self.Y_test is not None, "X_test and Y_test required for test_error."

    def __call__(self, learner: 'SelfTrainedGradientDescent'):
        """
        Executes at the end of each gradient step, computing only requested metrics.
        """
        # 1. Labeled Metrics
        if "lab_error" in self.metrics:
            preds_lab = learner.predict(self.X_lab)
            self.history_["lab_error"].append(np.mean(preds_lab != self.Y_lab))

        # 2. Unlabeled Metrics (Optimized to compute scores exactly once)
        if {"unl_error", "unl_usage", "unl_flipping_rate"} & self.metrics:
            scores_unl = learner.score(self.X_unl)
            preds_unl = np.where(scores_unl >= 0, 1, -1)

            if "unl_error" in self.metrics:
                self.history_["unl_error"].append(np.mean(preds_unl != self.Y_unl))

            if "unl_usage" in self.metrics:
                usage = np.sum(np.abs(scores_unl) >= learner.margin_threshold) / len(self.Y_unl)
                self.history_["unl_usage"].append(usage)

            if "unl_flipping_rate" in self.metrics:
                if learner.prev_scores_ is not None:
                    prev_preds = np.where(learner.prev_scores_ >= 0, 1, -1)
                    flipping_rate = np.mean(preds_unl != prev_preds)
                    self.history_["unl_flipping_rate"].append(flipping_rate)
                else:
                    self.history_["unl_flipping_rate"].append(0.0)

        # 3. Test Metrics
        if "test_error" in self.metrics:
            preds_test = learner.predict(self.X_test)
            self.history_["test_error"].append(np.mean(preds_test != self.Y_test))
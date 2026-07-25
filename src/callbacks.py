import torch

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Literal

from src.algorithms import SelfTrainedGradientDescent

MetricName = Literal[
    "lab_error", 
    "unl_error", 
    "test_error", 
    "unl_usage", 
    "unl_flipping_rate",
    "bias_term",
    "weight_signal_alignment",
    "label_score_alignment",
    "mean_score"
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

    - "lab_label_score_alignment": Tracks E[Y * score(X)] on the labeled set
    - "unl_label_score_alignment": Tracks E[Y * score(X)] on the unlabeled set
    - "test_label_score_alignment": Tracks E[Y * score(X)] on the test set

    - "lab_mean_score": Tracks E[score(X)] on the labeled set
    - "unl_mean_score": Tracks E[score(X)] on the unlabeled set
    - "test_mean_score": Tracks E[score(X)] on the test set

    - "bias_term": Tracks the value of the intercept (b)
    - "weight_vector_norm": Tracks the norm of the weight vector (||w||)
    - "weight_signal_alignment": Tracks the overlap between w and mu (requires mu)
    """
    X_lab: Optional[torch.Tensor] = None
    Y_lab: Optional[torch.Tensor] = None
    X_unl: Optional[torch.Tensor] = None
    Y_unl: Optional[torch.Tensor] = None
    X_test: Optional[torch.Tensor] = None
    Y_test: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None  # Required for weight_signal_alignment

    metrics: Set[MetricName] = field(
        default_factory=lambda: {"test_error"} 
    )

    history_: Dict[str, List[float]] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Initialize history dictionary and validate data dependencies."""
        for metric in self.metrics:
            self.history_[metric] = []

        # Validate standard data dependencies
        if "lab_error" in self.metrics:
            assert self.X_lab is not None and self.Y_lab is not None, "X_lab and Y_lab required for lab_error."
        if {"unl_error", "unl_usage", "unl_flipping_rate"} & self.metrics:
            assert self.X_unl is not None, "X_unl required for unlabeled metrics."
        if "unl_error" in self.metrics or "unl_usage" in self.metrics:
            assert self.Y_unl is not None, "Y_unl required for unl_error/unl_usage."
        if {"test_error", "label_score_alignment", "mean_score"} & self.metrics:
            assert self.X_test is not None, "X_test required for test metrics."
        if {"test_error", "label_score_alignment"} & self.metrics:
            assert self.Y_test is not None, "Y_test required for test errors/alignment."
            
        # Validate macroscopic tracking dependencies
        if "weight_signal_alignment" in self.metrics:
            assert self.mu is not None, "mu required for weight_signal_alignment."

    def __call__(self, learner: 'SelfTrainedGradientDescent'):
        """Executes at the end of each gradient step, computing only requested metrics."""
        
        # Use a local cache to store intermediate computations for this iteration
        cache = {}

        # --- Lazy Getters ---
        def get_scores_lab():
            if "scores_lab" not in cache:
                cache["scores_lab"] = learner.score(self.X_lab)
            return cache["scores_lab"]

        def get_preds_lab():
            if "preds_lab" not in cache:
                cache["preds_lab"] = learner.predict(self.X_lab)
            return cache["preds_lab"]

        def get_scores_unl():
            if "scores_unl" not in cache:
                cache["scores_unl"] = learner.score(self.X_unl)
            return cache["scores_unl"]

        def get_preds_unl():
            if "preds_unl" not in cache:
                cache["preds_unl"] = learner.predict(self.X_unl)
            return cache["preds_unl"]

        def get_scores_test():
            if "scores_test" not in cache:
                cache["scores_test"] = learner.score(self.X_test)
            return cache["scores_test"]

        def get_preds_test():
            if "preds_test" not in cache:
                cache["preds_test"] = learner.predict(self.X_test)
            return cache["preds_test"]


        # --- 1. Labeled Metrics ---
        if "lab_error" in self.metrics:
            error = (get_preds_lab() != self.Y_lab).float().mean().item()
            self.history_["lab_error"].append(error)
        
        if "lab_label_score_alignment" in self.metrics:
            alignment = torch.mean(self.Y_lab * get_scores_lab()).item()
            self.history_["lab_label_score_alignment"].append(alignment)

        if "lab_mean_score" in self.metrics:
            mean_score = torch.mean(get_scores_lab()).item()
            self.history_["lab_mean_score"].append(mean_score)


        # --- 2. Unlabeled Metrics ---
        if "unl_error" in self.metrics:
            error = (get_preds_unl() != self.Y_unl).float().mean().item()
            self.history_["unl_error"].append(error)

        if "unl_label_score_alignment" in self.metrics:
            alignment = torch.mean(self.Y_unl * get_scores_unl()).item()
            self.history_["unl_label_score_alignment"].append(alignment)

        if "unl_mean_score" in self.metrics:
            mean_score = torch.mean(get_scores_unl()).item()
            self.history_["unl_mean_score"].append(mean_score)


        # --- 3. Test Metrics ---
        if "test_error" in self.metrics:
            error = (get_preds_test() != self.Y_test).float().mean().item()
            self.history_["test_error"].append(error)
                
        if "test_label_score_alignment" in self.metrics:
            alignment = torch.mean(self.Y_test * get_scores_test()).item()
            self.history_["test_label_score_alignment"].append(alignment)

        if "test_mean_score" in self.metrics:
            mean_score = torch.mean(get_scores_test()).item()
            self.history_["test_mean_score"].append(mean_score)


        # --- 4. Pseudo-labeling Selection Metrics ---
        if "unl_usage" in self.metrics:
            usage = (torch.abs(get_scores_unl()) >= learner.cfg.margin_threshold).float().mean().item()
            self.history_["unl_usage"].append(usage)

        if "unl_flipping_rate" in self.metrics:
            if learner.prev_scores_ is not None:
                prev_preds = torch.where(learner.prev_scores_ >= 0, 1, -1)
                flipping_rate = (get_preds_unl() != prev_preds).float().mean().item()
                self.history_["unl_flipping_rate"].append(flipping_rate)
            else:
                self.history_["unl_flipping_rate"].append(0.0)


        # --- 5. Internal State Metrics ---
        if "bias_term" in self.metrics:
            # If learner.bias is a tensor, .item() safely extracts the float
            bias_val = learner.bias.item() if isinstance(learner.bias, torch.Tensor) else float(learner.bias)
            self.history_["bias_term"].append(bias_val)
        
        if "weight_vector_norm" in self.metrics:
            norm = torch.linalg.norm(learner.weights).item()
            self.history_["weight_vector_norm"].append(norm)
            
        if "weight_signal_alignment" in self.metrics:
            alignment = torch.mean(learner.weights * self.mu).item()
            self.history_["weight_signal_alignment"].append(alignment)
import torch

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Literal

from src.algorithms import SelfTrainedGradientDescent
from src.utils import compute_abstract_pseudo_residual_from, compute_population_error_from

MetricName = Literal[
    # Error terms
    "lab_error", 
    "unl_error", 
    "train_error",
    "test_error", 
    "population_error",
    # Label-Residual alignment terms
    "lab_label_residual_alignment",
    "unl_label_residual_alignment",
    "train_label_residual_alignment",
    "test_label_residual_alignment",
    # Mean Residual terms
    "lab_mean_residual",
    "unl_mean_residual",
    "train_mean_residual",
    "test_mean_residual",
    # Other Macroscopic terms
    "bias_term",
    "weight_vector_norm",
    "weight_signal_alignment",
    # Selection terms
    "unl_usage", 
    "unl_flipping_rate"
]

@dataclass
class TestEvaluatorCallback:
    """
    Evaluates the model dynamically based on requested tracking metrics.
    Possible metrics include:
    ---- 1. Error terms ----
    - "lab_error": Error on labeled data
    - "unl_error": Error on unlabeled data (requires Y_unl)
    - "train_error": Error on train data (requires X_train and Y_train)
    - "test_error": Error on test data (requires X_test and Y_test)
    ---- 2. Label-Residual alignment terms ----
    - "lab_label_residual_alignment": Tracks E[Y * g(X)] on the labeled set
    - "unl_label_residual_alignment": Tracks E[Y * g(X)] on the unlabeled set
    - "train_label_residual_alignment": Tracks E[Y * g(X)] on the train set
    - "test_label_residual_alignment": Tracks E[Y * g(X)] on the test set
    ---- 3. Mean Residual terms ----
    - "lab_mean_residual": Tracks E[g(X)] on the labeled set (mean residual)
    - "unl_mean_residual": Tracks E[g(X)] on the unlabeled set
    - "train_mean_residual": Tracks E[g(X)] on the train set
    - "test_mean_residual": Tracks E[g(X)] on the test set
    ---- 4. Other Macroscopic terms ----
    - "bias_term": Tracks the value of the intercept (b)
    - "weight_vector_norm": Tracks the norm of the weight vector (||w||)
    - "weight_signal_alignment": Tracks the overlap between w and mu (requires mu)
    ---- 5. Selection terms ----
    - "unl_usage": Fraction of unlabeled samples used for pseudo-labeling
    - "unl_flipping_rate": Rate of label flipping in unlabeled data across iterations
    """

    X_lab: Optional[torch.Tensor] = None
    Y_lab: Optional[torch.Tensor] = None
    X_unl: Optional[torch.Tensor] = None
    Y_unl: Optional[torch.Tensor] = None
    X_test: Optional[torch.Tensor] = None
    Y_test: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None  # Required for weight_signal_alignment
    sigma: Optional[float] = None # Required for population_error
    rho: Optional[float] = None # Required for population_error

    X_train: Optional[torch.Tensor] = field(init=False, default=None)
    Y_train: Optional[torch.Tensor] = field(init=False, default=None)

    metrics: Set[MetricName] = field(
        default_factory=lambda: {"test_error"} 
    )

    history_: Dict[str, List[float]] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Initialize history dictionary and validate data dependencies."""

        if "population_error" in self.metrics:
            self.metrics.update({"bias_term", "weight_vector_norm", "weight_signal_alignment"})

        if "train_mean_residual" in self.metrics:
            self.metrics.update({"lab_mean_residual", "unl_mean_residual"})

        if "train_label_residual_alignment" in self.metrics:
            self.metrics.update({"lab_label_residual_alignment", "unl_label_residual_alignment"})

        for metric in self.metrics:
            self.history_[metric] = []

        if self.X_lab is not None and self.X_unl is not None:
            self.X_train = torch.cat([self.X_lab, self.X_unl])
        if self.Y_lab is not None and self.Y_unl is not None:
            self.Y_train = torch.cat([self.Y_lab, self.Y_unl])  

        # Validate standard data dependencies
        if "lab_error" in self.metrics:
            assert self.X_lab is not None and self.Y_lab is not None, "X_lab and Y_lab required for lab_error."
        if {"unl_error", "unl_usage", "unl_flipping_rate"} & self.metrics:
            assert self.X_unl is not None, "X_unl required for unlabeled metrics."
        if "unl_error" in self.metrics or "unl_usage" in self.metrics:
            assert self.Y_unl is not None, "Y_unl required for unl_error/unl_usage."
        if {"test_error", "test_label_residual_alignment", "test_mean_residual"} & self.metrics:
            assert self.X_test is not None and self.Y_test is not None, "X_test and Y_test required for test metrics."
        if "train_error" in self.metrics:
            assert self.X_train is not None and self.Y_train is not None, "X_train and Y_train required for train metrics."
            
        # Validate macroscopic tracking dependencies
        if "weight_signal_alignment" in self.metrics:
            assert self.mu is not None, "mu required for weight_signal_alignment."

    def _compute_pseudo_residual(self, 
            learner: 'SelfTrainedGradientDescent', 
            preactivations: torch.Tensor, labels: torch.Tensor, 
            indicator_val: float, t: int, is_test_set: bool = False
        ) -> torch.Tensor:
        """
        Computes the pseudo-residual for the provided samples.
        """
        positive_margin = learner.cfg.positive_margin
        negative_margin = learner.cfg.negative_margin
        t_clamped = min(t, learner.cfg.n_iterations - 1)
        
        # --- FIX 2: Prevent IPW Inflation on Test Sets ---
        if is_test_set:
            coef = 0.0
            rho = 1.0
            unl_usage = 0.0
        else:
            coef = learner.cfg.get_pseudo_label_weight(t_clamped)
            n_lab = self.X_lab.shape[0] if self.X_lab is not None else 0
            n_unl = self.X_unl.shape[0] if self.X_unl is not None else 0
            rho = n_lab / (n_lab + n_unl) if (n_lab + n_unl) > 0 else 1.0
            
            if self.X_unl is not None:
                preactivations_unl = learner.compute_preactivation(self.X_unl)
                mask = learner.cfg.selection_function(
                    preactivations_unl, positive_margin, negative_margin
                )
                unl_usage = mask.double().mean().item()
            else:
                unl_usage = 0.0
            
        # 1. Callback uses the surrogate mask to match empirical and theoretical engines
        selection_mask = learner.cfg.selection_function(
            preactivations, positive_margin, negative_margin
        )
        indicator = torch.tensor(indicator_val, dtype=torch.float64, device=preactivations.device)
        
        # 2. Call the shared math engine
        return compute_abstract_pseudo_residual_from(
            preactivation=preactivations, label=labels, indicator=indicator,
            selection_mask=selection_mask, selection_rate=unl_usage,
            coef=coef, rho=rho, eta=learner.cfg.step_size, 
            loss_function=learner.cfg.loss_function
        )

    def __call__(self, learner: 'SelfTrainedGradientDescent', t: int):
        """
        Executes at the end of each gradient step, computing only requested metrics.
        """
        
        cache = {}

        # --- Lazy Getters ---
        def get_preactivations_lab():
            if "preactivations_lab" not in cache:
                cache["preactivations_lab"] = learner.compute_preactivation(self.X_lab)
            return cache["preactivations_lab"]

        def get_preactivations_unl():
            if "preactivations_unl" not in cache:
                cache["preactivations_unl"] = learner.compute_preactivation(self.X_unl)
            return cache["preactivations_unl"]

        def get_preactivations_test():
            if "preactivations_test" not in cache:
                cache["preactivations_test"] = learner.compute_preactivation(self.X_test)
            return cache["preactivations_test"]

        def get_preds_lab():
            if "preds_lab" not in cache:
                cache["preds_lab"] = learner.predict(self.X_lab)
            return cache["preds_lab"]

        def get_preds_unl():
            if "preds_unl" not in cache:
                cache["preds_unl"] = learner.predict(self.X_unl)
            return cache["preds_unl"]

        def get_preds_train():
            if "preds_train" not in cache:
                cache["preds_train"] = learner.predict(self.X_train)
            return cache["preds_train"]
        
        def get_preds_test():
            if "preds_test" not in cache:
                cache["preds_test"] = learner.predict(self.X_test)
            return cache["preds_test"]

        # --- 1. Labeled Metrics ---
        if "lab_error" in self.metrics:
            error = (get_preds_lab() != self.Y_lab).double().mean().item()
            self.history_["lab_error"].append(error)
        
        if "lab_label_residual_alignment" in self.metrics:
            g = self._compute_pseudo_residual(learner, get_preactivations_lab(), self.Y_lab, 1.0, t)
            self.history_["lab_label_residual_alignment"].append(torch.mean(self.Y_lab * g).item())

        if "lab_mean_residual" in self.metrics:
            g = self._compute_pseudo_residual(learner, get_preactivations_lab(), self.Y_lab, 1.0, t)
            self.history_["lab_mean_residual"].append(torch.mean(g).item())

        # --- 2. Unlabeled Metrics ---
        if "unl_error" in self.metrics:
            error = (get_preds_unl() != self.Y_unl).double().mean().item()
            self.history_["unl_error"].append(error)

        if "unl_label_residual_alignment" in self.metrics:
            g = self._compute_pseudo_residual(learner, get_preactivations_unl(), self.Y_unl, 0.0, t)
            self.history_["unl_label_residual_alignment"].append(torch.mean(self.Y_unl * g).item())

        if "unl_mean_residual" in self.metrics:
            g = self._compute_pseudo_residual(learner, get_preactivations_unl(), self.Y_unl, 0.0, t)
            self.history_["unl_mean_residual"].append(torch.mean(g).item())

        # --- 3. Train Metrics (Error Only!) ---
        if "train_error" in self.metrics:
            error = (get_preds_train() != self.Y_train).double().mean().item()
            self.history_["train_error"].append(error)

        n_lab = self.X_lab.shape[0] if self.X_lab is not None else 0
        n_unl = self.X_unl.shape[0] if self.X_unl is not None else 0
        rho = n_lab / (n_lab + n_unl) if (n_lab + n_unl) > 0 else 1.0

        if "train_mean_residual" in self.metrics:
            lab = self.history_["lab_mean_residual"][-1]
            unl = self.history_["unl_mean_residual"][-1]
            self.history_["train_mean_residual"].append(rho * lab + (1 - rho) * unl)

        if "train_label_residual_alignment" in self.metrics:
            lab = self.history_["lab_label_residual_alignment"][-1]
            unl = self.history_["unl_label_residual_alignment"][-1]
            self.history_["train_label_residual_alignment"].append(rho * lab + (1 - rho) * unl)

        # --- 4. Test Metrics ---
        if "test_error" in self.metrics:
            error = (get_preds_test() != self.Y_test).double().mean().item()
            self.history_["test_error"].append(error)
                
        if "test_label_residual_alignment" in self.metrics:
            g = self._compute_pseudo_residual(learner, get_preactivations_test(), self.Y_test, 1.0, t, is_test_set=True)
            self.history_["test_label_residual_alignment"].append(torch.mean(self.Y_test * g).item())

        if "test_mean_residual" in self.metrics:
            g = self._compute_pseudo_residual(learner, get_preactivations_test(), self.Y_test, 1.0, t, is_test_set=True)
            self.history_["test_mean_residual"].append(torch.mean(g).item())


        # --- 5. Pseudo-labeling Selection Metrics ---
        if "unl_usage" in self.metrics:
            preact_unl = get_preactivations_unl()
            mask = learner.cfg.selection_function(
                preact_unl, learner.cfg.positive_margin, learner.cfg.negative_margin
            )
            usage = mask.double().mean().item()
            self.history_["unl_usage"].append(usage)

        if "unl_flipping_rate" in self.metrics:
            if learner.prev_preactivations_ is not None:
                prev_preds = torch.where(learner.prev_preactivations_ >= 0, 1, -1)
                flipping_rate = (get_preds_unl() != prev_preds).double().mean().item()
                self.history_["unl_flipping_rate"].append(flipping_rate)
            else:
                self.history_["unl_flipping_rate"].append(0.0)


        # --- 6. Internal State Metrics ---
        if "bias_term" in self.metrics:
            bias_val = learner.bias.item() if isinstance(learner.bias, torch.Tensor) else float(learner.bias)
            self.history_["bias_term"].append(bias_val)
        
        if "weight_vector_norm" in self.metrics:
            # FIX 3: Dimension-normalized norm squared
            d = len(learner.weights)
            norm = (torch.linalg.norm(learner.weights) / (d**0.5)).item()
            self.history_["weight_vector_norm"].append(norm)
            
        if "weight_signal_alignment" in self.metrics:
            alignment = torch.mean(learner.weights * self.mu).item()
            self.history_["weight_signal_alignment"].append(alignment)

        if "population_error" in self.metrics:
            pop_err = compute_population_error_from(
                b=self.history_["bias_term"][-1],
                m=self.history_["weight_signal_alignment"][-1],
                tau=self.history_["weight_vector_norm"][-1],
                sigma=self.sigma,
                rho=self.rho
            )
            self.history_["population_error"].append(pop_err)
       
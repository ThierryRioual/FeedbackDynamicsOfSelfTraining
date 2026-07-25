import torch

from dataclasses import dataclass, field
from typing import List, Callable, Optional, Tuple

from src.config import AlgorithmConfig
from src.validation import validate_self_training_data, validate_gradient_step
from src.objectives import LossFunction, Penalty, LogisticLoss, RidgePenalty

@dataclass
class SelfTrainedGradientDescent:
    """
    A self-training algorithm that uses gradient descent for updating the model parameters. 
    (Implementation of Algorithm 1)
    """
    cfg: AlgorithmConfig
    callback: Optional[Callable[['SelfTrainedGradientDescent'], float]] = None

    bias: float = field(init=False, default=None) # b
    weights: torch.Tensor = field(init=False, default=None) # w
    prev_scores_: torch.Tensor = field(init=False, default=None) # Tracks which unlabeled samples are used for pseudo-labeling

    pseudo_label_param_schedule_: torch.Tensor = field(init=False, default=None)

    def __post_init__(self):
        """
        """
        t = torch.arange(self.cfg.n_iterations)
        t_clamped = torch.clamp(t, min=self.cfg.ramp_start, max=self.cfg.ramp_end)
        self.pseudo_label_param_schedule_ = self.cfg.pseudo_label_param * (t_clamped - self.cfg.ramp_start) / (self.cfg.ramp_end - self.cfg.ramp_start)
         

    def _compute_empirical_risk_gradient(self, 
                          scores_lab: torch.Tensor, 
                          Y_lab: torch.Tensor,
                          scores_unl: torch.Tensor, 
                          alpha: float) -> torch.Tensor:
        """
        Computes the empirical risk of the self-training algorithm. ($\\nabla R$ vector)
        """
        
        # Compute gradient from labeled data
        n_lab = scores_lab.shape[0]
        grad_lab = self.cfg.loss_function.gradient(scores_lab, Y_lab) / n_lab

        # Compute gradient from pseudo-labeled unlabeled data
        n_unl = scores_unl.shape[0]
        mask = torch.abs(scores_unl) >= self.cfg.margin_threshold
        n_psd = mask.sum()
        Y_psd = torch.where(scores_unl >= 0, 1, -1)
        if n_psd > 0:
            grad_unl = self.cfg.loss_function.gradient(scores_unl, Y_psd) * mask / n_psd
        else:
            grad_unl = torch.zeros_like(scores_unl)

        self.prev_scores_ = scores_unl

        n_total = n_lab + n_unl 

        return n_total * torch.concatenate([grad_lab, alpha * grad_unl])


    def _compute_gradient(self, bias: Optional[float], weights: torch.Tensor, 
                          X_lab: torch.Tensor, 
                          Y_lab: torch.Tensor,
                          X_unl: torch.Tensor, 
                          alpha: float) -> Tuple[float, torch.Tensor]:
        """
        Computes the gradient of the loss function with respect to model parameters weights.
        This includes contributions from labeled data, pseudo-labeled unlabeled data, and regularization.
        ($$ 1^\top \nabla R(r) / n, \sqrt{d}/n * X^\top \nabla R(r) + \lambda \nabla P(w) $$) 
        """

        bias = bias if bias is not None else 0

        d = torch.tensor(weights.shape[0])
        n = torch.tensor(X_lab.shape[0] + X_unl.shape[0])

        scores_lab = (X_lab @ weights) / torch.sqrt(d) + bias
        scores_unl = (X_unl @ weights) / torch.sqrt(d) + bias

        emp_risk = self._compute_empirical_risk_gradient(scores_lab, Y_lab, scores_unl, alpha) # \nabla R

        X_total = torch.concatenate([X_lab, X_unl])

        # Compute gradient of penalty term
        grad_pen = self.cfg.penalty_function.gradient(weights)

        return torch.mean(emp_risk), (torch.sqrt(d) / n) * (X_total.T @ emp_risk) + self.cfg.penalty_param * grad_pen

    def fit(self, X_lab: torch.Tensor, Y_lab: torch.Tensor, X_unl: torch.Tensor, 
            initial_bias: Optional[float] = None, initial_weights: Optional[torch.Tensor] = None) -> List[float]:
        """
        Fits the self-training model to the provided labeled and unlabeled data.
        """

        d = X_lab.shape[1]

        assert (self.cfg.include_bias) or (initial_bias is None), "Cannot initialize inital bias when inlcude_bias is set to False"
        self.bias = initial_bias if initial_bias is not None else 0.0
        self.weights = initial_weights if initial_weights is not None else torch.normal(mean=0.0, std=1.0, size=(d,)) 

        if self.callback is not None:
            self.callback(self)

        for t in range(self.cfg.n_iterations):
            psd_param = self.pseudo_label_param_schedule_[t] # \pi^t
        
            # Compute the gradient and update weights
            b_grad, w_grad = self._compute_gradient(self.bias, self.weights, X_lab, Y_lab, X_unl, psd_param)

            # Catch gradient divergence explicitly
            #validate_gradient_step(t, self.weights, grad)

            if self.cfg.include_bias:
                self.bias -= self.cfg.step_size * b_grad # Update bias

            self.weights -= self.cfg.step_size * w_grad # Update weights

            if self.callback is not None:
                self.callback(self)

        return self.bias, self.weights
    
    def score(self, X: torch.Tensor, bias: Optional[float] = None, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the raw scores (logits) for the given input data X using the learned weights.
        If weights are not provided, it uses the model's current weights.
        """
        assert (self.cfg.include_bias) or (bias is None), "Cannot include bias when inlcude_bias is set to False"

        if bias is None:
            bias = self.bias if (self.cfg.include_bias and self.bias is not None) else 0

        weights = weights if weights is not None else self.weights
        d = torch.tensor(weights.shape[0])

        return (X @ weights) / torch.sqrt(d) + bias
        
    def predict(self, X: torch.Tensor, bias: Optional[float] = None, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predicts the labels for the given input data X using the learned weights.
        If weights are not provided, it uses the model's current weights.
        """
        assert (self.cfg.include_bias) or (bias is None), "Cannot include bias when inlcude_bias is set to False"
        return torch.where(self.score(X, bias, weights) >= 0, 1, -1)
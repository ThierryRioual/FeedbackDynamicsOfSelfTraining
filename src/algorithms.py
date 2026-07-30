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
    callback: Optional[Callable[['SelfTrainedGradientDescent', int], None]] = None

    bias: float = field(init=False, default=None) # b
    weights: torch.Tensor = field(init=False, default=None) # w
    prev_preactivations_: torch.Tensor = field(init=False, default=None) # Tracks unlabeled preactivations for pseudo-labeling flipping rate

    def _compute_empirical_risk_gradient(self, 
                          preactivations_lab: torch.Tensor, 
                          Y_lab: torch.Tensor,
                          preactivations_unl: torch.Tensor, 
                          pi: float) -> torch.Tensor:
        """
        Computes the gradient of the empirical risk with respect to the preactivation vector r.
        Returns the pseudo-residual vector g = ∇_r R(r). ($\\nabla R$ vector)
        """
        
        # Compute gradient from labeled data
        n_lab = preactivations_lab.shape[0]
        grad_lab = self.cfg.loss_function.gradient(preactivations_lab, Y_lab) / n_lab

        # Compute gradient from pseudo-labeled unlabeled data
        n_unl = preactivations_unl.shape[0]
        mask = self.cfg.selection_function(
            preactivations_unl, 
            self.cfg.positive_margin, 
            self.cfg.negative_margin
        )
        n_psd = mask.sum()
        Y_psd = torch.where(preactivations_unl >= 0, 1, -1)
        if n_psd > 1e-10:
            grad_unl = self.cfg.loss_function.gradient(preactivations_unl, Y_psd) * mask / n_psd
        else:
            grad_unl = torch.zeros_like(preactivations_unl)

        self.prev_preactivations_ = preactivations_unl

        n_total = n_lab + n_unl 

        return n_total * torch.concatenate([grad_lab, pi * grad_unl])


    def _compute_gradient_step(self, bias: Optional[float], weights: torch.Tensor, 
                          X_lab: torch.Tensor, 
                          Y_lab: torch.Tensor,
                          X_unl: torch.Tensor, 
                          pi: float) -> Tuple[float, torch.Tensor]:
        """
        Computes the full gradient step for model parameters (bias and weights).
        First computes the preactivation r = Xw/√d + b, then the pseudo-residual g = ∇_r ℓ(r, y),
        and finally backpropagates through X to get the weight gradient plus the decay step.
        ($$ 1^\top \nabla R(r) / n, \sqrt{d}/n * X^\top \nabla R(r) + \lambda \nabla P(w) $$) 
        """

        bias = bias if bias is not None else 0

        d = torch.tensor(weights.shape[0])
        n = torch.tensor(X_lab.shape[0] + X_unl.shape[0])

        preactivations_lab = (X_lab @ weights) / torch.sqrt(d) + bias
        preactivations_unl = (X_unl @ weights) / torch.sqrt(d) + bias

        emp_risk = self._compute_empirical_risk_gradient(preactivations_lab, Y_lab, preactivations_unl, pi) # pseudo-residual

        X_total = torch.concatenate([X_lab, X_unl])

        # Compute the decay step (weight decay / penalty gradient)
        decay = self.cfg.penalty_function.gradient(weights)

        return torch.mean(emp_risk), (torch.sqrt(d) / n) * (X_total.T @ emp_risk) + self.cfg.penalty_param * decay

    def fit(self, X_lab: torch.Tensor, Y_lab: torch.Tensor, X_unl: torch.Tensor, 
            initial_bias: Optional[float] = None, initial_weights: Optional[torch.Tensor] = None) -> None:
        """
        Fits the self-training model to the provided labeled and unlabeled data.
        """

        d = X_lab.shape[1]

        assert (self.cfg.include_bias) or (initial_bias is None), "Cannot initialize inital bias when inlcude_bias is set to False"
        self.bias = initial_bias if initial_bias is not None else 0.0
        self.weights = initial_weights if initial_weights is not None else torch.normal(mean=0.0, std=1.0, size=(d,)) 

        if self.callback is not None:
            self.callback(self, 0)

        for t in range(self.cfg.n_iterations):
            psd_param = self.cfg.get_pseudo_label_weight(t) # \pi^t
        
            # Compute the gradient and update weights
            b_grad, w_grad = self._compute_gradient_step(self.bias, self.weights, X_lab, Y_lab, X_unl, psd_param)

            # Catch gradient divergence explicitly
            #validate_gradient_step(t, self.weights, grad)

            if self.cfg.include_bias:
                self.bias -= self.cfg.step_size * b_grad # Update bias

            self.weights -= self.cfg.step_size * w_grad # Update weights

            if self.callback is not None:
                self.callback(self, t + 1)

        return None
    
    def compute_preactivation(self, X: torch.Tensor, bias: Optional[float] = None, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the preactivation r = Xw/√d + b for the given input data X.
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
        Predicts the binary labels for the given input data X by thresholding the preactivation.
        If weights are not provided, it uses the model's current weights.
        """
        assert (self.cfg.include_bias) or (bias is None), "Cannot include bias when inlcude_bias is set to False"
        return torch.where(self.compute_preactivation(X, bias, weights) >= 0, 1, -1)
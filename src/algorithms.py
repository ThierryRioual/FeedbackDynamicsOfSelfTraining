from numpy.random import bit_generator
import subprocess
import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Tuple

from src.validation import validate_self_training_data, validate_gradient_step
from src.objectives import LossFunction, Penalty, LogisticLoss, RidgePenalty

@dataclass
class SelfTrainedGradientDescent:
    """
    A self-training algorithm that uses gradient descent for updating the model parameters. 
    (Implementation of Algorithm 1)
    """
    n_iterations: int # T
    margin_threshold: float # \kappa
    step_size: float # \eta
    penalty_param: float # \lambda
    pseudo_label_param: float # \alpha
    ramp_start: int # T_0
    ramp_end: int # T_1

    include_bias: bool = True

    loss_function: LossFunction = field(default_factory=LogisticLoss) # \ell
    penalty_function: Penalty = field(default_factory=RidgePenalty)

    bias: float = field(init=False, default=None) # b
    weights: np.ndarray = field(init=False, default=None) # w
    callback: Optional[Callable[['SelfTrainedGradientDescent'], float]] = None

    prev_scores_: np.ndarray = field(init=False, default=None)  
    # Tracks which unlabeled samples are used for pseudo-labeling

    @property
    def pseudo_label_param_schedule(self) -> np.ndarray:
        """
        Precompute the schedule for pseudo-labeling
        """
        func = lambda t: self.pseudo_label_param * (t - self.ramp_start) / (self.ramp_end - self.ramp_start) \
            if self.ramp_start <= t < self.ramp_end else \
            (0.0 if t < self.ramp_start else self.pseudo_label_param)  
        vfunc = np.vectorize(func)
        return vfunc(np.arange(self.n_iterations))                

    def _compute_empirical_risk_gradient(self, 
                          scores_lab: np.ndarray, 
                          Y_lab: np.ndarray,
                          scores_unl: np.ndarray, 
                          alpha: float) -> np.ndarray:
        """
        Computes the empirical risk of the self-training algorithm. (\nabla R vector)
        """
        
        # Compute gradient from labeled data
        n_lab = scores_lab.shape[0]
        grad_lab = self.loss_function.gradient(scores_lab, Y_lab) / n_lab

        # Compute gradient from pseudo-labeled unlabeled data
        n_unl = scores_unl.shape[0]
        mask = np.abs(scores_unl) >= self.margin_threshold
        n_psd = mask.sum()
        Y_psd = np.where(scores_unl >= 0, 1, -1)
        if n_psd > 0:
            grad_unl = self.loss_function.gradient(scores_unl, Y_psd) * mask / n_psd
        else:
            grad_unl = np.zeros_like(scores_unl)

        self.prev_scores_ = scores_unl

        n_total = n_lab + n_unl 

        return n_total * np.concatenate([grad_lab, alpha * grad_unl])


    def _compute_gradient(self, bias: Optional[float], weights: np.ndarray, 
                          X_lab: np.ndarray, 
                          Y_lab: np.ndarray,
                          X_unl: np.ndarray, 
                          alpha: float) -> Tuple[float, np.ndarray]:
        """
        Computes the gradient of the loss function with respect to model parameters weights.
        This includes contributions from labeled data, pseudo-labeled unlabeled data, and regularization.
        ( 1^\top \nabla R(r) / n, \sqrt{d}/n * X^\top \nabla R(r) + \lambda \nabla P(w) ) 
        """

        bias = bias if bias is not None else 0

        d = weights.shape[0]
        n = X_lab.shape[0] + X_unl.shape[0]

        scores_lab = (X_lab @ weights) / np.sqrt(d) + bias
        scores_unl = (X_unl @ weights) / np.sqrt(d) + bias

        emp_risk = self._compute_empirical_risk_gradient(scores_lab, Y_lab, scores_unl, alpha) # \nabla R

        X_total = np.concatenate([X_lab, X_unl])

        # Compute gradient of penalty term
        grad_pen = self.penalty_function.gradient(weights)

        return np.average(emp_risk), (np.sqrt(d) / n) * (X_total.T @ emp_risk) + self.penalty_param * grad_pen

    def fit(self, X_lab: np.ndarray, Y_lab: np.ndarray, X_unl: np.ndarray, 
            initial_bias: Optional[float] = None, initial_weights: Optional[np.ndarray] = None) -> List[float]:
        """
        Fits the self-training model to the provided labeled and unlabeled data.
        """
        N, M, d, _ = validate_self_training_data(X_lab, Y_lab, X_unl)

        assert (self.include_bias) or (initial_bias is None), "Cannot initialize inital bias when inlcude_bias is set to False"
        self.bias = initial_bias if initial_bias is not None else 0
        self.weights = initial_weights if initial_weights is not None else np.random.normal(0, 1, size=d) 

        if self.callback is not None:
            self.callback(self)

        for t in range(self.n_iterations):
            psd_param = self.pseudo_label_param_schedule[t] # \pi^t
        
            # Compute the gradient and update weights
            b_grad, w_grad = self._compute_gradient(self.bias, self.weights, X_lab, Y_lab, X_unl, psd_param)

            # Catch gradient divergence explicitly
            #validate_gradient_step(t, self.weights, grad)

            if self.include_bias:
                self.bias -= self.step_size * b_grad # Update bias

            self.weights -= self.step_size * w_grad # Update weights

            if self.callback is not None:
                self.callback(self)

        return self.bias, self.weights
    
    def score(self, X: np.ndarray, bias: Optional[float] = None, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the raw scores (logits) for the given input data X using the learned weights.
        If weights are not provided, it uses the model's current weights.
        """
        assert (self.include_bias) or (bias is None), "Cannot include bias when inlcude_bias is set to False"

        if bias is None:
            bias = self.bias if (self.include_bias and self.bias is not None) else 0

        weights = weights if weights is not None else self.weights
        d = weights.shape[0]

        return (X @ weights) / np.sqrt(d) + bias
        
    def predict(self, X: np.ndarray, bias: Optional[float] = None, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predicts the labels for the given input data X using the learned weights.
        If weights are not provided, it uses the model's current weights.
        """
        assert (self.include_bias) or (bias is None), "Cannot include bias when inlcude_bias is set to False"
        return np.where(self.score(X, bias, weights) >= 0, 1, -1)
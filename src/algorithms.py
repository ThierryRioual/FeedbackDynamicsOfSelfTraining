import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Optional

import validation
from objectives import LossFunction, Penalty, LogisticLoss, RidgePenalty

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

    loss_function: LossFunction = field(default_factory=LogisticLoss) # \ell
    penalty_function: Penalty = field(default_factory=RidgePenalty)

    weights: np.ndarray = field(init=False, default=None)
    callback: Optional[Callable[['SelfTrainedGradientDescent'], float]] = None

    prev_scores_: np.ndarray = field(init=False, default=None)  # Tracks which unlabeled samples are used for pseudo-labeling

    def _compute_gradient(self, weights: np.ndarray, 
                          X_lab_aug: np.ndarray, 
                          Y_lab: np.ndarray,
                          X_unl_aug: np.ndarray, 
                          alpha: float) -> np.ndarray:
        """
        Computes the gradient of the loss function with respect to model parameters weights.
        This includes contributions from labeled data, pseudo-labeled unlabeled data, and regularization.
        """

        # Compute gradient from labeled data
        grad_lab = self.loss_function.gradient(weights, X_lab_aug, Y_lab)

        # Compute gradient from pseudo-labeled unlabeled data
        scores_unl = X_unl_aug @ weights
        mask = np.abs(scores_unl) >= self.margin_threshold
        n_psd = mask.sum()
        grad_unl = np.zeros_like(weights)
        if n_psd > 0:
            X_psd = X_unl_aug[mask]
            scores_psd = scores_unl[mask]
            Y_psd = np.where(scores_psd >= 0, 1, -1)
            grad_unl = self.loss_function.gradient(weights, X_psd, Y_psd)

        self.prev_scores_ = scores_unl

        # Compute gradient of penalty term
        grad_pen = self.penalty_function.gradient(weights)

        return grad_lab + alpha * grad_unl + self.penalty_param * grad_pen

    def fit(self, X_lab: np.ndarray, Y_lab: np.ndarray, X_unl: np.ndarray, 
            initial_weights: np.ndarray = None) -> List[float]:
        """
        Fits the self-training model to the provided labeled and unlabeled data.
        """
        N, M, d, _ = validation.validate_self_training_data(X_lab, Y_lab, X_unl)

        self.weights = initial_weights if initial_weights is not None \
            else np.random.normal(0, 1/np.sqrt(d), size=d+1)  # +1 for bias term
        
        if self.callback is not None:
            self.callback(self)

        # Augment the labeled and unlabeled data with a bias term
        X_lab_aug = np.hstack([np.ones((N, 1)), X_lab])
        X_unl_aug = np.hstack([np.ones((M, 1)), X_unl])

        # Precompute the alpha schedule for pseudo-labeling
        func = lambda t: self.pseudo_label_param * (t - self.ramp_start) / (self.ramp_end - self.ramp_start) \
            if self.ramp_start <= t < self.ramp_end else \
            (0.0 if t < self.ramp_start else self.pseudo_label_param)  
        vfunc = np.vectorize(func)
        alpha_list = vfunc(np.arange(self.n_iterations))

        for t in range(self.n_iterations):
            alpha_t = alpha_list[t]
        
            # Compute the gradient and update weights
            grad = self._compute_gradient(self.weights, X_lab_aug, Y_lab, X_unl_aug, alpha_t)
            self.weights -= self.step_size * grad

            if self.callback is not None:
                self.callback(self)

        return self.weights
    
    def score(self, X: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
        """
        Computes the raw scores (logits) for the given input data X using the learned weights.
        If weights are not provided, it uses the model's current weights.
        """
        if weights is None:
            weights = self.weights
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_aug @ weights
    
    def predict(self, X: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
        """
        Predicts the labels for the given input data X using the learned weights.
        If weights are not provided, it uses the model's current weights.
        """
        scores = self.score(X, weights)
        return np.where(scores >= 0, 1, -1)
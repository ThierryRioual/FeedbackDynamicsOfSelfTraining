import numpy as np
from dataclasses import dataclass
from typing import List, Union
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

@dataclass
class SelfTrainingExperiment:
    X_lab: np.ndarray
    Y_lab: np.ndarray
    X_unl: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray

    def _resolve_schedule(self, param: Union[float, List[float], np.ndarray], 
                          expected_length: int, param_name: str) -> np.ndarray:
        """
        Validates and expands a scalar or sequence into a deterministic time-indexed schedule.
        """
        # If a scalar is provided, broadcast it to a constant sequence
        if isinstance(param, (int, float, np.number)):
            return np.full(expected_length, float(param))
        
        # If a list/array is provided, rigorously verify its dimension
        param_seq = np.asarray(param, dtype=float)
        if len(param_seq) != expected_length:
            raise ValueError(
                f"Dimension mismatch for '{param_name}': expected schedule of length "
                f"{expected_length}, got {len(param_seq)}."
            )
        
        return param_seq


@dataclass
class FullRefitSelfTrainer(SelfTrainingExperiment):
    
    def _fit_logistic(self, X_pseudo: np.ndarray, Y_pseudo: np.ndarray, 
                      gamma: float, lambd: float) -> LogisticRegression:
        N_lab = self.X_lab.shape[0]
        N_pseudo = X_pseudo.shape[0]

        if N_pseudo == 0:
            X_combined = self.X_lab
            Y_combined = self.Y_lab
            sample_weights = np.full(N_lab, 1.0 / N_lab)
        else:
            X_combined = np.vstack([self.X_lab, X_pseudo])
            Y_combined = np.concatenate([self.Y_lab, Y_pseudo])
            
            weights_lab = np.full(N_lab, 1.0 / N_lab)
            weights_pseudo = np.full(N_pseudo, gamma / N_pseudo)
            sample_weights = np.concatenate([weights_lab, weights_pseudo])
        
        C_param = 1.0 / lambd if lambd > 0 else 1e9

        clf = LogisticRegression(
            penalty='l2',
            C=C_param,
            fit_intercept=True, 
            solver='lbfgs',
            max_iter=1000,
            tol=1e-6
        )

        clf.fit(X_combined, Y_combined, sample_weight=sample_weights)
        return clf

    def run(self, kappa: Union[float, List[float], np.ndarray], 
                  gamma: Union[float, List[float], np.ndarray], 
                  lambd: Union[float, List[float], np.ndarray], 
                  T: int, **kwargs) -> List[float]:
        
        # 1. Resolve Hyperparameter Schedules
        kappa_seq = self._resolve_schedule(kappa, T, 'kappa')
        gamma_seq = self._resolve_schedule(gamma, T, 'gamma')
        lambd_seq = self._resolve_schedule(lambd, T + 1, 'lambd')

        empty_X = np.empty((0, self.X_lab.shape[1]))
        empty_Y = np.empty(0)
        
        # 2. Base Estimator (t=0)
        # Note: gamma is mathematically ignored when |I_t| = 0, so 1.0 is safely passed.
        clf = self._fit_logistic(empty_X, empty_Y, gamma=1.0, lambd=lambd_seq[0])
        errors = [1.0 - clf.score(self.X_test, self.Y_test)]

        # 3. Iterative Updates
        for t in range(T):
            scores = clf.decision_function(self.X_unl)
            mask = np.abs(scores) >= kappa_seq[t]

            if mask.sum() == 0:
                errors.append(errors[-1])
                continue

            Y_pseudo = np.where(scores[mask] >= 0, 1, -1)
            X_pseudo = self.X_unl[mask]

            clf = self._fit_logistic(X_pseudo, Y_pseudo, gamma_seq[t], lambd_seq[t + 1])
            errors.append(1.0 - clf.score(self.X_test, self.Y_test))

        return errors


@dataclass
class GradientStepSelfTrainer(SelfTrainingExperiment):
    
    def _compute_gradient(self, theta: np.ndarray, 
                          X_lab_aug: np.ndarray, 
                          X_unl_aug: np.ndarray, 
                          kappa: float, gamma: float, lambd: float) -> np.ndarray:
        
        N_lab = X_lab_aug.shape[0]
        
        margins_lab = self.Y_lab * (X_lab_aug @ theta)
        weights_lab = -self.Y_lab * expit(-margins_lab)
        grad_lab = (X_lab_aug.T @ weights_lab) / N_lab
        
        scores_unl = X_unl_aug @ theta
        mask = np.abs(scores_unl) >= kappa
        num_pseudo = mask.sum()
        
        grad_unl = np.zeros_like(theta)
        if num_pseudo > 0:
            X_pseudo = X_unl_aug[mask]
            scores_pseudo = scores_unl[mask]
            
            Y_pseudo = np.where(scores_pseudo >= 0, 1, -1)
            margins_pseudo = Y_pseudo * scores_pseudo
            weights_pseudo = -Y_pseudo * expit(-margins_pseudo)
            
            grad_unl = (gamma / num_pseudo) * (X_pseudo.T @ weights_pseudo)
            
        grad_pen = lambd * theta
        grad_pen[0] = 0.0  
        
        return grad_lab + grad_unl + grad_pen

    def run(self, kappa: Union[float, List[float], np.ndarray], 
                  gamma: Union[float, List[float], np.ndarray], 
                  lambd: Union[float, List[float], np.ndarray], 
                  T: int, 
                  eta: Union[float, List[float], np.ndarray] = 1.0,
                  **kwargs) -> List[float]:
        
        # 1. Resolve Hyperparameter Schedules
        kappa_seq = self._resolve_schedule(kappa, T, 'kappa')
        gamma_seq = self._resolve_schedule(gamma, T, 'gamma')
        lambd_seq = self._resolve_schedule(lambd, T + 1, 'lambd')
        eta_seq = self._resolve_schedule(eta, T, 'eta')

        X_lab_aug = np.hstack([np.ones((self.X_lab.shape[0], 1)), self.X_lab])
        X_unl_aug = np.hstack([np.ones((self.X_unl.shape[0], 1)), self.X_unl])
        X_test_aug = np.hstack([np.ones((self.X_test.shape[0], 1)), self.X_test])
        
        # 2. Step 0: Supervised Burn-in
        C_param = 1.0 / lambd_seq[0] if lambd_seq[0] > 0 else 1e9
        clf = LogisticRegression(penalty='l2', C=C_param, fit_intercept=True, solver='lbfgs')
        clf.fit(self.X_lab, self.Y_lab)
        
        theta = np.concatenate([clf.intercept_, clf.coef_[0]])
        
        test_scores = X_test_aug @ theta
        test_preds = np.where(test_scores >= 0, 1, -1)
        errors = [np.mean(test_preds != self.Y_test)]

        # 3. Step t: First-Order Updates
        for t in range(T):
            grad = self._compute_gradient(theta, X_lab_aug, X_unl_aug, 
                                          kappa_seq[t], gamma_seq[t], lambd_seq[t + 1])
            theta = theta - eta_seq[t] * grad
            
            test_scores = X_test_aug @ theta
            test_preds = np.where(test_scores >= 0, 1, -1)
            errors.append(np.mean(test_preds != self.Y_test))

        return errors
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set

from src.dgp import SpikedIsotropic
from src.algorithms import SelfTrainedGradientDescent

@dataclass
class StateEvolutionDynamics:
    """
    Tracks the evolution of the state variables (e.g., errors, usage rates) over iterations.
    This class is designed to store and update the state variables for each iteration of the algorithm.
    """
    dgp: SpikedIsotropic
    learner: SelfTrainedGradientDescent
    
    n_labeled: int # N
    n_unlabeled: int # M
    n_total: int = field(init=False) # n

    rng: np.random.Generator = field(init=False)

    initial_weights: Optional[np.ndarray] = None # w_0
    signal_: np.ndarray = field(init=False) # mu
    labels_: np.ndarray = field(init=False)

    weights_: List[np.ndarray] = field(init=False, default=list) # w
    weights_increments_: List[np.ndarray] = field(init=False, default=list) # v
    predictions_: List[np.ndarray] =field(init=False, default=list) # r

    signal_weights_alignments_: List[float] = field(init=False, default=list) # m
    labels_scores_covariances_: List[float] = field(init=False, default=list) # chi

    weights_covariance_: np.ndarray = field(init=False) # C_w
    scores_covariance_: np.ndarray = field(init=False) # C_g
    weights_memory_: np.ndarray = field(init=False) # Gamma_w
    scores_memory_: np.ndarray = field(init=False) # Gamma_g

    n_interations_: int = field(init=False) # T

    def __post_init__(self):
        """
        """
        self.rng = self.dgp.rng.copy()

        self.n_total - self.n_labeled + self.n_unlabeled
        self.signal_ = self.dgp.mu.copy()
        self.labels_ = self.rng.binomial(self.n_total, self.dgp.p)
        self.n_iterations_ = self.learner.n_iterations
    
        d = self.dgp.d  # Assuming dgp has dimension attribute
        T = self.n_iterations_
    
        # Pre-allocate arrays (shape: dimension x time)
        self.weights_ = np.zeros((d, T + 1))
        self.weights_increment_ = np.zeros((d, T + 1))
        self.predictions_ = np.zeros((d, T + 1))

        self.signal_weights_alignments_ = np.zeros(T + 1)
        self.labels_scores_covariance_ = np.zeros(T + 1)

        # Covariance matrices (shape: time x time)
        self.weight_covariance_ = np.zeros((T + 1, T + 1))
        self.score_covariance_ = np.zeros((T + 1, T + 1))
        self.weight_memory_ = np.zeros((T + 1, T + 1))
        self.score_memory_ = np.zeros((T + 1, T + 1))
    
        # Set initial conditions
        if self.initial_weights is not None:
            self.weights_[:, 0] = self.initial_weights
        else:
            self.weights_[:, 0] = self.learner.weights


    @staticmethod
    def _compute_projection_coef(X, x):
        """
        Computes the OLS projection coefficients using SVD.
        """
        # np.linalg.lstsq returns a tuple; the first element contains the solution
        coefficients, _, _, _ = np.linalg.lstsq(X, x, rcond=None)
        return coefficients


    def _compute_v(self, h, chi, W, Gamma_g, xi):
        return h + self.signal * chi / np.sqrt(self.d) + W @ Gamma_g + xi
    
    def _compute_r(self, Y, m, G, Gamma_w, omega):
        return m * self.labels_ + G @ Gamma_w + omega
    

    @staticmethod
    def compute_conditional_variance(cov_matrix: np.ndarray, t: int) -> float:
        """
        Computes the Schur complement for the conditional variance at time t.
        cov_matrix: The (t+1) x (t+1) empirical covariance matrix.
        """
        if t == 0:
            return cov_matrix[0, 0]
            
        # Extract blocks from the covariance matrix
        C_past = cov_matrix[:t, :t]
        c_cross = cov_matrix[:t, t]
        c_present = cov_matrix[t, t]
        
        # Compute Schur complement: C(t,t) - C(t, <t) * C(<t, <t)^-1 * C(<t, t)
        # Using lstsq is numerically safer than inv(C_past) @ c_cross
        inv_C_past_cross, _, _, _ = np.linalg.lstsq(C_past, c_cross, rcond=None)
        cond_var = c_present - np.dot(c_cross, inv_C_past_cross)
        
        # Ensure non-negativity against floating point errors
        return max(cond_var, 1e-12)

    def compute_innovations(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Samples the independent Gaussian innovations."""
        # 1. Compute conditional scalar variances
        var_xi = self.compute_conditional_variance(self.score_covariance_, t)
        var_omega = self.compute_conditional_variance(self.weight_covariance_, t)
        
        # 2. Sample independent isotropic Gaussians
        # \check{\xi}^t \sim N(0, var_xi * I_d)
        d = self.dgp.d
        xi_check = np.random.randn(d) * np.sqrt(var_xi)
        
        # \check{\omega}^t \sim N(0, var_omega * I_n)
        n = self.n_labeled + self.n_unlabeled
        omega_check = np.random.randn(n) * np.sqrt(var_omega)
        
        return xi_check, omega_check



    def compute_trajectory(self):

        for t in range(self.n_interations_+1):
            omega = None


    def update(self, iteration: int, lab_err: float, unl_err: float, test_err: float,
               unl_use: float, unl_flip_rate: float):
        """Update the state variables for a specific iteration."""
        self.lab_error[iteration] = lab_err
        self.unl_error[iteration] = unl_err
        self.test_error[iteration] = test_err
        self.unl_usage[iteration] = unl_use
        self.unl_flipping_rate[iteration] = unl_flip_rate
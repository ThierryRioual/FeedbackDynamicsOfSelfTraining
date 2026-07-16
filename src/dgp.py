import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

import validation

@dataclass
class SpikedIsotropic:
    """
    Represents a spiked isotropic distribution for generating synthetic datasets.
    The distribution is defined by a mean vector, isotropic noise, and a set of spikes with associated variances.
    """

    d: int
    mu: np.ndarray 
    sigma: float
    p: float
    seed: int
    s: int = 0 # Number of spikes
    spikes_val: list = field(default_factory=list)
    spikes_vect: np.ndarray = field(init=False, default=None)  # Will be initialized in __post_init__
    rng: np.random.Generator = field(init=False)
    V: np.ndarray = field(init=False)
    Lambda_sqrt: np.ndarray = field(init=False)

    def __post_init__(self):
        """
        Initializes the spiked isotropic distribution.
        Validates the parameters and sets up the random number generator.
        """
        if self.spikes_vect is None:
            self.spikes_vect = np.zeros((self.d, self.s))

        self.V = validation.validate_dgp_parameters(
            self.d, self.s, self.spikes_val, self.spikes_vect, self.mu, self.sigma, self.p
        )

        self.rng = np.random.default_rng(self.seed)
        self.Lambda_sqrt = np.diag(np.sqrt(np.asarray(self.spikes_val, dtype=float)))
    
    def _sample_class(self, n_samples: int, sign: int) -> np.ndarray:
        """
        Samples from the spiked isotropic distribution for a given class label.
        """

        Z = self.rng.standard_normal((n_samples, self.d))
        W = self.rng.standard_normal((n_samples, self.s))
        
        # Construct signal
        signal = sign * self.mu
        
        # Construct noise
        isotropic_noise = self.sigma * Z
        spiked_noise = W @ self.Lambda_sqrt @ self.V.T
        
        return signal + isotropic_noise + spiked_noise
 
    def sample(self, N: int, M: int, N_test: int) -> Tuple[np.ndarray, ...]:
        """
        Samples labeled, unlabeled, and test datasets from the spiked isotropic distribution.
        """

        n_pos_lab = self.rng.binomial(N, self.p)
        n_neg_lab = N - n_pos_lab
        m_pos_unl = self.rng.binomial(M, self.p)
        m_neg_unl = M - m_pos_unl
        n_pos_test = self.rng.binomial(N_test, self.p)
        n_neg_test = N_test - n_pos_test

        X_pos_lab = self._sample_class(n_pos_lab, sign=1)
        X_neg_lab = self._sample_class(n_neg_lab, sign=-1)
        X_lab = np.vstack([X_pos_lab, X_neg_lab])
        Y_lab = np.concatenate([np.ones(n_pos_lab), -np.ones(n_neg_lab)])

        X_pos_unl = self._sample_class(m_pos_unl, sign=1)
        X_neg_unl = self._sample_class(m_neg_unl, sign=-1)
        X_unl = np.vstack([X_pos_unl, X_neg_unl])
        Y_unl = np.concatenate([np.ones(m_pos_unl), -np.ones(m_neg_unl)])

        X_pos_test = self._sample_class(n_pos_test, sign=1)
        X_neg_test = self._sample_class(n_neg_test, sign=-1)
        X_test = np.vstack([X_pos_test, X_neg_test])
        Y_test = np.concatenate([np.ones(n_pos_test), -np.ones(n_neg_test)])

        shuffled_idx_lab = self.rng.permutation(N)
        shuffled_idx_unl = self.rng.permutation(M)
        shuffled_idx_test = self.rng.permutation(N_test)

        return (
            X_lab[shuffled_idx_lab], 
            Y_lab[shuffled_idx_lab], 
            X_unl[shuffled_idx_unl], 
            Y_unl[shuffled_idx_unl],
            X_test[shuffled_idx_test], 
            Y_test[shuffled_idx_test]
        )
import numpy as np
from dataclasses import dataclass, field

from typing import Tuple


@dataclass
class SpikedIsotropic:
    d: int
    s: int
    spikes_val: list
    spikes_vect: np.ndarray
    mu: np.ndarray 
    sigma: float
    p: float
    seed: int
    rng: np.random.Generator  = field(init=False) 
    V: np.ndarray = field(init=False)
    Lambda_sqrt: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.s > self.d:
            raise ValueError(f"Cannot have more spikes ({self.s}) than dimensions ({self.d}).")

        self.V = np.column_stack(self.spikes_vect)
        if not np.allclose(self.V.T @ self.V, np.eye(self.s)):
            raise ValueError("Spikes vectors must be orthonormal.")
        
        if self.p > 1.0 or self.p < 0.0:
            raise ValueError(f"Prior probability p ({self.s}) must be between 0.0 and 1.0.")

        self.rng = np.random.default_rng(self.seed)

        self.Lambda_sqrt = np.diag(np.sqrt(self.spikes_val))
    
    def _sample_class(self, n_samples: int, sign: int) -> np.ndarray:
        Z = self.rng.standard_normal((n_samples, self.d))
        W = self.rng.standard_normal((n_samples, self.s))
        
        # Construct signal
        signal = sign * self.mu
        
        # Construct noise
        isotropic_noise = self.sigma * Z
        spiked_noise = W @ self.Lambda_sqrt @ self.V.T
        
        return signal + isotropic_noise + spiked_noise
 
    def sample(self, N: int, M: int, N_test: int) -> Tuple[np.ndarray, ...]:

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
            X_test[shuffled_idx_test], 
            Y_test[shuffled_idx_test]
        )
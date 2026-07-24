import torch
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

from src.validation import validate_dgp_parameters

@dataclass
class SpikedIsotropic:
    """
    Represents a spiked isotropic distribution for generating synthetic datasets.
    The distribution is defined by a mean vector, isotropic noise, and a set of spikes with associated variances.
    """

    d: int
    mu: torch.Tensor 
    sigma: float
    p: float
    seed: int
    s: int = 0 # Number of spikes
    spikes_val: List[float] = field(default_factory=list)
    spikes_vect: Optional[torch.Tensor] = field(init=False, default=None)  # Will be initialized in __post_init__
    rng: torch.Generator = field(init=False)
    V: torch.Tensor = field(init=False)
    Lambda_sqrt: torch.Tensor = field(init=False)

    def __post_init__(self):
        """
        Initializes the spiked isotropic distribution.
        Validates the parameters and sets up the PyTorch random number generator.
        """
        if self.spikes_vect is None:
            self.spikes_vect = torch.zeros((self.d, self.s), dtype=torch.float64)

        self.V = validate_dgp_parameters(
            self.d, self.s, self.spikes_val, self.spikes_vect, self.mu, self.sigma, self.p
        )

        # Initialize the isolated PyTorch random number generator
        self.rng = torch.Generator().manual_seed(self.seed)
        
        # Setup Lambda_sqrt
        spikes_val_t = torch.as_tensor(self.spikes_val, dtype=torch.float64)
        self.Lambda_sqrt = torch.diag(torch.sqrt(spikes_val_t))
    
    def _sample_class(self, n_samples: int, sign: int) -> torch.Tensor:
        """
        Samples from the spiked isotropic distribution for a given class label.
        """
        # torch.randn takes the generator parameter to ensure reproducibility
        Z = torch.randn((n_samples, self.d), generator=self.rng, dtype=torch.float64)
        W = torch.randn((n_samples, self.s), generator=self.rng, dtype=torch.float64)
        
        # Construct signal
        signal = sign * self.mu
        
        # Construct noise (Matrix multiplication @ works natively in PyTorch)
        isotropic_noise = self.sigma * Z
        spiked_noise = W @ self.Lambda_sqrt @ self.V.T
        
        return signal + isotropic_noise + spiked_noise
 
    def sample(self, N: int, M: int, N_test: int) -> Tuple[torch.Tensor, ...]:
        """
        Samples labeled, unlabeled, and test datasets from the spiked isotropic distribution.
        """

        # PyTorch equivalent of np.random.binomial that respects the isolated generator
        n_pos_lab = int((torch.rand(N, generator=self.rng) < self.p).sum().item())
        n_neg_lab = N - n_pos_lab
        
        m_pos_unl = int((torch.rand(M, generator=self.rng) < self.p).sum().item())
        m_neg_unl = M - m_pos_unl
        
        n_pos_test = int((torch.rand(N_test, generator=self.rng) < self.p).sum().item())
        n_neg_test = N_test - n_pos_test

        X_pos_lab = self._sample_class(n_pos_lab, sign=1)
        X_neg_lab = self._sample_class(n_neg_lab, sign=-1)
        X_lab = torch.vstack([X_pos_lab, X_neg_lab])
        Y_lab = torch.cat([torch.ones(n_pos_lab, dtype=torch.float64), -torch.ones(n_neg_lab, dtype=torch.float64)])

        X_pos_unl = self._sample_class(m_pos_unl, sign=1)
        X_neg_unl = self._sample_class(m_neg_unl, sign=-1)
        X_unl = torch.vstack([X_pos_unl, X_neg_unl])
        Y_unl = torch.cat([torch.ones(m_pos_unl, dtype=torch.float64), -torch.ones(m_neg_unl, dtype=torch.float64)])

        X_pos_test = self._sample_class(n_pos_test, sign=1)
        X_neg_test = self._sample_class(n_neg_test, sign=-1)
        X_test = torch.vstack([X_pos_test, X_neg_test])
        Y_test = torch.cat([torch.ones(n_pos_test, dtype=torch.float64), -torch.ones(n_neg_test, dtype=torch.float64)])

        # PyTorch equivalent of np.random.permutation
        shuffled_idx_lab = torch.randperm(N, generator=self.rng)
        shuffled_idx_unl = torch.randperm(M, generator=self.rng)
        shuffled_idx_test = torch.randperm(N_test, generator=self.rng)

        return (
            X_lab[shuffled_idx_lab], 
            Y_lab[shuffled_idx_lab], 
            X_unl[shuffled_idx_unl], 
            Y_unl[shuffled_idx_unl],
            X_test[shuffled_idx_test], 
            Y_test[shuffled_idx_test]
        )
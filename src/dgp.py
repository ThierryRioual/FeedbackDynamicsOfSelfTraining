import torch
from dataclasses import dataclass, field
from typing import Tuple

from src.config import DataConfig
# Assuming validate_dgp_parameters is used elsewhere or remove the import if unused

@dataclass
class IsotropicGaussian:
    """
    Generates synthetic datasets where features are drawn from an isotropic Gaussian distribution.
    Labels are drawn according to the label_prior.
    """
    cfg: DataConfig
    seed: int
    rng: torch.Generator = field(init=False)
    _mu: torch.Tensor = field(init=False)

    def __post_init__(self):
        """Validates parameters and sets up the PyTorch random number generator."""
        self.rng = torch.Generator().manual_seed(self.seed)

        # 1. Safely resolve the ground-truth signal vector
        if self.cfg.signal_vector is not None:
            self._mu = self.cfg.signal_vector
        elif self.cfg.signal_prior is not None and self.cfg.dimensions is not None:
            # Note: Explicitly casting to float64 to match your Y tensors later
            self._mu = torch.tensor(
                [self.cfg.signal_prior() for _ in range(self.cfg.dimensions)],
                dtype=torch.float64 
            )
            self._mu = self._mu / torch.linalg.norm(self._mu)
        else:
            raise ValueError("DGP requires either signal_vector or (signal_prior + dimensions).")
            
        # Ensure dimensions exists for noise generation if not provided in config
        if self.cfg.dimensions is None:
            # dataclasses don't allow modifying frozen fields directly, but we don't need to.
            # We can just rely on self._mu.shape[0] for the dimensions.
            pass

    def _sample_class(self, n_samples: int, sign: int) -> torch.Tensor:
        """Samples from the isotropic Gaussian distribution for a given class label."""
        dim = self._mu.shape[0]
        noise = torch.randn((n_samples, dim), generator=self.rng, dtype=torch.float64)
        return sign * self._mu + self.cfg.scale * noise 
 
    def sample(self) -> Tuple[torch.Tensor, ...]:
        """
        Samples labeled, unlabeled, and test datasets.
        Sizes are pulled strictly from the DataConfig to guarantee theory/empirical alignment.
        """
        N = self.cfg.n_labeled
        M = self.cfg.n_unlabeled
        N_test = self.cfg.n_test
        
        if any(x is None for x in [N, M, N_test]):
            raise ValueError("DataConfig must have n_labeled, n_unlabeled, and n_test defined to sample data.")

        # PyTorch equivalent of np.random.binomial
        n_pos_lab = int((torch.rand(N, generator=self.rng) < self.cfg.label_prior).sum().item())
        n_neg_lab = N - n_pos_lab
        
        m_pos_unl = int((torch.rand(M, generator=self.rng) < self.cfg.label_prior).sum().item())
        m_neg_unl = M - m_pos_unl
        
        n_pos_test = int((torch.rand(N_test, generator=self.rng) < self.cfg.label_prior).sum().item())
        n_neg_test = N_test - n_pos_test

        # Labeled Set
        X_pos_lab = self._sample_class(n_pos_lab, sign=1)
        X_neg_lab = self._sample_class(n_neg_lab, sign=-1)
        X_lab = torch.vstack([X_pos_lab, X_neg_lab])
        Y_lab = torch.cat([torch.ones(n_pos_lab, dtype=torch.float64), -torch.ones(n_neg_lab, dtype=torch.float64)])

        # Unlabeled Set
        X_pos_unl = self._sample_class(m_pos_unl, sign=1)
        X_neg_unl = self._sample_class(m_neg_unl, sign=-1)
        X_unl = torch.vstack([X_pos_unl, X_neg_unl])
        Y_unl = torch.cat([torch.ones(m_pos_unl, dtype=torch.float64), -torch.ones(m_neg_unl, dtype=torch.float64)])

        # Test Set
        X_pos_test = self._sample_class(n_pos_test, sign=1)
        X_neg_test = self._sample_class(n_neg_test, sign=-1)
        X_test = torch.vstack([X_pos_test, X_neg_test])
        Y_test = torch.cat([torch.ones(n_pos_test, dtype=torch.float64), -torch.ones(n_neg_test, dtype=torch.float64)])

        # Shuffle indices
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
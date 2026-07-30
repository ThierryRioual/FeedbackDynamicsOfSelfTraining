import torch
from dataclasses import dataclass, field
from typing import Tuple, Optional

from src.config import DataConfig


@dataclass
class IsotropicGaussian:
    """
    Generates synthetic datasets where features are drawn from an isotropic Gaussian distribution.
    Labels are drawn according to the population-level parameters in DataConfig.

    Finite-dimensional quantities (n_train, n_test, dimensions) live here — not on DataConfig —
    because they describe a specific experiment, not the Laws of the Universe.
    """
    cfg: DataConfig
    n_train: int           # total number of training samples (N + M)
    n_test: int            # number of test samples
    dimensions: int        # feature dimension d
    seed: int = 42
    signal_vector: Optional[torch.Tensor] = None  # optionally override signal_law

    rng: torch.Generator = field(init=False)
    _mu: torch.Tensor = field(init=False)

    def __post_init__(self):
        """Sets up the RNG and resolves the ground-truth signal vector."""
        self.rng = torch.Generator().manual_seed(self.seed)

        # Resolve signal vector: explicit override > sample from prior
        if self.signal_vector is not None:
            self._mu = self.signal_vector
        else:
            self._mu = torch.tensor(
                [self.cfg.signal_law() for _ in range(self.dimensions)],
                dtype=torch.float64
            )

    # --- Convenience properties (expected stratified counts for display) ---

    @property
    def n_labeled(self) -> int:
        """Expected number of labeled samples: round(ρ * n_train)."""
        return round(self.cfg.supervision_ratio * self.n_train)

    @property
    def n_unlabeled(self) -> int:
        """Expected number of unlabeled samples: n_train - n_labeled."""
        return self.n_train - self.n_labeled

    @property
    def empirical_data_to_dimension_ratio(self) -> float:
        """The realized δ̂ = n_train / d for this experiment."""
        return self.n_train / self.dimensions

    def _sample_class(self, n_samples: int, sign: int) -> torch.Tensor:
        """Samples from the isotropic Gaussian distribution for a given class label."""
        d = self._mu.shape[0]
        noise = torch.randn((n_samples, d), generator=self.rng, dtype=torch.float64)
        return (sign * self._mu) / (d ** 0.5) + self.cfg.scale * noise

    def sample(self, stratified: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Samples labeled, unlabeled, and test datasets.

        Args:
            stratified: If True, forces exact counts based on population probabilities
                        (zero binomial variance, ideal for smooth SE comparison plots).
                        If False, uses pure i.i.d. random sampling (mathematically pure).
        """
        rho = self.cfg.supervision_ratio
        p = self.cfg.label_prior

        if stratified:
            # Exact counts — zero variance
            N = round(rho * self.n_train)
            M = self.n_train - N

            n_pos_lab = round(p * N)
            n_neg_lab = N - n_pos_lab

            m_pos_unl = round(p * M)
            m_neg_unl = M - m_pos_unl

            n_pos_test = round(p * self.n_test)
            n_neg_test = self.n_test - n_pos_test
        else:
            # I.I.D. random sampling — mathematically pure
            N = int((torch.rand(self.n_train, generator=self.rng) < rho).sum().item())
            M = self.n_train - N

            n_pos_lab = int((torch.rand(N, generator=self.rng) < p).sum().item())
            n_neg_lab = N - n_pos_lab

            m_pos_unl = int((torch.rand(M, generator=self.rng) < p).sum().item())
            m_neg_unl = M - m_pos_unl

            n_pos_test = int((torch.rand(self.n_test, generator=self.rng) < p).sum().item())
            n_neg_test = self.n_test - n_pos_test

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
        shuffled_idx_test = torch.randperm(self.n_test, generator=self.rng)

        return (
            X_lab[shuffled_idx_lab],
            Y_lab[shuffled_idx_lab],
            X_unl[shuffled_idx_unl],
            Y_unl[shuffled_idx_unl],
            X_test[shuffled_idx_test],
            Y_test[shuffled_idx_test]
        )
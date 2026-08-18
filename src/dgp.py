"""Conditional Gaussian design generation for realised quenched environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from src.config import DataConfig
from src.environment import FourCellSampleTypeLaw, QuenchedEnvironment, SampleTypeLaw


@dataclass
class IsotropicGaussian:
    """Generate ``X_i=Y_i mu/sqrt(d)+sigma U_i`` conditional on an environment.

    ``DataConfig`` retains the legacy product-law convenience path.  Passing an
    explicit ``sample_type_law`` or ``environment`` exposes the general
    quenched model without changing the Gaussian conditional design.
    """

    cfg: DataConfig
    n_train: int
    n_test: int
    dimensions: int
    seed: int = 42
    signal_vector: Optional[torch.Tensor] = None
    sample_type_law: Optional[SampleTypeLaw] = None
    environment: Optional[QuenchedEnvironment] = None
    test_label_prior: Optional[float] = None

    rng: torch.Generator = field(init=False)
    _mu: torch.Tensor = field(init=False)
    last_environment_: Optional[QuenchedEnvironment] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.n_train <= 0 or self.n_test < 0 or self.dimensions <= 0:
            raise ValueError("n_train and dimensions must be positive; n_test nonnegative")
        self.rng = torch.Generator().manual_seed(self.seed)
        if self.signal_vector is not None:
            mu = torch.as_tensor(self.signal_vector, dtype=torch.float64).detach().clone()
        elif self.environment is not None:
            mu = self.environment.mu.detach().clone()
        else:
            # Legacy callable laws have no generator argument.  Sampling remains
            # isolated at the object boundary through an explicit DGP seed for
            # all Gaussian/noise draws; callers wanting complete stream control
            # can supply signal_vector or an explicit environment.
            mu = torch.tensor([self.cfg.signal_law() for _ in range(self.dimensions)], dtype=torch.float64)
        if mu.shape != (self.dimensions,):
            raise ValueError(f"signal vector must have shape ({self.dimensions},)")
        self._mu = mu
        if self.environment is not None and self.environment.d != self.dimensions:
            raise ValueError("environment and dimensions disagree")
        if self.test_label_prior is not None and not 0 < self.test_label_prior < 1:
            raise ValueError("test_label_prior must lie in (0,1)")

    @property
    def law(self) -> SampleTypeLaw:
        return self.sample_type_law or FourCellSampleTypeLaw.product(
            label_prior=self.cfg.label_prior,
            supervision_ratio=self.cfg.supervision_ratio,
        )

    @property
    def n_labeled(self) -> int:
        if self.last_environment_ is not None:
            return self.last_environment_.N
        return round(self.cfg.supervision_ratio * self.n_train)

    @property
    def n_unlabeled(self) -> int:
        if self.last_environment_ is not None:
            return self.last_environment_.M
        return self.n_train - self.n_labeled

    @property
    def empirical_data_to_dimension_ratio(self) -> float:
        return self.n_train / self.dimensions

    def sample_environment(self, *, stratified: bool = False) -> QuenchedEnvironment:
        if self.environment is not None:
            self.last_environment_ = self.environment
            return self.environment
        if stratified:
            # Stratification is defined only for the legacy product law; it is a
            # variance-reduction extension rather than an iid sample from a
            # general four-cell law.
            if self.sample_type_law is not None:
                raise ValueError("stratified sampling is unavailable for a general joint sample-type law")
            p, rho, n = self.cfg.label_prior, self.cfg.supervision_ratio, self.n_train
            N = round(rho * n)
            categories = torch.cat((
                torch.zeros(round(p * N), dtype=torch.long),
                torch.full((N - round(p * N),), 2, dtype=torch.long),
                torch.ones(round(p * (n - N)), dtype=torch.long),
                torch.full((n - N - round(p * (n - N)),), 3, dtype=torch.long),
            ))
            categories = categories[torch.randperm(n, generator=self.rng)]
            Y = torch.where(categories < 2, 1.0, -1.0)
            Delta = torch.where((categories == 0) | (categories == 2), 1.0, 0.0)
        else:
            Y, Delta = self.law.sample(n=self.n_train, generator=self.rng, dtype=torch.float64, device=torch.device("cpu"))
        env = QuenchedEnvironment(self._mu, Y, Delta)
        self.last_environment_ = env
        return env

    def sample_design(self, environment: QuenchedEnvironment, *, generator: Optional[torch.Generator] = None, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(X,U)`` for fixed ``environment``; ``Delta`` is unused."""

        if environment.d != self.dimensions:
            raise ValueError("environment dimension disagrees with generator")
        generator = self.rng if generator is None else generator
        if noise is None:
            U = torch.randn((environment.n, environment.d), generator=generator, dtype=torch.float64, device=environment.mu.device)
        else:
            U = torch.as_tensor(noise, dtype=torch.float64, device=environment.mu.device).detach().clone()
            if U.shape != (environment.n, environment.d):
                raise ValueError("noise must have shape (n,d)")
        X = environment.Y[:, None] * environment.mu[None, :] / (environment.d ** 0.5) + self.cfg.scale * U
        return X, U

    def sample_test(self, *, generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        generator = self.rng if generator is None else generator
        p = self.test_label_prior if self.test_label_prior is not None else self.law.label_prior
        Y = torch.where(torch.rand(self.n_test, generator=generator) < p, 1.0, -1.0).to(torch.float64)
        U = torch.randn((self.n_test, self.dimensions), generator=generator, dtype=torch.float64)
        X = Y[:, None] * self._mu[None, :] / (self.dimensions ** 0.5) + self.cfg.scale * U
        return X, Y

    def sample_full(self, stratified: bool = False) -> Tuple[QuenchedEnvironment, torch.Tensor, torch.Tensor, torch.Tensor]:
        env = self.sample_environment(stratified=stratified)
        X, _ = self.sample_design(env)
        X_test, Y_test = self.sample_test()
        return env, X, X_test, Y_test

    def sample(self, stratified: bool = False) -> Tuple[torch.Tensor, ...]:
        """Backward-compatible split view derived from one full realisation."""

        env, X, X_test, Y_test = self.sample_full(stratified=stratified)
        return X[env.I_L], env.Y[env.I_L], X[env.I_U], env.Y[env.I_U], X_test, Y_test

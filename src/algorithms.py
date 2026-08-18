"""Finite-dimensional self-training gradient descent.

The only learning loop in this module is full-indexed and conditions on a
realised :class:`~src.environment.QuenchedEnvironment`.  The familiar split
``fit(X_lab, Y_lab, X_unl, ...)`` method is an adapter into that loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from src.config import AlgorithmConfig
from src.environment import QuenchedEnvironment
from src.initialization import (
    SelfTrainingInitialization,
    compute_pseudo_labels_from_scores,
    compute_scores,
)
from src.primitives import pseudo_labels, pseudo_residual, selection_rate


@dataclass(frozen=True)
class FiniteStep:
    """Frozen objects used by one finite objective/update."""

    t: int
    scores: torch.Tensor
    pseudo_labels: torch.Tensor
    selection: torch.Tensor
    omega: torch.Tensor
    g: torch.Tensor
    m: torch.Tensor
    chi: torch.Tensor
    zeta: torch.Tensor
    tau: torch.Tensor
    bias: torch.Tensor
    weight: torch.Tensor


@dataclass
class SelfTrainedGradientDescent:
    """Manuscript-faithful finite self-training gradient descent."""

    cfg: AlgorithmConfig
    callback: Optional[Callable[["SelfTrainedGradientDescent", int], None]] = None

    bias: Optional[torch.Tensor] = field(init=False, default=None)
    weights: Optional[torch.Tensor] = field(init=False, default=None)
    environment_: Optional[QuenchedEnvironment] = field(init=False, default=None)
    initialization_: Optional[SelfTrainingInitialization] = field(init=False, default=None)
    update_records_: list[FiniteStep] = field(init=False, default_factory=list)
    score_history_: list[torch.Tensor] = field(init=False, default_factory=list)
    weight_history_: list[torch.Tensor] = field(init=False, default_factory=list)
    residual_history_: list[torch.Tensor] = field(init=False, default_factory=list)
    prev_preactivations_: Optional[torch.Tensor] = field(init=False, default=None)

    def _validate_X(self, X: torch.Tensor, environment: QuenchedEnvironment) -> torch.Tensor:
        X = torch.as_tensor(X, dtype=torch.float64, device=environment.mu.device)
        if X.shape != (environment.n, environment.d):
            raise ValueError(f"X must have shape ({environment.n}, {environment.d})")
        if not torch.isfinite(X).all():
            raise ValueError("X must be finite")
        if environment.N == 0:
            raise ValueError("finite self-training requires at least one labelled observation")
        return X

    def _pseudo_weight(self, t: int) -> float:
        return self.cfg.get_pseudo_label_weight(t)

    @torch.no_grad()
    def fit_full(
        self,
        X: torch.Tensor,
        environment: QuenchedEnvironment,
        initialization: SelfTrainingInitialization,
    ) -> "SelfTrainedGradientDescent":
        """Fit the canonical full-indexed, quenched mathematical model."""

        X = self._validate_X(X, environment)
        if not isinstance(initialization, SelfTrainingInitialization):
            raise ValueError(
                "canonical fit_full requires an explicit SelfTrainingInitialization; "
                "it never creates Y_init from sign(r^0)"
            )
        init = initialization.for_environment(environment)
        if not self.cfg.include_bias and float(init.b_init) != 0.0:
            raise ValueError("b_init must be zero when include_bias=False")

        self.environment_ = environment
        self.initialization_ = init
        self.bias = torch.as_tensor(init.b_init, dtype=X.dtype, device=X.device).detach().clone().reshape(())
        if not self.cfg.include_bias:
            self.bias.zero_()
        self.weights = init.w_init.detach().clone()
        self.update_records_ = []
        self.score_history_ = []
        self.weight_history_ = [self.weights.detach().clone()]
        self.residual_history_ = []
        self.prev_preactivations_ = None

        # Initial callback retains the historical T+1 callback convention.  No
        # pseudo-residual is attributed to this state until the first objective
        # has been frozen below.
        if self.callback is not None:
            self.callback(self, 0)

        for t in range(self.cfg.n_iterations):
            b_current = self.bias
            w_current = self.weights
            scores = compute_scores(X, b_current if self.cfg.include_bias else 0.0, w_current)
            Yhat = pseudo_labels(t, scores, init.Y_init)
            selection = self.cfg.selection_function(scores, self.cfg.positive_margin, self.cfg.negative_margin)
            # Labeled selection is irrelevant, but setting it to zero makes the
            # stored weights exactly the finite manuscript convention.
            selection = selection * (1.0 - environment.Delta)
            omega = selection_rate(selection, environment.Delta)
            pi = self._pseudo_weight(t)
            g = pseudo_residual(
                scores=scores,
                Y=environment.Y,
                Delta=environment.Delta,
                Yhat=Yhat,
                selection=selection,
                omega=omega,
                pi=pi,
                eta=self.cfg.step_size,
                rho=environment.rho,
                loss_function=self.cfg.loss_function,
            )
            m = torch.dot(environment.mu, w_current) / environment.d
            chi = torch.dot(environment.Y, g) / environment.n
            zeta = torch.mean(g)
            tau = torch.linalg.vector_norm(w_current) / (environment.d ** 0.5)
            record = FiniteStep(
                t=t,
                scores=scores.detach().clone(),
                pseudo_labels=Yhat.detach().clone(),
                selection=selection.detach().clone(),
                omega=omega.detach().clone(),
                g=g.detach().clone(),
                m=m.detach().clone(),
                chi=chi.detach().clone(),
                zeta=zeta.detach().clone(),
                tau=tau.detach().clone(),
                bias=b_current.detach().clone(),
                weight=w_current.detach().clone(),
            )
            self.update_records_.append(record)
            self.score_history_.append(record.scores)
            self.residual_history_.append(record.g)

            # g already contains -eta times the frozen score gradient.
            decay = -self.cfg.step_size * self.cfg.penalty_param * self.cfg.penalty_function.gradient(w_current)
            self.prev_preactivations_ = scores[environment.I_U].detach().clone()
            self.weights = (w_current + decay + (environment.d ** 0.5 / environment.n) * (X.T @ g)).detach()
            self.bias = (b_current + zeta).detach() if self.cfg.include_bias else b_current.new_zeros(())
            self.weight_history_.append(self.weights.detach().clone())

            if self.callback is not None:
                self.callback(self, t + 1)
        return self

    def macroscopic_observables(self, t: int) -> dict[str, torch.Tensor]:
        """Finite observables attached to the exact objective at iteration ``t``."""

        if t < 0 or t >= len(self.update_records_):
            raise IndexError("t must index an executed update")
        step = self.update_records_[t]
        env = self.environment_
        assert env is not None
        return {
            "m": step.m,
            "chi": step.chi,
            "zeta": step.zeta,
            "tau": step.tau,
            "omega": step.omega,
            "C_w": torch.stack(self.weight_history_[: t + 1], dim=1).T @ step.weight / env.d,
            "C_g": torch.stack(self.residual_history_[: t + 1], dim=1).T @ step.g / env.n,
        }

    def finite_decomposition(self, X: torch.Tensor, U: torch.Tensor, t: int, *, sigma: float) -> dict[str, torch.Tensor]:
        """Return the finite forward/backward decomposition at an executed step."""

        if self.environment_ is None:
            raise RuntimeError("fit must be called first")
        env = self.environment_
        step = self.update_records_[t]
        X = torch.as_tensor(X, dtype=torch.float64, device=env.mu.device)
        U = torch.as_tensor(U, dtype=torch.float64, device=env.mu.device)
        if X.shape != (env.n, env.d) or U.shape != X.shape:
            raise ValueError("X and U must both have shape (n,d)")
        q = U @ step.weight / (env.d ** 0.5)
        p = U.T @ step.g / (env.n ** 0.5)
        r = step.bias + step.m * env.Y + float(sigma) * q
        noise_update = float(sigma) / (env.delta ** 0.5) * p
        return {
            "q": q,
            "p": p,
            "r": r,
            "direct_score": step.scores,
            "noise_update": noise_update,
            "direct_update_noise": (env.d ** 0.5 / env.n) * (X.T @ step.g) - step.chi * env.mu,
        }

    @torch.no_grad()
    def fit(
        self,
        X_lab: torch.Tensor,
        Y_lab: torch.Tensor | QuenchedEnvironment,
        X_unl: torch.Tensor | SelfTrainingInitialization | None = None,
        initial_bias: Optional[float] = None,
        initial_weights: Optional[torch.Tensor] = None,
        initial_pseudo_labels: Optional[torch.Tensor] = None,
        *,
        Y_unl: Optional[torch.Tensor] = None,
    ) -> "SelfTrainedGradientDescent":
        """Fit via the legacy split adapter or ``fit(X, environment, init)``.

        If true unlabeled labels are unavailable in the split API, the adapter
        constructs a bookkeeping environment whose unlabeled ``Y`` equals the
        supplied ``Y_init``.  Those coordinates are never consulted by the
        optimisation; users requiring a quenched ground truth should call
        :meth:`fit_full`.
        """

        if isinstance(Y_lab, QuenchedEnvironment):
            if not isinstance(X_unl, SelfTrainingInitialization):
                raise TypeError("canonical shorthand is fit(X, environment, initialization)")
            return self.fit_full(X_lab, Y_lab, X_unl)

        if X_unl is None or isinstance(X_unl, SelfTrainingInitialization):
            raise TypeError("legacy fit requires X_lab, Y_lab, and X_unl")
        X_lab = torch.as_tensor(X_lab, dtype=torch.float64)
        Y_lab_tensor = torch.as_tensor(Y_lab, dtype=torch.float64, device=X_lab.device)
        X_unl_tensor = torch.as_tensor(X_unl, dtype=torch.float64, device=X_lab.device)
        if X_lab.ndim != 2 or X_unl_tensor.ndim != 2 or X_lab.shape[1] != X_unl_tensor.shape[1]:
            raise ValueError("X_lab and X_unl must be two-dimensional with a common feature dimension")
        if Y_lab_tensor.shape != (X_lab.shape[0],):
            raise ValueError("Y_lab must have shape (N,)")
        if not torch.all((Y_lab_tensor == -1) | (Y_lab_tensor == 1)):
            raise ValueError("Y_lab entries must belong to {-1,+1}")
        N, M, d = X_lab.shape[0], X_unl_tensor.shape[0], X_lab.shape[1]
        if N == 0:
            raise ValueError("legacy fit requires at least one labelled observation")
        if initial_weights is None:
            initial_weights = torch.randn(d, dtype=torch.float64, device=X_lab.device)
        pi0 = self._pseudo_weight(0) if self.cfg.n_iterations else 0.0
        if initial_pseudo_labels is None and pi0 > 0 and M > 0:
            raise ValueError(
                "initial_pseudo_labels is required when the t=0 pseudo-labelled contribution is active; "
                "use compute_pseudo_labels_from_scores explicitly for the theorem-external endogenous experiment"
            )
        if initial_pseudo_labels is None:
            # No t=0 unlabeled term is active.  This placeholder is never used
            # at t=0 and is refreshed from scores thereafter.
            initial_pseudo_labels = torch.ones(M, dtype=torch.float64, device=X_lab.device)
        Y_init_unl = torch.as_tensor(initial_pseudo_labels, dtype=torch.float64, device=X_lab.device)
        if Y_init_unl.shape != (M,) or not torch.all((Y_init_unl == -1) | (Y_init_unl == 1)):
            raise ValueError("initial_pseudo_labels must be an M-vector with entries in {-1,+1}")
        if Y_unl is None:
            Y_unl_tensor = Y_init_unl
        else:
            Y_unl_tensor = torch.as_tensor(Y_unl, dtype=torch.float64, device=X_lab.device)
            if Y_unl_tensor.shape != (M,) or not torch.all((Y_unl_tensor == -1) | (Y_unl_tensor == 1)):
                raise ValueError("Y_unl must be an M-vector with entries in {-1,+1}")
        environment = QuenchedEnvironment(
            mu=torch.zeros(d, dtype=torch.float64, device=X_lab.device),
            Y=torch.cat((Y_lab_tensor, Y_unl_tensor)),
            Delta=torch.cat((torch.ones(N, dtype=torch.float64, device=X_lab.device), torch.zeros(M, dtype=torch.float64, device=X_lab.device))),
        )
        init = SelfTrainingInitialization.from_unlabeled_labels(
            environment,
            b_init=0.0 if initial_bias is None else initial_bias,
            w_init=initial_weights,
            Y_init_unlabeled=Y_init_unl,
        )
        return self.fit_full(torch.cat((X_lab, X_unl_tensor)), environment, init)

    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        if self.weights is None or self.bias is None:
            raise RuntimeError("fit must be called before prediction")
        return compute_scores(X, self.bias if self.cfg.include_bias else 0.0, self.weights)

    def compute_preactivation(self, X: torch.Tensor, bias: Optional[float] = None, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weights is None:
            return self.decision_function(X)
        return compute_scores(X, 0.0 if (not self.cfg.include_bias) else (self.bias if bias is None else bias), weights)

    def predict(self, X: torch.Tensor, bias: Optional[float] = None, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.compute_preactivation(X, bias=bias, weights=weights)
        return compute_pseudo_labels_from_scores(scores)

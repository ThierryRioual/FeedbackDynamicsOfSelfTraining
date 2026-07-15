import warnings
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.special import logit
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

_EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SelfTrainingConfig:
    N:           int          # labeled training points
    M:           int          # unlabeled points
    N_test:      int          # test points (fixed across runs)
    d:           int          # feature dimension
    mu_plus:     np.ndarray   # mean of class +1
    mu_minus:    np.ndarray   # mean of class -1
    p:           float        # confidence threshold
    T:           int          # self-training rounds
    n_runs:      int          # Monte Carlo repetitions
    n_jobs:      int          # joblib parallel workers
    seed:        int
    sigma_scale: float

    def __post_init__(self):
        if not (0.0 < self.p < 1.0):
            raise ValueError(f"p must be in (0, 1), got {self.p}")
        for name, val in [("N", self.N), ("M", self.M), ("N_test", self.N_test),
                          ("d", self.d), ("T", self.T), ("n_runs", self.n_runs)]:
            if not isinstance(val, int) or val < 1:
                raise ValueError(f"{name} must be a positive integer, got {val}")
        if self.sigma_scale <= 0:
            raise ValueError(f"sigma_scale must be positive, got {self.sigma_scale}")
        if self.mu_plus.shape != (self.d,) or self.mu_minus.shape != (self.d,):
            raise ValueError(f"mu_plus and mu_minus must have shape ({self.d},)")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def sample_data(n: int, cfg: SelfTrainingConfig, rng: np.random.Generator):
    X, y = make_blobs(n_samples=n, centers=[cfg.mu_minus, cfg.mu_plus],
                      cluster_std=np.sqrt(cfg.sigma_scale),
                      random_state=rng.integers(2**31))
    Y = 2 * y - 1  # {0, 1} -> {-1, +1}
    return X, Y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def fit_logistic(X: np.ndarray, Y: np.ndarray, penalty: str=None, C: float=1.0) -> LogisticRegression:
    solver = "lbfgs"
    clf = LogisticRegression(penalty=penalty, C=C, fit_intercept=True, 
                             max_iter=500, n_jobs=1, solver=solver)
    clf.fit(X, Y)
    return clf


# ---------------------------------------------------------------------------
# Self-training loop (single run)
# ---------------------------------------------------------------------------

def run_self_training(p: float, T: int,
                      X_lab: np.ndarray, Y_lab: np.ndarray,
                      X_unl: np.ndarray,
                      X_test: np.ndarray, Y_test: np.ndarray,
                      penalty: str=None, C: float=1.0) -> np.ndarray:

    kappa = logit(p)
    clf = fit_logistic(X_lab, Y_lab, penalty=penalty, C=C)
    errors = [1 - clf.score(X_test, Y_test)]

    for t in range(T):
        scores = clf.decision_function(X_unl)
        mask = np.abs(scores) >= kappa
        if mask.sum() == 0:
            errors.append(errors[-1])
            continue

        Y_pseudo = np.where(scores[mask] >= 0, 1, -1)
        clf = fit_logistic(np.vstack([X_lab, X_unl[mask]]),
                           np.concatenate([Y_lab, Y_pseudo]),
                           penalty=penalty, C=C)
        errors.append(1 - clf.score(X_test, Y_test))

    return np.array(errors)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(cfg: SelfTrainingConfig) -> np.ndarray:
    master_rng = np.random.default_rng(cfg.seed)
    X_test, Y_test = sample_data(cfg.N_test, cfg, master_rng)
    seeds = master_rng.integers(0, 2**31, size=cfg.n_runs)

    results = Parallel(n_jobs=cfg.n_jobs)(
        delayed(run_self_training)(cfg, int(s), X_test, Y_test) for s in seeds
    )
    return np.mean(results, axis=0)


# ---------------------------------------------------------------------------
# Bayes error & ratio
# ---------------------------------------------------------------------------

def estimate_bayes_error(cfg: SelfTrainingConfig,
                         X_test: np.ndarray, Y_test: np.ndarray,
                         n_large: int = 100_000) -> float:
    rng = np.random.default_rng(cfg.seed + 1)
    X_big, Y_big = sample_data(n_large, cfg, rng)
    return 1 - fit_logistic(X_big, Y_big).score(X_test, Y_test)


def stable_relative_excess(mean_e: np.ndarray, err_bayes: float, N: int) -> np.ndarray:
    num   = N * (mean_e    - err_bayes)
    denom = N * (mean_e[0] - err_bayes)
    if abs(denom) < 10 * _EPS:
        warnings.warn("Supervised excess ~ 0; ratio ill-defined.", RuntimeWarning)
        return np.full_like(mean_e, np.nan)
    return num / denom


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    d = 10
    # Means with variation across all dimensions: opposite spiral-like sign patterns
    mu_plus  = np.array([ 1, -1,  1, -1,  1, -1,  1, -1,  1, -1], dtype=float)
    mu_minus = np.array([-1,  1, -1,  1, -1,  1, -1,  1, -1,  1], dtype=float)

    cfg = SelfTrainingConfig(
        N=1000, M=2000, N_test=5000, d=d,
        mu_plus=mu_plus, mu_minus=mu_minus,
        p=0.8, T=5, n_runs=5000, n_jobs=4,
        seed=42, sigma_scale=1,
    )

    mean_e = run_experiment(cfg)

    rng = np.random.default_rng(cfg.seed)
    X_test, Y_test = sample_data(cfg.N_test, cfg, rng)
    err_bayes  = estimate_bayes_error(cfg, X_test, Y_test)
    rel_excess = stable_relative_excess(mean_e, err_bayes, N=cfg.N)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(cfg.T + 1), rel_excess, color="#4c8ef7", linewidth=2.0)
    ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Round $T$")
    ax.set_ylabel(r"$(\mathrm{Err}_T - \hat{L}_\mathrm{Bayes}) / (\mathrm{Err}_0 - \hat{L}_\mathrm{Bayes})$")
    ax.set_title("Relative Excess Test Error over Supervised Baseline")
    ax.set_xlim(0, cfg.T)
    plt.tight_layout()
    plt.savefig("self_training_rel_excess.png", dpi=150, bbox_inches="tight")
    plt.show()


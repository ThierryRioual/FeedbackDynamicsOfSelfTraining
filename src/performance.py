"""Population performance and Bayes benchmarks for the Gaussian mixture model."""

from __future__ import annotations

import math
from typing import Tuple

from scipy.stats import norm


def population_error(b: float, m: float, tau: float, sigma: float, p: float) -> float:
    scale = float(sigma) * float(tau)
    if scale == 0:
        return float(p) * float(float(b) + float(m) < 0) + (1 - float(p)) * float(float(b) - float(m) >= 0)
    return float(p) * norm.cdf((-float(b) - float(m)) / scale) + (1 - float(p)) * norm.cdf((float(b) - float(m)) / scale)


def bayes_parameters(signal_scale: float, sigma: float, p: float) -> Tuple[float, float, float]:
    """Return ``(b_star, m_star, tau_star)`` for ``w_star=mu``.

    Here ``signal_scale=||mu||/sqrt(d)``.  The formula therefore remains valid
    away from the special normalisation ``signal_scale=1``.
    """

    if signal_scale <= 0 or sigma < 0 or not 0 < p < 1:
        raise ValueError("require signal_scale>0, sigma>=0, and 0<p<1")
    # The optimal *normalised* intercept is
    # beta*=sigma^2/(2 signal_scale) log(p/(1-p)).  Since w*=mu has
    # tau*=signal_scale, the raw score intercept returned here is beta*tau.
    b_star = sigma**2 / 2 * math.log(p / (1 - p))
    return b_star, signal_scale**2, signal_scale

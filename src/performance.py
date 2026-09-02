"""Population performance and Bayes benchmarks for the Gaussian mixture model."""

from __future__ import annotations

import math
import sys
from typing import Tuple

from scipy.stats import norm


def population_error(b: float, m: float, tau: float, sigma: float, p: float) -> float:
    scale = float(sigma) * float(tau)
    if scale == 0:
        return float(p) * float(float(b) + float(m) < 0) + (1 - float(p)) * float(float(b) - float(m) >= 0)
    return float(p) * norm.cdf((-float(b) - float(m)) / scale) + (1 - float(p)) * norm.cdf((float(b) - float(m)) / scale)


def oracle_calibrated_bias_and_error(
    m: float,
    tau: float,
    sigma: float,
    p: float,
    *,
    machine_epsilon: float = sys.float_info.epsilon,
) -> Tuple[float, float]:
    """Return the prescribed oracle intercept and its population error.

    For the score ``b + m Y + sigma tau Z``, differentiating
    :func:`population_error` with respect to ``b`` gives

    ``2 b m = sigma**2 * tau**2 * log(p / (1 - p))``.

    ``(nan, nan)`` is returned when ``m`` is numerically zero or the resulting
    intercept is nonfinite.  The tolerance is scaled by the effective noise
    standard deviation and can be matched to the caller's floating-point
    dtype through ``machine_epsilon``.
    """

    alignment = float(m)
    weight_scale = float(tau)
    noise_scale = float(sigma) * weight_scale
    zero_tolerance = math.sqrt(float(machine_epsilon)) * max(
        1.0, abs(noise_scale)
    )
    if not math.isfinite(alignment) or abs(alignment) <= zero_tolerance:
        return float("nan"), float("nan")

    oracle_bias = (
        noise_scale**2
        / (2.0 * alignment)
        * math.log(float(p) / (1.0 - float(p)))
    )
    if not math.isfinite(oracle_bias):
        return float("nan"), float("nan")
    return oracle_bias, population_error(
        oracle_bias, alignment, weight_scale, sigma, p
    )


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

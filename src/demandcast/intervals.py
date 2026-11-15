"""Split-conformal prediction intervals.

Two methods, both calibrated on a window of recent hours that the interval model did not see:

* :func:`split_conformal_interval` wraps *any* point forecast: the half-width is a conformal
  quantile of absolute calibration residuals.
* :func:`cqr_interval` is conformalized quantile regression (Romano, Patterson & Candes 2019):
  it shifts a pair of quantile-regression bounds by a conformal quantile of how far the
  calibration actuals fall outside them (a negative shift narrows over-wide bounds).

With exchangeable calibration and test points both have finite-sample marginal coverage of at
least ``1 - alpha``. With ``log_scale=True`` the scores are computed on ``log1p`` counts, so
the correction is multiplicative: count noise grows with the level, and a single additive
half-width would be far too wide at night and too narrow at the evening peak.
"""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest score (``inf`` if that exceeds n)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(scores)
    if n == 0:
        raise ValueError("Need at least one calibration score.")
    k = math.ceil((n + 1) * (1 - alpha))
    return float(np.inf) if k > n else float(scores[k - 1])


def _fwd(x: np.ndarray, log_scale: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.clip(x, 0.0, None)) if log_scale else x


def _inv(x: np.ndarray, log_scale: bool, clip_zero: bool) -> np.ndarray:
    out = np.expm1(x) if log_scale else x
    return np.clip(out, 0.0, None) if clip_zero else out


def split_conformal_interval(
    cal_y: np.ndarray,
    cal_pred: np.ndarray,
    test_pred: np.ndarray,
    alpha: float,
    log_scale: bool = True,
    clip_zero: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric split-conformal interval around a point forecast."""
    scores = np.abs(_fwd(cal_y, log_scale) - _fwd(cal_pred, log_scale))
    q = conformal_quantile(scores, alpha)
    center = _fwd(test_pred, log_scale)
    return _inv(center - q, log_scale, clip_zero), _inv(center + q, log_scale, clip_zero)


def cqr_interval(
    cal_y: np.ndarray,
    cal_lo: np.ndarray,
    cal_hi: np.ndarray,
    test_lo: np.ndarray,
    test_hi: np.ndarray,
    alpha: float,
    log_scale: bool = True,
    clip_zero: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Conformalized quantile regression interval."""
    y = _fwd(cal_y, log_scale)
    lo, hi = _fwd(cal_lo, log_scale), _fwd(cal_hi, log_scale)
    scores = np.maximum(lo - y, y - hi)
    q = conformal_quantile(scores, alpha)
    t_lo, t_hi = _fwd(test_lo, log_scale) - q, _fwd(test_hi, log_scale) + q
    t_hi = np.maximum(t_hi, t_lo)  # a large negative shift must not invert the interval
    return _inv(t_lo, log_scale, clip_zero), _inv(t_hi, log_scale, clip_zero)

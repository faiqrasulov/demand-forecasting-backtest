"""Point and interval forecast metrics.

All functions take an optional boolean ``mask`` of hours to score (here: hours that were
actually recorded), so unrecorded hours never count towards or against a model.
"""

from __future__ import annotations

import numpy as np


def _prepare(*arrays: np.ndarray, mask: np.ndarray | None) -> list[np.ndarray]:
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("All arrays must have the same length.")
    keep = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not keep.any():
        raise ValueError("No points to score.")
    return [a[keep] for a in arrays]


def mae(y: np.ndarray, yhat: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean absolute error."""
    y, yhat = _prepare(y, yhat, mask=mask)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Root mean squared error."""
    y, yhat = _prepare(y, yhat, mask=mask)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def wape(y: np.ndarray, yhat: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Weighted absolute percentage error: sum|y - yhat| / sum|y| (total error / total demand)."""
    y, yhat = _prepare(y, yhat, mask=mask)
    denom = np.sum(np.abs(y))
    if denom == 0:
        raise ValueError("WAPE is undefined when all actuals are zero.")
    return float(np.sum(np.abs(y - yhat)) / denom)


def seasonal_naive_scale(
    y_train: np.ndarray, m: int, mask: np.ndarray | None = None
) -> float:
    """In-sample MAE of the seasonal naive forecast y_t = y_{t-m} on the training series.

    Only pairs where both y_t and y_{t-m} are scored (``mask``) are used.
    """
    y = np.asarray(y_train, dtype=np.float64)
    if len(y) <= m:
        raise ValueError("Training series must be longer than the seasonal period.")
    keep = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    pair = keep[m:] & keep[:-m]
    scale = float(np.mean(np.abs(y[m:] - y[:-m])[pair]))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Seasonal naive scale is zero or undefined.")
    return scale


def mase(
    y: np.ndarray, yhat: np.ndarray, scale: float, mask: np.ndarray | None = None
) -> float:
    """Mean absolute scaled error (Hyndman & Koehler 2006): MAE / in-sample seasonal-naive MAE."""
    return mae(y, yhat, mask=mask) / scale


def coverage(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Share of actuals inside the closed interval [lo, hi]."""
    y, lo, hi = _prepare(y, lo, hi, mask=mask)
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo: np.ndarray, hi: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean interval width hi - lo."""
    lo, hi = _prepare(lo, hi, mask=mask)
    return float(np.mean(hi - lo))

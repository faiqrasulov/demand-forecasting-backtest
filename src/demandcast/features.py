"""Leakage-safe features for hourly demand.

Calendar features (hour, weekday, holiday, working day) are known in advance. Lag features
are the only place where the target enters, and :func:`lag_features` refuses to build any lag
shorter than ``min_lag`` (the forecast horizon plus any data latency). With ``min_lag = 168``
one model is valid for every hour of a 7-day horizon (direct strategy), because every input
is at least a week old even for the last forecast hour.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer

WEEK = 168
DAY = 24


def seasonal_lags(min_lag: int, period: int = WEEK, n: int = 4) -> list[int]:
    """The ``n`` smallest multiples of ``period`` that are >= ``min_lag``."""
    first = period * max(1, math.ceil(min_lag / period))
    return [first + period * k for k in range(n)]


def calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Hour, weekday and holiday flags (all known in advance)."""
    idx = pd.DatetimeIndex(frame.index)
    return pd.DataFrame(
        {
            "hour": idx.hour.to_numpy(dtype=np.int64),
            "dayofweek": idx.dayofweek.to_numpy(dtype=np.int64),
            "workingday": frame["workingday"].to_numpy(dtype=np.int64),
            "holiday": frame["holiday"].to_numpy(dtype=np.int64),
        },
        index=frame.index,
    )


def lag_features(y: pd.Series, min_lag: int, n_seasons: int = 4) -> pd.DataFrame:
    """Lagged demand features that only look at least ``min_lag`` hours back.

    ``y`` must be on a regular hourly index. Columns:
    ``lag_<k>`` for the seasonal lags k >= min_lag (same hour, 1-4 weeks back for min_lag=168),
    ``lag_mean`` (mean of those), ``day_mean`` and ``week_mean`` (mean demand over the 24 h and
    168 h windows that end exactly ``min_lag`` hours before the target hour).
    """
    if min_lag < 1:
        raise ValueError("min_lag must be >= 1.")
    lags = seasonal_lags(min_lag, WEEK, n_seasons)
    if min(lags) < min_lag:  # defensive: seasonal_lags guarantees this
        raise ValueError(f"Lag {min(lags)} < min_lag {min_lag} would leak the horizon.")
    y = y.astype(np.float64)
    out = {f"lag_{k}": y.shift(k) for k in lags}
    out["lag_mean"] = pd.concat(out.values(), axis=1).mean(axis=1, skipna=True)
    base = y.shift(min_lag)
    out["day_mean"] = base.rolling(DAY, min_periods=1).mean()
    out["week_mean"] = base.rolling(WEEK, min_periods=DAY).mean()
    return pd.DataFrame(out, index=y.index)


def _periodic_basis(values: np.ndarray, period: float, n_knots: int) -> np.ndarray:
    knots = np.linspace(0.0, period, n_knots).reshape(-1, 1)
    spline = SplineTransformer(degree=3, knots=knots, extrapolation="periodic")
    return spline.fit(knots).transform(np.asarray(values, dtype=np.float64).reshape(-1, 1))


def _spline_basis(values: np.ndarray, lo: float, hi: float, n_knots: int) -> np.ndarray:
    knots = np.linspace(lo, hi, n_knots).reshape(-1, 1)
    spline = SplineTransformer(degree=3, knots=knots, extrapolation="linear")
    return spline.fit(knots).transform(np.asarray(values, dtype=np.float64).reshape(-1, 1))


def ridge_design(frame: pd.DataFrame, weather: bool) -> np.ndarray:
    """Fixed (data-independent) design matrix for the calendar ridge regression.

    Periodic cubic splines of hour-of-day, fitted separately for working and non-working days
    (commuter double peak vs. leisure afternoon hump), weekday dummies, holiday, two annual
    Fourier pairs and a linear trend. With ``weather=True`` adds a temperature spline,
    humidity, wind speed and weather-situation dummies.
    """
    idx = pd.DatetimeIndex(frame.index)
    hour = _periodic_basis(idx.hour.to_numpy(), 24.0, 13)
    working = frame["workingday"].to_numpy(dtype=np.float64)[:, None]
    doy = 2 * np.pi * idx.dayofyear.to_numpy(dtype=np.float64) / 365.25
    years = (idx - pd.Timestamp("2011-01-01")) / pd.Timedelta(days=365.25)
    blocks = [
        hour * working,
        hour * (1.0 - working),
        np.eye(7)[idx.dayofweek.to_numpy()],
        frame["holiday"].to_numpy(dtype=np.float64)[:, None],
        np.column_stack([np.sin(doy), np.cos(doy), np.sin(2 * doy), np.cos(2 * doy)]),
        np.asarray(years, dtype=np.float64)[:, None],
    ]
    if weather:
        code = np.clip(frame["weather"].to_numpy(dtype=np.int64), 1, 3)  # 4 is ~3 rows: merge
        blocks += [
            _spline_basis(frame["temp"].to_numpy(), -5.0, 40.0, 6),
            frame[["humidity"]].to_numpy(dtype=np.float64),
            frame[["windspeed"]].to_numpy(dtype=np.float64) / 10.0,
            np.eye(3)[code - 1],
        ]
    return np.hstack(blocks)

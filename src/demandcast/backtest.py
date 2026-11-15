"""Rolling-origin (expanding window) backtest splitter.

At each forecast origin ``T`` the forecaster may use data strictly before ``T - gap`` and must
predict the ``horizon`` hours ``[T, T + horizon)``. ``gap`` models data latency (hours that
have happened but are not yet available at forecast time). The last ``calibration`` hours of
the training window can be held out as a calibration set for conformal intervals.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    """Positional indices of one backtest fold (all arrays are sorted and contiguous)."""

    origin: pd.Timestamp
    train: np.ndarray  # everything available at the origin: [0, T - gap)
    proper_train: np.ndarray  # train minus the calibration window
    calibration: np.ndarray  # last `calibration` hours of train (empty if calibration=0)
    test: np.ndarray  # [T, T + horizon)


def make_origins(
    first: str | pd.Timestamp, n_origins: int, step_days: int
) -> list[pd.Timestamp]:
    """Evenly spaced forecast origins at midnight, ``step_days`` apart."""
    first = pd.Timestamp(first)
    if first != first.normalize():
        raise ValueError("Origins must be at midnight.")
    if n_origins < 1 or step_days < 1:
        raise ValueError("n_origins and step_days must be positive.")
    return [first + pd.Timedelta(days=step_days * k) for k in range(n_origins)]


class RollingOriginSplitter:
    """Expanding-window splitter over a regular hourly index.

    Parameters
    ----------
    origins: forecast origins (timestamps that must exist in the index).
    horizon: number of hours forecast from each origin.
    gap: hours before the origin that are *not* available for training (data latency).
    calibration: hours at the end of the training window reserved for calibration.
    min_train: minimum number of proper-training hours required at every origin.
    """

    def __init__(
        self,
        origins: Sequence[pd.Timestamp],
        horizon: int = 168,
        gap: int = 0,
        calibration: int = 0,
        min_train: int = 24 * 28,
    ) -> None:
        if horizon < 1 or gap < 0 or calibration < 0 or min_train < 1:
            raise ValueError("horizon >= 1, gap >= 0, calibration >= 0, min_train >= 1.")
        self.origins = sorted(pd.Timestamp(o) for o in origins)
        self.horizon = horizon
        self.gap = gap
        self.calibration = calibration
        self.min_train = min_train

    def split(self, index: pd.DatetimeIndex) -> Iterator[Split]:
        """Yield one :class:`Split` per origin, in time order."""
        index = pd.DatetimeIndex(index)
        if not index.is_monotonic_increasing or not index.is_unique:
            raise ValueError("Index must be sorted and unique.")
        if len(index) > 1 and (np.diff(index.asi8) != np.diff(index.asi8)[0]).any():
            raise ValueError("Index must be regular (no missing timestamps); reindex first.")
        for origin in self.origins:
            if origin not in index:
                raise ValueError(f"Origin {origin} is not in the index.")
            t = int(index.get_loc(origin))
            if t + self.horizon > len(index):
                raise ValueError(f"Origin {origin}: horizon runs past the end of the data.")
            train_end = t - self.gap
            cal_start = train_end - self.calibration
            if cal_start < self.min_train:
                raise ValueError(
                    f"Origin {origin}: only {max(cal_start, 0)} proper-training hours "
                    f"(< min_train={self.min_train})."
                )
            yield Split(
                origin=origin,
                train=np.arange(0, train_end),
                proper_train=np.arange(0, cal_start),
                calibration=np.arange(cal_start, train_end),
                test=np.arange(t, t + self.horizon),
            )

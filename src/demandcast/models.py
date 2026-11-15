"""Forecasting models behind one small interface.

``fit(train)`` receives only the hours available at the forecast origin. ``predict(frame,
index)`` receives a frame on the same regular hourly grid in which ``count`` is NaN for every
hour that is unknown at forecast time, and returns forecasts for ``index``. A model therefore
cannot read the target inside the horizon even by mistake; the lag-based models additionally
never look fewer than ``min_lag`` hours back (see :mod:`demandcast.features`).

Frame columns: ``count``, ``recorded``, ``holiday``, ``workingday`` and, in setting B only, the
weather columns.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from demandcast.data import WEATHER_COLS
from demandcast.features import (
    DAY,
    WEEK,
    calendar_features,
    lag_features,
    ridge_design,
    seasonal_lags,
)


class Forecaster(ABC):
    """Common interface: fit on the past, predict hours given a target-masked frame."""

    name: str = "forecaster"
    uses_weather: bool = False

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> Forecaster:
        """Fit on the hours available at the forecast origin."""

    @abstractmethod
    def predict(self, frame: pd.DataFrame, index: pd.DatetimeIndex) -> np.ndarray:
        """Forecast the hours in ``index`` (``frame`` must contain them and their past)."""


def _training_rows(train: pd.DataFrame) -> pd.Series:
    """Hours usable as training targets: recorded and not masked."""
    return train["recorded"].astype(bool) & train["count"].notna()


class SeasonalNaive(Forecaster):
    """Same hour one week earlier, or the mean over the last ``n_seasons`` weeks.

    With a data latency, the lags move back to the smallest whole weeks >= ``min_lag``.
    """

    def __init__(self, n_seasons: int = 1, min_lag: int = WEEK) -> None:
        self.lags = seasonal_lags(min_lag, WEEK, n_seasons)
        self.name = "seasonal_naive" if n_seasons == 1 else f"seasonal_naive_{n_seasons}w"

    def fit(self, train: pd.DataFrame) -> SeasonalNaive:
        return self

    def predict(self, frame: pd.DataFrame, index: pd.DatetimeIndex) -> np.ndarray:
        y = frame["count"]
        vals = np.column_stack([y.shift(k).loc[index].to_numpy(dtype=float) for k in self.lags])
        if np.isnan(vals).all(axis=1).any():
            raise ValueError("Not enough history for the seasonal naive forecast.")
        return np.nanmean(vals, axis=1)


class DailyETS(Forecaster):
    """Top-down: Holt-Winters on daily totals, split into hours by a recent hourly profile.

    Daily totals get a damped additive trend and multiplicative weekly seasonality
    (statsmodels). The hour-of-day shares come from the last ``profile_weeks`` weeks of
    training data, separately for working and non-working days; the day type of each future
    day is known from the calendar.
    """

    name = "daily_ets"

    def __init__(self, profile_weeks: int = 8) -> None:
        self.profile_weeks = profile_weeks

    def fit(self, train: pd.DataFrame) -> DailyETS:
        y = train["count"]
        day = y.index.normalize()
        complete = y.groupby(day).transform("size").to_numpy() == DAY
        y, day = y[complete], day[complete]
        daily = y.groupby(day).sum()
        if len(daily) < 28:
            raise ValueError("DailyETS needs at least 28 complete training days.")
        seasonal = "mul" if (daily > 0).all() else "add"
        model = ExponentialSmoothing(
            daily.to_numpy(dtype=float),
            trend="add",
            damped_trend=True,
            seasonal=seasonal,
            seasonal_periods=7,
            initialization_method="estimated",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fit = model.fit()
        self._last_day = daily.index[-1]

        recent = y.index >= self._last_day - pd.Timedelta(days=7 * self.profile_weeks - 1)
        hours = pd.DataFrame(
            {
                "count": y[recent].to_numpy(dtype=float),
                "hour": y.index[recent].hour,
                "working": train["workingday"].loc[y.index[recent]].to_numpy(dtype=bool),
            }
        )
        totals = hours.groupby(["working", "hour"])["count"].sum()
        self._profile = {}
        for working in (True, False):
            # fall back to the pooled profile if a day type is absent from the window
            t = totals.loc[working] if working in totals.index.get_level_values(0) else (
                totals.groupby(level="hour").sum()
            )
            shares = t.reindex(range(DAY), fill_value=0.0).to_numpy()
            self._profile[working] = shares / shares.sum()
        return self

    def predict(self, frame: pd.DataFrame, index: pd.DatetimeIndex) -> np.ndarray:
        index = pd.DatetimeIndex(index)
        steps = ((index.normalize() - self._last_day) / pd.Timedelta(days=1)).astype(int)
        steps = np.asarray(steps)
        if steps.min() < 1:
            raise ValueError("DailyETS can only forecast days after the training window.")
        daily = np.asarray(self._fit.forecast(int(steps.max())), dtype=float)
        working = frame["workingday"].loc[index].to_numpy(dtype=bool)
        share = np.where(
            working,
            self._profile[True][index.hour],
            self._profile[False][index.hour],
        )
        return np.clip(daily[steps - 1], 0.0, None) * share


class RidgeCalendar(Forecaster):
    """Ridge regression of log1p(demand) on calendar splines (+ weather in setting B)."""

    def __init__(self, weather: bool = False, alpha: float = 1.0) -> None:
        self.uses_weather = weather
        self.alpha = alpha
        self.name = "ridge_calendar"

    def fit(self, train: pd.DataFrame) -> RidgeCalendar:
        rows = train[_training_rows(train)]
        X = ridge_design(rows, self.uses_weather)
        self._model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self._model.fit(X, np.log1p(rows["count"].to_numpy(dtype=float)))
        return self

    def predict(self, frame: pd.DataFrame, index: pd.DatetimeIndex) -> np.ndarray:
        X = ridge_design(frame.loc[index], self.uses_weather)
        return np.clip(np.expm1(self._model.predict(X)), 0.0, None)


class GradientBoosting(Forecaster):
    """HistGradientBoosting on calendar + lag features (+ weather in setting B).

    Direct strategy: every lag is >= ``min_lag`` (the horizon), so a single model serves all
    168 forecast hours without feeding predictions back in. ``loss="poisson"`` (log link) for
    the point forecast of counts; ``loss="quantile"`` for interval bounds. Early stopping is
    off: its random validation split would mix future hours into training.
    """

    def __init__(
        self,
        weather: bool = False,
        min_lag: int = WEEK,
        loss: str = "poisson",
        quantile: float | None = None,
        max_iter: int = 300,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        random_state: int = 0,
    ) -> None:
        self.uses_weather = weather
        self.min_lag = min_lag
        self.name = "gbm_lag168" if loss == "poisson" else f"gbm_{loss}_{quantile}"
        params = {
            "loss": loss,
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "early_stopping": False,
            "random_state": random_state,
        }
        if loss == "quantile":
            if quantile is None or not 0 < quantile < 1:
                raise ValueError("quantile loss needs 0 < quantile < 1.")
            params["quantile"] = quantile
        self.params = params

    def features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Feature table for every hour of ``frame`` (lags computed within ``frame``)."""
        X = calendar_features(frame).join(lag_features(frame["count"], self.min_lag))
        if self.uses_weather:
            X = X.join(frame[WEATHER_COLS].astype(float))
        return X

    def fit(self, train: pd.DataFrame) -> GradientBoosting:
        keep = _training_rows(train).to_numpy()
        X = self.features(train)[keep]
        y = train["count"].to_numpy(dtype=float)[keep]
        self._model = HistGradientBoostingRegressor(**self.params).fit(X, y)
        return self

    def predict(self, frame: pd.DataFrame, index: pd.DatetimeIndex) -> np.ndarray:
        X = self.features(frame).loc[index]
        return np.clip(self._model.predict(X), 0.0, None)


class DirectGradientBoosting(Forecaster):
    """Direct multi-horizon strategy: one Poisson GBM per forecast day.

    The model for day ``d`` of the horizon only uses lags >= ``gap + 24 d``, which is exactly
    the information available for every hour of that day (day 1 may use yesterday, day 7 may
    not). Hours are routed to their day's model by their distance from the last training hour.
    """

    name = "gbm_direct"

    def __init__(
        self, weather: bool = False, horizon: int = WEEK, gap: int = 0, **gbm_kwargs
    ) -> None:
        self.uses_weather = weather
        self.gap = gap
        self.models = [
            GradientBoosting(weather=weather, min_lag=gap + DAY * d, **gbm_kwargs)
            for d in range(1, math.ceil(horizon / DAY) + 1)
        ]

    def fit(self, train: pd.DataFrame) -> DirectGradientBoosting:
        self._last = train.index[-1]
        for model in self.models:
            model.fit(train)
        return self

    def predict(self, frame: pd.DataFrame, index: pd.DatetimeIndex) -> np.ndarray:
        index = pd.DatetimeIndex(index)
        ahead = np.asarray((index - self._last) / pd.Timedelta(hours=1)).astype(int)
        day = np.ceil((ahead - self.gap) / DAY).astype(int)
        if day.min() < 1 or day.max() > len(self.models):
            raise ValueError("Requested hours lie outside the horizon the model was built for.")
        out = np.empty(len(index))
        for d in np.unique(day):
            sel = day == d
            out[sel] = self.models[d - 1].predict(frame, index[sel])
        return out

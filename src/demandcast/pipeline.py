"""Rolling-origin backtest of every model in two information settings.

Setting A: demand history + calendar only (what is actually known at the forecast origin).
Setting B: A + the *observed* weather of the forecast week, i.e. a perfect weather forecast.
B is an upper bound on what weather forecasts could add, not an operational setting.

At each origin the target is masked from the origin onwards (:func:`mask_future`), models are
refitted on the expanding window, and three interval methods are calibrated on the last
``calibration_hours`` before the origin.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from demandcast.backtest import RollingOriginSplitter, Split, make_origins
from demandcast.data import WEATHER_COLS
from demandcast.features import WEEK
from demandcast.intervals import cqr_interval, split_conformal_interval
from demandcast.metrics import seasonal_naive_scale
from demandcast.models import (
    DailyETS,
    DirectGradientBoosting,
    Forecaster,
    GradientBoosting,
    RidgeCalendar,
    SeasonalNaive,
)

SETTINGS = {"A": False, "B": True}  # setting -> uses observed weather
POINT_MODELS = [
    "seasonal_naive",
    "seasonal_naive_4w",
    "daily_ets",
    "ridge_calendar",
    "gbm_lag168",
    "gbm_direct",
]
INTERVAL_METHODS = ["seasonal_naive_conformal", "gbm_quantile", "gbm_cqr"]


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest design. Defaults reproduce the README."""

    first_origin: str = "2012-01-29"
    n_origins: int = 12
    step_days: int = 30
    horizon: int = WEEK
    gap: int = 0
    calibration_hours: int = 4 * WEEK
    levels: tuple[float, ...] = (0.8, 0.95)
    gbm_max_iter: int = 300
    gbm_learning_rate: float = 0.05
    seed: int = 0

    @property
    def min_lag(self) -> int:
        """Shortest lag any model may use: the horizon plus the data latency."""
        return self.horizon + self.gap

    @property
    def origins(self) -> list[pd.Timestamp]:
        return make_origins(self.first_origin, self.n_origins, self.step_days)


@dataclass
class BacktestResult:
    """Long tables of point and interval forecasts plus the MASE scale of each origin."""

    points: pd.DataFrame
    intervals: pd.DataFrame
    scales: dict[pd.Timestamp, float] = field(default_factory=dict)


def make_splitter(config: BacktestConfig) -> RollingOriginSplitter:
    return RollingOriginSplitter(
        config.origins,
        horizon=config.horizon,
        gap=config.gap,
        calibration=config.calibration_hours,
        min_train=2 * WEEK,
    )


def point_models(config: BacktestConfig, weather: bool) -> list[Forecaster]:
    """The six point forecasters, in reporting order."""
    return [
        SeasonalNaive(1, config.min_lag),
        SeasonalNaive(4, config.min_lag),
        DailyETS(),
        RidgeCalendar(weather=weather),
        _gbm(config, weather, "poisson"),
        DirectGradientBoosting(
            weather=weather,
            horizon=config.horizon,
            gap=config.gap,
            max_iter=config.gbm_max_iter,
            learning_rate=config.gbm_learning_rate,
            random_state=config.seed,
        ),
    ]


def _gbm(
    config: BacktestConfig, weather: bool, loss: str, quantile: float | None = None
) -> GradientBoosting:
    return GradientBoosting(
        weather=weather,
        min_lag=config.min_lag,
        loss=loss,
        quantile=quantile,
        max_iter=config.gbm_max_iter,
        learning_rate=config.gbm_learning_rate,
        random_state=config.seed,
    )


def mask_future(df: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Data as seen at the forecast origin: target hidden from ``T - gap`` to the horizon end."""
    known = df.iloc[: split.test[-1] + 1].copy()
    hidden = known.index[len(split.train) :]
    known.loc[hidden, "count"] = np.nan
    known.loc[hidden, "recorded"] = False
    return known


def setting_frame(frame: pd.DataFrame, weather: bool) -> pd.DataFrame:
    """Setting A physically drops the weather columns, so no model can use them."""
    return frame if weather else frame.drop(columns=WEATHER_COLS, errors="ignore")


def _ordered(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Repair quantile crossing (independently fitted lower/upper models)."""
    return np.minimum(lo, hi), np.maximum(lo, hi)


def run_backtest(
    df: pd.DataFrame,
    config: BacktestConfig | None = None,
    settings: tuple[str, ...] = ("A", "B"),
    log: Callable[[str], None] | None = None,
) -> BacktestResult:
    """Run the full rolling-origin backtest on a regularised hourly frame."""
    config = config or BacktestConfig()
    y_all = df["count"].to_numpy(dtype=float)
    rec_all = df["recorded"].to_numpy(dtype=bool)
    point_parts: list[pd.DataFrame] = []
    interval_parts: list[pd.DataFrame] = []
    scales: dict[pd.Timestamp, float] = {}

    for split in make_splitter(config).split(df.index):
        known = mask_future(df, split)
        test_index, cal_index = df.index[split.test], df.index[split.calibration]
        y_cal, rec_cal = y_all[split.calibration], rec_all[split.calibration]
        scales[split.origin] = seasonal_naive_scale(
            y_all[split.train], WEEK, rec_all[split.train]
        )
        base = {
            "origin": split.origin,
            "timestamp": test_index,
            "step": np.arange(1, config.horizon + 1),
            "y": y_all[split.test],
            "recorded": rec_all[split.test],
        }
        shared: dict[str, np.ndarray] = {}  # forecasts of models that ignore weather

        for setting in settings:
            weather = SETTINGS[setting]
            frame = setting_frame(known, weather)
            train = frame.iloc[split.train]
            for model in point_models(config, weather):
                if not model.uses_weather and model.name in shared:
                    yhat = shared[model.name]
                else:
                    yhat = model.fit(train).predict(frame, test_index)
                    if not model.uses_weather:
                        shared[model.name] = yhat
                point_parts.append(
                    pd.DataFrame({**base, "setting": setting, "model": model.name, "yhat": yhat})
                )

            proper = frame.iloc[split.proper_train]
            naive = SeasonalNaive(1, config.min_lag)
            naive_cal = naive.predict(frame, cal_index)[rec_cal]
            naive_test = naive.predict(frame, test_index)
            for level in config.levels:
                alpha = 1.0 - level
                bounds = {
                    "seasonal_naive_conformal": split_conformal_interval(
                        y_cal[rec_cal], naive_cal, naive_test, alpha
                    )
                }
                q_lo = _gbm(config, weather, "quantile", alpha / 2).fit(proper)
                q_hi = _gbm(config, weather, "quantile", 1 - alpha / 2).fit(proper)
                cal_lo, cal_hi = _ordered(
                    q_lo.predict(frame, cal_index), q_hi.predict(frame, cal_index)
                )
                test_lo, test_hi = _ordered(
                    q_lo.predict(frame, test_index), q_hi.predict(frame, test_index)
                )
                bounds["gbm_quantile"] = (test_lo, test_hi)
                bounds["gbm_cqr"] = cqr_interval(
                    y_cal[rec_cal], cal_lo[rec_cal], cal_hi[rec_cal], test_lo, test_hi, alpha
                )
                for method, (lo, hi) in bounds.items():
                    interval_parts.append(
                        pd.DataFrame(
                            {
                                **base,
                                "setting": setting,
                                "method": method,
                                "level": level,
                                "lo": lo,
                                "hi": hi,
                            }
                        )
                    )
        if log:
            log(f"origin {split.origin:%Y-%m-%d}: done")

    return BacktestResult(
        points=pd.concat(point_parts, ignore_index=True),
        intervals=pd.concat(interval_parts, ignore_index=True),
        scales=scales,
    )

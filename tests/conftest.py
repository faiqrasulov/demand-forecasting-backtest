"""Shared fixtures: small synthetic hourly demand series (no network, no real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demandcast.data import COLUMNS


def make_hourly(n_weeks: int = 10, seed: int = 0, missing: int = 12) -> pd.DataFrame:
    """Synthetic canonical frame: commuter/leisure hourly shapes, weekly cycle, Poisson noise.

    Starts on 2011-01-01 like the real data. ``missing`` random night hours are unrecorded.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2011-01-01", periods=n_weeks * 168, freq="h")
    hour = idx.hour.to_numpy()
    working = idx.day_of_week.to_numpy() < 5
    commute = 20 + 150 * np.exp(-((hour - 8) ** 2) / 3) + 200 * np.exp(-((hour - 17.5) ** 2) / 4)
    leisure = 15 + 140 * np.exp(-((hour - 14) ** 2) / 12)
    temp = 10 + 8 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 2, len(idx))
    level = np.where(working, commute, leisure) * (1 + 0.02 * (temp - 10))
    count = rng.poisson(np.clip(level, 1, None)).astype(float)
    recorded = np.ones(len(idx), dtype=bool)
    night = np.flatnonzero((hour >= 2) & (hour <= 4))
    recorded[rng.choice(night, size=missing, replace=False)] = False
    count[~recorded] = 0.0
    frame = pd.DataFrame(
        {
            "count": count,
            "recorded": recorded,
            "holiday": np.zeros(len(idx), dtype=bool),
            "workingday": working,
            "temp": temp,
            "feel_temp": temp + 2.0,
            "humidity": rng.uniform(0.3, 0.9, len(idx)),
            "windspeed": rng.uniform(0, 25, len(idx)),
            "weather": rng.choice([1, 1, 1, 2, 3], size=len(idx)).astype(np.int64),
        },
        index=idx,
    )
    frame.index.name = "timestamp"
    return frame[COLUMNS]


@pytest.fixture
def hourly() -> pd.DataFrame:
    return make_hourly()

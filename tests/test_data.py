import urllib.error

import numpy as np
import pandas as pd
import pytest

from demandcast import data
from demandcast.data import (
    COLUMNS,
    load_bike_hourly,
    openml_timestamps,
    openml_to_records,
    regularize,
    uci_to_records,
)


def _openml_like(stamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Frame with the OpenML 42712 layout (no date column, string categories)."""
    n = len(stamps)
    return pd.DataFrame(
        {
            "season": pd.Categorical(["spring"] * n),
            "year": stamps.year - 2011,
            "month": stamps.month,
            "hour": stamps.hour,
            "holiday": pd.Categorical(np.where(stamps.normalize() == "2011-01-03", "True",
                                               "False")),
            "weekday": (stamps.day_of_week + 1) % 7,
            "workingday": pd.Categorical(np.where(stamps.day_of_week < 5, "True", "False")),
            "weather": pd.Categorical(np.where(stamps.hour % 5 == 0, "misty", "clear")),
            "temp": np.linspace(5, 15, n),
            "feel_temp": np.linspace(6, 16, n),
            "humidity": np.full(n, 0.5),
            "windspeed": np.full(n, 10.0),
            "count": np.arange(1, n + 1),
        }
    )


def _gappy_stamps() -> pd.DatetimeIndex:
    full = pd.date_range("2011-01-01", "2011-01-10 23:00", freq="h")
    drop = full[(full.normalize() == "2011-01-05")]  # a whole missing day
    drop = drop.append(full[[3, 4, 50]])  # and some missing night hours
    return full.difference(drop)


def test_openml_timestamps_are_rebuilt_from_row_order():
    stamps = _gappy_stamps()
    rebuilt = openml_timestamps(_openml_like(stamps))
    pd.testing.assert_index_equal(rebuilt, stamps, check_names=False)


def test_openml_timestamps_reject_inconsistent_rows():
    frame = _openml_like(_gappy_stamps())
    frame.loc[30, "month"] = 2
    with pytest.raises(ValueError, match="disagree"):
        openml_timestamps(frame)


def test_regularize_fills_the_grid_and_flags_unrecorded_hours():
    stamps = _gappy_stamps()
    out = regularize(openml_to_records(_openml_like(stamps)))
    assert list(out.columns) == COLUMNS
    assert len(out) == 10 * 24 and out.index.is_monotonic_increasing
    missing = ~out.index.isin(stamps)
    assert (out["recorded"].to_numpy() == ~missing).all()
    assert (out.loc[missing, "count"] == 0).all()
    assert out["holiday"].dtype == bool and out.loc["2011-01-03", "holiday"].all()
    assert not out.loc["2011-01-05", "workingday"].isna().any()
    assert out[["temp", "weather"]].notna().all().all()
    assert set(out["weather"].unique()) <= {1, 2}


def test_uci_records_undo_the_normalisation():
    frame = pd.DataFrame(
        {
            "dteday": ["2011-01-01", "2011-01-01"],
            "hr": [0, 1],
            "holiday": [0, 0],
            "workingday": [0, 0],
            "weathersit": [1, 2],
            "temp": [0.24, 0.22],
            "atemp": [0.2879, 0.2727],
            "hum": [0.81, 0.8],
            "windspeed": [0.0, 0.2537],
            "cnt": [16, 40],
        }
    )
    rec = uci_to_records(frame)
    np.testing.assert_allclose(rec["temp"], [9.84, 9.02])
    np.testing.assert_allclose(rec["feel_temp"], [14.395, 13.635])
    np.testing.assert_allclose(rec["windspeed"], [0.0, 16.9979])
    assert rec.index[1] == pd.Timestamp("2011-01-01 01:00")


def test_offline_failure_is_clear(monkeypatch, tmp_path):
    def offline(_):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(data, "_fetch_openml", offline)
    monkeypatch.setattr(data, "_fetch_uci", offline)
    with pytest.raises(RuntimeError, match="offline"):
        load_bike_hourly(tmp_path)

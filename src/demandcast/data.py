"""Download the UCI Bike Sharing hourly data and turn it into a regular hourly series.

Primary source: OpenML dataset 42712 ("Bike_Sharing_Demand"), a copy of the UCI hourly file
with de-normalised weather columns but **no date column**. The timestamps are rebuilt from the
row order (the file is chronological) and validated against the year, month and weekday
columns. Fallback: the official UCI zip, which has explicit dates.

The raw file omits 165 of the 17,544 hours in 2011-2012: mostly 2-5 am hours with no
rentals, plus a few outages (January 2011 snow storms, hurricanes Irene and Sandy).
:func:`regularize` puts the data on a complete hourly grid, sets ``count = 0`` for the
missing hours (no recorded rentals) and adds a ``recorded`` flag so those hours are never
used as training targets or scored.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

OPENML_ID = 42712
UCI_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
START = pd.Timestamp("2011-01-01")
END = pd.Timestamp("2012-12-31 23:00")

# Columns of the canonical hourly frame (index: hourly DatetimeIndex).
WEATHER_COLS = ["temp", "feel_temp", "humidity", "windspeed", "weather"]
CALENDAR_COLS = ["holiday", "workingday"]
COLUMNS = ["count", "recorded", *CALENDAR_COLS, *WEATHER_COLS]

_WEATHER_CODES = {"clear": 1, "misty": 2, "rain": 3, "heavy_rain": 4}


def _as_bool(values: pd.Series) -> np.ndarray:
    """Map True/False flags stored as bool, 0/1 or 'True'/'False' strings to a bool array."""
    as_str = values.astype(str).str.strip().str.lower()
    return as_str.isin(["true", "1", "1.0"]).to_numpy()


def openml_timestamps(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Rebuild hourly timestamps for the OpenML copy (which has no date column).

    Rows are chronological; a new day starts whenever ``weekday`` changes, and the number of
    days advanced is the weekday difference modulo 7 (so a fully missing day would still be
    placed correctly). The result is checked against ``year``, ``month`` and ``weekday``.
    """
    weekday = frame["weekday"].to_numpy(dtype=np.int64)
    hour = frame["hour"].to_numpy(dtype=np.int64)
    steps = np.mod(np.diff(weekday), 7)
    day = np.concatenate([[0], np.cumsum(steps)])
    first_weekday = (START.day_of_week + 1) % 7  # UCI convention: 0 = Sunday
    if weekday[0] != first_weekday:
        raise ValueError("First row is not 2011-01-01; cannot rebuild timestamps.")
    stamps = pd.DatetimeIndex(
        START + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour, unit="h")
    )
    ok = (
        stamps.is_monotonic_increasing
        and stamps.is_unique
        and np.array_equal(stamps.year - START.year, frame["year"].to_numpy(dtype=np.int64))
        and np.array_equal(stamps.month, frame["month"].to_numpy(dtype=np.int64))
        and np.array_equal((stamps.day_of_week + 1) % 7, weekday)
    )
    if not ok:
        raise ValueError("Rebuilt timestamps disagree with the year/month/weekday columns.")
    return stamps


def openml_to_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert the OpenML frame to canonical hourly records (only the hours present)."""
    weather = frame["weather"].astype(str).map(_WEATHER_CODES)
    if weather.isna().any():
        raise ValueError(f"Unknown weather labels: {sorted(frame['weather'].astype(str).unique())}")
    out = pd.DataFrame(
        {
            "count": frame["count"].to_numpy(dtype=np.float64),
            "holiday": _as_bool(frame["holiday"]),
            "workingday": _as_bool(frame["workingday"]),
            "temp": frame["temp"].to_numpy(dtype=np.float64),
            "feel_temp": frame["feel_temp"].to_numpy(dtype=np.float64),
            "humidity": frame["humidity"].to_numpy(dtype=np.float64),
            "windspeed": frame["windspeed"].to_numpy(dtype=np.float64),
            "weather": weather.to_numpy(dtype=np.int64),
        },
        index=openml_timestamps(frame),
    )
    out.index.name = "timestamp"
    return out


def uci_to_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert UCI ``hour.csv`` to canonical records, undoing the UCI weather normalisation."""
    stamps = pd.DatetimeIndex(
        pd.to_datetime(frame["dteday"]) + pd.to_timedelta(frame["hr"].astype(int), unit="h")
    )
    out = pd.DataFrame(
        {
            "count": frame["cnt"].to_numpy(dtype=np.float64),
            "holiday": _as_bool(frame["holiday"]),
            "workingday": _as_bool(frame["workingday"]),
            "temp": frame["temp"].to_numpy(dtype=np.float64) * 41.0,
            "feel_temp": frame["atemp"].to_numpy(dtype=np.float64) * 50.0,
            "humidity": frame["hum"].to_numpy(dtype=np.float64),
            "windspeed": frame["windspeed"].to_numpy(dtype=np.float64) * 67.0,
            "weather": frame["weathersit"].to_numpy(dtype=np.int64),
        },
        index=stamps,
    )
    out.index.name = "timestamp"
    return out


def regularize(records: pd.DataFrame) -> pd.DataFrame:
    """Reindex records to a complete hourly grid.

    Missing hours get ``count = 0`` and ``recorded = False``; the calendar flags are taken
    from the same date (known in advance) and weather is forward/back filled.
    """
    if not records.index.is_unique:
        raise ValueError("Duplicate timestamps in records.")
    grid = pd.date_range(records.index.min().normalize(), records.index.max(), freq="h")
    out = records.reindex(grid)
    out.index.name = "timestamp"
    recorded = out["count"].notna().to_numpy()
    out["count"] = out["count"].fillna(0.0).astype(np.float64)
    out["recorded"] = recorded
    day = out.index.normalize()
    for col in CALENDAR_COLS:
        filled = out[col].groupby(day).transform("first")
        out[col] = filled.astype(bool).to_numpy()
    for col in WEATHER_COLS:
        out[col] = out[col].ffill().bfill()
    out["weather"] = out["weather"].astype(np.int64)
    return out[COLUMNS]


def _fetch_openml(data_home: Path) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, data_home=str(data_home))
    return openml_to_records(bunch.frame)


def _fetch_uci(data_home: Path) -> pd.DataFrame:
    path = data_home / "bike+sharing+dataset.zip"
    if not path.exists():
        with urllib.request.urlopen(UCI_URL, timeout=60) as resp:  # noqa: S310 (fixed URL)
            path.write_bytes(resp.read())
    with zipfile.ZipFile(path) as zf:
        frame = pd.read_csv(io.BytesIO(zf.read("hour.csv")))
    return uci_to_records(frame)


def load_bike_hourly(data_home: str | Path = "data") -> pd.DataFrame:
    """Return the regularised hourly Bike Sharing frame (17,544 rows, 2011-2012).

    Downloads on first use and caches under ``data_home``. Tries OpenML first, then the UCI
    zip; raises ``RuntimeError`` with a clear message if both fail (e.g. offline).
    """
    data_home = Path(data_home)
    data_home.mkdir(parents=True, exist_ok=True)
    errors = []
    for fetch in (_fetch_openml, _fetch_uci):
        try:
            records = fetch(data_home)
            break
        except Exception as exc:  # network errors, HTTP errors, parsing problems
            errors.append(f"{fetch.__name__}: {type(exc).__name__}: {exc}")
    else:
        raise RuntimeError(
            "Could not load the Bike Sharing data (are you offline?). Tried OpenML "
            f"data_id={OPENML_ID} and {UCI_URL}.\n" + "\n".join(errors)
        )
    frame = regularize(records)
    if frame.index[0] != START or frame.index[-1] != END:
        raise ValueError(f"Unexpected date range {frame.index[0]} .. {frame.index[-1]}.")
    return frame

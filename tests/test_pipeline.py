import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from conftest import make_hourly

from demandcast import cli
from demandcast.data import WEATHER_COLS
from demandcast.pipeline import (
    INTERVAL_METHODS,
    POINT_MODELS,
    BacktestConfig,
    make_splitter,
    mask_future,
    run_backtest,
    setting_frame,
)

SMALL = BacktestConfig(
    first_origin="2011-02-26", n_origins=2, step_days=7, calibration_hours=168, gbm_max_iter=15
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return make_hourly(n_weeks=10, seed=3)


@pytest.fixture(scope="module")
def result(df):
    return run_backtest(df, SMALL)


def test_backtest_shapes(result):
    n = SMALL.n_origins * 2 * SMALL.horizon
    assert len(result.points) == n * len(POINT_MODELS)
    assert len(result.intervals) == n * len(INTERVAL_METHODS) * len(SMALL.levels)
    assert set(result.points["model"]) == set(POINT_MODELS)
    assert np.isfinite(result.points["yhat"]).all()
    iv = result.intervals
    assert (iv["lo"] <= iv["hi"]).all() and (iv["lo"] >= 0).all()
    wide = iv[np.isclose(iv["level"], 0.95)]["hi"].to_numpy()
    narrow = iv[np.isclose(iv["level"], 0.8)]["hi"].to_numpy()
    assert (wide >= narrow - 1e-9).mean() > 0.95  # nested intervals (up to quantile noise)


def test_masking_hides_the_target_from_the_origin(df):
    split = next(make_splitter(SMALL).split(df.index))
    known = mask_future(df, split)
    t = len(split.train)
    assert known["count"].iloc[:t].notna().all()
    assert known["count"].iloc[t:].isna().all() and not known["recorded"].iloc[t:].any()
    assert not set(WEATHER_COLS) & set(setting_frame(known, weather=False).columns)


def test_backtest_forecasts_ignore_data_after_the_origin(df, result):
    """End to end: corrupt all demand at/after the first origin; that origin's point and
    interval forecasts must not change."""
    first = SMALL.origins[0]
    corrupted = df.copy()
    pos = corrupted.index.get_loc(first)
    corrupted.iloc[pos:, corrupted.columns.get_loc("count")] = 1e5
    again = run_backtest(corrupted, replace(SMALL, n_origins=1))
    a = result.points[result.points["origin"] == first].reset_index(drop=True)
    b = again.points
    np.testing.assert_array_equal(a["yhat"].to_numpy(), b["yhat"].to_numpy())
    ia = result.intervals[result.intervals["origin"] == first].reset_index(drop=True)
    np.testing.assert_array_equal(ia[["lo", "hi"]].to_numpy(), again.intervals[["lo", "hi"]])


def test_full_pipeline_writes_metrics_and_figures(df, tmp_path):
    metrics = cli.run(df, SMALL, tmp_path)
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["design"]["origins"] == ["2011-02-26 00:00", "2011-03-05 00:00"]
    for setting in ("A", "B"):
        for model in POINT_MODELS:
            v = saved["point"][setting][model]
            assert len(v["per_origin"]["mase"]) == 2 and v["mase"]["mean"] > 0
        for method in INTERVAL_METHODS:
            assert 0 <= saved["intervals"][setting][method]["80"]["coverage"]["mean"] <= 1
    # models that ignore weather are identical in both settings; weather models are not
    assert saved["point"]["A"]["daily_ets"] == saved["point"]["B"]["daily_ets"]
    assert "gbm_direct" in metrics["weather_oracle_gain"]
    assert len(saved["error_profiles"]["A"]["gbm_direct"]["wape_by_horizon_day"]) == 7
    for name in ("backtest_origins", "forecast_week", "model_comparison", "intervals_and_horizon"):
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 10_000

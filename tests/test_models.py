import numpy as np
import pandas as pd
import pytest
from conftest import make_hourly

from demandcast.models import (
    DailyETS,
    DirectGradientBoosting,
    GradientBoosting,
    RidgeCalendar,
    SeasonalNaive,
)

ORIGIN_POS = 7 * 168  # forecast origin: start of week 8
H = 168


def _models():
    return [
        SeasonalNaive(1),
        SeasonalNaive(4),
        DailyETS(),
        RidgeCalendar(weather=False),
        RidgeCalendar(weather=True),
        GradientBoosting(weather=False, max_iter=15),
        GradientBoosting(weather=True, max_iter=15),
        DirectGradientBoosting(weather=False, max_iter=15),
    ]


def test_seasonal_naive_is_exact_on_a_weekly_periodic_series():
    idx = pd.date_range("2011-01-01", periods=6 * 168, freq="h")
    pattern = np.random.default_rng(0).uniform(0, 500, 168)
    frame = pd.DataFrame({"count": np.tile(pattern, 6)}, index=idx)
    test = idx[4 * 168 : 5 * 168]
    for model in (SeasonalNaive(1), SeasonalNaive(4)):
        np.testing.assert_allclose(model.fit(frame).predict(frame, test), pattern, rtol=1e-12)


def test_seasonal_naive_moves_lags_back_for_data_latency():
    model = SeasonalNaive(n_seasons=2, min_lag=168 + 30)
    assert model.lags == [336, 504]


@pytest.mark.parametrize("model", _models(), ids=lambda m: f"{m.name}-{m.uses_weather}")
def test_forecasts_do_not_depend_on_the_future(model):
    """Fit on the past, then predict the horizon from an UNMASKED frame twice: once with the
    true future and once with every target value at/after the origin replaced by noise.
    Identical forecasts prove the model cannot read the horizon's (or later) demand."""
    df = make_hourly(n_weeks=10)
    test = df.index[ORIGIN_POS : ORIGIN_POS + H]
    model.fit(df.iloc[:ORIGIN_POS])
    a = model.predict(df, test)
    corrupted = df.copy()
    noise = np.random.default_rng(2).uniform(0, 5000, len(df) - ORIGIN_POS)
    corrupted.iloc[ORIGIN_POS:, corrupted.columns.get_loc("count")] = noise
    b = model.predict(corrupted, test)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (H,) and np.isfinite(a).all() and (a >= 0).all()


def test_direct_gbm_routes_each_day_to_its_own_model(hourly):
    model = DirectGradientBoosting(max_iter=15).fit(hourly.iloc[:ORIGIN_POS])
    test = hourly.index[ORIGIN_POS : ORIGIN_POS + H]
    pred = model.predict(hourly, test)
    assert [m.min_lag for m in model.models] == [24, 48, 72, 96, 120, 144, 168]
    for d in (1, 4, 7):
        hours = test[(d - 1) * 24 : d * 24]
        np.testing.assert_array_equal(
            pred[(d - 1) * 24 : d * 24], model.models[d - 1].predict(hourly, hours)
        )
    with pytest.raises(ValueError, match="outside the horizon"):
        model.predict(hourly, hourly.index[ORIGIN_POS + H : ORIGIN_POS + H + 1])


def test_daily_ets_splits_daily_totals_by_a_normalised_profile(hourly):
    model = DailyETS().fit(hourly.iloc[:ORIGIN_POS])
    for shares in model._profile.values():
        assert shares.sum() == pytest.approx(1.0)
    test = hourly.index[ORIGIN_POS : ORIGIN_POS + H]
    pred = model.predict(hourly, test)
    daily_pred = pd.Series(pred, index=test).groupby(test.normalize()).sum().to_numpy()
    np.testing.assert_allclose(daily_pred, model._fit.forecast(7), rtol=1e-9)


def test_daily_ets_recovers_a_noise_free_series():
    """Constant weekly cycle of daily totals and a fixed hourly profile: exact recovery."""
    idx = pd.date_range("2011-01-01", periods=10 * 168, freq="h")
    weekday_total = np.array([1000, 1100, 1200, 1300, 1400, 900, 800.0])
    profile = np.random.default_rng(0).uniform(0.5, 1.5, 24)
    profile /= profile.sum()
    count = weekday_total[idx.day_of_week] * profile[idx.hour]
    frame = pd.DataFrame(
        {"count": count, "recorded": True, "holiday": False, "workingday": idx.day_of_week < 5},
        index=idx,
    )
    model = DailyETS().fit(frame.iloc[:ORIGIN_POS])
    test = idx[ORIGIN_POS : ORIGIN_POS + H]
    np.testing.assert_allclose(model.predict(frame, test), count[ORIGIN_POS:ORIGIN_POS + H],
                               rtol=0.02)


def test_unrecorded_hours_are_not_training_targets(hourly):
    """Setting unrecorded hours to absurd values must not change a fitted ridge model."""
    a = RidgeCalendar().fit(hourly.iloc[:ORIGIN_POS])
    shifted = hourly.copy()
    unrec = ~shifted["recorded"].to_numpy()
    shifted.loc[shifted.index[unrec], "count"] = 1e6
    b = RidgeCalendar().fit(shifted.iloc[:ORIGIN_POS])
    test = hourly.index[ORIGIN_POS : ORIGIN_POS + H]
    np.testing.assert_allclose(a.predict(hourly, test), b.predict(hourly, test))


def test_quantile_gbm_requires_a_valid_quantile():
    with pytest.raises(ValueError):
        GradientBoosting(loss="quantile", quantile=None)

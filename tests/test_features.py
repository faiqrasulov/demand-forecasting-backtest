import re

import numpy as np
import pandas as pd
import pytest

from demandcast.features import lag_features, ridge_design, seasonal_lags


def test_seasonal_lags():
    assert seasonal_lags(168) == [168, 336, 504, 672]
    assert seasonal_lags(24, n=2) == [168, 336]
    assert seasonal_lags(170, n=2) == [336, 504]


@pytest.mark.parametrize("min_lag", [24, 48, 168, 175])
def test_lag_features_never_use_values_within_the_horizon(hourly, min_lag):
    """Changing every target value at/after the origin leaves the features of the next
    `min_lag` hours unchanged, i.e. they can be computed at the origin."""
    y = hourly["count"]
    t = 6 * 168
    perturbed = y.copy()
    perturbed.iloc[t:] = np.random.default_rng(1).uniform(0, 1e4, len(y) - t)
    a = lag_features(y, min_lag).iloc[t : t + min_lag]
    b = lag_features(perturbed, min_lag).iloc[t : t + min_lag]
    pd.testing.assert_frame_equal(a, b)
    # ... while the hour right after the horizon does see the change (the test has power)
    c = lag_features(y, min_lag).iloc[t + min_lag]
    d = lag_features(perturbed, min_lag).iloc[t + min_lag]
    assert not np.allclose(c.to_numpy(), d.to_numpy(), equal_nan=True)


@pytest.mark.parametrize("min_lag", [24, 168])
def test_lag_columns_respect_min_lag_and_are_exact_shifts(hourly, min_lag):
    y = hourly["count"]
    X = lag_features(y, min_lag)
    lags = [int(m.group(1)) for c in X.columns if (m := re.fullmatch(r"lag_(\d+)", c))]
    assert lags and min(lags) >= min_lag
    for k in lags:
        np.testing.assert_array_equal(X[f"lag_{k}"].to_numpy()[k:], y.to_numpy()[:-k])
    t = 900
    end = t - min_lag + 1
    np.testing.assert_allclose(X["day_mean"].iloc[t], y.iloc[end - 24 : end].mean())
    np.testing.assert_allclose(X["week_mean"].iloc[t], y.iloc[end - 168 : end].mean())


def test_lag_features_reject_bad_min_lag(hourly):
    with pytest.raises(ValueError):
        lag_features(hourly["count"], 0)


def test_ridge_design_is_deterministic_and_periodic(hourly):
    X_a = ridge_design(hourly, weather=False)
    X_b = ridge_design(hourly.iloc[100:200], weather=False)
    np.testing.assert_allclose(X_a[100:200], X_b)  # row-wise, no data-dependent fitting
    assert ridge_design(hourly, weather=True).shape[1] > X_a.shape[1]
    # hour-of-day basis is periodic: hours 23 and 0 are as close as hours 0 and 1
    # (rows 23-25 are Saturday/Sunday, so the non-working-day block, columns 12-23)
    hour_cols = X_a[:, 12:24]
    assert np.abs(hour_cols[23:26]).sum() > 0
    d_wrap = np.abs(hour_cols[23] - hour_cols[24]).sum()
    d_step = np.abs(hour_cols[24] - hour_cols[25]).sum()
    assert d_wrap == pytest.approx(d_step, rel=1e-9)

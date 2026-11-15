import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error

from demandcast.metrics import (
    coverage,
    mae,
    mase,
    mean_width,
    rmse,
    seasonal_naive_scale,
    wape,
)


def test_known_answers():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    yhat = np.array([12.0, 18.0, 33.0, 40.0])
    assert mae(y, yhat) == pytest.approx(7 / 4)
    assert rmse(y, yhat) == pytest.approx(np.sqrt(17 / 4))
    assert wape(y, yhat) == pytest.approx(7 / 100)
    # MASE: in-sample seasonal naive with m = 2 on [1, 3, 2, 6, 3] has errors |2-1|, |6-3|, |3-2|
    scale = seasonal_naive_scale(np.array([1.0, 3.0, 2.0, 6.0, 3.0]), m=2)
    assert scale == pytest.approx(5 / 3)
    assert mase(y, yhat, scale) == pytest.approx((7 / 4) / (5 / 3))


def test_agrees_with_sklearn_on_random_data():
    rng = np.random.default_rng(0)
    y, yhat = rng.gamma(2, 50, 500), rng.gamma(2, 50, 500)
    assert mae(y, yhat) == pytest.approx(mean_absolute_error(y, yhat))
    assert rmse(y, yhat) == pytest.approx(np.sqrt(mean_squared_error(y, yhat)))
    assert wape(y, yhat) == pytest.approx(mean_absolute_error(y, yhat) / y.mean())


def test_mask_excludes_hours():
    y = np.array([10.0, 0.0, 30.0])
    yhat = np.array([12.0, 50.0, 27.0])
    mask = np.array([True, False, True])
    assert mae(y, yhat, mask) == pytest.approx(2.5)
    assert wape(y, yhat, mask) == pytest.approx(5 / 40)
    # MASE scale: pairs touching an unscored point are dropped
    train = np.array([1.0, 100.0, 2.0, 4.0, 3.0])
    keep = np.array([True, False, True, True, True])
    assert seasonal_naive_scale(train, 1, keep) == pytest.approx((2 + 1) / 2)


def test_seasonal_naive_has_mase_one_in_sample_on_its_own_scale():
    y = np.random.default_rng(1).poisson(20, 400).astype(float)
    m = 24
    scale = seasonal_naive_scale(y, m)
    assert mase(y[m:], y[:-m], scale) == pytest.approx(1.0)


def test_interval_metrics_and_errors():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo, hi = np.array([0.0, 2.0, 3.5, 3.0]), np.array([2.0, 2.0, 5.0, 4.0])
    assert coverage(y, lo, hi) == pytest.approx(0.75)  # closed interval: boundaries count
    assert mean_width(lo, hi) == pytest.approx(1.125)
    with pytest.raises(ValueError):
        wape(np.zeros(3), np.ones(3))
    with pytest.raises(ValueError):
        mae(y, y, np.zeros(4, dtype=bool))
    with pytest.raises(ValueError):
        seasonal_naive_scale(np.ones(10), 3)  # zero scale

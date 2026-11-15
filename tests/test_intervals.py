import math

import numpy as np
import pytest

from demandcast.intervals import conformal_quantile, cqr_interval, split_conformal_interval
from demandcast.metrics import coverage


def test_conformal_quantile_known_answers():
    scores = np.arange(1.0, 20.0)  # n = 19
    assert conformal_quantile(scores, 0.1) == 18.0  # ceil(20 * 0.9) = 18th smallest
    assert conformal_quantile(scores, 0.5) == 10.0  # ceil(20 * 0.5) = 10th smallest
    assert conformal_quantile(scores[:5], 0.1) == np.inf  # ceil(6 * 0.9) = 6 > 5
    with pytest.raises(ValueError):
        conformal_quantile(scores, 1.0)


@pytest.mark.parametrize("alpha", [0.2, 0.05])
def test_split_conformal_matches_exact_coverage_on_iid_data(alpha):
    """For exchangeable data the expected coverage is exactly ceil((n+1)(1-a))/(n+1)."""
    rng = np.random.default_rng(0)
    n_cal, n_test, reps = 30, 200, 3000
    expected = math.ceil((n_cal + 1) * (1 - alpha)) / (n_cal + 1)
    covs = []
    for _ in range(reps):
        x_cal, x_test = rng.uniform(0, 10, n_cal), rng.uniform(0, 10, n_test)
        y_cal = 2 * x_cal + rng.standard_t(3, n_cal)
        y_test = 2 * x_test + rng.standard_t(3, n_test)
        lo, hi = split_conformal_interval(
            y_cal, 2 * x_cal, 2 * x_test, alpha, log_scale=False, clip_zero=False
        )
        covs.append(coverage(y_test, lo, hi))
    covs = np.asarray(covs)
    se = covs.std(ddof=1) / np.sqrt(reps)
    assert abs(covs.mean() - expected) < 4 * se
    assert covs.mean() >= 1 - alpha - 4 * se


@pytest.mark.parametrize("factor", [0.4, 2.5])  # quantile model too narrow / too wide
def test_cqr_reaches_nominal_coverage_from_a_miscalibrated_quantile_model(factor):
    """Heteroscedastic i.i.d. data; quantile bounds deliberately mis-scaled by `factor`."""
    rng = np.random.default_rng(1)
    alpha, n_cal, n_test, reps = 0.1, 200, 500, 400
    z = 1.6449  # standard normal 95th percentile -> central 90% band
    raw_cov, cqr_cov, raw_w, cqr_w = [], [], [], []
    for _ in range(reps):
        x_cal, x_test = rng.uniform(1, 5, n_cal), rng.uniform(1, 5, n_test)
        y_cal = x_cal + x_cal * rng.normal(size=n_cal)
        y_test = x_test + x_test * rng.normal(size=n_test)
        band = lambda x: (x - factor * z * x, x + factor * z * x)  # noqa: E731
        (c_lo, c_hi), (t_lo, t_hi) = band(x_cal), band(x_test)
        lo, hi = cqr_interval(y_cal, c_lo, c_hi, t_lo, t_hi, alpha,
                              log_scale=False, clip_zero=False)
        raw_cov.append(coverage(y_test, t_lo, t_hi))
        cqr_cov.append(coverage(y_test, lo, hi))
        raw_w.append(np.mean(t_hi - t_lo))
        cqr_w.append(np.mean(hi - lo))
    assert abs(np.mean(raw_cov) - (1 - alpha)) > 0.05  # the raw bands are clearly off
    expected = math.ceil((n_cal + 1) * (1 - alpha)) / (n_cal + 1)
    assert np.mean(cqr_cov) == pytest.approx(expected, abs=0.006)
    if factor > 1:
        assert np.mean(cqr_w) < np.mean(raw_w)  # over-wide bands get narrowed


def test_log_scale_intervals_are_multiplicative_and_non_negative():
    cal_y = np.array([10.0, 20.0, 40.0, 80.0, 160.0] * 10)
    cal_pred = cal_y * np.exp(np.tile([0.1, -0.1, 0.2, -0.2, 0.0], 10))
    test_pred = np.array([0.0, 5.0, 500.0])
    lo, hi = split_conformal_interval(cal_y, cal_pred, test_pred, alpha=0.1)
    assert (lo >= 0).all() and (lo <= test_pred).all() and (hi >= test_pred).all()
    # same relative half-width on the log1p scale for every level
    half = np.log1p(hi) - np.log1p(test_pred)
    np.testing.assert_allclose(half, half[0])
    lo2, hi2 = cqr_interval(cal_y, cal_pred, cal_pred, test_pred, test_pred, alpha=0.1)
    assert (lo2 >= 0).all() and (hi2 >= lo2).all()

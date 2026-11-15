import numpy as np
import pandas as pd
import pytest

from demandcast.backtest import RollingOriginSplitter, make_origins

INDEX = pd.date_range("2011-01-01", periods=10 * 168, freq="h")


def test_make_origins_spacing_and_midnight():
    origins = make_origins("2011-02-01", 3, 30)
    assert origins == [pd.Timestamp("2011-02-01"), pd.Timestamp("2011-03-03"),
                       pd.Timestamp("2011-04-02")]
    with pytest.raises(ValueError):
        make_origins("2011-02-01 05:00", 3, 7)


@pytest.mark.parametrize("gap", [0, 5, 24])
@pytest.mark.parametrize("calibration", [0, 168])
def test_splitter_never_trains_at_or_after_origin(gap, calibration):
    origins = [INDEX[5 * 168], INDEX[6 * 168 + 24], INDEX[8 * 168]]
    splitter = RollingOriginSplitter(
        origins, horizon=168, gap=gap, calibration=calibration, min_train=24
    )
    previous_train = None
    for s in splitter.split(INDEX):
        t = INDEX.get_loc(s.origin)
        # training data strictly before the origin, and the latency gap is excluded
        assert INDEX[s.train].max() < s.origin
        assert s.train.max() == t - gap - 1
        assert not np.isin(np.arange(t - gap, t + 168), s.train).any()
        # test window is exactly the next `horizon` hours
        np.testing.assert_array_equal(s.test, np.arange(t, t + 168))
        # calibration is the tail of the training window; proper train is the rest
        assert len(s.calibration) == calibration
        np.testing.assert_array_equal(np.concatenate([s.proper_train, s.calibration]), s.train)
        if calibration:
            assert s.calibration.max() == s.train.max()
        # expanding window: every later training set contains the earlier one
        if previous_train is not None:
            assert np.isin(previous_train, s.train).all()
        previous_train = s.train


def test_splitter_validates_inputs():
    irregular = INDEX.delete(100)
    with pytest.raises(ValueError, match="regular"):
        list(RollingOriginSplitter([INDEX[5 * 168]]).split(irregular))
    with pytest.raises(ValueError, match="not in the index"):
        list(RollingOriginSplitter([pd.Timestamp("2030-01-01")]).split(INDEX))
    with pytest.raises(ValueError, match="past the end"):
        list(RollingOriginSplitter([INDEX[-100]]).split(INDEX))
    with pytest.raises(ValueError, match="min_train"):
        list(RollingOriginSplitter([INDEX[200]], calibration=168).split(INDEX))

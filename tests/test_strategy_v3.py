from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from bt_helpers import EUR
from tradingbot.backtesting.costs import CostModel
from tradingbot.backtesting.engine import BacktestConfig, run_backtest
from tradingbot.config import load_config
from tradingbot.data.collectors import SyntheticCollector
from tradingbot.data.normalizer import normalize_ohlcv
from tradingbot.features import FeatureParams, compute_features
from tradingbot.signals import Action, Side
from tradingbot.strategies import VPABreakoutStrategy, get_strategy_def

T = pd.Timestamp("2024-03-04 10:00", tz="UTC")


@pytest.fixture
def strat():
    return VPABreakoutStrategy(load_config().strategy_v3)


def row(open_=100.0, close=100.0, hi=101.0, lo=99.0, vr=1.0, atr=1.0):
    return {
        "open": open_, "close": close, "range_high": hi, "range_low": lo,
        "volume_ratio": vr, "atr": atr,
    }  # fmt: skip


def ev(strat, cur, position=None):
    return strat.evaluate("EURUSD", "1h", T, row(), cur, position)


def test_long_on_high_volume_close_above_a_tight_range(strat):
    (sig,) = ev(strat, row(open_=100.5, close=102.0, vr=2.0))
    assert sig.action == Action.ENTER_LONG
    assert sig.stop_distance == pytest.approx(strat.params.atr_stop_mult * 1.0)


def test_short_on_high_volume_close_below_a_tight_range(strat):
    (sig,) = ev(strat, row(open_=99.5, close=98.0, vr=2.0))
    assert sig.action == Action.ENTER_SHORT


def test_every_condition_is_needed(strat):
    base = dict(open_=100.5, close=102.0, vr=2.0)
    assert ev(strat, row(**{**base, "vr": 1.0})) == []  # volume not high
    assert ev(strat, row(**{**base, "close": 101.0})) == []  # not above the range (on it)
    assert ev(strat, row(**{**base, "open_": 103.0})) == []  # down candle
    assert ev(strat, row(**base, hi=110.0, lo=90.0)) == []  # range too wide (20 ATR)


def test_exit_on_opposite_high_volume_candle_and_no_stacking(strat):
    (x,) = ev(strat, row(open_=100.0, close=99.0, vr=2.0), Side.LONG)
    assert x.action == Action.EXIT_LONG
    assert ev(strat, row(open_=100.0, close=99.0, vr=1.0), Side.LONG) == []
    (x,) = ev(strat, row(open_=100.0, close=101.0, vr=2.0), Side.SHORT)
    assert x.action == Action.EXIT_SHORT
    assert ev(strat, row(open_=100.5, close=102.0, vr=2.0), Side.LONG) == []


def test_missing_features_give_no_signal(strat):
    assert ev(strat, row(vr=float("nan"))) == []
    assert ev(strat, row(atr=0.0, vr=2.0, close=102.0)) == []


def test_range_excludes_the_current_candle():
    idx = pd.date_range("2024-01-01", periods=30, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0}, index=idx
    )
    df.loc[idx[25], "high"] = 150.0  # a spike on candle 25
    f = compute_features(df, FeatureParams(range_window=20))
    assert f["range_high"].iloc[25] == 101.0  # the spike is not in its own range
    assert f["range_high"].iloc[26] == 150.0  # it is in the next candle's range
    assert f["range_high"].iloc[:20].isna().all()


def test_registered_and_configured():
    sdef = get_strategy_def("v3")
    assert sdef.config_attr == "strategy_v3"
    assert FeatureParams.from_v3(load_config().strategy_v3).range_window == 20


def _damaged(df, i):
    bad = df.copy()
    close = bad.iloc[i, bad.columns.get_loc("close")] * 0.97
    bad.iloc[i, bad.columns.get_loc("close")] = close
    bad.iloc[i, bad.columns.get_loc("low")] = min(bad.iloc[i]["low"], close)
    return bad


def test_damaging_the_next_candle_never_changes_a_v3_decision_or_fill():
    raw = SyntheticCollector(seed=5).fetch(
        "EURUSD", "1h", datetime(2022, 1, 3, tzinfo=UTC), datetime(2023, 6, 2, tzinfo=UTC)
    )
    df = normalize_ohlcv(raw)[0]
    app = load_config()
    app = app.model_copy(
        update={"strategy_v3": app.strategy_v3.model_copy(update={"vol_mult": 1.2})}
    )
    fp = FeatureParams.from_v3(app.strategy_v3)
    strat = VPABreakoutStrategy(app.strategy_v3)
    cfg = BacktestConfig(costs=CostModel.from_config(app.costs))

    def go(d):
        return run_backtest(compute_features(d, fp), strat, EUR, cfg, "EURUSD", "1h")

    base = go(df)
    bars_ = [df.index.get_loc(t) for t in base.signals["bar_time"]]
    assert len(bars_) >= 4, "synthetic data produced too few V3 signals to test"
    assert np.isfinite(len(bars_))
    for b in bars_[1:6]:
        dmg = go(_damaged(df, b + 1))
        pd.testing.assert_frame_equal(
            base.signals[base.signals["bar_time"] <= df.index[b]].reset_index(drop=True),
            dmg.signals[dmg.signals["bar_time"] <= df.index[b]].reset_index(drop=True),
        )

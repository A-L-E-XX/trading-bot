from datetime import UTC, datetime

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
from tradingbot.strategies import MeanReversionBBStrategy, get_strategy_def

T = pd.Timestamp("2024-03-04 10:00", tz="UTC")


@pytest.fixture
def strat():
    return MeanReversionBBStrategy(load_config().strategy_v2)


def row(close=100.0, sma=100.0, std=1.0, rsi=50.0, atr=2.0):
    return {"close": close, "sma": sma, "bb_std": std, "rsi": rsi, "atr": atr}


def ev(strat, cur, position=None):
    return strat.evaluate("EURUSD", "1h", T, row(), cur, position)


def test_long_when_below_lower_band_and_oversold(strat):
    (sig,) = ev(strat, row(close=97.0, rsi=25.0))  # lower band = 100 - 2*1 = 98
    assert sig.action == Action.ENTER_LONG
    assert sig.stop_distance == pytest.approx(strat.params.atr_stop_mult * 2.0)


def test_short_when_above_upper_band_and_overbought(strat):
    (sig,) = ev(strat, row(close=103.0, rsi=75.0))
    assert sig.action == Action.ENTER_SHORT


def test_needs_both_band_and_rsi(strat):
    assert ev(strat, row(close=97.0, rsi=45.0)) == []  # outside the band, RSI not oversold
    assert ev(strat, row(close=99.0, rsi=20.0)) == []  # oversold, but inside the band
    assert ev(strat, row(close=98.0, rsi=20.0)) == []  # exactly on the band is not outside


def test_exits_at_the_mean_and_never_enters_while_in_a_trade(strat):
    (x,) = ev(strat, row(close=100.0), Side.LONG)
    assert x.action == Action.EXIT_LONG and x.stop_distance is None
    assert ev(strat, row(close=99.0), Side.LONG) == []  # not back at the mean yet
    (x,) = ev(strat, row(close=100.0), Side.SHORT)
    assert x.action == Action.EXIT_SHORT
    assert ev(strat, row(close=97.0, rsi=10.0), Side.LONG) == []  # no stacking


def test_no_short_when_disallowed():
    params = load_config().strategy_v2.model_copy(update={"allow_short": False})
    assert ev(MeanReversionBBStrategy(params), row(close=103.0, rsi=75.0)) == []


@pytest.mark.parametrize("bad", ["sma", "bb_std", "rsi", "atr"])
def test_missing_indicator_means_no_signal(strat, bad):
    cur = row(close=97.0, rsi=25.0)
    cur[bad] = float("nan")
    assert ev(strat, cur) == []


def test_zero_volatility_means_no_signal(strat):
    assert ev(strat, row(close=97.0, rsi=25.0, std=0.0)) == []
    assert ev(strat, row(close=97.0, rsi=25.0, atr=0.0)) == []


def test_registry_builds_matching_features_and_strategy():
    app = load_config()
    d = get_strategy_def("v2")
    assert d.strategy_cls is MeanReversionBBStrategy
    fp = d.feature_params(d.params(app))
    assert (
        fp.sma_period == app.strategy_v2.bb_period and fp.rsi_period == app.strategy_v2.rsi_period
    )
    with pytest.raises(ValueError, match="unknown strategy"):
        get_strategy_def("v9")


def test_bb_std_is_causal(make_ohlcv):
    df = make_ohlcv("EURUSD", "1h", seed=2)
    full = compute_features(df)
    cut = compute_features(df.iloc[:400])
    pd.testing.assert_series_equal(full["bb_std"].iloc[:400], cut["bb_std"])


def _damaged(df, i):
    bad = df.copy()
    close = bad.iloc[i, bad.columns.get_loc("close")] * 0.97
    bad.iloc[i, bad.columns.get_loc("close")] = close
    bad.iloc[i, bad.columns.get_loc("low")] = min(bad.iloc[i]["low"], close)
    return bad


def test_damaging_the_next_candle_never_changes_a_v2_decision_or_fill():
    raw = SyntheticCollector(seed=5).fetch(
        "EURUSD", "1h", datetime(2022, 1, 3, tzinfo=UTC), datetime(2023, 1, 2, tzinfo=UTC)
    )
    df = normalize_ohlcv(raw)[0]
    app = load_config()
    fp = FeatureParams.from_v2(app.strategy_v2)
    strat = MeanReversionBBStrategy(app.strategy_v2)
    cfg = BacktestConfig(costs=CostModel.from_config(app.costs))

    def go(d):
        return run_backtest(compute_features(d, fp), strat, EUR, cfg, "EURUSD", "1h")

    base = go(df)
    bars_ = [df.index.get_loc(t) for t in base.signals["bar_time"]]
    assert len(bars_) >= 6
    for b in bars_[2:8]:
        dmg = go(_damaged(df, b + 1))
        decided = base.signals["bar_time"] <= df.index[b]
        pd.testing.assert_frame_equal(
            base.signals[decided].reset_index(drop=True),
            dmg.signals[dmg.signals["bar_time"] <= df.index[b]].reset_index(drop=True),
        )
        t_open = df.index[b + 1]
        cols = ["side", "entry_time", "entry_price"]
        pd.testing.assert_frame_equal(
            base.trades[base.trades["entry_time"] <= t_open][cols].reset_index(drop=True),
            dmg.trades[dmg.trades["entry_time"] <= t_open][cols].reset_index(drop=True),
        )

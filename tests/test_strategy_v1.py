import pandas as pd
import pytest

from tradingbot.config import load_config
from tradingbot.features import FeatureParams, compute_features
from tradingbot.signals import Action, Side, Signal
from tradingbot.strategies import EmaTrendAtrStrategy

T = pd.Timestamp("2024-01-02 10:00", tz="UTC")


@pytest.fixture
def strat():
    return EmaTrendAtrStrategy(load_config().strategy_v1)


def row(fast, slow, close=110.0, trend=100.0, atr=2.0, ready=True):
    return {
        "close": close,
        "ema_fast": fast,
        "ema_slow": slow,
        "ema_trend": trend,
        "atr": atr,
        "ready": ready,
    }


def up_cross():
    return row(9.0, 10.0), row(11.0, 10.0)


def down_cross():
    return row(11.0, 10.0), row(9.0, 10.0)


def test_signal_requires_stop_distance_for_entries():
    with pytest.raises(ValueError):
        Signal("EURUSD", "1h", T, Action.ENTER_LONG, "s", "1", "r", 1.0, None)
    Signal("EURUSD", "1h", T, Action.EXIT_LONG, "s", "1", "r", 1.0, None)  # exits need none


def test_long_entry_on_up_cross_above_trend(strat):
    prev, cur = up_cross()
    cur["close"], cur["atr"] = 105.0, 2.5
    (sig,) = strat.evaluate("EURUSD", "1h", T, prev, cur, None)
    assert sig.action == Action.ENTER_LONG
    assert sig.stop_distance == pytest.approx(2.0 * 2.5)  # atr_stop_mult 2.0
    assert sig.strategy == "ema_trend_atr" and sig.strategy_version == "1.0.0"
    assert sig.reference_price == 105.0 and sig.bar_time == T and sig.reason


def test_short_entry_on_down_cross_below_trend(strat):
    prev, cur = down_cross()
    cur["close"] = 95.0
    (sig,) = strat.evaluate("EURUSD", "1h", T, prev, cur, None)
    assert sig.action == Action.ENTER_SHORT


def test_trend_filter_blocks_entries(strat):
    prev, cur = up_cross()
    cur["close"] = 95.0  # below trend EMA: no long
    assert strat.evaluate("EURUSD", "1h", T, prev, cur, None) == []
    prev, cur = down_cross()
    cur["close"] = 110.0  # above trend EMA: no short
    assert strat.evaluate("EURUSD", "1h", T, prev, cur, None) == []


def test_shorts_can_be_disabled():
    params = load_config().strategy_v1.model_copy(update={"allow_short": False})
    prev, cur = down_cross()
    cur["close"] = 95.0
    assert EmaTrendAtrStrategy(params).evaluate("EURUSD", "1h", T, prev, cur, None) == []


def test_exit_on_opposite_cross_and_reversal_when_trend_allows(strat):
    prev, cur = down_cross()
    cur["close"] = 110.0  # above trend: exit the long, no short
    out = strat.evaluate("EURUSD", "1h", T, prev, cur, Side.LONG)
    assert [s.action for s in out] == [Action.EXIT_LONG]
    cur["close"] = 95.0  # below trend: exit the long AND reverse to short
    out = strat.evaluate("EURUSD", "1h", T, prev, cur, Side.LONG)
    assert [s.action for s in out] == [Action.EXIT_LONG, Action.ENTER_SHORT]
    prev, cur = up_cross()
    out = strat.evaluate("EURUSD", "1h", T, prev, cur, Side.SHORT)
    assert [s.action for s in out] == [Action.EXIT_SHORT, Action.ENTER_LONG]


def test_no_second_entry_while_in_position(strat):
    prev, cur = up_cross()
    assert strat.evaluate("EURUSD", "1h", T, prev, cur, Side.LONG) == []


def test_no_signals_during_warmup_or_without_atr(strat):
    prev, cur = up_cross()
    assert strat.evaluate("EURUSD", "1h", T, prev, {**cur, "ready": False}, None) == []
    assert strat.evaluate("EURUSD", "1h", T, {**prev, "ready": False}, cur, None) == []
    assert strat.evaluate("EURUSD", "1h", T, prev, {**cur, "atr": float("nan")}, None) == []
    assert strat.evaluate("EURUSD", "1h", T, prev, {**cur, "atr": 0.0}, None) == []


def test_equal_emas_are_not_a_cross(strat):
    # prev fast == slow then fast > slow counts as an up-cross exactly once
    assert strat.evaluate("EURUSD", "1h", T, row(10.0, 10.0), row(11.0, 10.0), None)
    assert strat.evaluate("EURUSD", "1h", T, row(11.0, 10.0), row(12.0, 10.0), None) == []


# ------------------------------------------------------------------ data -> strategy -> signals


def test_signals_from_data_without_any_broker(make_ohlcv, strat):
    """M5 acceptance: data -> features -> signals, reproducibly, with no broker."""
    from datetime import UTC, datetime

    df = make_ohlcv(
        "EURUSD", "1h", start=datetime(2022, 1, 3, tzinfo=UTC), end=datetime(2023, 1, 2, tzinfo=UTC)
    )
    feats = compute_features(df, FeatureParams.from_config(load_config().strategy_v1))
    a = strat.generate_signals(feats, "EURUSD", "1h")
    b = strat.generate_signals(feats.copy(), "EURUSD", "1h")
    assert a == b and len(a) > 5

    warmup_end = feats.index[FeatureParams().strategy_warmup_rows]
    assert all(s.bar_time >= warmup_end for s in a)  # nothing before EMA200 exists
    position, entries = None, [s for s in a if s.action.is_entry]
    assert entries and all(s.stop_distance > 0 for s in entries)
    for s in a:  # entries/exits must alternate consistently
        if s.action == Action.ENTER_LONG:
            assert position is None
            position = Side.LONG
        elif s.action == Action.ENTER_SHORT:
            assert position is None
            position = Side.SHORT
        elif s.action == Action.EXIT_LONG:
            assert position == Side.LONG
            position = None
        else:
            assert position == Side.SHORT
            position = None


def test_signals_do_not_depend_on_future_bars(make_ohlcv, strat):
    feats = compute_features(make_ohlcv("BTCUSD", "1h"))
    full = strat.generate_signals(feats, "BTCUSD", "1h")
    cut = 900
    partial = strat.generate_signals(feats.iloc[:cut], "BTCUSD", "1h")
    assert partial == [s for s in full if s.bar_time <= feats.index[cut - 1]]


def test_generate_signals_checks_columns(strat, make_ohlcv):
    with pytest.raises(ValueError, match="missing columns"):
        strat.generate_signals(make_ohlcv("EURUSD", "1h"), "EURUSD", "1h")


def test_damaging_the_next_candle_never_changes_a_signal(make_ohlcv, strat):
    from datetime import UTC, datetime

    df = make_ohlcv(
        "EURUSD", "1h", start=datetime(2022, 1, 3, tzinfo=UTC), end=datetime(2023, 1, 2, tzinfo=UTC)
    )
    params = FeatureParams.from_config(load_config().strategy_v1)
    base = strat.generate_signals(compute_features(df, params), "EURUSD", "1h")
    assert len(base) >= 6
    for sig in base[2:8]:
        b = df.index.get_loc(sig.bar_time)
        bad = df.copy()
        bad.iloc[b + 1, bad.columns.get_loc("close")] *= 0.97
        bad.iloc[b + 1, bad.columns.get_loc("low")] = min(
            bad.iloc[b + 1]["low"], bad.iloc[b + 1]["close"]
        )
        again = strat.generate_signals(compute_features(bad, params), "EURUSD", "1h")
        assert [s for s in again if s.bar_time <= sig.bar_time] == [
            s for s in base if s.bar_time <= sig.bar_time
        ]

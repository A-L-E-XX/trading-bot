from datetime import UTC, datetime

import pandas as pd
import pytest

from bt_helpers import EUR, JPY, XAU, Scripted, bars, cfg
from tradingbot.backtesting.costs import CostModel
from tradingbot.backtesting.engine import BacktestConfig, run_backtest
from tradingbot.config import load_config
from tradingbot.features import FeatureParams, compute_features
from tradingbot.signals import Action as A
from tradingbot.strategies import EmaTrendAtrStrategy


def run(df, script, spec=EUR, config=None):
    return run_backtest(df, Scripted(script), spec, config or cfg(), spec.symbol, "1h")


# ------------------------------------------------------------------ timing: no look-ahead


def test_entry_fills_at_next_candle_open_not_signal_price():
    df = bars([1.10, 1.11, 1.12, 1.13, 1.14])
    r = run(df, {1: [(A.ENTER_LONG, 0.01)], 3: [(A.EXIT_LONG, None)]})
    (t,) = r.trades.to_dict("records")
    assert t["entry_time"] == df.index[2] and t["entry_price"] == pytest.approx(1.12)
    assert t["exit_time"] == df.index[4] and t["exit_price"] == pytest.approx(1.14)
    assert t["pnl_usd"] == pytest.approx(0.02 * 1000)  # 0.01 lot = 1,000 units
    assert t["bars_held"] == 2 and t["exit_reason"] == "signal"


def test_signal_on_last_candle_cannot_execute():
    df = bars([1.10, 1.10, 1.10, 1.10])
    r = run(df, {3: [(A.ENTER_LONG, 0.01)]})
    assert r.trades.empty


def test_no_signal_before_second_candle():
    df = bars([1.10, 1.10, 1.10])
    r = run(df, {0: [(A.ENTER_LONG, 0.01)]})
    assert r.trades.empty


# ------------------------------------------------------------------ costs and P&L


def test_long_trade_pays_spread_and_slippage():
    costs = CostModel(spread_multiplier=1.0, slippage_spread_fraction=0.5)  # 0.0001 + 0.00005
    df = bars([1.1000, 1.1000, 1.1000, 1.1020, 1.1050])
    r = run(df, {1: [(A.ENTER_LONG, 0.01)], 3: [(A.EXIT_LONG, None)]}, config=cfg(costs))
    t = r.trades.iloc[0]
    assert t["entry_price"] == pytest.approx(1.1000 + 0.0001 + 0.00005)  # buys at the ask + slip
    assert t["exit_price"] == pytest.approx(1.1050 - 0.00005)  # sells at the bid - slip
    assert t["pnl_usd"] == pytest.approx(4.80)
    assert t["cost_usd"] == pytest.approx(0.20)  # one spread + two slips on 1,000 units
    assert t["pnl_usd"] + t["cost_usd"] == pytest.approx(5.00)  # frictionless result


def test_short_trade_pays_spread_and_slippage():
    costs = CostModel(spread_multiplier=1.0, slippage_spread_fraction=0.5)
    df = bars([1.1000, 1.1000, 1.1000, 1.0980, 1.0950])
    r = run(df, {1: [(A.ENTER_SHORT, 0.01)], 3: [(A.EXIT_SHORT, None)]}, config=cfg(costs))
    t = r.trades.iloc[0]
    assert t["entry_price"] == pytest.approx(1.1000 - 0.00005)  # sells at the bid - slip
    assert t["exit_price"] == pytest.approx(1.0950 + 0.0001 + 0.00005)  # buys back at ask + slip
    assert t["pnl_usd"] == pytest.approx(4.80)


def test_commission_is_charged_each_side():
    costs = CostModel(0.0, 0.0, commission_per_lot=7.0)  # 0.01 lot -> $0.07 per side
    df = bars([1.10, 1.10, 1.10, 1.10, 1.10])
    r = run(df, {1: [(A.ENTER_LONG, 0.01)], 3: [(A.EXIT_LONG, None)]}, config=cfg(costs))
    assert r.trades.iloc[0]["pnl_usd"] == pytest.approx(-0.14)


def test_usdjpy_profit_is_converted_from_yen():
    df = bars([150.0, 150.0, 150.0, 151.0, 151.0])
    r = run(df, {1: [(A.ENTER_LONG, 1.0)], 3: [(A.EXIT_LONG, None)]}, JPY)
    assert r.trades.iloc[0]["pnl_usd"] == pytest.approx(1000 / 151.0)  # 1,000 JPY at 151


def test_gold_lot_is_one_ounce():
    df = bars([2000.0, 2000.0, 2000.0, 2010.0, 2010.0])
    r = run(df, {1: [(A.ENTER_LONG, 20.0)], 3: [(A.EXIT_LONG, None)]}, XAU)
    assert r.trades.iloc[0]["pnl_usd"] == pytest.approx(10.0)  # $10 move x 1 oz


# ------------------------------------------------------------------ stops


def test_long_stop_hit_intrabar_fills_at_stop():
    df = bars([1.10] * 6, lows=[1.10, 1.10, 1.10, 1.095, 1.10, 1.10])
    r = run(df, {1: [(A.ENTER_LONG, 0.003)]})
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "stop" and t["exit_time"] == df.index[3]
    assert t["stop_price"] == pytest.approx(1.097) and t["exit_price"] == pytest.approx(1.097)
    assert t["pnl_usd"] == pytest.approx(-3.0) and t["r_multiple"] == pytest.approx(-1.0)
    assert t["bars_held"] == 2


def test_gap_through_stop_fills_at_the_open():
    df = bars([1.10, 1.10, 1.10, 1.090, 1.090], lows=[1.10, 1.10, 1.10, 1.088, 1.088])
    r = run(df, {1: [(A.ENTER_LONG, 0.003)]})
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "stop" and t["exit_price"] == pytest.approx(1.090)  # worse than stop
    assert t["pnl_usd"] == pytest.approx(-10.0)


def test_short_stop_is_triggered_by_the_ask_price():
    costs = CostModel(spread_multiplier=1.0, slippage_spread_fraction=0.0)  # spread 0.0001
    opens = [1.10] * 5
    miss = run(bars(opens, highs=[1.10, 1.10, 1.10, 1.1018, 1.10]), {1: [(A.ENTER_SHORT, 0.002)]},
               config=cfg(costs))  # fmt: skip
    assert miss.trades.iloc[0]["exit_reason"] == "end_of_data"  # ask high 1.1019 < stop 1.1020
    hit = run(bars(opens, highs=[1.10, 1.10, 1.10, 1.10195, 1.10]), {1: [(A.ENTER_SHORT, 0.002)]},
              config=cfg(costs))  # fmt: skip
    t = hit.trades.iloc[0]
    assert t["exit_reason"] == "stop" and t["exit_price"] == pytest.approx(1.1020)
    assert t["pnl_usd"] == pytest.approx(-2.0)


def test_stop_can_trigger_on_the_entry_candle():
    df = bars([1.10] * 5, lows=[1.10, 1.10, 1.095, 1.10, 1.10])
    r = run(df, {1: [(A.ENTER_LONG, 0.003)]})
    t = r.trades.iloc[0]
    assert t["entry_time"] == t["exit_time"] == df.index[2] and t["bars_held"] == 1


# ------------------------------------------------------------------ equity, positions, ordering


def test_equity_marks_open_trades_to_market():
    df = bars(
        [1.10] * 5, highs=[1.10, 1.10, 1.12, 1.15, 1.15], closes=[1.10, 1.10, 1.12, 1.15, 1.15]
    )
    r = run(df, {1: [(A.ENTER_LONG, 0.05)]})
    assert list(r.equity.iloc[:4]) == pytest.approx([100.0, 100.0, 120.0, 150.0])
    assert r.trades.iloc[0]["exit_reason"] == "end_of_data"
    assert (
        r.equity.iloc[-1]
        == pytest.approx(150.0)
        == pytest.approx(r.trades.iloc[0]["balance_after"])
    )


def test_short_is_marked_at_the_ask():
    costs = CostModel(1.0, 0.0)  # spread 0.0001
    df = bars([1.10] * 4, closes=[1.10, 1.10, 1.08, 1.08], lows=[1.10, 1.10, 1.08, 1.08])
    r = run(df, {1: [(A.ENTER_SHORT, 0.05)]}, config=cfg(costs))
    assert r.equity.iloc[2] == pytest.approx(100 + (1.10 - 1.0801) * 1000)


def test_flat_account_final_equity_equals_start_plus_pnl():
    df = bars([1.10, 1.10, 1.10, 1.11, 1.11, 1.11, 1.10, 1.10])
    r = run(df, {1: [(A.ENTER_LONG, 0.05)], 3: [(A.EXIT_LONG, None)], 5: [(A.ENTER_SHORT, 0.05)],
                 6: [(A.EXIT_SHORT, None)]})  # fmt: skip
    assert r.equity.iloc[-1] == pytest.approx(100.0 + r.trades["pnl_usd"].sum())
    assert len(r.equity) == len(df)


def test_exit_then_reverse_at_the_same_open():
    df = bars([1.10] * 7)
    script = {1: [(A.ENTER_LONG, 0.01)], 3: [(A.ENTER_SHORT, 0.01), (A.EXIT_LONG, None)]}
    r = run(df, script)  # entry listed first, but the exit must execute first
    first, second = r.trades.to_dict("records")
    assert first["side"] == "long" and first["exit_time"] == df.index[4]
    assert second["side"] == "short" and second["entry_time"] == df.index[4]


def test_entry_while_in_position_is_ignored():
    df = bars([1.10] * 6)
    r = run(df, {1: [(A.ENTER_LONG, 0.01)], 2: [(A.ENTER_LONG, 0.01)]})
    assert len(r.trades) == 1 and r.ignored_entry_signals == 1


def test_account_blown_halts_the_backtest():
    df = bars([1.10, 1.10, 1.10, 1.00, 1.00, 1.00], lows=[1.10, 1.10, 1.10, 1.00, 1.00, 1.00])
    r = run(df, {1: [(A.ENTER_LONG, 1.0)]}, config=cfg(lots=50.0))
    assert r.halted_reason == "account_blown" and r.trades.iloc[0]["exit_reason"] == "account_blown"
    assert len(r.equity) == 4 and r.metrics["ruined"] is True


def test_no_trades_gives_flat_equity():
    df = bars([1.10] * 5)
    r = run(df, {})
    assert r.trades.empty and list(r.equity) == [100.0] * 5 and r.metrics["trades"] == 0


def test_risk_per_trade_is_reported_not_enforced():
    df = bars([1.10] * 6)
    r = run(df, {1: [(A.ENTER_LONG, 0.01)]})  # a 100-pip stop on 1,000 units is $10 = 10% of $100
    t = r.trades.iloc[0]
    assert t["stop_risk_usd"] == pytest.approx(10.0) and t[
        "stop_risk_pct_of_balance"
    ] == pytest.approx(10.0)
    assert r.metrics["trades_over_risk_limit"] == 1  # above the 5% limit, but the trade still ran


def test_signal_log_is_recorded():
    df = bars([1.10] * 6)
    r = run(df, {1: [(A.ENTER_LONG, 0.01)], 3: [(A.EXIT_LONG, None)]})
    assert list(r.signals["action"]) == ["enter_long", "exit_long"]


# ------------------------------------------------------------------ input checks


def test_input_validation():
    df = bars([1.10] * 5)
    with pytest.raises(ValueError, match="missing columns"):
        run(df.drop(columns=["ready"]), {})
    with pytest.raises(ValueError, match="sorted"):
        run(df.iloc[::-1], {})
    with pytest.raises(ValueError, match="not allowed"):
        run(df, {}, config=cfg(lots=0.005))


# ------------------------------------------------------------------ the real strategy end to end


@pytest.fixture(scope="module")
def real_run_inputs():
    from tradingbot.data.collectors import SyntheticCollector
    from tradingbot.data.normalizer import normalize_ohlcv

    raw = SyntheticCollector(seed=5).fetch(
        "EURUSD", "1h", datetime(2022, 1, 3, tzinfo=UTC), datetime(2023, 1, 2, tzinfo=UTC)
    )
    df, _ = normalize_ohlcv(raw)
    app = load_config()
    feats = compute_features(df, FeatureParams.from_config(app.strategy_v1))
    config = BacktestConfig(costs=CostModel.from_config(app.costs))
    return feats, EmaTrendAtrStrategy(app.strategy_v1), config


def test_backtest_is_reproducible(real_run_inputs):
    feats, strat, config = real_run_inputs
    a = run_backtest(feats, strat, EUR, config, "EURUSD", "1h")
    b = run_backtest(feats.copy(), strat, EUR, config, "EURUSD", "1h")
    assert len(a.trades) > 5
    pd.testing.assert_frame_equal(a.trades, b.trades)
    pd.testing.assert_series_equal(a.equity, b.equity)
    assert a.metrics == b.metrics


def test_backtest_has_no_look_ahead(real_run_inputs):
    """Trades that finished before the cut must be identical whether or not later data exists."""
    feats, strat, config = real_run_inputs
    full = run_backtest(feats, strat, EUR, config, "EURUSD", "1h")
    cut = 4000
    part = run_backtest(feats.iloc[:cut], strat, EUR, config, "EURUSD", "1h")
    last = feats.index[cut - 1]

    def finished(res, upto=None):
        t = res.trades[res.trades["exit_reason"] != "end_of_data"]
        if upto is not None:
            t = t[t["exit_time"] <= upto]
        return t.reset_index(drop=True)

    assert len(finished(part)) >= 3
    pd.testing.assert_frame_equal(finished(part), finished(full, last))
    pd.testing.assert_series_equal(part.equity.iloc[: cut - 10], full.equity.iloc[: cut - 10])


def test_future_price_changes_do_not_change_past_results(real_run_inputs):
    feats, strat, config = real_run_inputs
    base = run_backtest(feats, strat, EUR, config, "EURUSD", "1h")
    edited = feats.copy()
    cols = ["open", "high", "low", "close"]
    edited.iloc[5000:, [edited.columns.get_loc(c) for c in cols]] *= 1.2
    after = run_backtest(edited, strat, EUR, config, "EURUSD", "1h")
    n = 4990
    pd.testing.assert_series_equal(base.equity.iloc[:n], after.equity.iloc[:n])


# ------------------------------------------------------------------ one-candle look-ahead detector
# Truncating history cannot detect a strategy that peeks ONE candle ahead (the peeked candle is
# still present for every earlier decision). Instead, damage the candle after a decision and make
# sure the decision, and the fill at that candle's open, do not change.


def _damaged(df: pd.DataFrame, i: int) -> pd.DataFrame:
    bad = df.copy()
    close = bad.iloc[i, bad.columns.get_loc("close")] * 0.97
    bad.iloc[i, bad.columns.get_loc("close")] = close
    bad.iloc[i, bad.columns.get_loc("low")] = min(bad.iloc[i]["low"], close)
    return bad


@pytest.fixture(scope="module")
def raw_eurusd():
    from tradingbot.data.collectors import SyntheticCollector
    from tradingbot.data.normalizer import normalize_ohlcv

    raw = SyntheticCollector(seed=5).fetch(
        "EURUSD", "1h", datetime(2022, 1, 3, tzinfo=UTC), datetime(2023, 1, 2, tzinfo=UTC)
    )
    return normalize_ohlcv(raw)[0]


def test_damaging_the_next_candle_never_changes_a_decision_or_its_fill(raw_eurusd):
    app = load_config()
    params = FeatureParams.from_config(app.strategy_v1)
    strat = EmaTrendAtrStrategy(app.strategy_v1)
    config = BacktestConfig(costs=CostModel.from_config(app.costs))

    def go(df):
        return run_backtest(compute_features(df, params), strat, EUR, config, "EURUSD", "1h")

    base = go(raw_eurusd)
    sig_bars = [raw_eurusd.index.get_loc(t) for t in base.signals["bar_time"]]
    assert len(sig_bars) >= 6
    for b in sig_bars[2:8]:  # candle b closes -> signal; candle b + 1 is the one we damage
        dmg = go(_damaged(raw_eurusd, b + 1))
        decided = base.signals["bar_time"] <= raw_eurusd.index[b]
        pd.testing.assert_frame_equal(
            base.signals[decided].reset_index(drop=True),
            dmg.signals[dmg.signals["bar_time"] <= raw_eurusd.index[b]].reset_index(drop=True),
        )
        t_open = raw_eurusd.index[b + 1]  # fills at that candle's open cannot see its range
        cols = ["side", "entry_time", "entry_price"]
        pd.testing.assert_frame_equal(
            base.trades[base.trades["entry_time"] <= t_open][cols].reset_index(drop=True),
            dmg.trades[dmg.trades["entry_time"] <= t_open][cols].reset_index(drop=True),
        )

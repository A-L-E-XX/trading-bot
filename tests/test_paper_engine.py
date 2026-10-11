import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from tradingbot.backtesting.costs import CostModel
from tradingbot.backtesting.engine import BacktestConfig, run_backtest
from tradingbot.backtesting.instruments import load_specs
from tradingbot.backtesting.runner import DEFAULT_SPECS_PATH
from tradingbot.config import load_config
from tradingbot.data.collectors import SyntheticCollector
from tradingbot.data.normalizer import normalize_ohlcv
from tradingbot.execution import PaperAccount, PaperEngine
from tradingbot.features import compute_features
from tradingbot.risk import RiskLimits
from tradingbot.strategies import get_strategy_def

APP = load_config()
SPECS = load_specs(DEFAULT_SPECS_PATH)
COSTS = CostModel.from_config(APP.costs)
LIMITS = RiskLimits.from_config(APP.risk)
COLS = ["side", "entry_time", "entry_price", "exit_time", "exit_price", "lots", "pnl_usd",
        "exit_reason"]  # fmt: skip


def data(symbol="EURUSD", tf="1h", bars=700, seed=5):
    raw = SyntheticCollector(seed=seed).fetch(
        symbol, tf, datetime(2022, 1, 3, tzinfo=UTC), datetime(2023, 6, 1, tzinfo=UTC)
    )
    return normalize_ohlcv(raw)[0].iloc[:bars]


def engine(df_symbol="EURUSD", strategy="v1", account=None, tf="1h", **kw):
    sdef = get_strategy_def(strategy)
    return PaperEngine(
        sdef, sdef.params(APP), SPECS[df_symbol], COSTS,
        account or PaperAccount(100.0, 0.01), LIMITS, df_symbol, tf,
        feature_window=kw.pop("feature_window", None), **kw,
    )  # fmt: skip


def feed(eng, df):
    for t, r in df.iterrows():
        eng.on_bar(t, r.open, r.high, r.low, r.close, r.volume)


def backtest(df, symbol="EURUSD", strategy="v1", account=None):
    account = account or PaperAccount(100.0, 0.01)
    sdef = get_strategy_def(strategy)
    feats = compute_features(df, sdef.feature_params(sdef.params(APP)))
    cfg = BacktestConfig(
        initial_balance=account.initial_balance, lots=account.lots, costs=COSTS,
        lot_value_scale=account.lot_value_scale, risk_sizing=account.risk_sizing,
        max_lots=account.max_lots, risk_limits=LIMITS,
    )  # fmt: skip
    return run_backtest(
        feats, sdef.strategy_cls(sdef.params(APP)), SPECS[symbol], cfg, symbol, "1h"
    )


def as_frame(trades):
    out = pd.DataFrame(trades)[COLS].copy()
    for c in ("entry_time", "exit_time"):
        out[c] = pd.to_datetime(out[c], utc=True).astype("datetime64[ns, UTC]")
    return out.reset_index(drop=True)


@pytest.mark.parametrize(
    "strategy,account",
    [
        ("v1", PaperAccount(100.0, 0.01)),
        ("v2", PaperAccount(100.0, 0.01, lot_value_scale=0.01, risk_sizing=True)),
        ("v2", PaperAccount(20.0, 0.01, lot_value_scale=0.01, risk_sizing=True)),
    ],
)
def test_replay_reproduces_the_backtest_trade_for_trade(strategy, account):
    df = data(bars=700)
    eng = engine(strategy=strategy, account=account)
    feed(eng, df)
    bt = backtest(df, strategy=strategy, account=account)
    done = bt.trades[bt.trades["exit_reason"] != "end_of_data"]  # paper is still holding that one
    assert len(done) >= 3, "scenario too quiet to prove anything"
    pd.testing.assert_frame_equal(
        as_frame(eng.trades)[: len(done)], as_frame(done.to_dict("records")), check_exact=False
    )
    assert len(eng.trades) == len(done)


def test_halt_and_kill_switch_match_the_backtest():
    df = data(symbol="EURUSD", bars=500)
    acct = PaperAccount(100.0, 0.01)
    eng = engine(account=acct)
    feed(eng, df)
    bt = backtest(df, account=acct)
    assert eng.halted == bt.halted_reason


def test_duplicate_and_out_of_order_bars_are_ignored():
    df = data(bars=300)
    eng = engine()
    feed(eng, df)
    n_trades, n_events, last = len(eng.trades), len(eng.events), eng.last_bar_time
    r = df.iloc[100]
    out = eng.on_bar(df.index[100], r.open, r.high, r.low, r.close, r.volume)
    assert [e["event"] for e in out] == ["bar_ignored"]
    feed(eng, df)  # replaying the whole feed changes nothing but adds ignore events
    assert len(eng.trades) == n_trades and eng.last_bar_time == last
    assert {e["event"] for e in eng.events[n_events:]} == {"bar_ignored"}


@pytest.mark.parametrize(
    "bad",
    [
        dict(open=float("nan")),
        dict(close=-1.0),
        dict(high=0.5),  # high below the body
        dict(low=99.0),  # low above the body
    ],
)
def test_invalid_candle_is_refused_and_not_traded(bad):
    eng = engine()
    t = pd.Timestamp("2024-03-04 10:00", tz="UTC")
    bar = dict(open=1.10, high=1.11, low=1.09, close=1.10, volume=1.0) | bad
    out = eng.on_bar(t, **bar)
    assert [e["event"] for e in out] == ["bar_rejected"]
    assert eng.last_bar_time is None and eng.bars_seen == 0


def test_naive_timestamp_is_rejected():
    with pytest.raises(ValueError, match="timezone"):
        engine().on_bar(pd.Timestamp("2024-03-04 10:00"), 1.1, 1.11, 1.09, 1.1)


def test_restart_from_saved_state_continues_identically(tmp_path: Path):
    df = data(bars=700)
    straight = engine(account=PaperAccount(20.0, 0.01, 0.01, True), strategy="v2")
    feed(straight, df)

    state, log = tmp_path / "state.json", tmp_path / "events.jsonl"
    cut = 350
    a = engine(account=PaperAccount(20.0, 0.01, 0.01, True), strategy="v2",
               state_path=state, event_log_path=log)  # fmt: skip
    feed(a, df.iloc[:cut])
    del a  # "crash"
    b = engine(account=PaperAccount(20.0, 0.01, 0.01, True), strategy="v2", history=df.iloc[:cut],
               state_path=state, event_log_path=log)  # fmt: skip
    feed(b, df.iloc[cut:])
    pd.testing.assert_frame_equal(as_frame(b.trades), as_frame(straight.trades))
    assert b.balance == pytest.approx(straight.balance)
    kinds = [json.loads(line)["event"] for line in log.read_text().splitlines()]
    assert "position_opened" in kinds and "position_closed" in kinds


def test_state_for_another_symbol_or_stale_history_is_refused(tmp_path: Path):
    df = data(bars=300)
    state = tmp_path / "s.json"
    a = engine(state_path=state)
    feed(a, df)
    with pytest.raises(ValueError, match="does not match"):
        engine(df_symbol="GBPUSD", state_path=state, history=df)
    with pytest.raises(ValueError, match="supply history"):
        engine(state_path=state, history=df.iloc[:200])


def test_event_log_is_append_only_json_lines_with_every_decision(tmp_path: Path):
    df = data(bars=500)
    log = tmp_path / "events.jsonl"
    eng = engine(event_log_path=log)
    feed(eng, df)
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == len(eng.events)
    kinds = {e["event"] for e in lines}
    assert {"signal", "position_opened", "position_closed"} <= kinds
    assert all("bar_time" in e and "symbol" in e for e in lines)


def test_entries_rejected_by_the_risk_engine_are_logged_with_a_reason():
    # $20 normal account: any V2 stop is far too wide for a 3% budget, so nothing may open
    eng = engine(strategy="v2", account=PaperAccount(20.0, 0.01, 1.0, True))
    feed(eng, data(bars=500))
    rej = [e for e in eng.events if e["event"] == "order_rejected"]
    assert rej and all(e["reason"] == "stop_too_wide_for_account" for e in rej)
    assert not eng.trades


def test_spread_override_is_used_for_fills():
    df = data(bars=900)
    acct = PaperAccount(100.0, 0.01, lot_value_scale=0.01, risk_sizing=True)
    wide, normal = engine(strategy="v2", account=acct), engine(strategy="v2", account=acct)
    for t, r in df.iterrows():
        wide.on_bar(t, r.open, r.high, r.low, r.close, r.volume, spread_override=0.0020)
        normal.on_bar(t, r.open, r.high, r.low, r.close, r.volume)
    assert wide.trades and sum(t["cost_usd"] for t in wide.trades) > 3 * sum(
        t["cost_usd"] for t in normal.trades
    )

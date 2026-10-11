import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradingbot.backtesting.costs import CostModel
from tradingbot.backtesting.instruments import load_specs
from tradingbot.backtesting.runner import DEFAULT_SPECS_PATH
from tradingbot.config import load_config
from tradingbot.data.collectors import MT5Collector, MT5Error, SyntheticCollector
from tradingbot.data.normalizer import normalize_ohlcv
from tradingbot.execution import PaperAccount, PaperEngine
from tradingbot.execution.cli import main as paper_main
from tradingbot.execution.live import (
    FeedError,
    closed_candles,
    run_session,
    saved_last_bar,
    seed_history,
)
from tradingbot.risk import RiskLimits
from tradingbot.strategies import get_strategy_def

APP, SPECS = load_config(), load_specs(DEFAULT_SPECS_PATH)
OFFSET = 2  # the fake server runs at UTC+2
SRC = Path(__file__).resolve().parents[1] / "src" / "tradingbot"


class Tick:
    def __init__(self, bid, ask, t):
        self.bid, self.ask, self.time = bid, ask, t


class ScriptedMT5:
    """Minimal MetaTrader5 stand-in: serves candles up to ``visible`` (the last one is forming)."""

    TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1 = 16385, 16388, 16408
    ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_TRADE_MODE_REAL = 0, 2

    def __init__(self, df: pd.DataFrame, visible: int, fail_first=0, demo=True):
        self.df, self.visible, self.fail, self.demo = df, visible, fail_first, demo
        self.calls = 0

    def initialize(self, **kw):
        return True

    def shutdown(self):
        pass

    def last_error(self):
        return (-1, "scripted")

    def account_info(self):
        mode = self.ACCOUNT_TRADE_MODE_DEMO if self.demo else self.ACCOUNT_TRADE_MODE_REAL
        return type("A", (), {"trade_mode": mode, "login": 1, "server": "x", "currency": "USD"})()

    def symbol_select(self, name, enable):
        return True

    def symbol_info_tick(self, name):
        t = int((self.df.index[self.visible - 1] + pd.Timedelta(hours=OFFSET)).timestamp())
        return Tick(1.0998, 1.1000, t)

    def copy_rates_from_pos(self, symbol, tf, pos, count):
        self.calls += 1
        if self.calls <= self.fail:
            return None
        part = self.df.iloc[max(0, self.visible - count) : self.visible]
        secs = ((part.index + pd.Timedelta(hours=OFFSET)).astype("int64") // 10**9).to_numpy()
        out = np.zeros(
            len(part),
            dtype=[
                ("time", "i8"),
                ("open", "f8"),
                ("high", "f8"),
                ("low", "f8"),
                ("close", "f8"),
                ("tick_volume", "u8"),
            ],  # fmt: skip
        )
        out["time"] = secs
        for c in ("open", "high", "low", "close"):
            out[c] = part[c].to_numpy()
        out["tick_volume"] = part["volume"].to_numpy()
        return out


@pytest.fixture(scope="module")
def df():
    raw = SyntheticCollector(seed=5).fetch(
        "EURUSD", "1h", datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 3, 1, tzinfo=UTC)
    )
    return normalize_ohlcv(raw)[0].iloc[:600]


def collector(mt5):
    return MT5Collector(mt5_module=mt5, server_utc_offset_hours=OFFSET)


def make_engine(history=None, **kw):
    sdef = get_strategy_def("v2")
    return PaperEngine(
        sdef, sdef.params(APP), SPECS["EURUSD"], CostModel.from_config(APP.costs),
        PaperAccount(100.0, 0.01, 0.01, True), RiskLimits.from_config(APP.risk), "EURUSD", "1h",
        history=history, feature_window=500, **kw,
    )  # fmt: skip


def now_after(df, i):
    return lambda: df.index[i] + pd.Timedelta(minutes=1)  # shortly after candle i opened


# ---------------------------------------------------------------- the collector's live methods
def test_fetch_recent_converts_server_time_to_utc_and_keeps_the_forming_candle(df):
    col = collector(ScriptedMT5(df, visible=300))
    rates = col.fetch_recent("EURUSD", "1h", 50)
    assert len(rates) == 50 and rates["time"].iloc[-1] == df.index[299]
    assert str(rates["time"].dt.tz) == "UTC"
    assert rates["volume"].iloc[0] == df["volume"].iloc[250]


def test_quote_returns_bid_ask_spread_and_utc_time(df):
    q = collector(ScriptedMT5(df, visible=300)).quote("EURUSD")
    assert q["bid"] == 1.0998 and q["ask"] == 1.1000 and q["spread"] == pytest.approx(0.0002)
    assert q["time"] == df.index[299]


def test_live_methods_refuse_real_accounts(df):
    with pytest.raises(MT5Error, match="NOT a demo"):
        collector(ScriptedMT5(df, visible=300, demo=False)).fetch_recent("EURUSD", "1h", 10)


def test_empty_rates_raise_a_clear_error(df):
    with pytest.raises(MT5Error, match="no recent"):
        collector(ScriptedMT5(df, visible=300, fail_first=1)).fetch_recent("EURUSD", "1h", 10)


# ---------------------------------------------------------------- closed candles only
def test_closed_candles_drops_the_forming_candle(df):
    rates = collector(ScriptedMT5(df, visible=300)).fetch_recent("EURUSD", "1h", 100)
    out = closed_candles(rates, "1h", df.index[299] + pd.Timedelta(minutes=5))
    assert out.index[-1] == df.index[298] and len(out) == 99


def test_a_wrong_server_offset_is_detected():
    t = pd.date_range("2024-03-04 08:00", periods=5, freq="1h", tz="UTC")
    rates = pd.DataFrame(
        {"time": t, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}
    )
    # "now" is 10:30 but the second-to-last candle opened at 11:00: impossible, offset is wrong
    with pytest.raises(FeedError, match="server-utc-offset"):
        closed_candles(rates, "1h", pd.Timestamp("2024-03-04 10:30", tz="UTC"))


def test_duplicate_or_unsorted_candles_are_refused():
    t = pd.DatetimeIndex(["2024-03-04 08:00", "2024-03-04 08:00", "2024-03-04 09:00"], tz="UTC")
    rates = pd.DataFrame(
        {"time": t, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}
    )
    with pytest.raises(FeedError, match="strictly increasing"):
        closed_candles(rates, "1h", pd.Timestamp("2024-03-05", tz="UTC"))


# ---------------------------------------------------------------- the session loop
def test_session_catches_up_oldest_first_and_never_processes_the_forming_candle(df):
    mt5 = ScriptedMT5(df, visible=400)  # candle 399 is still forming
    eng = make_engine(history=df.iloc[:380])  # engine is 20 candles behind
    res = run_session(
        collector(mt5), eng, max_polls=1, now=now_after(df, 399), sleep=lambda s: None
    )
    assert res.processed == 19 and eng.last_bar_time == df.index[398]
    assert eng.bars_seen == 19


def test_session_picks_up_new_candles_as_time_passes(df):
    mt5 = ScriptedMT5(df, visible=400)
    eng = make_engine(history=df.iloc[:399])
    clock = {"i": 399}

    def sleep(_):  # a new candle closes while we wait
        mt5.visible += 1
        clock["i"] += 1

    def clock_now():
        return df.index[clock["i"]] + pd.Timedelta(minutes=1)

    res = run_session(collector(mt5), eng, max_polls=4, now=clock_now, sleep=sleep)
    assert res.processed == 3 and eng.last_bar_time == df.index[401]


def test_restarting_the_session_does_not_double_count(df):
    mt5 = ScriptedMT5(df, visible=450)
    eng = make_engine(history=df.iloc[:400])
    run_session(collector(mt5), eng, max_polls=1, now=now_after(df, 449), sleep=lambda s: None)
    n_trades, balance = len(eng.trades), eng.balance
    res = run_session(
        collector(mt5), eng, max_polls=1, now=now_after(df, 449), sleep=lambda s: None
    )
    assert res.processed == 0 and len(eng.trades) == n_trades and eng.balance == balance


def test_transient_feed_errors_are_logged_and_recovered(df):
    mt5 = ScriptedMT5(df, visible=410, fail_first=2)  # first two requests return nothing
    eng = make_engine(history=df.iloc[:400])
    sleeps = []
    res = run_session(collector(mt5), eng, max_polls=3, now=now_after(df, 409), sleep=sleeps.append)
    assert res.errors == 2 and res.processed == 9
    errs = [e for e in eng.events if e["event"] == "feed_error"]
    assert [e["consecutive"] for e in errs] == [1, 2]
    assert sleeps[1] > sleeps[0]  # backs off


def test_too_many_errors_in_a_row_stops_the_session(df):
    mt5 = ScriptedMT5(df, visible=410, fail_first=99)
    res = run_session(collector(mt5), make_engine(history=df.iloc[:400]), max_consecutive_errors=3,
                      now=now_after(df, 409), sleep=lambda s: None)  # fmt: skip
    assert "3 feed errors in a row" in res.stopped_because and res.errors == 3


def test_wrong_offset_stops_instead_of_trading_on_bad_times(df):
    mt5 = ScriptedMT5(df, visible=410)
    eng = make_engine(history=df.iloc[:400])
    res = run_session(collector(mt5), eng, max_consecutive_errors=2,
                      now=now_after(df, 380), sleep=lambda s: None)  # fmt: skip
    assert res.processed == 0 and eng.last_bar_time == df.index[399]


def test_spread_check_event_compares_live_to_model_spread(df):
    mt5 = ScriptedMT5(df, visible=410)
    eng = make_engine(history=df.iloc[:400])
    run_session(collector(mt5), eng, max_polls=1, now=now_after(df, 409), sleep=lambda s: None)
    (chk,) = [e for e in eng.events if e["event"] == "spread_check"]
    assert chk["live_spread"] == pytest.approx(0.0002) and chk["ratio"] > 0


# ---------------------------------------------------------------- history seeding / resume
def test_seed_history_merges_and_recent_wins_on_overlap(df):
    stored, recent = df.iloc[:300].copy(), df.iloc[250:350].copy()
    recent.loc[recent.index[0], "close"] = 9.99
    out = seed_history(stored, recent)
    assert len(out) == 350 and out.index.is_monotonic_increasing and not out.index.has_duplicates
    assert out.loc[df.index[250], "close"] == 9.99
    assert seed_history(stored, recent, up_to=df.index[320]).index[-1] == df.index[320]
    with pytest.raises(FeedError):
        seed_history(None, pd.DataFrame())


def test_saved_last_bar_reads_the_state_file(tmp_path, df):
    state = tmp_path / "s.json"
    assert saved_last_bar(state) is None
    eng = make_engine(history=df.iloc[:300], state_path=state)
    eng.on_bar(df.index[300], *df.iloc[300][["open", "high", "low", "close", "volume"]])
    assert saved_last_bar(state) == df.index[300]


# ---------------------------------------------------------------- safety + CLI
def test_there_is_no_code_that_can_send_orders_to_the_broker():
    forbidden = ("order_send", "order_check", "order_calc", "positions_close", "Buy(", "Sell(")
    for rel in ("data/collectors/mt5.py", "execution/live.py", "execution/paper.py",
                "execution/cli.py"):  # fmt: skip
        text = (SRC / rel).read_text(encoding="utf-8")
        assert not [w for w in forbidden if w in text], rel


def test_cli_status_reads_a_saved_session(tmp_path, df, capsys):
    sd = tmp_path / "sess"
    eng = make_engine(history=df.iloc[:300], state_path=sd / "state.json")
    eng.on_bar(df.index[300], *df.iloc[300][["open", "high", "low", "close", "volume"]])
    assert paper_main(["status", "--name", "sess", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "EURUSD 1h" in out and "balance $100.00" in out and "open position: none" in out
    assert paper_main(["status", "--name", "nope", "--out-dir", str(tmp_path)]) == 1


def test_cli_run_explains_when_metatrader5_is_missing(monkeypatch, capsys, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "MetaTrader5", None)
    code = paper_main(["run", "--strategy", "v2", "--symbol", "EURUSD", "--timeframe", "1h",
                       "--out-dir", str(tmp_path), "--once"])  # fmt: skip
    assert code == 1 and "uv sync --extra mt5" in capsys.readouterr().err


def test_event_log_lines_are_valid_json(tmp_path, df):
    log = tmp_path / "e.jsonl"
    eng = make_engine(history=df.iloc[:300], event_log_path=log)
    run_session(collector(ScriptedMT5(df, visible=330)), eng, max_polls=1,
                now=now_after(df, 329), sleep=lambda s: None)  # fmt: skip
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert rows and all("event" in r for r in rows)

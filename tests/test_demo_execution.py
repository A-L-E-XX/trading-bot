from types import SimpleNamespace as NS

import pandas as pd
import pytest

from tradingbot.data.collectors.mt5 import MT5Collector, MT5Error
from tradingbot.execution.broker import DemoBroker, OrderRefused
from tradingbot.execution.cli import main as paper_main
from tradingbot.execution.demo import DemoExecutor
from tradingbot.signals import Action

MAGIC = 770001


class FakeTradingMT5:
    """MetaTrader5 stand-in that records every order. Loss at stop = lots * distance * 100."""

    ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_TRADE_MODE_REAL = 0, 2
    TRADE_ACTION_DEAL, ORDER_TYPE_BUY, ORDER_TYPE_SELL = 1, 0, 1
    POSITION_TYPE_BUY, POSITION_TYPE_SELL = 0, 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
    TRADE_RETCODE_DONE = 10009

    def __init__(self, demo=True, equity=100.0, retcode=10009):
        self.demo, self.equity, self.retcode = demo, equity, retcode
        self.sent: list[dict] = []
        self.open_positions: list[NS] = []
        self.deals: dict[int, list[NS]] = {}
        self._next = 1000

    def initialize(self, **kw):
        return True

    def shutdown(self):
        pass

    def last_error(self):
        return (-1, "fake")

    def account_info(self):
        mode = self.ACCOUNT_TRADE_MODE_DEMO if self.demo else self.ACCOUNT_TRADE_MODE_REAL
        return NS(trade_mode=mode, equity=self.equity, balance=self.equity, currency="USD")

    def symbol_info(self, name):
        return NS(volume_min=0.01, volume_step=0.01, volume_max=50.0, digits=2, filling_mode=2)

    def symbol_info_tick(self, name):
        return NS(bid=100.0, ask=100.1, time=0)

    def order_calc_profit(self, order_type, symbol, lots, price, sl):
        return -abs(price - sl) * lots * 100

    def order_send(self, request):
        self.sent.append(request)
        if self.retcode != self.TRADE_RETCODE_DONE:
            return NS(retcode=self.retcode, comment="rejected", order=0, price=0.0)
        if "position" in request:
            self.open_positions = [
                p for p in self.open_positions if p.ticket != request["position"]
            ]
            return NS(retcode=self.retcode, order=request["position"], price=request["price"])
        self._next += 1
        typ = self.POSITION_TYPE_BUY if request["type"] == self.ORDER_TYPE_BUY else 1
        self.open_positions.append(
            NS(ticket=self._next, magic=request["magic"], type=typ, volume=request["volume"],
               price_open=request["price"], sl=request["sl"], profit=0.0)
        )  # fmt: skip
        return NS(retcode=self.retcode, order=self._next, price=request["price"])

    def positions_get(self, symbol=None):
        return tuple(self.open_positions)

    def history_deals_get(self, position=None):
        return self.deals.get(position)


def make_broker(mt5=None, **kw):
    mt5 = mt5 or FakeTradingMT5()
    col = MT5Collector(symbol_suffix="m", mt5_module=mt5)
    col.connect() if mt5.demo else setattr(col, "_connected", True)
    return DemoBroker(col, "USTEC", magic=MAGIC, **kw), mt5


# ------------------------------------------------------------------------------ the broker
def test_refuses_to_trade_on_a_non_demo_account_and_sends_nothing():
    broker, mt5 = make_broker(FakeTradingMT5(demo=False))
    with pytest.raises(OrderRefused, match="NOT a demo"):
        broker.open("long", 0.05, 2.0)
    with pytest.raises(OrderRefused, match="NOT a demo"):
        broker.close(NS(ticket=1, side="long", lots=0.01))
    assert mt5.sent == []


def test_demo_check_is_repeated_before_every_order():
    broker, mt5 = make_broker()
    broker.open("long", 0.05, 2.0)
    mt5.demo = False  # the terminal is switched to a real account mid-session
    with pytest.raises(OrderRefused, match="NOT a demo"):
        broker.open("long", 0.05, 2.0)
    assert len(mt5.sent) == 1


@pytest.mark.parametrize("dist", [0, -1.0, None, float("nan")])
def test_never_sends_an_order_without_a_valid_stop(dist):
    broker, mt5 = make_broker()
    with pytest.raises(OrderRefused):
        broker.open("long", 0.05, dist)
    assert mt5.sent == []


def test_order_carries_stop_magic_and_a_step_rounded_volume():
    broker, mt5 = make_broker(max_risk_pct=100.0)
    res = broker.open("long", 0.057, 2.0)
    (req,) = mt5.sent
    assert req["sl"] == pytest.approx(100.1 - 2.0) and req["magic"] == MAGIC
    assert req["volume"] == pytest.approx(0.05) and res["lots"] == pytest.approx(0.05)
    broker.open("short", 0.02, 2.0)
    assert mt5.sent[1]["sl"] == pytest.approx(100.0 + 2.0)


def test_lots_are_scaled_down_to_the_real_risk_limit():
    broker, mt5 = make_broker(max_risk_pct=3.0)  # equity 100 -> limit $3 ; loss = lots*dist*100
    res = broker.open("long", 1.0, 2.0)  # 1.0 lot would lose $200
    assert res["lots"] == pytest.approx(0.01) and res["risk_at_stop"] == pytest.approx(2.0)
    assert res["risk_at_stop"] <= 3.0


def test_minimum_lot_too_risky_means_no_order():
    broker, mt5 = make_broker(max_risk_pct=1.0)  # limit $1 ; min lot at dist 2 loses $2
    with pytest.raises(OrderRefused, match="minimum lot"):
        broker.open("long", 0.5, 2.0)
    assert mt5.sent == []


def test_hard_lot_cap():
    broker, mt5 = make_broker(max_lots=0.03, max_risk_pct=100.0)
    assert broker.open("long", 5.0, 0.5)["lots"] == pytest.approx(0.03)


def test_broker_rejection_is_reported_not_swallowed():
    broker, _ = make_broker(FakeTradingMT5(retcode=10006))
    with pytest.raises(OrderRefused, match="10006"):
        broker.open("long", 0.01, 2.0)


def test_only_positions_with_our_magic_are_seen_and_closed():
    broker, mt5 = make_broker()
    broker.open("long", 0.01, 2.0)
    mt5.open_positions.append(
        NS(ticket=5, magic=1, type=0, volume=1.0, price_open=1, sl=0, profit=0)
    )  # someone else's trade
    assert [p.ticket for p in broker.positions()] == [1001]
    assert len(broker.close_all()) == 1
    assert [p.ticket for p in mt5.open_positions] == [5]


# ------------------------------------------------------------------------------ the executor
T0 = pd.Timestamp("2025-01-01 00:00", tz="UTC")


def make_exec(tmp_path, mt5=None):
    broker, mt5 = make_broker(mt5)
    notes = []
    risk = NS(
        evaluate_entry=lambda **kw: NS(
            approved=True, lots=0.05, risk_usd=0.6, reason=NS(value="ok"), detail=""
        )
    )
    eng = NS(
        halted=None, pending=[], pos=None, risk=risk, spec=None, symbol="USTEC",
        account=NS(risk_sizing=True, lots=0.01, max_lots=None), last_bar_time=T0,
        note=lambda kind, t, **f: notes.append((kind, f)),
    )  # fmt: skip
    ex = DemoExecutor(broker, eng, tmp_path / "demo_state.json")
    return ex, eng, mt5, notes


def enter(t=T0, long=True):
    return NS(
        action=Action.ENTER_LONG if long else Action.ENTER_SHORT,
        bar_time=t, stop_distance=2.0, reason="test",
    )  # fmt: skip


def kinds(notes):
    return [k for k, _ in notes]


def test_pending_entry_opens_one_position_and_is_never_sent_twice(tmp_path):
    ex, eng, mt5, notes = make_exec(tmp_path)
    eng.pending = [enter()]
    ex.sync()
    ex.sync()
    assert len(mt5.sent) == 1 and "demo_order_sent" in kinds(notes)
    # the paper engine now fills it a candle later; still exactly one broker position
    eng.pending, eng.pos = [], NS(entry_time="x")
    ex.sync()
    assert len(mt5.sent) == 1 and len(mt5.open_positions) == 1


def test_restart_does_not_resend_the_same_signal(tmp_path):
    ex, eng, mt5, _ = make_exec(tmp_path)
    eng.pending = [enter()]
    ex.sync()
    mt5.open_positions.clear()  # broker lost it (e.g. stop hit) while we were down
    ex2 = DemoExecutor(ex.broker, eng, tmp_path / "demo_state.json")
    ex2.sync()
    assert len(mt5.sent) == 1


def test_failed_order_is_logged_and_not_retried(tmp_path):
    ex, eng, mt5, notes = make_exec(tmp_path, FakeTradingMT5(retcode=10006))
    eng.pending = [enter()]
    ex.sync()
    ex.sync()
    assert kinds(notes).count("demo_order_failed") == 1 and len(mt5.sent) == 1


def test_exit_signal_closes_the_position_with_its_result(tmp_path):
    ex, eng, mt5, notes = make_exec(tmp_path)
    eng.pending = [enter()]
    ex.sync()
    ticket = ex.ticket
    mt5.deals[ticket] = [NS(profit=1.5, commission=-0.1, swap=0.0, fee=0.0)]
    eng.pending, eng.pos = [NS(action=Action.EXIT_LONG, bar_time=T0)], NS(entry_time="x")
    ex.sync()
    assert mt5.open_positions == [] and ex.ticket is None
    closed = dict(notes)["demo_position_closed"]
    assert closed["broker_pnl"] == pytest.approx(1.4)


def test_halted_paper_engine_flattens_and_blocks_new_entries(tmp_path):
    ex, eng, mt5, notes = make_exec(tmp_path)
    eng.pending = [enter()]
    ex.sync()
    eng.halted = "kill_switch"
    ex.sync()
    assert mt5.open_positions == []
    eng.pending = [enter(T0 + pd.Timedelta(hours=4))]
    ex.sync()
    assert mt5.open_positions == [] and len(mt5.sent) == 2  # open + close only


def test_unknown_position_with_our_magic_is_closed_as_an_orphan(tmp_path):
    ex, eng, mt5, notes = make_exec(tmp_path)
    mt5.open_positions.append(
        NS(ticket=77, magic=MAGIC, type=0, volume=0.1, price_open=100, sl=90, profit=0)
    )
    ex.sync()
    assert mt5.open_positions == [] and "demo_orphan_closed" in kinds(notes)


def test_broker_flat_while_paper_open_is_logged_once_and_never_reentered(tmp_path):
    ex, eng, mt5, notes = make_exec(tmp_path)
    eng.pending = [enter()]
    ex.sync()
    mt5.open_positions.clear()  # stop-loss hit at the broker
    eng.pending, eng.pos = [], NS(entry_time="e1")
    ex.sync()
    ex.sync()
    assert "demo_position_closed" in kinds(notes)
    assert kinds(notes).count("demo_mismatch") == 1 and len(mt5.sent) == 1


def test_risk_engine_rejection_means_no_order(tmp_path):
    ex, eng, mt5, notes = make_exec(tmp_path)
    eng.risk = NS(
        evaluate_entry=lambda **kw: NS(
            approved=False, lots=0, risk_usd=0, reason=NS(value="max_open_positions"), detail=""
        )
    )
    eng.pending = [enter()]
    ex.sync()
    assert mt5.sent == [] and "demo_order_skipped" in kinds(notes)


# ------------------------------------------------------------------------------ guard rails
def test_only_the_broker_module_can_send_orders():
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "tradingbot"
    forbidden = ("order_send", "order_check", "positions_close", "Buy(", "Sell(")
    for path in src.rglob("*.py"):
        if path.name == "broker.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert not [w for w in forbidden if w in text], path.name
    text = (src / "execution" / "broker.py").read_text(encoding="utf-8")
    for fn in ("def open(", "def close("):
        body = text[text.index(fn) :]
        assert body.index("_require_demo()") < body.index("order_send")


def test_demo_orders_are_off_unless_asked(monkeypatch, tmp_path, capsys):
    import sys

    monkeypatch.setitem(sys.modules, "MetaTrader5", None)
    paper_main(["run", "--strategy", "v2", "--symbol", "EURUSD", "--timeframe", "1h", "--once",
                "--out-dir", str(tmp_path)])  # fmt: skip
    assert "DEMO ORDERS ON" not in capsys.readouterr().out


def test_flatten_command_reports_a_missing_terminal(monkeypatch, capsys):
    import sys

    monkeypatch.setitem(sys.modules, "MetaTrader5", None)
    assert paper_main(["flatten", "--symbol", "USTEC"]) == 1
    assert "MetaTrader5" in capsys.readouterr().err


def test_collector_error_type_is_shared():
    assert issubclass(OrderRefused, MT5Error)

"""Mirrors the paper engine's decisions onto an MT5 DEMO account, with reconciliation.

The paper engine stays the brain (strategy, risk engine, kill switch). This class only turns its
intent into broker orders, one candle EARLIER than the paper fill (the paper engine fills at the
next candle's open, which it can only see one candle later; here we act as soon as the signal
candle has closed). Differences between the two are logged, because they measure real execution.

Rules:
- intent = the paper position (or a pending entry) minus any pending exit;
- a pending entry is sent once (its signal time is remembered, also across restarts);
- if the paper engine is halted (kill switch etc.) every position of ours is closed;
- a position at the broker that we do not know about (same magic) is closed and reported;
- if the broker is flat while the paper engine holds a position (stop hit at the broker, a failed
  order) we log it and do NOT re-enter.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from tradingbot.data.collectors.mt5 import MT5Error
from tradingbot.execution.broker import DemoBroker, OrderRefused
from tradingbot.execution.paper import PaperEngine
from tradingbot.signals import Action


class DemoExecutor:
    def __init__(self, broker: DemoBroker, engine: PaperEngine, state_path: Path | str):
        self.broker = broker
        self.engine = engine
        self.path = Path(state_path)
        self.ticket: int | None = None
        self.sent_signal: str | None = None  # bar_time of the last entry signal we acted on
        if self.path.exists():
            st = json.loads(self.path.read_text(encoding="utf-8"))
            self.ticket, self.sent_signal = st.get("ticket"), st.get("sent_signal")

    # -------------------------------------------------------------------------- persistence
    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"ticket": self.ticket, "sent_signal": self.sent_signal}), encoding="utf-8"
        )
        os.replace(tmp, self.path)

    def _note(self, kind: str, **fields) -> None:
        self.engine.note(kind, self.engine.last_bar_time, **fields)

    # -------------------------------------------------------------------------- the sync step
    def sync(self) -> None:
        """Bring the broker in line with the paper engine. Call after every poll."""
        eng = self.engine
        ours = {p.ticket: p for p in self.broker.positions()}

        # 1. a position we were tracking has disappeared at the broker (stop hit, manual close)
        if self.ticket is not None and self.ticket not in ours:
            pnl = self.broker.closed_pnl(self.ticket)
            self._note("demo_position_closed", ticket=self.ticket, broker_pnl=pnl)
            self.ticket = None
            self._save()

        # 2. positions with our magic that we do not know about: close them
        for t, p in list(ours.items()):
            if t != self.ticket:
                self._note("demo_orphan_closed", ticket=t, side=p.side, lots=p.lots)
                self.broker.close(p, "tb orphan")
                ours.pop(t)

        have = ours.get(self.ticket) if self.ticket is not None else None

        # 3. desired state
        if eng.halted:
            if have:
                self._close(have, f"paper engine halted: {eng.halted}")
            return
        pending = list(eng.pending)
        exit_pending = any(s.action in (Action.EXIT_LONG, Action.EXIT_SHORT) for s in pending)
        enter = next(
            (s for s in pending if s.action in (Action.ENTER_LONG, Action.ENTER_SHORT)), None
        )
        paper_open = eng.pos is not None

        if have and (exit_pending or (not paper_open and enter is None)):
            self._close(have, "exit signal" if exit_pending else "paper position closed")
        elif not have and paper_open and enter is None and self.ticket is None:
            # broker is flat but paper holds a trade: log once per paper trade, never re-enter
            key = f"flat:{eng.pos.entry_time}"
            if self.sent_signal != key:
                self.sent_signal = key
                self._note("demo_mismatch", detail="broker flat while paper position is open")
                self._save()
        elif not have and not paper_open and enter is not None:
            key = str(enter.bar_time)
            if self.sent_signal != key:
                self.sent_signal = key  # remembered even if the order fails: never retry blindly
                self._save()
                self._open(enter)

    # -------------------------------------------------------------------------- actions
    def _open(self, sig) -> None:
        eng = self.engine
        side = "long" if sig.action == Action.ENTER_LONG else "short"
        assert eng.risk is not None
        tick = self.broker._tick()
        price = float(tick.ask if side == "long" else tick.bid)
        dec = eng.risk.evaluate_entry(
            symbol=eng.symbol,
            stop_distance=float(sig.stop_distance),
            price=price,
            spec=eng.spec,
            requested_lots=None if eng.account.risk_sizing else eng.account.lots,
            max_lots=eng.account.max_lots,
        )
        if not dec.approved:
            self._note("demo_order_skipped", reason=dec.reason.value, detail=dec.detail)
            return
        try:
            res = self.broker.open(side, dec.lots, float(sig.stop_distance), comment=sig.reason)
        except (OrderRefused, MT5Error) as exc:
            self._note("demo_order_failed", side=side, lots=dec.lots, error=str(exc))
            return
        self.ticket = res["ticket"]
        self._save()
        self._note(
            "demo_order_sent", paper_lots=dec.lots, paper_risk_usd=dec.risk_usd,
            signal_time=str(sig.bar_time), **res,
        )  # fmt: skip

    def _close(self, position, why: str) -> None:
        try:
            res = self.broker.close(position, "tb " + why)
        except (OrderRefused, MT5Error) as exc:
            self._note("demo_close_failed", ticket=position.ticket, error=str(exc))
            return
        pnl = self.broker.closed_pnl(position.ticket)
        self._note("demo_position_closed", reason=why, broker_pnl=pnl, **res)
        self.ticket = None
        self._save()

    def flatten(self) -> list[dict]:
        """Close everything of ours (also used by the panic command)."""
        done = self.broker.close_all()
        self.ticket = None
        self._save()
        return done

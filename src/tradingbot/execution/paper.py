"""Paper trading engine (Milestone 9): one strategy on one symbol/timeframe, fake money.

Feed it CLOSED candles one at a time with ``on_bar``. It then does, in this exact order (the same
order and fill model as the backtester, so a replay of history reproduces the backtest):

1. Execute the orders decided at the previous close, at THIS candle's open (exits first).
2. Check the stop against this candle's high/low (a gap through it fills at the open).
3. Mark equity, feed the risk engine, and flatten + halt if the kill switch trips.
4. Ask the strategy for signals from the candle that just closed; they wait for the next candle.

Every decision is written to an append-only JSON-lines event log. State is saved atomically after
every candle, so a restart carries on exactly where it stopped; a candle that was already processed
is ignored (idempotent), so replaying a feed twice is harmless. It never talks to a broker.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tradingbot.backtesting.costs import CostModel
from tradingbot.backtesting.instruments import InstrumentSpec
from tradingbot.features import compute_features
from tradingbot.risk import OpenPosition, RiskEngine, RiskLimits
from tradingbot.signals import Action, Side, Signal
from tradingbot.strategies import StrategyDef

OHLCV = ["open", "high", "low", "close", "volume"]
STATE_VERSION = 1


@dataclass(frozen=True)
class PaperAccount:
    initial_balance: float
    lots: float = 0.01  # used when risk_sizing is False
    lot_value_scale: float = 1.0
    risk_sizing: bool = False
    max_lots: float | None = None


@dataclass
class _Pos:
    side: str
    entry_time: str
    entry_price: float
    stop_price: float
    stop_distance: float
    entry_reason: str
    lots: float
    units: float
    risk_usd: float
    bars: int = 0


class PaperEngine:
    def __init__(
        self,
        sdef: StrategyDef,
        strategy_params,
        spec: InstrumentSpec,
        costs: CostModel,
        account: PaperAccount,
        limits: RiskLimits,
        symbol: str,
        timeframe: str,
        history: pd.DataFrame | None = None,
        state_path: Path | str | None = None,
        event_log_path: Path | str | None = None,
        feature_window: int | None = 1500,
    ):
        self.sdef, self.params = sdef, strategy_params
        self.strategy = sdef.strategy_cls(strategy_params)
        self.feature_params = sdef.feature_params(strategy_params)
        self.spec, self.costs, self.account, self.limits = spec, costs, account, limits
        self.symbol, self.timeframe = symbol, timeframe
        self.state_path = Path(state_path) if state_path else None
        self.event_log_path = Path(event_log_path) if event_log_path else None
        self.feature_window = feature_window
        self.per_lot_units = spec.contract_size * account.lot_value_scale
        self.events: list[dict] = []
        self.trades: list[dict] = []
        self.equity_curve: list[tuple[str, float]] = []

        self.balance = account.initial_balance
        self.pos: _Pos | None = None
        self.pending: list[Signal] = []
        self.halted: str | None = None
        self.last_bar_time: pd.Timestamp | None = None
        self.bars_seen = 0
        self.buffer = pd.DataFrame(columns=OHLCV, dtype="float64")
        self.risk: RiskEngine | None = None

        if history is not None and len(history):
            self.buffer = history[OHLCV].astype("float64").copy()
            self.last_bar_time = history.index[-1]
        if self.state_path and self.state_path.exists():
            self._restore(json.loads(self.state_path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------------ logging / state
    def _log(self, kind: str, bar_time, **fields) -> dict:
        ev = {"event": kind, "bar_time": str(bar_time), "symbol": self.symbol, **fields}
        self.events.append(ev)
        if self.event_log_path:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.event_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ev, default=str) + "\n")
        return ev

    def note(self, kind: str, bar_time, **fields) -> dict:
        """Write an extra event (used by the live feed for spread checks and feed errors)."""
        return self._log(kind, bar_time, **fields)

    def _signal_dict(self, s: Signal) -> dict:
        return {
            "bar_time": str(s.bar_time),
            "action": s.action.value,
            "reason": s.reason,
            "reference_price": s.reference_price,
            "stop_distance": s.stop_distance,
        }

    def _save(self) -> None:
        if not self.state_path:
            return
        r = self.risk
        state = {
            "version": STATE_VERSION,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "balance": self.balance,
            "halted": self.halted,
            "last_bar_time": None if self.last_bar_time is None else str(self.last_bar_time),
            "bars_seen": self.bars_seen,
            "pos": None if self.pos is None else self.pos.__dict__,
            "pending": [self._signal_dict(s) for s in self.pending],
            "trades": self.trades,
            "risk": None
            if r is None
            else {
                "equity": r.state.equity,
                "peak_equity": r.state.peak_equity,
                "day": str(r.state.day),
                "day_start_equity": r.state.day_start_equity,
                "kill_switch": r.state.kill_switch,
                "kill_detail": r.state.kill_detail,
                "daily_halt": r.state.daily_halt,
                "positions": {k: v.__dict__ for k, v in r.state.positions.items()},
            },
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, self.state_path)  # atomic: a crash never leaves a half-written state

    def _restore(self, st: dict) -> None:
        if st.get("version") != STATE_VERSION or (st["symbol"], st["timeframe"]) != (
            self.symbol,
            self.timeframe,
        ):
            raise ValueError("state file does not match this engine (version/symbol/timeframe)")
        saved_last = pd.Timestamp(st["last_bar_time"]) if st["last_bar_time"] else None
        if saved_last is not None and (
            self.last_bar_time is None or self.last_bar_time != saved_last
        ):
            raise ValueError(
                f"history ends at {self.last_bar_time} but the saved state ends at {saved_last}: "
                "supply history up to the last processed candle"
            )
        self.balance, self.halted = st["balance"], st["halted"]
        self.last_bar_time, self.bars_seen = saved_last, st["bars_seen"]
        self.pos = _Pos(**st["pos"]) if st["pos"] else None
        self.trades = st["trades"]
        self.pending = [
            Signal(
                self.symbol,
                self.timeframe,
                pd.Timestamp(d["bar_time"]),
                Action(d["action"]),
                self.strategy.name,
                self.strategy.version,
                d["reason"],
                d["reference_price"],
                d["stop_distance"],
            )
            for d in st["pending"]
        ]
        if st["risk"]:
            r = st["risk"]
            self.risk = RiskEngine(
                self.limits, r["equity"], pd.Timestamp(r["day"]), self.account.lot_value_scale
            )
            s = self.risk.state
            s.equity, s.peak_equity = r["equity"], r["peak_equity"]
            s.day, s.day_start_equity = pd.Timestamp(r["day"]), r["day_start_equity"]
            s.kill_switch, s.kill_detail, s.daily_halt = (
                r["kill_switch"],
                r["kill_detail"],
                r["daily_halt"],
            )
            s.positions = {k: OpenPosition(**v) for k, v in r["positions"].items()}

    # ------------------------------------------------------------------------------ trading
    def _unrealized(self, close: float, spread: float) -> float:
        p = self.pos
        assert p is not None
        # a short is marked at the ask (close + spread)
        move = close - p.entry_price if p.side == "long" else p.entry_price - (close + spread)
        return move * p.units * self.spec.quote_to_usd(close)

    def _close(self, bar_time, exit_price: float, reason: str, spread: float, slip: float) -> None:
        p = self.pos
        assert p is not None
        move = exit_price - p.entry_price if p.side == "long" else p.entry_price - exit_price
        commission = 2 * self.costs.commission_per_lot * p.lots
        pnl = move * p.units * self.spec.quote_to_usd(exit_price) - commission
        cost = (spread + 2 * slip) * p.units * self.spec.quote_to_usd(p.entry_price) + commission
        self.balance += pnl
        trade = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": p.side,
            "entry_time": p.entry_time,
            "entry_price": p.entry_price,
            "exit_time": str(bar_time),
            "exit_price": exit_price,
            "lots": p.lots,
            "stop_price": p.stop_price,
            "stop_distance": p.stop_distance,
            "stop_risk_usd": p.risk_usd,
            "exit_reason": reason,
            "entry_reason": p.entry_reason,
            "pnl_usd": pnl,
            "cost_usd": cost,
            "bars_held": p.bars,
            "balance_after": self.balance,
        }
        self.trades.append(trade)
        self._log("position_closed", bar_time, **trade)
        assert self.risk is not None
        self.risk.register_close(self.symbol)
        self.pos = None

    def _open(self, bar_time, sig: Signal, o: float, spread: float, slip: float) -> None:
        assert self.risk is not None
        side = Side.LONG if sig.action == Action.ENTER_LONG else Side.SHORT
        fill = o + spread + slip if side == Side.LONG else o - slip
        dist = float(sig.stop_distance)
        dec = self.risk.evaluate_entry(
            symbol=self.symbol,
            stop_distance=dist,
            price=fill,
            spec=self.spec,
            requested_lots=None if self.account.risk_sizing else self.account.lots,
            max_lots=self.account.max_lots,
        )
        if not dec.approved:
            self._log("order_rejected", bar_time, action=sig.action.value, reason=dec.reason.value,
                      detail=dec.detail)  # fmt: skip
            return
        stop = fill - dist if side == Side.LONG else fill + dist
        self.pos = _Pos(side.value, str(bar_time), fill, stop, dist, sig.reason, dec.lots,
                        dec.lots * self.per_lot_units, dec.risk_usd)  # fmt: skip
        self.risk.register_open(self.symbol, dec.lots, dec.risk_usd)
        self._log("position_opened", bar_time, side=side.value, lots=dec.lots, fill=fill,
                  stop_price=stop, risk_usd=dec.risk_usd, risk_pct=dec.risk_pct,
                  reason=sig.reason)  # fmt: skip

    def on_bar(
        self,
        bar_time: pd.Timestamp,
        open: float,  # noqa: A002
        high: float,
        low: float,
        close: float,
        volume: float = 0.0,
        spread_override: float | None = None,
    ) -> list[dict]:
        """Process one CLOSED candle. Returns the events it produced."""
        n_before = len(self.events)
        if bar_time.tzinfo is None:
            raise ValueError("bar_time must be timezone-aware (UTC)")
        if self.last_bar_time is not None and bar_time <= self.last_bar_time:
            self._log("bar_ignored", bar_time, detail="already processed or out of order")
            return self.events[n_before:]
        vals = (open, high, low, close)
        if (
            not all(math.isfinite(v) and v > 0 for v in vals)
            or high < max(open, close, low)
            or (low > min(open, close, high))
        ):
            self._log("bar_rejected", bar_time, detail="invalid OHLC", ohlc=list(vals))
            return self.events[n_before:]  # fail closed: do not advance, do not trade on bad data

        spread = self.costs.spread(self.spec) if spread_override is None else spread_override
        slip = self.costs.slippage_spread_fraction * spread
        if self.risk is None:
            self.risk = RiskEngine(
                self.limits, self.account.initial_balance, bar_time, self.account.lot_value_scale
            )
        row = pd.DataFrame([[open, high, low, close, volume]], columns=OHLCV,
                           index=pd.DatetimeIndex([bar_time], name="time"))  # fmt: skip
        self.buffer = pd.concat([self.buffer, row]) if len(self.buffer) else row
        self.last_bar_time = bar_time
        self.bars_seen += 1

        if self.halted is None:
            # 1) execute what was decided at the previous close, at this candle's open
            for sig in sorted(self.pending, key=lambda s: s.action.is_entry):
                if sig.action.is_entry:
                    if self.pos is None:
                        self._open(bar_time, sig, open, spread, slip)
                elif self.pos is not None and (
                    (sig.action == Action.EXIT_LONG and self.pos.side == "long")
                    or (sig.action == Action.EXIT_SHORT and self.pos.side == "short")
                ):
                    px = open - slip if self.pos.side == "long" else open + spread + slip
                    self._close(bar_time, px, "signal", spread, slip)
            self.pending = []

            # 2) stop check against this candle's range (including the entry candle)
            if self.pos is not None:
                self.pos.bars += 1
                if self.pos.side == "long" and low <= self.pos.stop_price:
                    self._close(
                        bar_time, min(open, self.pos.stop_price) - slip, "stop", spread, slip
                    )
                elif self.pos.side == "short" and high + spread >= self.pos.stop_price:
                    px = max(open + spread, self.pos.stop_price) + slip
                    self._close(bar_time, px, "stop", spread, slip)

            # 3) mark equity, update risk, flatten + halt if the kill switch trips
            eq = self.balance + (self._unrealized(close, spread) if self.pos else 0.0)
            if eq <= 0:
                if self.pos is not None:
                    px = close - slip if self.pos.side == "long" else close + spread + slip
                    self._close(bar_time, px, "account_blown", spread, slip)
                self.halted = "account_blown"
                eq = self.balance
                self._log("halted", bar_time, reason=self.halted)
            else:
                for ev in self.risk.update_equity(eq, bar_time):
                    if ev != "new_day":
                        self._log(ev, bar_time, equity=eq)
                    if ev == "kill_switch":
                        if self.pos is not None:
                            px = close - slip if self.pos.side == "long" else close + spread + slip
                            self._close(bar_time, px, "kill_switch", spread, slip)
                            eq = self.balance
                        self.halted = "kill_switch"
                        self._log("halted", bar_time, reason=self.halted)
            self.equity_curve.append((str(bar_time), eq))

            # 4) decide at the close; acted on at the next candle's open
            if self.halted is None and self.bars_seen >= 2:
                win = (
                    self.buffer
                    if not self.feature_window
                    else self.buffer.iloc[-self.feature_window :]
                )
                feats = compute_features(win, self.feature_params)
                prev, cur = feats.iloc[-2].to_dict(), feats.iloc[-1].to_dict()
                current = Side(self.pos.side) if self.pos else None
                for sig in self.strategy.evaluate(
                    self.symbol, self.timeframe, bar_time, prev, cur, current
                ):
                    self.pending.append(sig)
                    d = self._signal_dict(sig)
                    self._log("signal", bar_time, **{k: v for k, v in d.items() if k != "bar_time"})
        else:
            self._log("bar_skipped", bar_time, detail=f"engine halted: {self.halted}")

        if self.feature_window and len(self.buffer) > 2 * self.feature_window:
            self.buffer = self.buffer.iloc[-self.feature_window :]
        self._save()
        return self.events[n_before:]

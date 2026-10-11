"""Risk engine (Milestone 8): the one place that decides whether a trade may be opened.

Principles
- Fail closed: anything invalid or unexpected rejects the trade, never approves it.
- Exits are never blocked (the engine only judges ENTRIES).
- Every rejection carries a stable reason code and a plain-language reason, so it can be logged.
- The kill switch is latched: after it trips, only a deliberate ``reset_kill_switch()`` re-arms it.
- Pure logic, no clock, no broker, no I/O: the caller passes the time, so it is fully testable and
  behaves identically in backtests and in paper trading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from tradingbot.backtesting.instruments import InstrumentSpec
from tradingbot.config import RiskConfig


class Reason(StrEnum):
    APPROVED = "approved"
    KILL_SWITCH = "kill_switch_active"
    DAILY_LOSS = "daily_loss_limit_reached"
    MAX_POSITIONS = "max_open_positions"
    DUPLICATE_SYMBOL = "position_already_open_for_symbol"
    MAX_EXPOSURE = "max_total_exposure_lots"
    INVALID_STOP = "invalid_stop_distance"
    STOP_TOO_WIDE = "stop_too_wide_for_account"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class RiskLimits:
    max_risk_per_trade_pct: float
    max_open_positions: int
    max_total_exposure_lots: float
    daily_loss_limit_pct: float
    kill_switch_drawdown_pct: float

    @classmethod
    def from_config(cls, cfg: RiskConfig) -> RiskLimits:
        return cls(
            cfg.max_risk_per_trade_pct,
            cfg.max_open_positions,
            cfg.max_total_exposure_lots,
            cfg.daily_loss_limit_pct,
            cfg.kill_switch_drawdown_pct,
        )


@dataclass(frozen=True)
class Decision:
    approved: bool
    lots: float
    reason: Reason
    detail: str
    risk_usd: float = 0.0
    risk_pct: float = 0.0


@dataclass(frozen=True)
class OpenPosition:
    symbol: str
    lots: float
    risk_usd: float


@dataclass
class AccountState:
    equity: float
    peak_equity: float
    day: pd.Timestamp  # UTC calendar day the daily loss is measured from
    day_start_equity: float
    kill_switch: bool = False
    kill_detail: str | None = None
    daily_halt: bool = False
    positions: dict[str, OpenPosition] = field(default_factory=dict)


def size_lots(
    risk_budget_usd: float,
    stop_distance: float,
    price: float,
    spec: InstrumentSpec,
    lot_value_scale: float = 1.0,
    max_lots: float | None = None,
) -> float:
    """Largest lot size (rounded DOWN to the lot step) whose stop loses at most the budget.

    Returns 0.0 when even the smallest allowed lot would risk more than the budget.
    """
    unit_risk = stop_distance * spec.contract_size * lot_value_scale * spec.quote_to_usd(price)
    if not (unit_risk > 0 and math.isfinite(unit_risk)) or risk_budget_usd <= 0:
        return 0.0
    steps = math.floor(risk_budget_usd / unit_risk / spec.volume_step + 1e-9)
    lots = round(steps * spec.volume_step, 8)
    lots = min(lots, spec.volume_max, max_lots if max_lots else math.inf)
    return lots if lots >= spec.volume_min - 1e-12 else 0.0


def risk_usd(
    lots: float,
    stop_distance: float,
    price: float,
    spec: InstrumentSpec,
    lot_value_scale: float = 1.0,
) -> float:
    return lots * stop_distance * spec.contract_size * lot_value_scale * spec.quote_to_usd(price)


class RiskEngine:
    def __init__(
        self,
        limits: RiskLimits,
        initial_equity: float,
        now: pd.Timestamp,
        lot_value_scale: float = 1.0,
    ):
        if not (initial_equity > 0 and math.isfinite(initial_equity)):
            raise ValueError("initial_equity must be a positive number")
        if not (lot_value_scale > 0 and math.isfinite(lot_value_scale)):
            raise ValueError("lot_value_scale must be positive")
        self.limits = limits
        # exposure limits are in normal-lot equivalents: a cent-account lot counts for 1/100
        self.lot_value_scale = lot_value_scale
        day = now.tz_convert("UTC").normalize() if now.tzinfo else now.normalize()
        self.state = AccountState(initial_equity, initial_equity, day, initial_equity)

    # ------------------------------------------------------------------ account tracking
    def update_equity(self, equity: float, now: pd.Timestamp) -> list[str]:
        """Feed the latest equity. Returns events: new_day, daily_loss_limit, kill_switch."""
        s, events = self.state, []
        if not math.isfinite(equity):
            self._trip("equity is not a finite number")
            return ["kill_switch"]
        day = now.tz_convert("UTC").normalize() if now.tzinfo else now.normalize()
        if day > s.day:
            s.day, s.day_start_equity, s.daily_halt = day, s.equity, False
            events.append("new_day")
        s.equity = equity
        s.peak_equity = max(s.peak_equity, equity)
        if (
            not s.daily_halt
            and s.day_start_equity > 0
            and (s.day_start_equity - equity) / s.day_start_equity * 100
            >= self.limits.daily_loss_limit_pct
        ):
            s.daily_halt = True
            events.append("daily_loss_limit")
        drawdown = (s.peak_equity - equity) / s.peak_equity * 100 if s.peak_equity > 0 else 100.0
        if not s.kill_switch and drawdown >= self.limits.kill_switch_drawdown_pct:
            self._trip(f"drawdown {drawdown:.1f}% reached {self.limits.kill_switch_drawdown_pct}%")
            events.append("kill_switch")
        return events

    def _trip(self, detail: str) -> None:
        self.state.kill_switch, self.state.kill_detail = True, detail

    def reset_kill_switch(self) -> None:
        """Deliberate human action only. Re-arms from the current equity as the new peak."""
        self.state.kill_switch, self.state.kill_detail = False, None
        self.state.peak_equity = self.state.equity

    def register_open(self, symbol: str, lots: float, risk_usd_: float) -> None:
        self.state.positions[symbol] = OpenPosition(symbol, lots, risk_usd_)

    def register_close(self, symbol: str) -> None:
        self.state.positions.pop(symbol, None)

    # ------------------------------------------------------------------------- decisions
    def _no(self, reason: Reason, detail: str) -> Decision:
        return Decision(False, 0.0, reason, detail)

    def evaluate_entry(
        self,
        *,
        symbol: str,
        stop_distance: float | None,
        price: float,
        spec: InstrumentSpec,
        requested_lots: float | None = None,
        max_lots: float | None = None,
    ) -> Decision:
        """Approve (with a lot size) or reject an entry. ``requested_lots`` None = size by risk."""
        try:
            return self._evaluate(symbol, stop_distance, price, spec, requested_lots, max_lots)
        except Exception as exc:  # fail closed
            return self._no(Reason.INVALID_INPUT, f"{type(exc).__name__}: {exc}")

    def _evaluate(self, symbol, stop_distance, price, spec, requested, max_lots) -> Decision:
        s, lim, scale = self.state, self.limits, self.lot_value_scale
        if s.kill_switch:
            return self._no(Reason.KILL_SWITCH, s.kill_detail or "kill switch is active")
        if s.daily_halt:
            return self._no(Reason.DAILY_LOSS, f"lost {lim.daily_loss_limit_pct}% today")
        if not (s.equity > 0 and math.isfinite(price) and price > 0):
            return self._no(Reason.INVALID_INPUT, "equity and price must be positive numbers")
        if stop_distance is None or not (stop_distance > 0 and math.isfinite(stop_distance)):
            return self._no(Reason.INVALID_STOP, f"stop distance {stop_distance!r}")
        if symbol in s.positions:
            return self._no(Reason.DUPLICATE_SYMBOL, f"{symbol} already has a position")
        if len(s.positions) >= lim.max_open_positions:
            return self._no(Reason.MAX_POSITIONS, f"{len(s.positions)} positions already open")

        budget = s.equity * lim.max_risk_per_trade_pct / 100
        if requested is None:
            lots = size_lots(budget, stop_distance, price, spec, scale, max_lots)
        else:
            spec.check_lot(requested)
            lots = requested
        if lots <= 0:
            return self._no(
                Reason.STOP_TOO_WIDE,
                f"even the minimum lot risks more than {lim.max_risk_per_trade_pct}% of equity",
            )
        usd = risk_usd(lots, stop_distance, price, spec, scale)
        if usd > budget * (1 + 1e-9):
            return self._no(Reason.STOP_TOO_WIDE, f"risk ${usd:.2f} exceeds budget ${budget:.2f}")
        exposure = (sum(p.lots for p in s.positions.values()) + lots) * scale
        if exposure > lim.max_total_exposure_lots + 1e-12:
            return self._no(
                Reason.MAX_EXPOSURE, f"total exposure would be {exposure:.2f} normal lots"
            )
        return Decision(True, lots, Reason.APPROVED, "ok", usd, usd / s.equity * 100)

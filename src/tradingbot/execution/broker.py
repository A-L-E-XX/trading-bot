"""The ONLY module that can send orders to MetaTrader 5, and only on a DEMO account.

Safety rules enforced here (each one has a test):
- the account must be a demo account, re-checked before EVERY order and close (fail closed);
- every entry carries a stop-loss; an order without one is refused;
- lots are rounded to the broker's step and capped by a hard limit;
- the loss at the stop is computed by the broker (``order_calc_profit``) and the order is scaled
  down, or refused, if it would exceed the risk limit on the real account equity;
- every position is tagged with a magic number, and only positions with that magic are touched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tradingbot.data.collectors.mt5 import MT5Collector, MT5Error


class OrderRefused(MT5Error):
    """The order was not sent, or the broker did not accept it."""


@dataclass(frozen=True)
class BrokerPosition:
    ticket: int
    side: str  # "long" | "short"
    lots: float
    price_open: float
    sl: float
    profit: float


class DemoBroker:
    def __init__(
        self,
        collector: MT5Collector,
        symbol: str,
        *,
        magic: int,
        max_lots: float = 1.0,
        max_risk_pct: float = 3.0,
        deviation: int = 50,
    ):
        if max_lots <= 0 or max_risk_pct <= 0:
            raise ValueError("max_lots and max_risk_pct must be positive")
        self.col = collector
        self.symbol = symbol
        self.magic = magic
        self.max_lots = max_lots
        self.max_risk_pct = max_risk_pct
        self.deviation = deviation

    # ------------------------------------------------------------------ helpers
    @property
    def _mt5(self):
        return self.col._mt5

    @property
    def broker_symbol(self) -> str:
        return self.col.broker_symbol(self.symbol)

    def _require_demo(self) -> None:
        if not self.col.account_is_demo():  # re-checked every time, never cached
            raise OrderRefused("refusing to trade: this MT5 account is NOT a demo account")

    def _info(self):
        info = self._mt5.symbol_info(self.broker_symbol)
        if info is None:
            raise OrderRefused(f"no symbol info for {self.broker_symbol!r}")
        return info

    def _tick(self):
        tick = self._mt5.symbol_info_tick(self.broker_symbol)
        if tick is None or not tick.ask or not tick.bid:
            raise OrderRefused("no live quote (market closed or symbol not visible)")
        return tick

    def _filling(self, info) -> int:
        flags = getattr(info, "filling_mode", 0)
        if flags & 2:
            return self._mt5.ORDER_FILLING_IOC
        if flags & 1:
            return self._mt5.ORDER_FILLING_FOK
        return self._mt5.ORDER_FILLING_RETURN

    @staticmethod
    def _floor_to_step(value: float, step: float) -> float:
        return round(math.floor(value / step + 1e-9) * step, 8)

    # ------------------------------------------------------------------ reading
    def equity(self) -> float:
        return float(self.col.account_info().equity)

    def positions(self) -> list[BrokerPosition]:
        """Open positions for this symbol that carry OUR magic number (nothing else)."""
        raw = self._mt5.positions_get(symbol=self.broker_symbol)
        if raw is None:
            raise MT5Error(f"positions_get failed: {self._mt5.last_error()}")
        out = []
        for p in raw:
            if getattr(p, "magic", None) != self.magic:
                continue
            side = "long" if p.type == self._mt5.POSITION_TYPE_BUY else "short"
            out.append(BrokerPosition(p.ticket, side, p.volume, p.price_open, p.sl, p.profit))
        return out

    def closed_pnl(self, ticket: int) -> float | None:
        """Realised result (profit + commission + swap + fee) of a position, or None if unknown."""
        deals = self._mt5.history_deals_get(position=ticket)
        if not deals:
            return None
        return float(
            sum(
                getattr(d, "profit", 0.0)
                + getattr(d, "commission", 0.0)
                + getattr(d, "swap", 0.0)
                + getattr(d, "fee", 0.0)
                for d in deals
            )
        )

    def risk_at_stop(self, side: str, lots: float, price: float, sl: float) -> float:
        """Loss (positive, account currency) the BROKER says this trade has at its stop."""
        order_type = self._mt5.ORDER_TYPE_BUY if side == "long" else self._mt5.ORDER_TYPE_SELL
        pnl = self._mt5.order_calc_profit(order_type, self.broker_symbol, lots, price, sl)
        if pnl is None:
            raise OrderRefused(f"cannot compute the loss at the stop: {self._mt5.last_error()}")
        return abs(float(pnl))

    # ------------------------------------------------------------------ trading
    def open(self, side: str, lots: float, stop_distance: float, comment: str = "tb") -> dict:
        """Open a market position with a stop. Returns what was actually done."""
        self._require_demo()
        if side not in ("long", "short"):
            raise OrderRefused(f"bad side {side!r}")
        if not (stop_distance and stop_distance > 0 and math.isfinite(stop_distance)):
            raise OrderRefused("refusing an order without a valid stop distance")
        info = self._info()
        step, vmin = float(info.volume_step), float(info.volume_min)
        vmax = min(float(info.volume_max), self.max_lots)
        tick = self._tick()
        price = float(tick.ask if side == "long" else tick.bid)
        sl = price - stop_distance if side == "long" else price + stop_distance
        digits = int(getattr(info, "digits", 5))
        sl = round(sl, digits)
        if sl <= 0:
            raise OrderRefused("stop price would be below zero")

        lots = min(self._floor_to_step(lots, step), self._floor_to_step(vmax, step))
        equity = self.equity()
        limit = equity * self.max_risk_pct / 100
        risk = self.risk_at_stop(side, lots, price, sl) if lots >= vmin else 0.0
        while lots >= vmin and risk > limit:  # scale down to the real risk limit
            lots = self._floor_to_step(lots - step, step) if lots - step >= vmin else 0.0
            risk = self.risk_at_stop(side, lots, price, sl) if lots >= vmin else 0.0
        if lots < vmin:
            raise OrderRefused(
                f"even the minimum lot {vmin} risks more than {self.max_risk_pct}% of the account "
                f"(limit {limit:.2f}); trade skipped"
            )

        request = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": self.broker_symbol,
            "volume": lots,
            "type": self._mt5.ORDER_TYPE_BUY if side == "long" else self._mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": comment[:30],
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(info),
        }
        result = self._mt5.order_send(request)
        self._check(result, "open")
        return {
            "ticket": int(result.order),
            "side": side,
            "lots": lots,
            "requested_price": price,
            "fill_price": float(getattr(result, "price", price) or price),
            "sl": sl,
            "risk_at_stop": risk,
            "equity": equity,
        }

    def close(self, position: BrokerPosition, comment: str = "tb close") -> dict:
        self._require_demo()
        info = self._info()
        tick = self._tick()
        is_long = position.side == "long"
        request = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": self.broker_symbol,
            "volume": position.lots,
            "type": self._mt5.ORDER_TYPE_SELL if is_long else self._mt5.ORDER_TYPE_BUY,
            "position": position.ticket,
            "price": float(tick.bid if is_long else tick.ask),
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": comment[:30],
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(info),
        }
        result = self._mt5.order_send(request)
        self._check(result, "close")
        return {"ticket": position.ticket, "price": float(getattr(result, "price", 0.0) or 0.0)}

    def close_all(self) -> list[dict]:
        """Panic button: close every position that carries our magic number."""
        return [self.close(p, "tb panic") for p in self.positions()]

    def _check(self, result, what: str) -> None:
        if result is None:
            raise OrderRefused(f"{what}: no answer from the terminal ({self._mt5.last_error()})")
        if result.retcode != self._mt5.TRADE_RETCODE_DONE:
            raise OrderRefused(
                f"{what}: broker answered {result.retcode} {getattr(result, 'comment', '')}"
            )

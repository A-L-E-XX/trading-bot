"""Strategy V4: opening-range breakout scalper (spec: docs/strategy_v4.md)."""

from __future__ import annotations

import pandas as pd

from tradingbot.config import StrategyV4Config
from tradingbot.signals import Action, Side, Signal
from tradingbot.strategies.base import Strategy

LONG_WHY = "first close above the opening range"
SHORT_WHY = "first close below the opening range"
FLAT_EXIT = "end of session: flatten"
NEEDED = ("close", "orb_high", "orb_low", "orb_signal", "orb_flatten", "atr")


class OrbScalperStrategy(Strategy):
    required_features = NEEDED

    def __init__(self, params: StrategyV4Config):
        self.params = params
        self.name = params.name
        self.version = params.version

    def _signal(self, symbol, timeframe, bar_time, cur, action, reason, stop_distance=None):
        return Signal(
            symbol=symbol,
            timeframe=timeframe,
            bar_time=bar_time,
            action=action,
            strategy=self.name,
            strategy_version=self.version,
            reason=reason,
            reference_price=float(cur["close"]),
            stop_distance=stop_distance,
        )

    def evaluate(self, symbol, timeframe, bar_time, prev, cur, position) -> list[Signal]:
        # Decides from the current closed candle only.
        out: list[Signal] = []
        flatten = bool(cur["orb_flatten"])
        if position == Side.LONG and flatten:
            out.append(self._signal(symbol, timeframe, bar_time, cur, Action.EXIT_LONG, FLAT_EXIT))
        elif position == Side.SHORT and flatten:
            out.append(self._signal(symbol, timeframe, bar_time, cur, Action.EXIT_SHORT, FLAT_EXIT))
        if position is not None or flatten:
            return out
        if any(pd.isna(cur[k]) for k in NEEDED) or cur["atr"] <= 0:
            return []
        p = self.params
        hi, lo, close = cur["orb_high"], cur["orb_low"], cur["close"]
        if (hi - lo) < p.min_range_atr * cur["atr"]:
            return []
        if cur["orb_signal"] > 0:  # stop = the opposite side of the opening range
            out.append(
                self._signal(
                    symbol, timeframe, bar_time, cur, Action.ENTER_LONG, LONG_WHY, float(close - lo)
                )
            )
        elif cur["orb_signal"] < 0 and p.allow_short:
            out.append(
                self._signal(
                    symbol,
                    timeframe,
                    bar_time,
                    cur,
                    Action.ENTER_SHORT,
                    SHORT_WHY,
                    float(hi - close),
                )
            )
        return out

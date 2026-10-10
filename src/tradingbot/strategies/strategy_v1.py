"""Strategy V1: EMA trend-following with an ATR stop (spec: docs/strategy_v1.md)."""

from __future__ import annotations

import pandas as pd

from tradingbot.config import StrategyV1Config
from tradingbot.signals import Action, Side, Signal
from tradingbot.strategies.base import Strategy

UP_EXIT = "EMA fast crossed above slow: exit short"
DOWN_EXIT = "EMA fast crossed below slow: exit long"
LONG_WHY = "EMA fast crossed above slow and close above trend EMA"
SHORT_WHY = "EMA fast crossed below slow and close below trend EMA"


class EmaTrendAtrStrategy(Strategy):
    required_features = ("close", "ema_fast", "ema_slow", "ema_trend", "atr")

    def __init__(self, params: StrategyV1Config):
        self.params = params
        self.name = params.name
        self.version = params.version

    def _signal(
        self, symbol, timeframe, bar_time, cur, action, reason, stop_distance=None
    ) -> Signal:
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

    def _entry(self, symbol, timeframe, bar_time, cur, action, reason, distance) -> Signal:
        return self._signal(symbol, timeframe, bar_time, cur, action, reason, distance)

    def _exit(self, symbol, timeframe, bar_time, cur, action, reason) -> Signal:
        return self._signal(symbol, timeframe, bar_time, cur, action, reason)

    def evaluate(self, symbol, timeframe, bar_time, prev, cur, position) -> list[Signal]:
        # No signals until every indicator (EMA200 is the slowest) is valid on both bars.
        if not (bool(prev["ready"]) and bool(cur["ready"])):
            return []
        if pd.isna(cur["atr"]) or cur["atr"] <= 0:
            return []

        crossed_up = prev["ema_fast"] <= prev["ema_slow"] and cur["ema_fast"] > cur["ema_slow"]
        crossed_down = prev["ema_fast"] >= prev["ema_slow"] and cur["ema_fast"] < cur["ema_slow"]
        dist = float(self.params.atr_stop_mult * cur["atr"])
        out: list[Signal] = []

        if crossed_up:
            if position == Side.SHORT:
                out.append(self._exit(symbol, timeframe, bar_time, cur, Action.EXIT_SHORT, UP_EXIT))
            if position != Side.LONG and cur["close"] > cur["ema_trend"]:
                out.append(
                    self._entry(symbol, timeframe, bar_time, cur, Action.ENTER_LONG, LONG_WHY, dist)
                )
        elif crossed_down:
            if position == Side.LONG:
                out.append(
                    self._exit(symbol, timeframe, bar_time, cur, Action.EXIT_LONG, DOWN_EXIT)
                )
            if (
                self.params.allow_short
                and position != Side.SHORT
                and cur["close"] < cur["ema_trend"]
            ):
                out.append(
                    self._entry(
                        symbol, timeframe, bar_time, cur, Action.ENTER_SHORT, SHORT_WHY, dist
                    )
                )
        return out

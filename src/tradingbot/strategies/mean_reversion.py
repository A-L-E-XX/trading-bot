"""Strategy V2: Bollinger/RSI mean reversion with an ATR stop (spec: docs/strategy_v2.md)."""

from __future__ import annotations

import pandas as pd

from tradingbot.config import StrategyV2Config
from tradingbot.signals import Action, Side, Signal
from tradingbot.strategies.base import Strategy

LONG_WHY = "close below lower band and RSI oversold"
SHORT_WHY = "close above upper band and RSI overbought"
LONG_EXIT = "close back at or above the mean: exit long"
SHORT_EXIT = "close back at or below the mean: exit short"


class MeanReversionBBStrategy(Strategy):
    required_features = ("close", "sma", "bb_std", "rsi", "atr")

    def __init__(self, params: StrategyV2Config):
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
        # Decides from the current closed candle only (``prev`` is not needed for this strategy).
        if any(pd.isna(cur[k]) for k in ("close", "sma", "bb_std", "rsi", "atr")):
            return []
        if cur["atr"] <= 0 or cur["bb_std"] <= 0:
            return []
        p = self.params
        close, mean = cur["close"], cur["sma"]
        lower, upper = mean - p.bb_k * cur["bb_std"], mean + p.bb_k * cur["bb_std"]
        dist = float(p.atr_stop_mult * cur["atr"])
        out: list[Signal] = []

        if position == Side.LONG and close >= mean:
            out.append(self._signal(symbol, timeframe, bar_time, cur, Action.EXIT_LONG, LONG_EXIT))
        elif position == Side.SHORT and close <= mean:
            out.append(
                self._signal(symbol, timeframe, bar_time, cur, Action.EXIT_SHORT, SHORT_EXIT)
            )
        elif position is None:
            if close < lower and cur["rsi"] < p.rsi_low:
                out.append(
                    self._signal(
                        symbol, timeframe, bar_time, cur, Action.ENTER_LONG, LONG_WHY, dist
                    )
                )
            elif p.allow_short and close > upper and cur["rsi"] > p.rsi_high:
                out.append(
                    self._signal(
                        symbol, timeframe, bar_time, cur, Action.ENTER_SHORT, SHORT_WHY, dist
                    )
                )
        return out

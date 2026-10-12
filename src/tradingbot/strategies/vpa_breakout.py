"""Strategy V3: volume-confirmed breakout from a tight range (spec: docs/strategy_v3.md)."""

from __future__ import annotations

import pandas as pd

from tradingbot.config import StrategyV3Config
from tradingbot.signals import Action, Side, Signal
from tradingbot.strategies.base import Strategy

LONG_WHY = "accumulation breakout: close above a tight range on high volume"
SHORT_WHY = "distribution breakdown: close below a tight range on high volume"
LONG_EXIT = "high-volume down candle: exit long"
SHORT_EXIT = "high-volume up candle: exit short"
NEEDED = ("open", "close", "range_high", "range_low", "volume_ratio", "atr")


class VPABreakoutStrategy(Strategy):
    required_features = NEEDED

    def __init__(self, params: StrategyV3Config):
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
        if any(pd.isna(cur[k]) for k in NEEDED) or cur["atr"] <= 0:
            return []
        p = self.params
        high_volume = cur["volume_ratio"] >= p.vol_mult
        up, down = cur["close"] > cur["open"], cur["close"] < cur["open"]
        dist = float(p.atr_stop_mult * cur["atr"])
        out: list[Signal] = []

        if position == Side.LONG and down and high_volume:
            out.append(self._signal(symbol, timeframe, bar_time, cur, Action.EXIT_LONG, LONG_EXIT))
        elif position == Side.SHORT and up and high_volume:
            out.append(
                self._signal(symbol, timeframe, bar_time, cur, Action.EXIT_SHORT, SHORT_EXIT)
            )
        elif position is None:
            tight = (cur["range_high"] - cur["range_low"]) <= p.max_range_atr * cur["atr"]
            if tight and high_volume and up and cur["close"] > cur["range_high"]:
                out.append(
                    self._signal(
                        symbol, timeframe, bar_time, cur, Action.ENTER_LONG, LONG_WHY, dist
                    )
                )
            elif (
                p.allow_short
                and tight
                and high_volume
                and down
                and (cur["close"] < cur["range_low"])
            ):
                out.append(
                    self._signal(
                        symbol, timeframe, bar_time, cur, Action.ENTER_SHORT, SHORT_WHY, dist
                    )
                )
        return out

"""Small helpers for building hand-checkable backtest scenarios."""

from __future__ import annotations

import pandas as pd

from tradingbot.backtesting.costs import CostModel
from tradingbot.backtesting.engine import BacktestConfig
from tradingbot.backtesting.instruments import InstrumentSpec
from tradingbot.signals import Action, Signal
from tradingbot.strategies.base import Strategy

EUR = InstrumentSpec("EURUSD", 100000.0, 1e-5, 5, 0.01, 0.01, 10.0, "EUR", "USD")  # spread 0.0001
JPY = InstrumentSpec("USDJPY", 100000.0, 1e-3, 3, 0.01, 0.01, 10.0, "USD", "JPY")  # spread 0.010
XAU = InstrumentSpec("XAUUSD", 100.0, 1e-3, 3, 0.01, 0.01, 240.0, "XAU", "USD")  # spread 0.24

NO_COSTS = CostModel(spread_multiplier=0.0, slippage_spread_fraction=0.0)
START = pd.Timestamp("2024-01-02 00:00", tz="UTC")


def bars(opens, highs=None, lows=None, closes=None) -> pd.DataFrame:
    """Hourly candles. Highs/lows/closes default to the open so only what you set matters."""
    highs = highs or opens
    lows = lows or opens
    closes = closes or opens
    idx = pd.date_range(START, periods=len(opens), freq="1h", name="time")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "ready": True}, index=idx
    )


def cfg(costs: CostModel = NO_COSTS, lots: float = 0.01, balance: float = 100.0) -> BacktestConfig:
    return BacktestConfig(initial_balance=balance, lots=lots, costs=costs)


class Scripted(Strategy):
    """Emits fixed signals at fixed candle numbers (decided at that candle's close)."""

    name = "scripted"
    version = "0"
    required_features = ()

    def __init__(self, script: dict[int, list[tuple[Action, float | None]]]):
        self.script = script

    def evaluate(self, symbol, timeframe, bar_time, prev, cur, position):
        i = int((bar_time - START) / pd.Timedelta(hours=1))
        return [
            Signal(symbol, timeframe, bar_time, action, "scripted", "0", "test", cur["close"], dist)
            for action, dist in self.script.get(i, [])
        ]

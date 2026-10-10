"""Strategy interface. The same strategy object is used for backtest, paper and (later) live."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import pandas as pd

from tradingbot.signals import Action, Side, Signal


class Strategy(ABC):
    name: str
    version: str
    required_features: tuple[str, ...] = ()

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        timeframe: str,
        bar_time: pd.Timestamp,
        prev: Mapping[str, Any],
        cur: Mapping[str, Any],
        position: Side | None,
    ) -> list[Signal]:
        """Decide at the close of ``cur`` given the previous bar's features and the open position.

        Must use only ``prev`` and ``cur`` (past and present information), never future bars.
        """

    def generate_signals(self, features: pd.DataFrame, symbol: str, timeframe: str) -> list[Signal]:
        """Signals over a whole history, with no broker and no stops.

        The position is tracked from the signals alone (every entry/exit is assumed to happen),
        which
        makes this a pure data -> strategy -> signals pipeline. The backtester instead feeds the
        *real* position (which stops can close) into ``evaluate``.
        """
        missing = [c for c in (*self.required_features, "ready") if c not in features.columns]
        if missing:
            raise ValueError(f"features are missing columns: {missing}")
        signals: list[Signal] = []
        position: Side | None = None
        rows = features.to_dict("records")
        times = features.index
        for i in range(1, len(rows)):
            out = self.evaluate(symbol, timeframe, times[i], rows[i - 1], rows[i], position)
            for sig in out:
                signals.append(sig)
                if sig.action == Action.ENTER_LONG:
                    position = Side.LONG
                elif sig.action == Action.ENTER_SHORT:
                    position = Side.SHORT
                else:
                    position = None
        return signals

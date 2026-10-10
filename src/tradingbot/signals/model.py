"""The signal object: what a strategy says. Strategies never place orders - they emit signals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class Action(StrEnum):
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"

    @property
    def is_entry(self) -> bool:
        return self in (Action.ENTER_LONG, Action.ENTER_SHORT)


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class Signal:
    """A decision made at the *close* of candle ``bar_time`` (its open time).

    It can only be acted on from the next candle's open onwards, never at the signal candle's
    price.
    ``stop_distance`` (entries only) is a price distance; the engine places the stop at the real
    fill price -/+ that distance. ``reference_price`` is the signal candle's close, for logging.
    """

    symbol: str
    timeframe: str
    bar_time: pd.Timestamp
    action: Action
    strategy: str
    strategy_version: str
    reason: str
    reference_price: float
    stop_distance: float | None = None

    def __post_init__(self) -> None:
        if self.action.is_entry and (self.stop_distance is None or self.stop_distance <= 0):
            raise ValueError("entry signals need a positive stop_distance")

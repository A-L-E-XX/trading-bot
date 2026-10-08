"""Collector interface: anything that can produce raw OHLCV candles for a symbol/timeframe."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd


class Collector(Protocol):
    name: str

    def fetch(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Return raw candles with a ``time`` column (UTC) plus open/high/low/close/volume.

        ``start``/``end`` are tz-aware UTC datetimes. The result may be unsorted or contain
        duplicates; the normalizer cleans that up and the validator reports on it.
        """
        ...

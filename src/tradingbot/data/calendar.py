"""Which candles are *expected* to be missing (market closed)?

This is deliberately generous around the weekend boundaries because brokers close and reopen
at slightly different times. Anything else missing is reported by the validator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CRYPTO_PREFIXES = ("BTC", "ETH", "LTC", "XRP", "SOL")


def is_24_7(symbol: str) -> bool:
    """Crypto trades around the clock; forex/metals have a weekend break."""
    return symbol.upper().startswith(CRYPTO_PREFIXES)


def expected_closed_mask(index: pd.DatetimeIndex, symbol: str, timeframe: str) -> np.ndarray:
    """Boolean array: True where a candle starting at that UTC time is expected to not exist."""
    if is_24_7(symbol):
        return np.zeros(len(index), dtype=bool)
    dow = np.asarray(index.dayofweek)
    hour = np.asarray(index.hour)
    if timeframe == "1d":
        return dow >= 5  # Saturday and Sunday
    saturday = dow == 5
    friday_evening = (dow == 4) & (hour >= 21)
    sunday_before_open = (dow == 6) & (hour < 21)
    return saturday | friday_evening | sunday_before_open

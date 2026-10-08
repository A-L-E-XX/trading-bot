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


_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def slot_name(slot: int) -> str:
    return f"{_DAY_NAMES[slot // 24]} {slot % 24:02d}:00"


def recurring_closure_slots(
    open_grid: pd.DatetimeIndex,
    missing: pd.DatetimeIndex,
    *,
    min_rate: float = 0.25,
    min_occurrences: int = 20,
    min_chunks: int = 3,
) -> np.ndarray:
    """Weekly (weekday, hour) slots where candles are routinely absent -> a broker break.

    Brokers pause some instruments at fixed times (gold's daily break, the rollover hour, the
    Sunday opening hour). Such a slot is treated as a closure when candles are missing in at least
    ``min_rate`` of its occurrences in at least ``min_chunks`` of 4 equal time chunks. Requiring
    several chunks means one long outage in a single period is NOT mistaken for a closure.

    ``open_grid`` is the expected candle grid (statically closed hours already removed) and
    ``missing`` the candles absent from it. Returns a boolean array of length 168 (Mon 00:00 = 0).
    """
    n_chunks = 4
    if len(open_grid) == 0 or len(missing) == 0:
        return np.zeros(168, dtype=bool)
    chunk = np.minimum(np.arange(len(open_grid)) * n_chunks // len(open_grid), n_chunks - 1)
    slot = np.asarray(open_grid.dayofweek) * 24 + np.asarray(open_grid.hour)
    occurrences = np.zeros((n_chunks, 168))
    np.add.at(occurrences, (chunk, slot), 1)
    pos = open_grid.get_indexer(missing)
    pos = pos[pos >= 0]
    absent = np.zeros((n_chunks, 168))
    np.add.at(absent, (chunk[pos], slot[pos]), 1)
    rate = absent / np.maximum(occurrences, 1)
    chunks_affected = ((rate >= min_rate) & (occurrences > 0)).sum(axis=0)
    return (chunks_affected >= min_chunks) & (occurrences.sum(axis=0) >= min_occurrences)

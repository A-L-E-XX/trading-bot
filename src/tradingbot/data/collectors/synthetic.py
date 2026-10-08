"""Deterministic synthetic candles for development and tests (NOT real market data)."""

from __future__ import annotations

import zlib
from datetime import datetime

import numpy as np
import pandas as pd

from tradingbot.data.calendar import expected_closed_mask
from tradingbot.timeframes import TIMEFRAME_SECONDS, timeframe_freq

_START_PRICE = {"BTCUSD": 60000.0, "XAUUSD": 2500.0, "USDJPY": 150.0}


class SyntheticCollector:
    """Random-walk candles. Same (seed, symbol, timeframe, range) -> identical output."""

    name = "synthetic"

    def __init__(
        self, seed: int = 42, hourly_volatility: float = 0.0015, drift_per_hour: float = 0.0
    ):
        self.seed = seed
        self.hourly_volatility = hourly_volatility
        self.drift_per_hour = drift_per_hour

    def fetch(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        times = pd.date_range(
            start, end, freq=timeframe_freq(timeframe), tz="UTC", inclusive="left"
        )
        times = times[~expected_closed_mask(times, symbol, timeframe)]
        n = len(times)
        rng = np.random.default_rng(self.seed + zlib.crc32(f"{symbol}|{timeframe}".encode()))
        hours = TIMEFRAME_SECONDS[timeframe] / 3600
        sigma = self.hourly_volatility * np.sqrt(hours)
        log_ret = rng.normal(self.drift_per_hour * hours, sigma, n)
        close = _START_PRICE.get(symbol, 1.1 if symbol.endswith("USD") else 100.0) * np.exp(
            np.cumsum(log_ret)
        )
        open_ = np.concatenate([[close[0] / np.exp(log_ret[0])], close[:-1]]) if n else close
        wick_up = np.abs(rng.normal(0, sigma * 0.5, n))
        wick_dn = np.abs(rng.normal(0, sigma * 0.5, n))
        high = np.maximum(open_, close) * np.exp(wick_up)
        low = np.minimum(open_, close) * np.exp(-wick_dn)
        volume = np.round(rng.lognormal(6.0, 0.5, n))
        return pd.DataFrame(
            {
                "time": times,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

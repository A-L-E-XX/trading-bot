"""Supported timeframes and their pandas equivalents."""

from __future__ import annotations

import pandas as pd

TIMEFRAME_SECONDS: dict[str, int] = {"1h": 3600, "4h": 4 * 3600, "1d": 24 * 3600}
_PANDAS_FREQ: dict[str, str] = {"1h": "1h", "4h": "4h", "1d": "1D"}


def check_timeframe(timeframe: str) -> str:
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(
            f"unsupported timeframe {timeframe!r}; use one of {list(TIMEFRAME_SECONDS)}"
        )
    return timeframe


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    return pd.Timedelta(seconds=TIMEFRAME_SECONDS[check_timeframe(timeframe)])


def timeframe_freq(timeframe: str) -> str:
    return _PANDAS_FREQ[check_timeframe(timeframe)]

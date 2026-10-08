from datetime import UTC, datetime

import pytest

from tradingbot.data.collectors import SyntheticCollector
from tradingbot.data.normalizer import normalize_ohlcv

START = datetime(2023, 1, 2, tzinfo=UTC)  # a Monday
END = datetime(2023, 3, 1, tzinfo=UTC)


@pytest.fixture
def start():
    return START


@pytest.fixture
def end():
    return END


@pytest.fixture
def make_ohlcv():
    """Factory: clean canonical frame for any symbol/timeframe."""

    def _make(symbol="BTCUSD", timeframe="1h", start=START, end=END, seed=1):
        raw = SyntheticCollector(seed=seed).fetch(symbol, timeframe, start, end)
        df, _ = normalize_ohlcv(raw)
        return df

    return _make


@pytest.fixture
def ohlcv(make_ohlcv):
    return make_ohlcv()

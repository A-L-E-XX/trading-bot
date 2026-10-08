"""Indicator primitives. Every value at row t depends only on rows <= t (no look-ahead).

Warm-up: each indicator returns NaN until it has enough history. Wilder-style smoothing (RSI, ATR)
and EMA are seeded from the first observation rather than an SMA seed (as TA-Lib does), so the
first ~3*period values differ slightly from TA-Lib and converge afterwards. This choice is
deliberate: it is simple, causal and fully reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(close: pd.Series) -> pd.Series:
    return close.pct_change(fill_method=None)


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average, alpha = 2 / (period + 1), NaN for the first period-1 rows."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100]. A flat series gives 50; only-gains gives 100."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    return out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    return ranges.max(axis=1)  # first bar: high - low (no previous close)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range with Wilder smoothing."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def volatility(log_ret: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of log returns (per candle, not annualised)."""
    return log_ret.rolling(window, min_periods=window).std(ddof=1)


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume relative to its rolling average (NaN where the average is zero)."""
    avg = sma(volume, window)
    return volume / avg.where(avg > 0)

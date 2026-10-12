"""Feature engine: canonical OHLCV in, strategy-ready feature frame out."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradingbot.config import StrategyV1Config
from tradingbot.features import indicators as ind

REQUIRED_INPUT = ["open", "high", "low", "close", "volume"]
FEATURE_COLUMNS = [
    "ret",
    "log_ret",
    "sma",
    "bb_std",
    "ema_fast",
    "ema_slow",
    "ema_trend",
    "rsi",
    "atr",
    "volatility",
    "volume_ratio",
    "range_high",
    "range_low",
]
# Strategy V1 needs these; the `ready` flag is True only once all of them are valid.
STRATEGY_V1_REQUIRED = ["ema_fast", "ema_slow", "ema_trend", "atr"]


@dataclass(frozen=True)
class FeatureParams:
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    atr_period: int = 14
    rsi_period: int = 14
    sma_period: int = 20
    volatility_window: int = 20
    volume_window: int = 20
    range_window: int = 20
    orb: bool = False  # add the opening-range columns (strategy V4 only)
    orb_tz: str = "America/New_York"
    orb_open_minute: int = 570
    orb_range_minutes: int = 30
    orb_entry_window_minutes: int = 120
    orb_flatten_minute: int = 955

    @classmethod
    def from_config(cls, strategy: StrategyV1Config) -> FeatureParams:
        return cls(
            ema_fast=strategy.ema_fast,
            ema_slow=strategy.ema_slow,
            ema_trend=strategy.ema_trend,
            atr_period=strategy.atr_period,
        )

    @classmethod
    def from_v2(cls, strategy) -> FeatureParams:
        return cls(
            sma_period=strategy.bb_period,
            rsi_period=strategy.rsi_period,
            atr_period=strategy.atr_period,
        )

    @classmethod
    def from_v3(cls, strategy) -> FeatureParams:
        return cls(atr_period=strategy.atr_period, range_window=strategy.range_window)

    @classmethod
    def from_v4(cls, strategy) -> FeatureParams:
        return cls(
            atr_period=strategy.atr_period,
            orb=True,
            orb_tz=strategy.timezone,
            orb_open_minute=strategy.open_minute,
            orb_range_minutes=strategy.range_minutes,
            orb_entry_window_minutes=strategy.entry_window_minutes,
            orb_flatten_minute=strategy.flatten_minute,
        )

    @property
    def strategy_warmup_rows(self) -> int:
        """Rows needed before every Strategy V1 feature is valid."""
        return max(self.ema_fast, self.ema_slow, self.ema_trend, self.atr_period + 1)


def compute_features(ohlcv: pd.DataFrame, params: FeatureParams | None = None) -> pd.DataFrame:
    """Return the input columns plus feature columns and a boolean ``ready`` column.

    Deterministic and causal: row t uses only rows <= t. The input is never modified.
    """
    params = params or FeatureParams()
    missing = [c for c in REQUIRED_INPUT if c not in ohlcv.columns]
    if missing:
        raise ValueError(f"input is missing columns: {missing}")
    if not ohlcv.index.is_monotonic_increasing or ohlcv.index.has_duplicates:
        raise ValueError("input must be sorted by time with unique timestamps (run the normalizer)")

    close, high, low, volume = ohlcv["close"], ohlcv["high"], ohlcv["low"], ohlcv["volume"]
    out = ohlcv[REQUIRED_INPUT].copy()
    out["ret"] = ind.simple_returns(close)
    out["log_ret"] = ind.log_returns(close)
    out["sma"] = ind.sma(close, params.sma_period)
    out["bb_std"] = ind.rolling_std(close, params.sma_period)
    out["ema_fast"] = ind.ema(close, params.ema_fast)
    out["ema_slow"] = ind.ema(close, params.ema_slow)
    out["ema_trend"] = ind.ema(close, params.ema_trend)
    out["rsi"] = ind.rsi(close, params.rsi_period)
    out["atr"] = ind.atr(high, low, close, params.atr_period)
    out["volatility"] = ind.volatility(out["log_ret"], params.volatility_window)
    out["volume_ratio"] = ind.volume_ratio(volume, params.volume_window)
    # Range of the PREVIOUS range_window candles (shifted so the current candle is not in it).
    out["range_high"] = (
        out["high"].rolling(params.range_window, min_periods=params.range_window).max().shift(1)
    )
    out["range_low"] = (
        out["low"].rolling(params.range_window, min_periods=params.range_window).min().shift(1)
    )
    if params.orb:
        _add_opening_range(out, params)
    out["ready"] = out[STRATEGY_V1_REQUIRED].notna().all(axis=1)
    return out


def _add_opening_range(out: pd.DataFrame, p: FeatureParams) -> None:
    """Opening range of each local trading day plus the first breakout close (all causal).

    ``orb_high``/``orb_low``: high/low of the candles that start inside the opening range; visible
    only from the first candle that starts at or after the range end (so it is complete).
    ``orb_signal``: +1/-1 on the FIRST candle of the day (inside the entry window) that closes
    above/below the range, else 0. ``orb_flatten``: candle starts at or after the flatten time.
    """
    local = out.index.tz_convert(p.orb_tz)
    minute = pd.Series(local.hour * 60 + local.minute, index=out.index)
    day = pd.Series(local.date, index=out.index)
    start, end = p.orb_open_minute, p.orb_open_minute + p.orb_range_minutes
    in_range = (minute >= start) & (minute < end)
    rng_hi = out["high"].where(in_range).groupby(day).transform("max")
    rng_lo = out["low"].where(in_range).groupby(day).transform("min")
    has_open = (in_range & (minute == start)).groupby(day).transform("max")  # open candle present
    after = (minute >= end) & has_open.astype(bool)
    out["orb_high"] = rng_hi.where(after)
    out["orb_low"] = rng_lo.where(after)
    window = after & (minute < end + p.orb_entry_window_minutes)
    up = window & (out["close"] > out["orb_high"])
    down = window & (out["close"] < out["orb_low"])
    first = (up | down) & ((up | down).groupby(day).cumsum() == 1)
    out["orb_signal"] = (up.astype(int) - down.astype(int)).where(first, 0)
    out["orb_flatten"] = minute >= p.orb_flatten_minute

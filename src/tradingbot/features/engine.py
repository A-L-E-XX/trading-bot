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
    "ema_fast",
    "ema_slow",
    "ema_trend",
    "rsi",
    "atr",
    "volatility",
    "volume_ratio",
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

    @classmethod
    def from_config(cls, strategy: StrategyV1Config) -> FeatureParams:
        return cls(
            ema_fast=strategy.ema_fast,
            ema_slow=strategy.ema_slow,
            ema_trend=strategy.ema_trend,
            atr_period=strategy.atr_period,
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
    out["ema_fast"] = ind.ema(close, params.ema_fast)
    out["ema_slow"] = ind.ema(close, params.ema_slow)
    out["ema_trend"] = ind.ema(close, params.ema_trend)
    out["rsi"] = ind.rsi(close, params.rsi_period)
    out["atr"] = ind.atr(high, low, close, params.atr_period)
    out["volatility"] = ind.volatility(out["log_ret"], params.volatility_window)
    out["volume_ratio"] = ind.volume_ratio(volume, params.volume_window)
    out["ready"] = out[STRATEGY_V1_REQUIRED].notna().all(axis=1)
    return out

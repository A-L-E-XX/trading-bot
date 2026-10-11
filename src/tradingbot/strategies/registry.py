"""Which strategies exist, how their features are built and which parameters to stress-test."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tradingbot.config import AppConfig
from tradingbot.features import FeatureParams
from tradingbot.strategies.base import Strategy
from tradingbot.strategies.mean_reversion import MeanReversionBBStrategy
from tradingbot.strategies.strategy_v1 import EmaTrendAtrStrategy


@dataclass(frozen=True)
class StrategyDef:
    key: str
    config_attr: str
    strategy_cls: Callable[..., Strategy]
    feature_params: Callable[..., FeatureParams]
    sensitivity_fields: tuple[str, ...]  # moved +/-20% in the research gate

    def params(self, app: AppConfig):
        return getattr(app, self.config_attr)


REGISTRY: dict[str, StrategyDef] = {
    "v1": StrategyDef(
        "v1",
        "strategy_v1",
        EmaTrendAtrStrategy,
        FeatureParams.from_config,
        ("ema_fast", "ema_slow", "ema_trend", "atr_stop_mult"),
    ),
    "v2": StrategyDef(
        "v2",
        "strategy_v2",
        MeanReversionBBStrategy,
        FeatureParams.from_v2,
        ("bb_period", "bb_k", "rsi_low", "rsi_high", "atr_stop_mult"),
    ),
}


def get_strategy_def(key: str) -> StrategyDef:
    if key not in REGISTRY:
        raise ValueError(f"unknown strategy {key!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[key]

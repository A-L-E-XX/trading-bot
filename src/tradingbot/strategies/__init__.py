"""Strategies. Each turns features into signals; none talks to a broker."""

from tradingbot.strategies.base import Strategy
from tradingbot.strategies.mean_reversion import MeanReversionBBStrategy
from tradingbot.strategies.registry import REGISTRY, StrategyDef, get_strategy_def
from tradingbot.strategies.strategy_v1 import EmaTrendAtrStrategy
from tradingbot.strategies.vpa_breakout import VPABreakoutStrategy

__all__ = [
    "REGISTRY",
    "EmaTrendAtrStrategy",
    "MeanReversionBBStrategy",
    "Strategy",
    "VPABreakoutStrategy",
    "StrategyDef",
    "get_strategy_def",
]

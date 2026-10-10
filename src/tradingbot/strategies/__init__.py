"""Strategies. Each turns features into signals; none talks to a broker."""

from tradingbot.strategies.base import Strategy
from tradingbot.strategies.strategy_v1 import EmaTrendAtrStrategy

__all__ = ["EmaTrendAtrStrategy", "Strategy"]

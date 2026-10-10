"""Transaction costs: bid/ask spread, slippage and commission."""

from __future__ import annotations

from dataclasses import dataclass, replace

from tradingbot.backtesting.instruments import InstrumentSpec
from tradingbot.config import CostsConfig


@dataclass(frozen=True)
class CostModel:
    """Candle prices are BID prices. A buy fills at ask (= bid + spread), a sell at bid.

    Slippage is added against us on every fill (buys fill higher, sells lower) and is a fraction of
    the spread, so it scales sensibly across BTC, gold and forex.
    """

    spread_multiplier: float = 1.0
    slippage_spread_fraction: float = 0.25
    commission_per_lot: float = 0.0  # USD per lot per side

    @classmethod
    def from_config(cls, cfg: CostsConfig) -> CostModel:
        return cls(cfg.spread_multiplier, cfg.slippage_spread_fraction, cfg.commission_per_lot)

    def spread(self, spec: InstrumentSpec) -> float:
        return spec.spread_price(self.spread_multiplier)

    def slippage(self, spec: InstrumentSpec) -> float:
        return self.slippage_spread_fraction * self.spread(spec)

    def with_overrides(self, **kwargs) -> CostModel:
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})

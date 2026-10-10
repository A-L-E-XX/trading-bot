"""Instrument specifications from the broker (config/instrument_specs.json via `tb-data specs`)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    contract_size: float  # units per 1.00 lot
    point: float
    digits: int
    volume_min: float
    volume_step: float
    spread_points: float  # spread seen when the specs were exported
    currency_base: str
    currency_profit: str
    volume_max: float = 200.0

    def quote_to_usd(self, price: float) -> float:
        """USD value of one unit of the profit (quote) currency at ``price``."""
        if self.currency_profit == "USD":
            return 1.0
        if self.currency_base == "USD":  # USDJPY, USDCAD: 1 JPY = 1 / USDJPY dollars
            return 1.0 / price
        raise NotImplementedError(
            f"{self.symbol}: needs a cross rate to convert {self.currency_profit} to USD"
        )

    def spread_price(self, multiplier: float = 1.0) -> float:
        return self.spread_points * self.point * multiplier

    def check_lot(self, lots: float) -> None:
        steps = lots / self.volume_step
        if lots < self.volume_min - 1e-12 or abs(steps - round(steps)) > 1e-6:
            raise ValueError(
                f"{self.symbol}: lot {lots} not allowed "
                f"(min {self.volume_min}, step {self.volume_step})"
            )


def load_specs(path: Path | str) -> dict[str, InstrumentSpec]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. On the PC with MT5 run: uv run --extra mt5 tb-data specs"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    specs = {}
    for symbol, s in raw["symbols"].items():
        specs[symbol] = InstrumentSpec(
            symbol=symbol,
            contract_size=float(s["trade_contract_size"]),
            point=float(s["point"]),
            digits=int(s["digits"]),
            volume_min=float(s["volume_min"]),
            volume_step=float(s["volume_step"]),
            spread_points=float(s["spread"]),
            currency_base=s["currency_base"],
            currency_profit=s["currency_profit"],
            volume_max=float(s.get("volume_max", 200.0)),
        )
    return specs


def get_spec(specs: dict[str, InstrumentSpec], symbol: str) -> InstrumentSpec:
    if symbol not in specs:
        raise ValueError(f"no broker specs for {symbol}: run `uv run --extra mt5 tb-data specs`")
    return specs[symbol]

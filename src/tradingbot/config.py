"""Configuration: typed TOML settings (strategy/risk/etc.) + env-based secrets."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.toml"

Timeframe = Literal["1h", "4h", "1d"]


class AccountConfig(BaseModel):
    initial_balance: float = Field(gt=0)
    fixed_lot: float = Field(gt=0)
    max_drawdown_pct: float = Field(gt=0, le=100)
    currency: str = "USD"
    # Cent accounts: one lot is worth 1/100 (0.01). Verify against your broker's specs.
    lot_value_scale: float = Field(default=1.0, gt=0)
    # True: size each trade so its stop risks risk.max_risk_per_trade_pct (instead of fixed_lot)
    risk_sizing: bool = False
    max_lots: float | None = Field(default=None, gt=0)


class InstrumentsConfig(BaseModel):
    symbols: list[str] = Field(min_length=1)
    symbol_suffix: str = ""

    def broker_symbol(self, symbol: str) -> str:
        """Symbol name as the broker spells it (suffix applied)."""
        return f"{symbol}{self.symbol_suffix}"


class TimeframesConfig(BaseModel):
    primary: list[Timeframe] = Field(min_length=1)


class StrategyV1Config(BaseModel):
    name: str
    version: str
    ema_fast: int = Field(gt=0)
    ema_slow: int = Field(gt=0)
    ema_trend: int = Field(gt=0)
    atr_period: int = Field(gt=0)
    atr_stop_mult: float = Field(gt=0)
    allow_short: bool = True

    @model_validator(mode="after")
    def _ema_ordering(self) -> StrategyV1Config:
        if not self.ema_fast < self.ema_slow < self.ema_trend:
            raise ValueError("require ema_fast < ema_slow < ema_trend")
        return self


class CostsConfig(BaseModel):
    spread_multiplier: float = Field(ge=0)
    slippage_spread_fraction: float = Field(ge=0)
    commission_per_lot: float = Field(ge=0)


class RiskConfig(BaseModel):
    max_risk_per_trade_pct: float = Field(gt=0, le=100)
    max_open_positions: int = Field(gt=0)
    max_total_exposure_lots: float = Field(gt=0)
    daily_loss_limit_pct: float = Field(gt=0, le=100)
    kill_switch_drawdown_pct: float = Field(gt=0, le=100)


class ResearchGateConfig(BaseModel):
    min_oos_profit_factor: float = Field(gt=0)
    min_oos_trades: int = Field(gt=0)
    max_oos_drawdown_pct: float = Field(gt=0, le=100)
    max_param_sensitivity_drop_pct: float = Field(gt=0, le=100)
    min_stress_profit_factor: float = Field(gt=0)
    max_mc_breach_probability_pct: float = Field(ge=0, le=100)
    min_profitable_folds_pct: float = Field(ge=0, le=100)


class AppConfig(BaseModel):
    """Everything in config/default.toml, validated."""

    account: AccountConfig
    instruments: InstrumentsConfig
    timeframes: TimeframesConfig
    strategy_v1: StrategyV1Config
    costs: CostsConfig
    risk: RiskConfig
    research_gate: ResearchGateConfig

    @model_validator(mode="after")
    def _risk_consistent_with_account(self) -> AppConfig:
        if self.risk.kill_switch_drawdown_pct > self.account.max_drawdown_pct:
            raise ValueError("kill switch must trigger at or before the max drawdown")
        return self


class Settings(BaseSettings):
    """Environment-based settings and secrets (prefix TB_). Never logged or committed."""

    model_config = SettingsConfigDict(
        env_prefix="TB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,  # blank lines like "TB_MT5_LOGIN=" mean "not set"
    )

    mode: Literal["backtest", "paper"] = "backtest"  # live is deliberately not an option
    log_level: str = "INFO"
    data_dir: Path = Path("data_store")
    output_dir: Path = Path("outputs")

    mt5_login: int | None = None
    mt5_password: SecretStr | None = None
    mt5_server: str | None = None
    mt5_path: str | None = None


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    with open(path, "rb") as fh:
        return AppConfig.model_validate(tomllib.load(fh))


def load_settings() -> Settings:
    return Settings()


def with_account_overrides(
    app: AppConfig,
    balance: float | None = None,
    lot_scale: float | None = None,
    risk_sizing: bool | None = None,
    max_lots: float | None = None,
    risk_pct: float | None = None,
) -> AppConfig:
    """Copy of the config with command-line account/risk overrides applied (None = keep)."""
    acct = {
        k: v
        for k, v in {
            "initial_balance": balance,
            "lot_value_scale": lot_scale,
            "risk_sizing": risk_sizing,
            "max_lots": max_lots,
        }.items()
        if v is not None
    }
    out = app.model_copy(update={"account": app.account.model_copy(update=acct)})
    if risk_pct is not None:
        out = out.model_copy(
            update={"risk": out.risk.model_copy(update={"max_risk_per_trade_pct": risk_pct})}
        )
    return out

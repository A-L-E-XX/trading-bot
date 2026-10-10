import pytest
from pydantic import ValidationError

import tradingbot
from tradingbot.config import AppConfig, Settings, load_config


def test_version():
    assert tradingbot.__version__ == "0.1.0"


def test_default_config_loads_and_matches_decisions():
    cfg = load_config()
    assert cfg.account.initial_balance == 100.0
    assert cfg.account.fixed_lot == 0.01
    assert cfg.account.max_drawdown_pct == 30.0
    assert cfg.timeframes.primary == ["1h", "4h", "1d"]
    assert "BTCUSD" in cfg.instruments.symbols
    assert "XAUUSD" in cfg.instruments.symbols
    assert (
        len(cfg.instruments.symbols) == 11
    )  # BTC + XAU + 5 majors + NZDUSD, USDCHF, XAGUSD, ETHUSD


def test_broker_symbol_suffix():
    cfg = load_config()
    cfg.instruments.symbol_suffix = ".m"
    assert cfg.instruments.broker_symbol("EURUSD") == "EURUSD.m"


def test_bad_ema_ordering_rejected():
    data = load_config().model_dump()
    data["strategy_v1"]["ema_fast"] = 100
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)


def test_kill_switch_cannot_exceed_max_drawdown():
    data = load_config().model_dump()
    data["risk"]["kill_switch_drawdown_pct"] = 50.0
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)


def test_live_mode_is_not_allowed(monkeypatch):
    monkeypatch.setenv("TB_MODE", "live")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_secrets_never_in_repr(monkeypatch):
    monkeypatch.setenv("TB_MT5_PASSWORD", "hunter2")
    s = Settings(_env_file=None)
    assert "hunter2" not in repr(s)
    assert "hunter2" not in str(s)
    assert s.mt5_password.get_secret_value() == "hunter2"


def test_blank_env_values_are_treated_as_unset(tmp_path):
    """Regression: a copied .env.example has blank TB_MT5_LOGIN= and must still load."""
    env = tmp_path / ".env"
    env.write_text("TB_MODE=backtest  # comment\nTB_MT5_LOGIN=\nTB_MT5_PASSWORD=\nTB_MT5_SERVER=\n")
    s = Settings(_env_file=env)
    assert s.mt5_login is None
    assert s.mt5_password is None
    assert s.mode == "backtest"


def test_example_env_file_loads():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / ".env.example"
    assert Settings(_env_file=example).mt5_login is None

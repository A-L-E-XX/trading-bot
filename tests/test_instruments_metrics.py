from pathlib import Path

import pandas as pd
import pytest

from bt_helpers import EUR, JPY, XAU
from tradingbot.backtesting.costs import CostModel
from tradingbot.backtesting.instruments import load_specs
from tradingbot.backtesting.metrics import compute_metrics, max_drawdown
from tradingbot.config import load_config

SPECS = Path(__file__).resolve().parents[1] / "config" / "instrument_specs.json"


def test_quote_to_usd_conversion():
    assert EUR.quote_to_usd(1.1) == 1.0  # profit already in USD
    assert JPY.quote_to_usd(150.0) == pytest.approx(1 / 150.0)  # USDJPY profit is in JPY
    odd = EUR.__class__("EURGBP", 1e5, 1e-5, 5, 0.01, 0.01, 10, "EUR", "GBP")
    with pytest.raises(NotImplementedError):
        odd.quote_to_usd(0.85)


def test_spread_and_lot_rules():
    assert EUR.spread_price() == pytest.approx(0.0001)
    assert EUR.spread_price(3.0) == pytest.approx(0.0003)
    assert XAU.spread_price() == pytest.approx(0.24)
    EUR.check_lot(0.01)
    EUR.check_lot(0.05)
    for bad in (0.005, 0.015, 0.0):
        with pytest.raises(ValueError):
            EUR.check_lot(bad)


def test_real_broker_specs_load():
    specs = load_specs(SPECS)
    assert set(specs) == set(load_config().instruments.symbols)
    assert specs["EURUSD"].contract_size == 100000 and specs["EURUSD"].spread_points == 8
    assert specs["XAUUSD"].contract_size == 100 and specs["BTCUSD"].contract_size == 1
    assert specs["USDJPY"].currency_profit == "JPY"
    for spec in specs.values():
        spec.check_lot(load_config().account.fixed_lot)  # 0.01 lot is valid everywhere
    with pytest.raises(FileNotFoundError, match="tb-data specs"):
        load_specs(SPECS.parent / "missing.json")


def test_cost_model():
    m = CostModel(spread_multiplier=2.0, slippage_spread_fraction=0.25)
    assert m.spread(EUR) == pytest.approx(0.0002)
    assert m.slippage(EUR) == pytest.approx(0.00005)
    assert (
        m.with_overrides(spread_multiplier=None, commission_per_lot=7.0).commission_per_lot == 7.0
    )
    assert CostModel.from_config(load_config().costs).slippage_spread_fraction == 0.25


def test_max_drawdown_hand_example():
    eq = pd.Series([100.0, 110.0, 99.0, 105.0, 80.0, 120.0])
    usd, frac = max_drawdown(eq)
    assert usd == pytest.approx(30.0) and frac == pytest.approx(30 / 110)


def test_compute_metrics_hand_example():
    trades = pd.DataFrame({"pnl_usd": [10.0, -5.0, 15.0, -10.0], "bars_held": [2, 3, 4, 1]})
    idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    equity = pd.Series([100.0] * 50 + [110.0] * 50, index=idx)
    m = compute_metrics(trades, equity, 100.0, 30.0)
    assert m["trades"] == 4 and m["win_rate"] == 0.5
    assert m["net_profit_usd"] == 10.0
    assert m["profit_factor"] == pytest.approx(25 / 15)
    assert m["expectancy_usd"] == pytest.approx(2.5)
    assert m["avg_win_usd"] == 12.5 and m["avg_loss_usd"] == -7.5
    assert m["largest_loss_usd"] == -10.0 and m["return_pct"] == pytest.approx(10.0)
    assert m["max_drawdown_pct"] == 0.0 and m["breached_max_drawdown"] is False
    assert m["time_in_market_pct"] == pytest.approx(10.0)
    assert m["ruined"] is False


def test_drawdown_breach_and_ruin_flags():
    idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    m = compute_metrics(pd.DataFrame({"pnl_usd": [-71.0], "bars_held": [3]}),
                        pd.Series([100.0, 60.0, 29.0, 0.0], index=idx), 100.0, 30.0)  # fmt: skip
    assert m["breached_max_drawdown"] is True and m["ruined"] is True


def test_metrics_with_no_trades():
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    m = compute_metrics(pd.DataFrame(), pd.Series([100.0] * 5, index=idx), 100.0, 30.0)
    assert m["trades"] == 0 and m["profit_factor"] is None and m["return_pct"] == 0.0

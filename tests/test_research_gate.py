import math

import numpy as np

from tradingbot.backtesting.costs import CostModel
from tradingbot.backtesting.instruments import load_specs
from tradingbot.backtesting.runner import DEFAULT_SPECS_PATH
from tradingbot.config import load_config
from tradingbot.data.collectors import SyntheticCollector
from tradingbot.data.pipeline import build_dataset
from tradingbot.data.storage import DatasetStore
from tradingbot.research import evaluate_gate, monte_carlo
from tradingbot.research.cli import main as research_main
from tradingbot.research.gate import CHECKS, profit_factor


def test_profit_factor_edge_cases():
    base = {"trades": 2, "gross_profit_usd": 6.0, "gross_loss_usd": 3.0}
    assert profit_factor(base) == 2.0
    assert profit_factor({**base, "gross_loss_usd": 0.0}) == math.inf
    assert profit_factor({"trades": 0}) is None


def test_monte_carlo_is_seeded_and_needs_enough_trades():
    pnl = [1.0, -1.0, 2.0, -3.0, 0.5, 1.5, -0.5, 1.0]
    assert monte_carlo(pnl, 100, 30) == monte_carlo(pnl, 100, 30)
    assert monte_carlo(pnl[:4], 100, 30) is None


def test_monte_carlo_all_winners_never_breach_and_big_losers_always_do():
    win = monte_carlo(np.full(20, 1.0), 100, 30)
    assert win["mc_breach_probability_pct"] == 0 and win["mc_ruin_probability_pct"] == 0
    lose = monte_carlo(np.full(20, -10.0), 100, 30)
    assert lose["mc_breach_probability_pct"] == 100 and lose["mc_ruin_probability_pct"] == 100


def test_gate_is_deterministic_and_reports_every_check(make_ohlcv):
    app, specs = load_config(), load_specs(DEFAULT_SPECS_PATH)
    df = make_ohlcv("EURUSD", "1h", seed=5)
    args = (df, specs["EURUSD"], app, CostModel.from_config(app.costs), "EURUSD", "1h")
    a, b = evaluate_gate(*args, mc_sims=200), evaluate_gate(*args, mc_sims=200)
    assert a.row() == b.row()
    assert set(a.checks) == set(CHECKS)


def test_no_trades_means_fail_not_pass(make_ohlcv):
    app, specs = load_config(), load_specs(DEFAULT_SPECS_PATH)
    flat = make_ohlcv("EURUSD", "1d", seed=1).iloc[:300]  # too short to be ready/trade much
    res = evaluate_gate(flat, specs["EURUSD"], app, CostModel.from_config(app.costs),
                        "EURUSD", "1d", mc_sims=100)  # fmt: skip
    assert not res.passed


def test_research_cli_writes_report(tmp_path, start, end):
    data = tmp_path / "data"
    build_dataset(SyntheticCollector(seed=3), "EURUSD", "4h", start, end, DatasetStore(data))
    out = tmp_path / "out"
    code = research_main(["--symbols", "EURUSD", "--timeframes", "4h", "--data-dir", str(data),
                          "--out-dir", str(out), "--mc-sims", "100"])  # fmt: skip
    assert code == 0
    for name in ("gate.csv", "gate.json", "gate_report.md"):
        assert (out / name).exists()

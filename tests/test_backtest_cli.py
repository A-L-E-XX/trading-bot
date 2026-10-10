"""End to end: stored data -> tb-backtest -> reproducible output files."""

import json

import pandas as pd

from tradingbot.backtesting.cli import main
from tradingbot.data.collectors import SyntheticCollector
from tradingbot.data.pipeline import build_dataset
from tradingbot.data.storage import DatasetStore


def _prepare(tmp_path, start, end):
    data = tmp_path / "data"
    build_dataset(SyntheticCollector(seed=3), "EURUSD", "4h", start, end, DatasetStore(data))
    return data


def _run(tmp_path, data, out, *extra):
    return main(["--symbols", "EURUSD", "--timeframes", "4h", "--data-dir", str(data),
                 "--out-dir", str(out), "--no-plots", *extra])  # fmt: skip


def test_cli_writes_all_outputs_and_is_deterministic(tmp_path, start, end):
    data = _prepare(tmp_path, start, end)
    assert _run(tmp_path, data, tmp_path / "a") == 0
    assert _run(tmp_path, data, tmp_path / "b") == 0
    folder = tmp_path / "a" / "EURUSD_4h"
    for name in ("trades.csv", "signals.csv", "equity.csv", "summary.json"):
        assert (folder / name).exists()
    assert (tmp_path / "a" / "report.md").exists() and (tmp_path / "a" / "summary.csv").exists()
    for name in ("trades.csv", "signals.csv", "equity.csv", "summary.json"):
        assert (folder / name).read_bytes() == (tmp_path / "b" / "EURUSD_4h" / name).read_bytes()
    meta = json.loads((folder / "summary.json").read_text())
    assert meta["data"]["sha256"] == DatasetStore(data).metadata("EURUSD", "4h")["sha256"]


def test_spread_stress_never_helps(tmp_path, start, end):
    data = _prepare(tmp_path, start, end)
    _run(tmp_path, data, tmp_path / "base")
    _run(tmp_path, data, tmp_path / "stress", "--spread-multiplier", "3")
    base = pd.read_csv(tmp_path / "base" / "summary.csv").iloc[0]
    stress = pd.read_csv(tmp_path / "stress" / "summary.csv").iloc[0]
    assert stress["total_cost_usd"] >= base["total_cost_usd"]


def test_cli_fails_cleanly_without_data(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _run(tmp_path, empty, tmp_path / "o") == 1

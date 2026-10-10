"""Run backtests on stored datasets and write reproducible outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from tradingbot import __version__  # noqa: E402
from tradingbot.backtesting.costs import CostModel  # noqa: E402
from tradingbot.backtesting.engine import BacktestConfig, BacktestResult, run_backtest  # noqa: E402
from tradingbot.backtesting.instruments import InstrumentSpec  # noqa: E402
from tradingbot.config import DEFAULT_CONFIG_PATH, AppConfig  # noqa: E402
from tradingbot.data.storage import DatasetStore  # noqa: E402
from tradingbot.features import FeatureParams, compute_features  # noqa: E402
from tradingbot.strategies import EmaTrendAtrStrategy  # noqa: E402

DEFAULT_SPECS_PATH = DEFAULT_CONFIG_PATH.parent / "instrument_specs.json"

# Chart colours: the reference palette's first categorical slot on its light surface.
_SURFACE, _INK, _INK_2, _GRID, _LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"

KEY_COLUMNS = [
    "symbol",
    "timeframe",
    "trades",
    "win_rate",
    "net_profit_usd",
    "return_pct",
    "profit_factor",
    "max_drawdown_pct",
    "breached_max_drawdown",
    "largest_loss_usd",
    "trades_over_risk_limit",
    "max_stop_risk_pct",
    "total_cost_usd",
    "ruined",
]


@dataclass
class RunOutput:
    result: BacktestResult
    meta: dict


def run_one(
    store: DatasetStore,
    symbol: str,
    timeframe: str,
    app: AppConfig,
    spec: InstrumentSpec,
    costs: CostModel,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> RunOutput:
    """Load (checksum-verified) data, compute features on the full history, replay, report."""
    ohlcv = store.load(symbol, timeframe, verify=True)
    features = compute_features(ohlcv, FeatureParams.from_config(app.strategy_v1))
    window = features
    if start is not None:
        window = window[window.index >= start]
    if end is not None:
        window = window[window.index < end]
    if len(window) < 3:
        raise ValueError(f"{symbol} {timeframe}: fewer than 3 candles in the requested window")

    config = BacktestConfig(
        initial_balance=app.account.initial_balance,
        lots=app.account.fixed_lot,
        costs=costs,
        max_drawdown_limit_pct=app.account.max_drawdown_pct,
        max_risk_per_trade_pct=app.risk.max_risk_per_trade_pct,
    )
    result = run_backtest(
        window, EmaTrendAtrStrategy(app.strategy_v1), spec, config, symbol, timeframe
    )
    stored = store.metadata(symbol, timeframe)
    meta = {
        "symbol": symbol,
        "timeframe": timeframe,
        "tradingbot_version": __version__,
        "data": {"sha256": stored["sha256"], "source": stored["source"], "rows": stored["rows"]},
        "window": {
            "first_candle": str(window.index[0]),
            "last_candle": str(window.index[-1]),
            "candles": len(window),
        },  # fmt: skip
        "strategy": app.strategy_v1.model_dump(),
        "account": {"initial_balance": config.initial_balance, "lots": config.lots},
        "costs": {
            "spread_multiplier": costs.spread_multiplier,
            "slippage_spread_fraction": costs.slippage_spread_fraction,
            "commission_per_lot": costs.commission_per_lot,
            "spread_price": costs.spread(spec),
            "slippage_price": costs.slippage(spec),
        },
        "halted_reason": result.halted_reason,
        "ignored_entry_signals": result.ignored_entry_signals,
        "metrics": result.metrics,
    }
    return RunOutput(result, meta)


def plot_equity(result: BacktestResult, initial_balance: float, path: Path) -> None:
    """Equity (top) and drawdown (bottom): two measures, two panels, one shared time axis."""
    eq = result.equity
    peak = eq.cummax()
    dd = (peak - eq) / peak.where(peak > 0) * 100
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(9, 5.2), sharex=True, height_ratios=[3, 1.4], facecolor=_SURFACE
    )
    for ax in (a1, a2):
        ax.set_facecolor(_SURFACE)
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(_GRID)
        ax.tick_params(colors=_INK_2, labelsize=8, length=0)
    a1.plot(eq.index, eq.values, color=_LINE, linewidth=1.6)
    a1.axhline(initial_balance, color=_INK_2, linewidth=0.8, linestyle=(0, (4, 3)))
    a1.annotate(f"start ${initial_balance:,.0f}", (eq.index[0], initial_balance), color=_INK_2,
                fontsize=8, xytext=(4, 4), textcoords="offset points")  # fmt: skip
    a1.set_ylabel("Equity (USD)", color=_INK_2, fontsize=9)
    m = result.metrics
    a1.set_title(
        f"{result.symbol} {result.timeframe}  |  {m['trades']} trades, "
        f"net ${m['net_profit_usd']:,.2f}, max drawdown {m['max_drawdown_pct']:.1f}%",
        loc="left", color=_INK, fontsize=10.5, fontweight="bold",
    )  # fmt: skip
    a2.fill_between(dd.index, dd.values, 0, color=_INK_2, alpha=0.35, linewidth=0)
    a2.plot(dd.index, dd.values, color=_INK_2, linewidth=1.0)
    a2.invert_yaxis()
    a2.set_ylabel("Drawdown (%)", color=_INK_2, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=_SURFACE)
    plt.close(fig)


def write_run(out_dir: Path, run: RunOutput, plots: bool = True) -> Path:
    r, meta = run.result, run.meta
    folder = out_dir / f"{r.symbol}_{r.timeframe}"
    folder.mkdir(parents=True, exist_ok=True)
    r.trades.to_csv(folder / "trades.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    r.signals.to_csv(folder / "signals.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    r.equity.to_csv(folder / "equity.csv", date_format="%Y-%m-%dT%H:%M:%SZ")
    (folder / "summary.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    if plots:
        plot_equity(r, meta["account"]["initial_balance"], folder / "equity.png")
    return folder


def summary_table(runs: list[RunOutput]) -> pd.DataFrame:
    rows = []
    for run in runs:
        row = {k: run.meta["metrics"].get(k) for k in KEY_COLUMNS[2:]}
        rows.append({"symbol": run.meta["symbol"], "timeframe": run.meta["timeframe"], **row})
    return pd.DataFrame(rows)[KEY_COLUMNS]


_SHORT_NAMES = {
    "win_rate": "win_%",
    "net_profit_usd": "net_$",
    "return_pct": "ret_%",
    "max_drawdown_pct": "maxDD_%",
    "breached_max_drawdown": "DD>limit",
    "largest_loss_usd": "worst_$",
    "trades_over_risk_limit": "over_risk",
    "max_stop_risk_pct": "max_risk_%",
    "total_cost_usd": "costs_$",
}

_HOW_TO_READ = (
    "How to read it: each row is ONE symbol and timeframe with its own account. `over_risk` counts "
    "trades whose stop risked more than the per-trade risk limit (not enforced here; the risk "
    "engine, Milestone 8, will reject them). Results are in-sample and use one cost snapshot, so "
    "they are not evidence of an edge until the research gate (Milestone 7) is passed."
)


def write_report(out_dir: Path, runs: list[RunOutput]) -> pd.DataFrame:
    table = summary_table(runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "summary.csv", index=False)
    meta = runs[0].meta
    show = table.copy()
    show["win_rate"] = (show["win_rate"] * 100).round(0)
    show = show.rename(columns=_SHORT_NAMES).round(2)
    shown = show.to_markdown(index=False) if _has_tabulate() else show.to_string(index=False)
    costs = meta["costs"]
    text = [
        "# Backtest report",
        "",
        f"Strategy `{meta['strategy']['name']}` v{meta['strategy']['version']}, "
        f"account ${meta['account']['initial_balance']:,.0f}, fixed {meta['account']['lots']} lot, "
        f"spread x{costs['spread_multiplier']}, "
        f"slippage {costs['slippage_spread_fraction']} x spread per fill.",
        "",
        shown,
        "",
        _HOW_TO_READ,
    ]
    (out_dir / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return table


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True

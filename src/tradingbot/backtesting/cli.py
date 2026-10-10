"""Command line: ``uv run tb-backtest`` - a reproducible backtest from one command."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from tradingbot.backtesting.costs import CostModel
from tradingbot.backtesting.instruments import load_specs
from tradingbot.backtesting.runner import (
    DEFAULT_SPECS_PATH,
    run_one,
    summary_table,
    write_report,
    write_run,
)
from tradingbot.config import load_config, load_settings, with_account_overrides
from tradingbot.data.storage import DatasetIntegrityError, DatasetStore


def _ts(text: str | None) -> pd.Timestamp | None:
    return pd.Timestamp(text, tz="UTC") if text else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tb-backtest", description=__doc__)
    p.add_argument("--symbols", nargs="+", help="default: all symbols in config")
    p.add_argument("--timeframes", nargs="+", choices=["1h", "4h", "1d"], help="default: all")
    p.add_argument(
        "--start", help="UTC date; only trade from here (features still use earlier data)"
    )
    p.add_argument("--end", help="UTC date (exclusive)")
    p.add_argument("--spread-multiplier", type=float, help="stress test: 2.0 doubles the spread")
    p.add_argument(
        "--slippage-fraction", type=float, help="slippage per fill, as a fraction of spread"
    )
    p.add_argument("--commission-per-lot", type=float)
    p.add_argument("--data-dir", help="dataset folder (default: TB_DATA_DIR / data_store)")
    p.add_argument("--specs", default=str(DEFAULT_SPECS_PATH), help="broker contract specs JSON")
    p.add_argument("--out-dir", default="outputs/backtests", help="where trades/equity/report go")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--enforce-risk",
        action="store_true",
        help="skip entries whose stop risks more than risk.max_risk_per_trade_pct",
    )
    p.add_argument("--balance", type=float, help="starting balance, e.g. 20")
    p.add_argument(
        "--lot-scale", type=float, help="value of one lot vs the spec (cent account: 0.01)"
    )
    p.add_argument(
        "--risk-sizing", action="store_true", help="size each trade by risk %% of balance"
    )
    p.add_argument("--risk-pct", type=float, help="per-trade risk limit in %% of balance")
    p.add_argument("--max-lots", type=float, help="cap on lots per trade when sizing by risk")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app, settings = load_config(), load_settings()
    app = with_account_overrides(
        app,
        args.balance,
        args.lot_scale,
        True if args.risk_sizing else None,
        args.max_lots,
        args.risk_pct,
    )
    store = DatasetStore(args.data_dir or settings.data_dir)
    specs = load_specs(args.specs)
    costs = CostModel.from_config(app.costs).with_overrides(
        spread_multiplier=args.spread_multiplier,
        slippage_spread_fraction=args.slippage_fraction,
        commission_per_lot=args.commission_per_lot,
    )
    symbols = args.symbols or app.instruments.symbols
    timeframes = args.timeframes or app.timeframes.primary
    out_dir = Path(args.out_dir)

    runs, failures = [], 0
    for symbol in symbols:
        for tf in timeframes:
            t0 = time.time()
            try:
                run = run_one(
                    store,
                    symbol,
                    tf,
                    app,
                    specs[symbol],
                    costs,
                    _ts(args.start),
                    _ts(args.end),
                    enforce_risk_limit=args.enforce_risk,
                )
            except (FileNotFoundError, KeyError, ValueError, DatasetIntegrityError) as exc:
                failures += 1
                print(f"FAILED {symbol} {tf}: {exc}", file=sys.stderr)
                continue
            write_run(out_dir, run, plots=not args.no_plots)
            runs.append(run)
            m = run.meta["metrics"]
            print(f"{symbol} {tf}: {m['trades']} trades, net ${m['net_profit_usd']:.2f}, "
                  f"maxDD {m['max_drawdown_pct']:.1f}% ({time.time() - t0:.1f}s)")  # fmt: skip
    if not runs:
        print("Nothing was run.", file=sys.stderr)
        return 1
    table = write_report(out_dir, runs)
    print("\n" + summary_table(runs).round(2).to_string(index=False))
    print(f"\nWrote {len(table)} runs to {out_dir}/ (report.md, summary.csv, one folder per run)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

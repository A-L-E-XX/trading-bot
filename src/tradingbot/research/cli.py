"""Command line: ``uv run tb-research`` - run the research gate on stored datasets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from tradingbot.backtesting.costs import CostModel
from tradingbot.backtesting.instruments import get_spec, load_specs
from tradingbot.backtesting.runner import DEFAULT_SPECS_PATH
from tradingbot.config import load_config, load_settings, with_account_overrides
from tradingbot.data.storage import DatasetIntegrityError, DatasetStore
from tradingbot.research.gate import CHECKS, GateResult, evaluate_gate

_NOTE = (
    "How to read it: a candidate passes only if EVERY check passes. Parameters were fixed before "
    "the out-of-sample period was examined and the per-trade risk limit is enforced. "
    "{n} candidates were tested, so a few passes could still be luck: treat a pass as "
    "'worth paper trading', not as proof. If nothing passes, Strategy V1 does not continue."
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tb-research", description=__doc__)
    p.add_argument("--symbols", nargs="+")
    p.add_argument("--timeframes", nargs="+", choices=["1h", "4h", "1d"])
    p.add_argument("--data-dir")
    p.add_argument("--specs", default=str(DEFAULT_SPECS_PATH))
    p.add_argument("--out-dir", default="outputs/research")
    p.add_argument("--oos-fraction", type=float, default=0.4, help="share of history held out")
    p.add_argument("--mc-sims", type=int, default=2000)
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


def write_report(out_dir: Path, results: list[GateResult], app) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([r.row() for r in results])
    table.to_csv(out_dir / "gate.csv", index=False)
    (out_dir / "gate.json").write_text(
        json.dumps([r.row() for r in results], indent=2, default=str) + "\n", encoding="utf-8"
    )
    g = app.research_gate
    show = table[
        ["symbol", "timeframe", "PASS", "oos_trades", "oos_profit_factor", "oos_net_usd",
         "oos_drawdown_pct", "worst_param_drop_pct", "stress2x_profit_factor",
         "mc_breach_probability_pct", "profitable_folds_pct"]
    ].round(2)  # fmt: skip
    fails = {c: int((~table[f"ok_{c}"]).sum()) for c in CHECKS}
    text = [
        "# Research gate report",
        "",
        f"Strategy `{app.strategy_v1.name}`: EMA {app.strategy_v1.ema_fast}/"
        f"{app.strategy_v1.ema_slow}/{app.strategy_v1.ema_trend}, stop "
        f"{app.strategy_v1.atr_stop_mult} x ATR, risk limit {app.risk.max_risk_per_trade_pct}% "
        f"per trade (enforced), account ${app.account.initial_balance:,.0f}, "
        f"{app.account.fixed_lot} lot.",
        "",
        "Thresholds: OOS profit factor >= "
        f"{g.min_oos_profit_factor}, OOS trades >= {g.min_oos_trades}, OOS drawdown <= "
        f"{g.max_oos_drawdown_pct}%, worst parameter drop <= {g.max_param_sensitivity_drop_pct}%, "
        f"profit factor at 2x spread >= {g.min_stress_profit_factor}, Monte Carlo drawdown-breach "
        f"probability <= {g.max_mc_breach_probability_pct}%, profitable folds >= "
        f"{g.min_profitable_folds_pct}%.",
        "",
        f"**{int(table['PASS'].sum())} of {len(table)} candidates passed.**",
        "",
        "Checks failed (count of candidates): " + ", ".join(f"{c} {n}" for c, n in fails.items()),
        "",
        show.to_markdown(index=False),
        "",
        _NOTE.format(n=len(table)),
    ]
    (out_dir / "gate_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return table


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
    costs = CostModel.from_config(app.costs)
    symbols = args.symbols or app.instruments.symbols
    timeframes = args.timeframes or app.timeframes.primary
    results, failures = [], 0
    for symbol in symbols:
        for tf in timeframes:
            t0 = time.time()
            try:
                ohlcv = store.load(symbol, tf, verify=True)
                res = evaluate_gate(
                    ohlcv, get_spec(specs, symbol), app, costs, symbol, tf,
                    oos_fraction=args.oos_fraction, mc_sims=args.mc_sims,
                )  # fmt: skip
            except (FileNotFoundError, KeyError, ValueError, DatasetIntegrityError) as exc:
                failures += 1
                print(f"FAILED {symbol} {tf}: {exc}", file=sys.stderr)
                continue
            results.append(res)
            verdict = "PASS" if res.passed else "fail"
            oos = res.details["oos_trades"]
            print(f"{symbol} {tf}: {verdict} (OOS {oos} trades) ({time.time() - t0:.1f}s)")
    if not results:
        print("Nothing was run.", file=sys.stderr)
        return 1
    table = write_report(Path(args.out_dir), results, app)
    print(f"\n{int(table['PASS'].sum())} of {len(table)} candidates passed. "
          f"Report: {args.out_dir}/gate_report.md")  # fmt: skip
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line: ``uv run --extra mt5 tb-paper run`` - paper trading on live DEMO candles.

run      connect to your MT5 DEMO account, then paper trade one strategy on one symbol/timeframe
status   show the saved state of a paper session
Read-only: it never sends an order to the broker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from tradingbot.backtesting.costs import CostModel
from tradingbot.backtesting.instruments import get_spec, load_specs
from tradingbot.backtesting.runner import DEFAULT_SPECS_PATH
from tradingbot.config import load_config, load_settings, with_account_overrides
from tradingbot.data.collectors import MT5Collector, MT5Error
from tradingbot.data.storage import DatasetStore
from tradingbot.execution.live import (
    FeedError,
    closed_candles,
    run_session,
    saved_last_bar,
    seed_history,
)
from tradingbot.execution.paper import PaperAccount, PaperEngine
from tradingbot.risk import RiskLimits
from tradingbot.strategies import get_strategy_def


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tb-paper", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="paper trade on live demo candles")
    r.add_argument("--strategy", choices=["v1", "v2"], required=True)
    r.add_argument("--symbol", required=True)
    r.add_argument("--timeframe", choices=["1h", "4h", "1d"], required=True)
    r.add_argument("--balance", type=float, help="starting paper balance (default: config)")
    r.add_argument("--lot-scale", type=float, help="cent account: 0.01")
    r.add_argument("--risk-sizing", action="store_true")
    r.add_argument("--risk-pct", type=float)
    r.add_argument("--max-lots", type=float)
    r.add_argument("--name", help="session name (default strategy_symbol_timeframe)")
    r.add_argument("--out-dir", default="outputs/paper")
    r.add_argument("--data-dir")
    r.add_argument("--specs", default=str(DEFAULT_SPECS_PATH))
    r.add_argument("--poll-seconds", type=float, default=60.0)
    r.add_argument("--once", action="store_true", help="process available candles once and exit")
    r.add_argument(
        "--server-utc-offset",
        type=float,
        default=0.0,
        help="MT5 server time minus UTC in hours; use the SAME value as for tb-data download",
    )
    r.add_argument("--server-timezone")
    s = sub.add_parser("status", help="show a saved session")
    s.add_argument("--name", required=True)
    s.add_argument("--out-dir", default="outputs/paper")
    return p


def cmd_status(args) -> int:
    path = Path(args.out_dir) / args.name / "state.json"
    if not path.exists():
        log = path.with_name("events.jsonl")
        if log.exists():
            print(f"Session started but no candle has been processed yet (log: {log}).")
            return 0
        print(f"No session found at {path.parent} (it has never been started).", file=sys.stderr)
        return 1
    st = json.loads(path.read_text(encoding="utf-8"))
    wins = sum(1 for t in st["trades"] if t["pnl_usd"] > 0)
    print(f"{st['symbol']} {st['timeframe']}: last candle {st['last_bar_time']}, "
          f"balance ${st['balance']:.2f}, {len(st['trades'])} trades ({wins} winners), "
          f"halted: {st['halted']}")  # fmt: skip
    print("open position:", json.dumps(st["pos"]) if st["pos"] else "none")
    if st["risk"]:
        r = st["risk"]
        print(f"equity ${r['equity']:.2f} (peak ${r['peak_equity']:.2f}), "
              f"kill switch: {r['kill_switch']}, daily halt: {r['daily_halt']}")  # fmt: skip
    return 0


def cmd_run(args) -> int:
    app, settings = load_config(), load_settings()
    app = with_account_overrides(
        app, args.balance, args.lot_scale, True if args.risk_sizing else None, args.max_lots,
        args.risk_pct,
    )  # fmt: skip
    sdef = get_strategy_def(args.strategy)
    specs = load_specs(args.specs)
    spec = get_spec(specs, args.symbol)
    name = args.name or f"{args.strategy}_{args.symbol}_{args.timeframe}"
    folder = Path(args.out_dir) / name
    state_path = folder / "state.json"

    col = MT5Collector(
        login=settings.mt5_login,
        password=settings.mt5_password.get_secret_value() if settings.mt5_password else None,
        server=settings.mt5_server,
        path=settings.mt5_path,
        symbol_suffix=app.instruments.symbol_suffix,
        server_utc_offset_hours=args.server_utc_offset,
        server_timezone=args.server_timezone,
    )
    try:
        col.connect()  # refuses anything that is not a demo account
        recent = closed_candles(
            col.fetch_recent(args.symbol, args.timeframe, 1500),
            args.timeframe,
            pd.Timestamp.now(tz="UTC"),
        )
        store = DatasetStore(args.data_dir or settings.data_dir)
        stored = (
            store.load(args.symbol, args.timeframe)
            if store.exists(args.symbol, args.timeframe)
            else None
        )
        resume_at = saved_last_bar(state_path)
        history = seed_history(stored, recent, up_to=resume_at)
        if resume_at is not None and history.index[-1] != resume_at:
            raise FeedError(
                f"cannot resume: saved session ended at {resume_at} but the data ends at "
                f"{history.index[-1]}. Download recent data (tb-data download) and retry."
            )
        acct = app.account
        engine = PaperEngine(
            sdef, sdef.params(app), spec, CostModel.from_config(app.costs),
            PaperAccount(acct.initial_balance, acct.fixed_lot, acct.lot_value_scale,
                         acct.risk_sizing, acct.max_lots),
            RiskLimits.from_config(app.risk), args.symbol, args.timeframe, history=history,
            state_path=state_path, event_log_path=folder / "events.jsonl",
        )  # fmt: skip
        print(f"Paper session '{name}' on a DEMO account: {args.strategy} {args.symbol} "
              f"{args.timeframe}, balance ${engine.balance:.2f}, "
              f"last candle {engine.last_bar_time}. Ctrl+C to stop.")  # fmt: skip
        engine.note(
            "session_started", engine.last_bar_time, strategy=args.strategy,
            timeframe=args.timeframe, balance=engine.balance, once=bool(args.once),
        )  # fmt: skip
        try:
            result = run_session(col, engine, poll_seconds=args.poll_seconds,
                                 max_polls=1 if args.once else None)  # fmt: skip
        except KeyboardInterrupt:
            print("\nStopped by you. State is saved; run the same command to resume.")
            return 0
        print(f"Session ended: {result.stopped_because}. Candles processed: {result.processed}, "
              f"feed errors: {result.errors}. Trades so far: {len(engine.trades)}, "
              f"balance ${engine.balance:.2f}.")  # fmt: skip
        return 0 if result.errors == 0 or result.stopped_because == "max_polls" else 1
    except (MT5Error, FeedError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        col.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_status(args) if args.cmd == "status" else cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())

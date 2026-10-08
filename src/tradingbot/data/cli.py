"""Command line: ``uv run tb-data <command>``.

mt5-check   connect to your MT5 DEMO account, list matching symbols, estimate the server offset
download    build datasets from mt5 | csv | synthetic
validate    re-validate stored datasets and verify checksums
info        list stored datasets
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from tradingbot.config import load_config, load_settings
from tradingbot.data.collectors import CSVCollector, MT5Collector, MT5Error, SyntheticCollector
from tradingbot.data.pipeline import DataValidationError, build_dataset, validate_stored
from tradingbot.data.storage import DatasetIntegrityError, DatasetStore


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def _mt5_collector(args, cfg, settings) -> MT5Collector:
    return MT5Collector(
        login=settings.mt5_login,
        password=settings.mt5_password.get_secret_value() if settings.mt5_password else None,
        server=settings.mt5_server,
        path=settings.mt5_path,
        symbol_suffix=cfg.instruments.symbol_suffix,
        server_utc_offset_hours=getattr(args, "server_utc_offset", 0.0),
        server_timezone=getattr(args, "server_timezone", None),
    )


def cmd_mt5_check(args, cfg, settings) -> int:
    col = _mt5_collector(args, cfg, settings)
    try:
        col.connect()
        info = col.account_info()
        print(f"Connected: login={info.login} server={info.server} currency={info.currency} (DEMO)")
        print("\nSymbols (our name -> what your broker calls it):")
        for ours, found in col.find_symbols(cfg.instruments.symbols).items():
            print(f"  {ours:8} -> {', '.join(found) if found else 'NOT FOUND'}")
        offset = col.estimate_server_offset_hours(cfg.instruments.symbols[2])
        print(
            f"\nEstimated server time offset from UTC: {offset} hours "
            "(reliable only while markets are open)"
        )
        print(
            "If broker names have a suffix (e.g. EURUSD.m), "
            "set symbol_suffix in config/default.toml."
        )
        return 0
    except MT5Error as exc:
        print(f"MT5 error: {exc}", file=sys.stderr)
        return 1
    finally:
        col.close()


def cmd_download(args, cfg, settings) -> int:
    symbols = args.symbols or cfg.instruments.symbols
    timeframes = args.timeframes or cfg.timeframes.primary
    start, end = _utc(args.start), _utc(args.end) if args.end else datetime.now(UTC)
    store = DatasetStore(args.data_dir or settings.data_dir)
    if args.source == "synthetic":
        collector = SyntheticCollector(seed=args.seed)
    elif args.source == "csv":
        if not args.csv_dir:
            print("--csv-dir is required for --source csv", file=sys.stderr)
            return 2
        collector = CSVCollector(args.csv_dir, args.server_utc_offset, args.server_timezone)
    else:
        collector = _mt5_collector(args, cfg, settings)
        if args.server_utc_offset == 0.0 and not args.server_timezone:
            print(
                "NOTE: assuming broker server time == UTC. Run `tb-data mt5-check` and pass "
                "--server-utc-offset (or --server-timezone) if it differs."
            )
    failures = 0
    try:
        for symbol in symbols:
            for tf in timeframes:
                try:
                    result = build_dataset(
                        collector, symbol, tf, start, end, store, allow_invalid=args.allow_invalid
                    )
                    print(result.report.summary())
                except (DataValidationError, MT5Error, FileNotFoundError, ValueError) as exc:
                    failures += 1
                    print(f"FAILED {symbol} {tf}: {exc}", file=sys.stderr)
    finally:
        if isinstance(collector, MT5Collector):
            collector.close()
    print(f"\nDone. {failures} failure(s). Data directory: {store.root}")
    return 1 if failures else 0


def cmd_validate(args, cfg, settings) -> int:
    store = DatasetStore(args.data_dir or settings.data_dir)
    datasets = store.list_datasets()
    if not datasets:
        print(f"No datasets found in {store.root}")
        return 1
    bad = 0
    for symbol, tf in datasets:
        try:
            report = validate_stored(store, symbol, tf)
        except DatasetIntegrityError as exc:
            bad += 1
            print(f"CHECKSUM MISMATCH {symbol} {tf}: {exc}")
            continue
        bad += 0 if report.ok else 1
        print(report.summary())
    return 1 if bad else 0


def cmd_info(args, cfg, settings) -> int:
    store = DatasetStore(args.data_dir or settings.data_dir)
    for symbol, tf in store.list_datasets():
        m = store.metadata(symbol, tf)
        print(
            f"{symbol:8} {tf:3} {m['rows']:>8} rows  {m['start']} -> {m['end']}  "
            f"source={m['source']}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tb-data", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--data-dir", help="dataset folder (default: TB_DATA_DIR / data_store)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("mt5-check", help="check MT5 demo connection and symbol names")

    d = sub.add_parser("download", help="build datasets")
    d.add_argument("--source", choices=["mt5", "csv", "synthetic"], required=True)
    d.add_argument("--symbols", nargs="+")
    d.add_argument("--timeframes", nargs="+", choices=["1h", "4h", "1d"])
    d.add_argument("--start", default="2020-01-01", help="UTC date, e.g. 2020-01-01")
    d.add_argument("--end", help="UTC date (default: now)")
    d.add_argument("--csv-dir")
    d.add_argument("--seed", type=int, default=42, help="synthetic data seed")
    d.add_argument(
        "--server-utc-offset", type=float, default=0.0, help="broker time = UTC + this many hours"
    )
    d.add_argument(
        "--server-timezone", help="IANA zone of broker time (handles DST), e.g. Europe/Athens"
    )
    d.add_argument(
        "--allow-invalid", action="store_true", help="store data even if validation errors exist"
    )

    sub.add_parser("validate", help="re-validate stored datasets")
    sub.add_parser("info", help="list stored datasets")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg, settings = load_config(), load_settings()
    handler = {
        "mt5-check": cmd_mt5_check,
        "download": cmd_download,
        "validate": cmd_validate,
        "info": cmd_info,
    }
    return handler[args.command](args, cfg, settings)


if __name__ == "__main__":
    raise SystemExit(main())

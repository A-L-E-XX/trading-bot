"""Live paper session (Milestone 10): MT5 candles in, paper engine out. READ-ONLY.

This module only READS market data from a DEMO account and feeds the paper engine. There is no code
path that sends, modifies or closes an order at the broker (a test enforces that).

How it behaves
- Only fully CLOSED candles are processed (the forming candle is never used).
- It processes every closed candle it has not seen yet, oldest first, so a pause or restart catches
  up. A candle processed before is ignored by the engine (idempotent).
- Orders decided at one candle's close are filled at the next candle's open price from the data, so
  events for a fill appear once that next candle has closed (a delay of one candle in the log, not
  in the price).
- A transient MT5 problem is logged and retried with backoff; too many failures in a row stop the
  session with an error so that a supervisor notices.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tradingbot.data.collectors.mt5 import MT5Collector, MT5Error
from tradingbot.execution.paper import PaperEngine

TF_DELTA = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}
CLOCK_TOLERANCE = pd.Timedelta(minutes=10)


class FeedError(RuntimeError):
    pass


def closed_candles(
    rates: pd.DataFrame, timeframe: str, now: pd.Timestamp, tolerance=CLOCK_TOLERANCE
) -> pd.DataFrame:
    """Drop the forming candle (the last row) and return canonical UTC-indexed OHLCV.

    Also refuses data that looks wrong: the newest "closed" candle must really have closed already
    (otherwise the server time offset is probably misconfigured).
    """
    if len(rates) < 2:
        return rates.iloc[0:0].set_index("time") if len(rates) else pd.DataFrame()
    df = rates.iloc[:-1].copy().set_index("time")
    df.index.name = "time"
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise FeedError("candles from the broker are not strictly increasing")
    if df.index[-1] + TF_DELTA[timeframe] > now + tolerance:
        raise FeedError(
            f"newest closed candle {df.index[-1]} ({timeframe}) has not closed yet at {now}: "
            "check --server-utc-offset / --server-timezone"
        )
    return df[["open", "high", "low", "close", "volume"]].astype("float64")


@dataclass
class SessionResult:
    processed: int
    errors: int
    stopped_because: str


def run_session(
    collector: MT5Collector,
    engine: PaperEngine,
    *,
    poll_seconds: float = 60.0,
    candles_to_fetch: int = 1500,
    max_consecutive_errors: int = 30,
    max_polls: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
) -> SessionResult:
    """Poll for new closed candles and feed them to the engine until stopped.

    ``max_polls=1`` processes whatever is available once and returns (used for catch-up/cron).
    """
    processed = errors = consecutive = polls = 0
    while True:
        polls += 1
        try:
            rates = collector.fetch_recent(engine.symbol, engine.timeframe, candles_to_fetch)
            df = closed_candles(rates, engine.timeframe, now())
            fresh = df if engine.last_bar_time is None else df[df.index > engine.last_bar_time]
            quote = collector.quote(engine.symbol)
            for t, r in fresh.iterrows():
                engine.on_bar(t, r.open, r.high, r.low, r.close, r.volume)
                processed += 1
            if len(fresh) and quote:
                model = engine.costs.spread(engine.spec)
                ratio = quote["spread"] / model if model else None
                engine.note("spread_check", fresh.index[-1], live_spread=quote["spread"],
                            model_spread=model, ratio=ratio)  # fmt: skip
            consecutive = 0
        except (MT5Error, FeedError) as exc:
            errors += 1
            consecutive += 1
            engine.note("feed_error", engine.last_bar_time, error=str(exc), consecutive=consecutive)
            if consecutive >= max_consecutive_errors:
                return SessionResult(processed, errors, f"{consecutive} feed errors in a row")
            sleep(min(300.0, poll_seconds * 2 ** min(consecutive, 5)))
            if max_polls is not None and polls >= max_polls:
                return SessionResult(processed, errors, "max_polls")
            continue
        if engine.halted:
            return SessionResult(processed, errors, f"engine halted: {engine.halted}")
        if max_polls is not None and polls >= max_polls:
            return SessionResult(processed, errors, "max_polls")
        sleep(poll_seconds)


def seed_history(
    stored: pd.DataFrame | None, recent: pd.DataFrame, up_to: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Stored dataset + recent closed candles (recent wins on overlap), cut at ``up_to``."""
    parts = [x for x in (stored, recent) if x is not None and len(x)]
    if not parts:
        raise FeedError("no history available: download data first (tb-data download)")
    merged = pd.concat([p[["open", "high", "low", "close", "volume"]] for p in parts])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    if up_to is not None:
        merged = merged[merged.index <= up_to]
    return merged


def saved_last_bar(state_path: Path) -> pd.Timestamp | None:
    if not state_path.exists():
        return None
    raw = json.loads(state_path.read_text(encoding="utf-8")).get("last_bar_time")
    return pd.Timestamp(raw) if raw else None

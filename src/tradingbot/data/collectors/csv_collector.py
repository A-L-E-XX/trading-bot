"""Read candles from CSV files (including MetaTrader 5 "Export bars" files)."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from tradingbot.data.normalizer import to_utc

_MT5_TF = {"1h": "H1", "4h": "H4", "1d": "D1"}


class CSVCollector:
    """Looks for ``<SYMBOL>_<tf>.csv`` or ``<SYMBOL>_<H1|H4|D1>.csv`` (case-insensitive).

    Accepts comma/tab/semicolon separated files. MT5 exports with ``<DATE> <TIME> <OPEN> ...``
    headers are understood. ``utc_offset_hours`` converts broker time to UTC.
    """

    name = "csv"

    def __init__(
        self, directory: Path | str, utc_offset_hours: float = 0.0, timezone: str | None = None
    ):
        self.directory = Path(directory)
        self.utc_offset_hours = utc_offset_hours
        self.timezone = timezone

    def _find(self, symbol: str, timeframe: str) -> Path:
        wanted = {f"{symbol}_{timeframe}.csv".lower(), f"{symbol}_{_MT5_TF[timeframe]}.csv".lower()}
        for path in self.directory.glob("*.csv"):
            if path.name.lower() in wanted:
                return path
        raise FileNotFoundError(
            f"no CSV for {symbol} {timeframe} in {self.directory} (looked for {sorted(wanted)})"
        )

    def fetch(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        df = pd.read_csv(self._find(symbol, timeframe), sep=None, engine="python")
        df.columns = [re.sub(r"[<>\s]", "", str(c)).lower() for c in df.columns]
        if "date" in df.columns and "time" in df.columns:  # MT5 export: separate date and time
            date = df.pop("date").astype(str).str.replace(".", "-", regex=False)
            df["time"] = date + " " + df["time"].astype(str)
        elif "date" in df.columns:
            df = df.rename(columns={"date": "time"})
        if "time" not in df.columns:
            raise ValueError("CSV needs a 'time' (or 'date') column")
        df["time"] = to_utc(
            df["time"], utc_offset_hours=self.utc_offset_hours, timezone=self.timezone
        )
        df = df.dropna(subset=["time"])
        df = df[(df["time"] >= pd.Timestamp(start)) & (df["time"] < pd.Timestamp(end))]
        return df.reset_index(drop=True)

"""Dataset storage: one CSV + one JSON metadata sidecar per (symbol, timeframe).

CSV (not parquet) keeps the install light and the files human-readable. The metadata holds a
SHA-256 of the CSV bytes so a dataset can be verified and regenerated identically.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from tradingbot.data.normalizer import OHLCV_COLUMNS
from tradingbot.data.validators import ValidationReport

_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class DatasetIntegrityError(RuntimeError):
    """Stored data does not match its recorded checksum."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DatasetStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def csv_path(self, symbol: str, timeframe: str) -> Path:
        return self.root / symbol / f"{timeframe}.csv"

    def meta_path(self, symbol: str, timeframe: str) -> Path:
        return self.root / symbol / f"{timeframe}.meta.json"

    def exists(self, symbol: str, timeframe: str) -> bool:
        return (
            self.csv_path(symbol, timeframe).exists() and self.meta_path(symbol, timeframe).exists()
        )

    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        *,
        source: str,
        report: ValidationReport,
        notes: dict | None = None,
    ) -> dict:
        csv, meta = self.csv_path(symbol, timeframe), self.meta_path(symbol, timeframe)
        csv.parent.mkdir(parents=True, exist_ok=True)
        df[OHLCV_COLUMNS].to_csv(csv, date_format=_TIME_FORMAT, lineterminator="\n")
        metadata = {
            "symbol": symbol,
            "timeframe": timeframe,
            "source": source,
            "timezone": "UTC",
            "rows": len(df),
            "start": report.start,
            "end": report.end,
            "sha256": sha256_file(csv),
            "created_utc": datetime.now(UTC).strftime(_TIME_FORMAT),
            "validation": report.to_dict(),
            "notes": notes or {},
        }
        meta.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return metadata

    def metadata(self, symbol: str, timeframe: str) -> dict:
        return json.loads(self.meta_path(symbol, timeframe).read_text(encoding="utf-8"))

    def load(self, symbol: str, timeframe: str, *, verify: bool = True) -> pd.DataFrame:
        csv = self.csv_path(symbol, timeframe)
        if not self.exists(symbol, timeframe):
            raise FileNotFoundError(f"no stored dataset for {symbol} {timeframe} under {self.root}")
        if verify and sha256_file(csv) != self.metadata(symbol, timeframe)["sha256"]:
            raise DatasetIntegrityError(f"{csv} does not match its recorded checksum")
        df = pd.read_csv(csv, index_col="time", parse_dates=["time"])
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        df.index = idx.as_unit("ns")
        df.index.name = "time"
        return df.astype("float64")

    def list_datasets(self) -> list[tuple[str, str]]:
        found = []
        for meta in sorted(self.root.glob("*/*.meta.json")):
            found.append((meta.parent.name, meta.name.removesuffix(".meta.json")))
        return found

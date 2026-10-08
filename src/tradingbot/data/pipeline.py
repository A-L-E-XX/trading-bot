"""fetch -> normalize -> validate -> store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from tradingbot.data.collectors.base import Collector
from tradingbot.data.normalizer import NormalizationStats, normalize_ohlcv
from tradingbot.data.storage import DatasetStore
from tradingbot.data.validators import ValidationReport, validate_ohlcv
from tradingbot.timeframes import check_timeframe


class DataValidationError(RuntimeError):
    def __init__(self, report: ValidationReport):
        super().__init__("dataset failed validation:\n" + report.summary())
        self.report = report


@dataclass
class BuildResult:
    metadata: dict | None
    report: ValidationReport
    normalization: NormalizationStats
    saved: bool


def build_dataset(
    collector: Collector,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    store: DatasetStore,
    *,
    allow_invalid: bool = False,
) -> BuildResult:
    """Download, clean, validate and store one dataset. Invalid data is NOT saved by default."""
    check_timeframe(timeframe)
    raw = collector.fetch(symbol, timeframe, start, end)
    df, stats = normalize_ohlcv(raw)
    report = validate_ohlcv(df, symbol, timeframe)
    if not report.ok and not allow_invalid:
        raise DataValidationError(report)
    metadata = store.save(
        df,
        symbol,
        timeframe,
        source=collector.name,
        report=report,
        notes={
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "normalization": stats.to_dict(),
        },
    )
    return BuildResult(metadata, report, stats, saved=True)


def validate_stored(store: DatasetStore, symbol: str, timeframe: str) -> ValidationReport:
    """Re-validate a stored dataset (also verifies its checksum)."""
    return validate_ohlcv(store.load(symbol, timeframe, verify=True), symbol, timeframe)

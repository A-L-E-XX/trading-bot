"""Data validation: timestamps, ordering, duplicates, gaps, and basic price sanity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from tradingbot.data.calendar import (
    expected_closed_mask,
    is_24_7,
    recurring_closure_slots,
    slot_name,
)
from tradingbot.data.normalizer import OHLCV_COLUMNS
from tradingbot.timeframes import timeframe_freq

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    code: str
    severity: Severity
    count: int
    detail: str
    examples: tuple[str, ...] = ()


@dataclass
class ValidationReport:
    symbol: str
    timeframe: str
    rows: int
    start: str | None
    end: str | None
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self) -> set[str]:
        return {i.code for i in self.issues}

    def summary(self) -> str:
        head = f"{self.symbol} {self.timeframe}: {self.rows} rows, {self.start} -> {self.end}"
        status = "OK" if self.ok else "FAILED"
        lines = [f"{head} [{status}]"]
        for i in self.issues:
            ex = f" e.g. {', '.join(i.examples)}" if i.examples else ""
            lines.append(f"  {i.severity.upper():7} {i.code}: {i.count} - {i.detail}{ex}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "rows": self.rows,
            "start": self.start,
            "end": self.end,
            "ok": self.ok,
            "issues": [
                {
                    "code": i.code,
                    "severity": i.severity,
                    "count": i.count,
                    "detail": i.detail,
                    "examples": list(i.examples),
                }
                for i in self.issues
            ],
        }


def _fmt(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    *,
    max_missing_fraction: float = 0.02,
    max_abs_return: float = 0.30,
    max_flat_fraction: float = 0.05,
) -> ValidationReport:
    """Validate a canonical OHLCV frame. ``report.ok`` is False if any *error* was found."""
    report = ValidationReport(symbol, timeframe, len(df), None, None)

    def add(code: str, severity: Severity, count: int, detail: str, examples=()) -> None:
        if count:
            report.issues.append(Issue(code, severity, int(count), detail, tuple(examples)))

    def first_examples(mask, n: int = 3) -> list[str]:
        return [_fmt(t) for t in df.index[np.asarray(mask)][:n]]

    if len(df) == 0:
        add("empty", "error", 1, "dataset has no rows")
        return report
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        add("bad_index", "error", 1, "index must be a tz-aware (UTC) DatetimeIndex")
        return report
    absent = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if absent:
        add("missing_columns", "error", len(absent), f"missing columns: {absent}")
        return report

    idx = df.index
    report.start, report.end = _fmt(idx.min()), _fmt(idx.max())

    # --- timestamps -------------------------------------------------------------
    dup = idx.duplicated(keep=False)
    add(
        "duplicate_timestamps",
        "error",
        dup.sum(),
        "repeated candle timestamps",
        first_examples(dup),
    )
    steps_back = np.asarray(idx[1:] < idx[:-1])
    add("out_of_order", "error", steps_back.sum(), "timestamps not in ascending order")
    off_hour = np.asarray((idx.minute != 0) | (idx.second != 0))
    add(
        "off_hour_timestamps",
        "error",
        off_hour.sum(),
        "candles must start on the hour",
        first_examples(off_hour),
    )

    # --- values -----------------------------------------------------------------
    vals = df[OHLCV_COLUMNS].to_numpy(dtype="float64")
    non_finite = ~np.isfinite(vals).all(axis=1)
    add(
        "non_finite_values",
        "error",
        non_finite.sum(),
        "NaN or infinite values",
        first_examples(non_finite),
    )

    o, h, lo, c, v = (df[k].to_numpy(dtype="float64") for k in OHLCV_COLUMNS)
    with np.errstate(invalid="ignore"):
        nonpos = (np.minimum.reduce([o, h, lo, c]) <= 0) & ~non_finite
        bad_hl = (h < lo) & ~non_finite
        bad_high = (h < np.maximum(o, c)) & ~non_finite
        bad_low = (lo > np.minimum(o, c)) & ~non_finite
        neg_vol = (v < 0) & ~non_finite
    add("non_positive_price", "error", nonpos.sum(), "price <= 0", first_examples(nonpos))
    add("high_below_low", "error", bad_hl.sum(), "high < low", first_examples(bad_hl))
    add(
        "high_not_highest",
        "error",
        bad_high.sum(),
        "high < max(open, close)",
        first_examples(bad_high),
    )
    add("low_not_lowest", "error", bad_low.sum(), "low > min(open, close)", first_examples(bad_low))
    add("negative_volume", "error", neg_vol.sum(), "volume < 0", first_examples(neg_vol))

    # --- gaps (only meaningful on a clean, sorted, unique index) -------------------
    clean_idx = pd.DatetimeIndex(idx.unique().sort_values())
    full = pd.date_range(clean_idx[0], clean_idx[-1], freq=timeframe_freq(timeframe), tz="UTC")
    missing = full.difference(clean_idx)
    if len(missing):
        closed = expected_closed_mask(missing, symbol, timeframe)
        unexpected = missing[~closed]
        recurring_note = ""
        if not is_24_7(symbol) and len(unexpected):
            open_grid = full[~expected_closed_mask(full, symbol, timeframe)]
            slots = recurring_closure_slots(open_grid, unexpected)
            unexpected_slot = np.asarray(unexpected.dayofweek) * 24 + np.asarray(unexpected.hour)
            routine = slots[unexpected_slot]
            if routine.any():
                names = ", ".join(slot_name(int(i)) for i in np.flatnonzero(slots)[:6])
                recurring_note = (
                    f"; {int(routine.sum())} candles in recurring broker-break slots "
                    f"ignored ({names})"
                )
                unexpected = unexpected[~routine]
        frac = len(unexpected) / max(len(full), 1)
        sev: Severity = "error" if frac > max_missing_fraction else "warning"
        add(
            "missing_candles",
            sev,
            len(unexpected),
            f"{frac:.2%} of expected candles missing (limit {max_missing_fraction:.0%}); "
            f"{int(closed.sum())} weekend/closed candles ignored{recurring_note}",
            [_fmt(t) for t in unexpected[:3]],
        )
    off_grid = clean_idx.difference(full)
    add(
        "off_grid_timestamps",
        "warning",
        len(off_grid),
        "timestamps not on the expected candle grid (daylight-saving shift?)",
        [_fmt(t) for t in off_grid[:3]],
    )

    # --- sanity warnings -----------------------------------------------------------
    flat = (h == lo) & ~non_finite
    if flat.mean() > max_flat_fraction:
        add(
            "many_flat_candles",
            "warning",
            flat.sum(),
            f"more than {max_flat_fraction:.0%} candles have high == low",
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = np.abs(np.diff(c) / c[:-1])
    big = np.concatenate([[False], np.nan_to_num(ret, nan=0.0) > max_abs_return])
    add(
        "extreme_moves",
        "warning",
        big.sum(),
        f"close-to-close move > {max_abs_return:.0%}",
        first_examples(big),
    )

    return report

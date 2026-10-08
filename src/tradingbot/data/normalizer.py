"""Turn raw collector output into the canonical OHLCV format.

Canonical format: DataFrame indexed by ``time`` (tz-aware UTC, ns), sorted ascending, unique,
float64 columns ``open, high, low, close, volume``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
_ALIASES = {
    "tick_volume": "volume",
    "tickvol": "volume",
    "vol": "volume",
    "date": "time",
    "datetime": "time",
    "timestamp": "time",
}


@dataclass(frozen=True)
class NormalizationStats:
    rows_in: int
    rows_out: int
    duplicates_removed: int
    reordered: bool
    unparseable_rows_dropped: int

    def to_dict(self) -> dict:
        return asdict(self)


# When several columns mean "volume" (MT5 exports have both <TICKVOL> and <VOL>), the first match
# in this order wins. Tick volume is preferred: real volume is usually 0 for forex and metals.
_VOLUME_PRIORITY = ["tick_volume", "tickvol", "volume", "vol"]


def _canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    names = [str(c).strip().lower() for c in df.columns]
    winner = next((v for v in _VOLUME_PRIORITY if v in names), None)
    keep = [i for i, n in enumerate(names) if n not in _VOLUME_PRIORITY or n == winner]
    out = df.iloc[:, keep].copy()
    out.columns = [_ALIASES.get(names[i], names[i]) for i in keep]
    return out


def to_utc(times: pd.Series, *, utc_offset_hours: float = 0.0, timezone: str | None = None):
    """Convert broker/server timestamps to UTC.

    ``timezone`` (IANA name, e.g. "Europe/Athens") handles daylight saving; ``utc_offset_hours``
    is a fixed offset (server time = UTC + offset). Naive inputs are interpreted as server time;
    tz-aware inputs are simply converted.
    """
    parsed = pd.to_datetime(times, errors="coerce")
    if parsed.dt.tz is not None:
        return parsed.dt.tz_convert("UTC")
    if timezone:
        local = parsed.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
        return local.dt.tz_convert("UTC")
    shifted = parsed - pd.Timedelta(hours=utc_offset_hours)
    return shifted.dt.tz_localize("UTC")


def _as_utc_ns_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert("UTC").as_unit("ns")


def normalize_ohlcv(raw: pd.DataFrame) -> tuple[pd.DataFrame, NormalizationStats]:
    """Normalize a raw frame (``time`` column or datetime index + OHLC[V] columns)."""
    df = raw.copy()
    if isinstance(df.index, pd.DatetimeIndex) and "time" not in {str(c).lower() for c in df}:
        df = df.reset_index(names="time")
    df = _canonical_columns(df)
    missing = [c for c in ("time", "open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"raw data is missing required columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    rows_in = len(df)
    times = pd.to_datetime(df["time"], errors="coerce")
    bad_time = times.isna()
    df = df.loc[~bad_time].copy()
    times = times.loc[~bad_time]
    index = _as_utc_ns_index(pd.DatetimeIndex(times))
    df = df[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce").astype("float64")
    df.index = index
    df.index.name = "time"

    reordered = not df.index.is_monotonic_increasing
    df = df.sort_index(kind="stable")
    dup = df.index.duplicated(keep="last")
    duplicates_removed = int(dup.sum())
    df = df.loc[~dup]

    stats = NormalizationStats(
        rows_in=rows_in,
        rows_out=len(df),
        duplicates_removed=duplicates_removed,
        reordered=bool(reordered),
        unparseable_rows_dropped=int(bad_time.sum()),
    )
    return df, stats

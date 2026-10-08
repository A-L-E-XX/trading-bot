import numpy as np
import pandas as pd
import pytest

from tradingbot.data.collectors import SyntheticCollector
from tradingbot.data.normalizer import normalize_ohlcv, to_utc
from tradingbot.data.validators import validate_ohlcv

# ---------------------------------------------------------------- normalizer


def test_normalizer_sorts_dedupes_and_types(start, end):
    raw = SyntheticCollector().fetch("BTCUSD", "1h", start, end)
    messy = pd.concat([raw.iloc[::-1], raw.iloc[:5]], ignore_index=True)
    df, stats = normalize_ohlcv(messy)
    assert stats.reordered and stats.duplicates_removed == 5
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert str(df.index.dtype) == "datetime64[ns, UTC]"
    assert all(df[c].dtype == "float64" for c in df.columns)
    assert len(df) == len(raw)


def test_normalizer_drops_unparseable_times_and_accepts_aliases():
    raw = pd.DataFrame(
        {
            "Date": ["2023-01-02 00:00", "garbage", "2023-01-02 01:00"],
            "Open": [1, 1, 1],
            "High": [2, 2, 2],
            "Low": [0.5, 0.5, 0.5],
            "Close": [1.5, 1.5, 1.5],
            "tick_volume": [10, 10, 10],
        }
    )
    df, stats = normalize_ohlcv(raw)
    assert stats.unparseable_rows_dropped == 1 and len(df) == 2
    assert "volume" in df.columns


def test_normalizer_requires_ohlc():
    with pytest.raises(ValueError):
        normalize_ohlcv(pd.DataFrame({"time": [1], "open": [1]}))


def test_to_utc_fixed_offset_and_timezone():
    naive = pd.Series(pd.to_datetime(["2023-01-02 12:00", "2023-07-03 12:00"]))
    fixed = to_utc(naive, utc_offset_hours=2)
    assert fixed.iloc[0] == pd.Timestamp("2023-01-02 10:00", tz="UTC")
    assert fixed.iloc[1] == pd.Timestamp("2023-07-03 10:00", tz="UTC")
    athens = to_utc(naive, timezone="Europe/Athens")  # UTC+2 winter, UTC+3 summer
    assert athens.iloc[0] == pd.Timestamp("2023-01-02 10:00", tz="UTC")
    assert athens.iloc[1] == pd.Timestamp("2023-07-03 09:00", tz="UTC")


# ---------------------------------------------------------------- validators


def test_clean_synthetic_data_validates(make_ohlcv):
    for symbol in ("BTCUSD", "EURUSD", "XAUUSD"):
        for tf in ("1h", "4h", "1d"):
            report = validate_ohlcv(make_ohlcv(symbol, tf), symbol, tf)
            assert report.ok, report.summary()
            assert not report.issues, report.summary()  # weekends must not count as gaps


def test_detects_duplicates(ohlcv):
    bad = pd.concat([ohlcv, ohlcv.iloc[[10]]])
    assert "duplicate_timestamps" in validate_ohlcv(bad, "BTCUSD", "1h").codes()


def test_detects_out_of_order(ohlcv):
    bad = ohlcv.iloc[::-1]
    report = validate_ohlcv(bad, "BTCUSD", "1h")
    assert "out_of_order" in report.codes() and not report.ok


def _mutate(df, row, col, value):
    bad = df.copy()
    bad.iloc[row, bad.columns.get_loc(col)] = value
    return bad


def test_detects_nan_and_inf(ohlcv):
    assert (
        "non_finite_values"
        in validate_ohlcv(_mutate(ohlcv, 5, "close", np.nan), "BTCUSD", "1h").codes()
    )
    assert (
        "non_finite_values"
        in validate_ohlcv(_mutate(ohlcv, 5, "high", np.inf), "BTCUSD", "1h").codes()
    )


def test_detects_non_positive_price(ohlcv):
    assert (
        "non_positive_price"
        in validate_ohlcv(_mutate(ohlcv, 3, "low", 0.0), "BTCUSD", "1h").codes()
    )


def test_detects_inconsistent_ohlc(ohlcv):
    row = ohlcv.iloc[7]
    bad = _mutate(ohlcv, 7, "high", row["low"] * 0.5)
    codes = validate_ohlcv(bad, "BTCUSD", "1h").codes()
    assert {"high_below_low", "high_not_highest"} <= codes
    bad = _mutate(ohlcv, 7, "low", row["high"] * 2)
    assert "low_not_lowest" in validate_ohlcv(bad, "BTCUSD", "1h").codes()


def test_detects_negative_volume(ohlcv):
    assert (
        "negative_volume"
        in validate_ohlcv(_mutate(ohlcv, 2, "volume", -1.0), "BTCUSD", "1h").codes()
    )


def test_detects_off_hour_timestamp(ohlcv):
    bad = ohlcv.copy()
    idx = bad.index.to_series()
    idx.iloc[4] = idx.iloc[4] + pd.Timedelta(minutes=17)
    bad.index = pd.DatetimeIndex(idx)
    assert "off_hour_timestamps" in validate_ohlcv(bad, "BTCUSD", "1h").codes()


def test_small_gap_is_warning_large_gap_is_error(ohlcv):
    few = ohlcv.drop(ohlcv.index[[10, 11]])
    report = validate_ohlcv(few, "BTCUSD", "1h")
    assert report.ok and "missing_candles" in report.codes()
    many = ohlcv.drop(ohlcv.index[100:400])
    report = validate_ohlcv(many, "BTCUSD", "1h")
    assert not report.ok and "missing_candles" in report.codes()


def test_weekend_gap_expected_for_forex_but_not_crypto(make_ohlcv):
    forex = make_ohlcv("EURUSD", "1h")
    assert "missing_candles" not in validate_ohlcv(forex, "EURUSD", "1h").codes()
    # the same frame (which has weekend holes) is NOT acceptable for a 24/7 crypto symbol
    assert "missing_candles" in validate_ohlcv(forex, "BTCUSD", "1h").codes()


def test_extreme_move_is_warning(ohlcv):
    bad = ohlcv.copy()
    bad.iloc[50:, :4] = bad.iloc[50:, :4] * 2
    report = validate_ohlcv(bad, "BTCUSD", "1h")
    assert "extreme_moves" in report.codes() and report.ok


def test_empty_and_bad_index():
    assert "empty" in validate_ohlcv(pd.DataFrame(), "BTCUSD", "1h").codes()
    naive = pd.DataFrame(
        {c: [1.0] for c in ["open", "high", "low", "close", "volume"]},
        index=pd.to_datetime(["2023-01-01"]),
    )
    assert "bad_index" in validate_ohlcv(naive, "BTCUSD", "1h").codes()


def test_report_summary_and_dict(ohlcv):
    report = validate_ohlcv(ohlcv.drop(ohlcv.index[[10]]), "BTCUSD", "1h")
    assert "BTCUSD 1h" in report.summary()
    assert report.to_dict()["ok"] is True

import json

import pandas as pd
import pytest

from tradingbot.data import DatasetIntegrityError, DatasetStore, DataValidationError, build_dataset
from tradingbot.data.collectors import SyntheticCollector
from tradingbot.data.pipeline import validate_stored


def test_pipeline_roundtrip_and_metadata(tmp_path, start, end):
    store = DatasetStore(tmp_path)
    result = build_dataset(SyntheticCollector(seed=7), "EURUSD", "1h", start, end, store)
    assert result.saved and result.report.ok
    meta = store.metadata("EURUSD", "1h")
    assert meta["source"] == "synthetic" and meta["timezone"] == "UTC"
    assert meta["rows"] == len(store.load("EURUSD", "1h"))
    assert meta["validation"]["ok"] is True
    assert validate_stored(store, "EURUSD", "1h").ok
    assert store.list_datasets() == [("EURUSD", "1h")]


def test_loaded_data_is_identical_to_saved(tmp_path, make_ohlcv):
    store = DatasetStore(tmp_path)
    df = make_ohlcv("BTCUSD", "4h")
    from tradingbot.data.validators import validate_ohlcv

    store.save(df, "BTCUSD", "4h", source="test", report=validate_ohlcv(df, "BTCUSD", "4h"))
    loaded = store.load("BTCUSD", "4h")
    pd.testing.assert_frame_equal(df, loaded, check_freq=False)
    assert str(loaded.index.dtype) == "datetime64[ns, UTC]"


def test_dataset_regenerates_identically(tmp_path, start, end):
    """M3 acceptance: the same dataset can be regenerated (identical checksum) and validated."""
    a, b = DatasetStore(tmp_path / "a"), DatasetStore(tmp_path / "b")
    for store in (a, b):
        build_dataset(SyntheticCollector(seed=3), "XAUUSD", "1d", start, end, store)
    assert a.metadata("XAUUSD", "1d")["sha256"] == b.metadata("XAUUSD", "1d")["sha256"]


def test_tampering_is_detected(tmp_path, start, end):
    store = DatasetStore(tmp_path)
    build_dataset(SyntheticCollector(), "EURUSD", "1h", start, end, store)
    csv = store.csv_path("EURUSD", "1h")
    csv.write_text(csv.read_text().replace(".", ",", 1))
    with pytest.raises(DatasetIntegrityError):
        store.load("EURUSD", "1h")


def test_invalid_data_is_not_saved(tmp_path, start, end):
    class BrokenCollector(SyntheticCollector):
        def fetch(self, *args, **kwargs):
            df = super().fetch(*args, **kwargs)
            df.loc[3, "high"] = df.loc[3, "low"] * 0.5
            return df

    store = DatasetStore(tmp_path)
    with pytest.raises(DataValidationError) as err:
        build_dataset(BrokenCollector(), "EURUSD", "1h", start, end, store)
    assert "high_below_low" in err.value.report.codes()
    assert not store.exists("EURUSD", "1h")
    # an explicit override stores it, with the problem recorded in the metadata
    build_dataset(BrokenCollector(), "EURUSD", "1h", start, end, store, allow_invalid=True)
    assert json.loads(store.meta_path("EURUSD", "1h").read_text())["validation"]["ok"] is False


def test_unsupported_timeframe(tmp_path, start, end):
    with pytest.raises(ValueError):
        build_dataset(SyntheticCollector(), "EURUSD", "5m", start, end, DatasetStore(tmp_path))


def test_synthetic_is_deterministic_and_seed_dependent(start, end):
    a = SyntheticCollector(seed=1).fetch("EURUSD", "1h", start, end)
    b = SyntheticCollector(seed=1).fetch("EURUSD", "1h", start, end)
    c = SyntheticCollector(seed=2).fetch("EURUSD", "1h", start, end)
    pd.testing.assert_frame_equal(a, b)
    assert not a["close"].equals(c["close"])

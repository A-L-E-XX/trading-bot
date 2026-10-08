from collections import namedtuple
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from tradingbot.data import DatasetStore, build_dataset
from tradingbot.data.cli import main
from tradingbot.data.collectors import CSVCollector, MT5Collector, MT5Error

START = datetime(2023, 1, 2, tzinfo=UTC)
END = datetime(2023, 1, 9, tzinfo=UTC)

# ------------------------------------------------------------------ CSV collector


def test_csv_collector_reads_mt5_export_format(tmp_path):
    rows = ["<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"]
    t = pd.date_range("2023-01-02 00:00", periods=48, freq="1h")
    for i, ts in enumerate(t):
        p = 1.07 + i * 0.0001
        rows.append(
            f"{ts:%Y.%m.%d}\t{ts:%H:%M:%S}\t{p:.5f}\t{p + 0.0003:.5f}\t{p - 0.0003:.5f}"
            f"\t{p + 0.0001:.5f}\t100\t0\t2"
        )
    (tmp_path / "EURUSD_H1.csv").write_text("\n".join(rows))
    store = DatasetStore(tmp_path / "store")
    result = build_dataset(CSVCollector(tmp_path), "EURUSD", "1h", START, END, store)
    assert result.report.ok
    df = store.load("EURUSD", "1h")
    assert len(df) == 48 and df["volume"].iloc[0] == 100


def test_csv_collector_utc_offset_and_missing_file(tmp_path):
    t = pd.date_range("2023-01-02 02:00", periods=5, freq="1h")
    pd.DataFrame(
        {"time": t, "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1, "volume": 5}
    ).to_csv(tmp_path / "BTCUSD_1h.csv", index=False)
    raw = CSVCollector(tmp_path, utc_offset_hours=2).fetch("BTCUSD", "1h", START, END)
    assert raw["time"].iloc[0] == pd.Timestamp("2023-01-02 00:00", tz="UTC")
    with pytest.raises(FileNotFoundError):
        CSVCollector(tmp_path).fetch("EURUSD", "1h", START, END)


# ------------------------------------------------------------------ MT5 collector (fake terminal)

Account = namedtuple("Account", "login server currency trade_mode")
Sym = namedtuple("Sym", "name")
Tick = namedtuple("Tick", "time")


class FakeMT5:
    TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1 = 16385, 16388, 16408
    ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_TRADE_MODE_REAL = 0, 2

    def __init__(self, demo=True, init_ok=True, server_offset_hours=2):
        self.demo, self.init_ok, self.offset = demo, init_ok, server_offset_hours
        self.shutdown_called = False
        self.requests = []

    def initialize(self, **kwargs):
        return self.init_ok

    def shutdown(self):
        self.shutdown_called = True

    def last_error(self):
        return (-1, "fake error")

    def account_info(self):
        mode = self.ACCOUNT_TRADE_MODE_DEMO if self.demo else self.ACCOUNT_TRADE_MODE_REAL
        return Account(123, "Fake-Demo", "USD", mode)

    def symbols_get(self):
        return [Sym("EURUSD.m"), Sym("EURUSDpro"), Sym("BTCUSD"), Sym("XAUUSD")]

    def symbol_select(self, name, enable):
        return name != "NOPE"

    def symbol_info_tick(self, name):
        now = datetime.now(UTC).timestamp() + self.offset * 3600
        return Tick(int(now))

    def copy_rates_range(self, symbol, tf, date_from, date_to):
        self.requests.append((symbol, tf, date_from, date_to))
        # server time = UTC + offset, returned as epoch seconds
        t = pd.date_range("2023-01-02 02:00", periods=24, freq="1h")  # = 00:00 UTC for offset 2
        secs = (t - pd.Timestamp("1970-01-01")).total_seconds().astype("int64")
        dtype = [
            ("time", "i8"),
            ("open", "f8"),
            ("high", "f8"),
            ("low", "f8"),
            ("close", "f8"),
            ("tick_volume", "u8"),
            ("spread", "i4"),
            ("real_volume", "u8"),
        ]
        arr = np.zeros(len(t), dtype=dtype)
        arr["time"] = secs
        arr["open"], arr["high"], arr["low"], arr["close"] = 1.0, 1.2, 0.9, 1.1
        arr["tick_volume"] = 50
        return arr


def test_mt5_collector_converts_server_time_to_utc():
    fake = FakeMT5()
    col = MT5Collector(mt5_module=fake, server_utc_offset_hours=2)
    raw = col.fetch("EURUSD", "1h", START, END)
    assert raw["time"].iloc[0] == pd.Timestamp("2023-01-02 00:00", tz="UTC")
    assert list(raw.columns) == ["time", "open", "high", "low", "close", "volume"]
    assert raw["volume"].iloc[0] == 50
    assert fake.requests[0][1] == FakeMT5.TIMEFRAME_H1


def test_mt5_collector_applies_symbol_suffix():
    fake = FakeMT5()
    MT5Collector(mt5_module=fake, symbol_suffix=".m", server_utc_offset_hours=2).fetch(
        "EURUSD", "1h", START, END
    )
    assert fake.requests[0][0] == "EURUSD.m"


def test_mt5_collector_refuses_real_accounts():
    fake = FakeMT5(demo=False)
    with pytest.raises(MT5Error, match="NOT a demo"):
        MT5Collector(mt5_module=fake).connect()
    assert fake.shutdown_called


def test_mt5_collector_reports_failures():
    with pytest.raises(MT5Error, match="initialize failed"):
        MT5Collector(mt5_module=FakeMT5(init_ok=False)).connect()
    with pytest.raises(MT5Error, match="not available"):
        MT5Collector(mt5_module=FakeMT5()).fetch("NOPE", "1h", START, END)


def test_mt5_symbol_discovery_and_offset_estimate():
    col = MT5Collector(mt5_module=FakeMT5(server_offset_hours=3))
    found = col.find_symbols(["EURUSD", "GBPUSD"])
    assert found["EURUSD"] == ["EURUSD.m", "EURUSDpro"] and found["GBPUSD"] == []
    assert col.estimate_server_offset_hours("EURUSD") == 3.0


def test_missing_metatrader5_package_gives_clear_error():
    with pytest.raises(MT5Error, match="uv sync --extra mt5"):
        MT5Collector().connect()


# ------------------------------------------------------------------ CLI


def test_cli_download_info_validate(tmp_path, capsys):
    d = str(tmp_path)
    args = [
        "--data-dir",
        d,
        "download",
        "--source",
        "synthetic",
        "--symbols",
        "EURUSD",
        "BTCUSD",
        "--timeframes",
        "1h",
        "1d",
        "--start",
        "2023-01-02",
        "--end",
        "2023-02-01",
    ]
    assert main(args) == 0
    assert main(["--data-dir", d, "validate"]) == 0
    assert main(["--data-dir", d, "info"]) == 0
    out = capsys.readouterr().out
    assert "EURUSD" in out and "BTCUSD" in out and "0 failure" in out


def test_cli_validate_flags_tampered_data(tmp_path, capsys):
    d = str(tmp_path)
    main(
        [
            "--data-dir",
            d,
            "download",
            "--source",
            "synthetic",
            "--symbols",
            "EURUSD",
            "--timeframes",
            "1h",
            "--start",
            "2023-01-02",
            "--end",
            "2023-02-01",
        ]
    )
    csv = tmp_path / "EURUSD" / "1h.csv"
    csv.write_text(csv.read_text() + "\n")
    assert main(["--data-dir", d, "validate"]) == 1
    assert "CHECKSUM MISMATCH" in capsys.readouterr().out


def test_cli_csv_source_requires_directory(tmp_path):
    assert main(["--data-dir", str(tmp_path), "download", "--source", "csv"]) == 2

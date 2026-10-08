"""Market-data collectors: MT5 (demo), CSV files, and deterministic synthetic data."""

from tradingbot.data.collectors.base import Collector
from tradingbot.data.collectors.csv_collector import CSVCollector
from tradingbot.data.collectors.mt5 import MT5Collector, MT5Error
from tradingbot.data.collectors.synthetic import SyntheticCollector

__all__ = ["CSVCollector", "Collector", "MT5Collector", "MT5Error", "SyntheticCollector"]

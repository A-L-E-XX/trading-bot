"""MetaTrader 5 collector (DEMO accounts only).

The ``MetaTrader5`` Python package only runs on Windows next to a running MT5 terminal. The module
is imported lazily (or injected for tests), so the rest of the project works everywhere.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from tradingbot.data.normalizer import to_utc

_TF_ATTR = {"1h": "TIMEFRAME_H1", "4h": "TIMEFRAME_H4", "1d": "TIMEFRAME_D1"}
_CHUNK_DAYS = {"1h": 365, "4h": 4 * 365, "1d": 10 * 365}  # keeps each request well under bar limits


class MT5Error(RuntimeError):
    pass


class MT5Collector:
    name = "mt5"

    def __init__(
        self,
        *,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        path: str | None = None,
        symbol_suffix: str = "",
        server_utc_offset_hours: float = 0.0,
        server_timezone: str | None = None,
        mt5_module=None,
    ):
        self._login, self._password, self._server, self._path = login, password, server, path
        self.symbol_suffix = symbol_suffix
        self.server_utc_offset_hours = server_utc_offset_hours
        self.server_timezone = server_timezone
        self._mt5 = mt5_module
        self._connected = False

    # -- connection ---------------------------------------------------------------
    def connect(self) -> None:
        if self._connected:
            return
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5  # type: ignore[import-not-found]
            except ImportError as exc:
                raise MT5Error(
                    "The MetaTrader5 package is not installed (Windows only). "
                    "Run: uv sync --extra mt5"
                ) from exc
            self._mt5 = mt5
        kwargs: dict = {}
        if self._path:
            kwargs["path"] = self._path
        if self._login:
            kwargs.update(login=self._login, password=self._password, server=self._server)
        if not self._mt5.initialize(**kwargs):
            raise MT5Error(f"MT5 initialize failed: {self._mt5.last_error()}")
        self._connected = True
        if not self.account_is_demo():
            self.close()
            raise MT5Error("Refusing to continue: this MT5 account is NOT a demo account.")

    def close(self) -> None:
        if self._mt5 is not None and self._connected:
            self._mt5.shutdown()
        self._connected = False

    def account_info(self):
        info = self._mt5.account_info()
        if info is None:
            raise MT5Error("No account information (is the terminal logged in?)")
        return info

    def account_is_demo(self) -> bool:
        return self.account_info().trade_mode == self._mt5.ACCOUNT_TRADE_MODE_DEMO

    # -- discovery helpers (used by `tb-data mt5-check`) -----------------------------
    def broker_symbol(self, symbol: str) -> str:
        return f"{symbol}{self.symbol_suffix}"

    def find_symbols(self, wanted: list[str]) -> dict[str, list[str]]:
        """For each wanted symbol, the broker symbols whose name starts with it."""
        all_names = [s.name for s in (self._mt5.symbols_get() or [])]
        return {
            w: sorted(n for n in all_names if n.upper().startswith(w.upper()))[:8] for w in wanted
        }

    def estimate_server_offset_hours(self, symbol: str) -> float | None:
        """Server-time offset from UTC, from the latest tick (needs the market open)."""
        tick = self._mt5.symbol_info_tick(self.broker_symbol(symbol))
        if tick is None:
            return None
        server_now = datetime.fromtimestamp(tick.time, tz=UTC).replace(tzinfo=None)
        delta_h = (server_now - datetime.now(UTC).replace(tzinfo=None)).total_seconds() / 3600
        return float(round(delta_h))

    # -- data -----------------------------------------------------------------------
    def fetch(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        self.connect()
        broker_symbol = self.broker_symbol(symbol)
        if not self._mt5.symbol_select(broker_symbol, True):
            raise MT5Error(
                f"symbol {broker_symbol!r} not available on this broker: {self._mt5.last_error()}"
            )
        tf = getattr(self._mt5, _TF_ATTR[timeframe])
        # Request a padded range (MT5 interprets dates in server time) and trim after converting.
        lo, hi = start - timedelta(days=1), end + timedelta(days=1)
        step = timedelta(days=_CHUNK_DAYS[timeframe])
        frames, cursor = [], lo
        while cursor < hi:
            chunk_end = min(cursor + step, hi)
            rates = self._mt5.copy_rates_range(broker_symbol, tf, cursor, chunk_end)
            if rates is not None and len(rates):
                frames.append(pd.DataFrame(np.asarray(rates)))
            cursor = chunk_end
        if not frames:
            raise MT5Error(
                f"MT5 returned no {timeframe} candles for {broker_symbol}. Open its chart in the "
                "terminal and scroll back so the history downloads, or check the date range."
            )
        raw = pd.concat(frames, ignore_index=True)
        server_time = pd.to_datetime(raw["time"], unit="s")
        raw["time"] = to_utc(
            server_time,
            utc_offset_hours=self.server_utc_offset_hours,
            timezone=self.server_timezone,
        )
        raw = raw.rename(columns={"tick_volume": "volume"})
        raw = raw[["time", "open", "high", "low", "close", "volume"]]
        return raw[
            (raw["time"] >= pd.Timestamp(start)) & (raw["time"] < pd.Timestamp(end))
        ].reset_index(drop=True)

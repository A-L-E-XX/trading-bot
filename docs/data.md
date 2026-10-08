# Market data (Milestone 3) and features (Milestone 4)

## Canonical format

Every dataset, whatever its source, is stored as:

- `data_store/<SYMBOL>/<timeframe>.csv` with columns `time,open,high,low,close,volume`
- `data_store/<SYMBOL>/<timeframe>.meta.json` with source, row count, period, SHA-256 checksum, validation result

`time` is the **candle open time in UTC** (`2023-01-02T10:00:00Z`). `volume` is **tick volume** (number of
price updates), not traded amount. That is the only volume MT5 provides for forex and metals.

## Pipeline

`collector.fetch` → `normalize_ohlcv` (UTC, sort, de-duplicate, float64) → `validate_ohlcv` → `DatasetStore.save`.
Datasets that fail validation are **not saved** unless you pass `--allow-invalid`.

## What the validator checks

| Check | Severity |
|---|---|
| Empty data, wrong index, missing columns | error |
| Duplicate or out-of-order timestamps | error |
| Candles not starting on the hour | error |
| NaN / infinite values, price ≤ 0, negative volume | error |
| high < low, high < max(open, close), low > min(open, close) | error |
| Missing candles. Weekends and **recurring broker breaks** (same weekly time slot missing across the dataset, e.g. gold's daily break) are ignored for forex/gold; crypto trades 24/7 so nothing is ignored | warning, **error above 2%** |
| Timestamps off the expected grid (often a daylight-saving shift) | warning |
| Close-to-close move > 30%, or more than 5% flat candles | warning |

## Commands (run on the Windows PC that has MT5)

```powershell
uv sync --extra mt5                       # installs the MetaTrader5 package (Windows only)
# fill TB_MT5_LOGIN, TB_MT5_PASSWORD, TB_MT5_SERVER in .env  (demo account only)
uv run tb-data mt5-check                  # connection, broker symbol names, server time offset
uv run tb-data download --source mt5 --start 2021-01-01 --server-utc-offset 2
uv run tb-data validate                   # re-check every stored dataset and its checksum
uv run tb-data info
```

Other sources: `--source csv --csv-dir <folder>` (files named `EURUSD_H1.csv` or `EURUSD_1h.csv`, MT5
"Export bars" format supported) and `--source synthetic` (fake random-walk data for development only).

## Known limitations

- **Broker time zone.** MT5 timestamps are in broker server time, not UTC. Use `mt5-check` to see the offset
  and pass `--server-utc-offset N`, or `--server-timezone Europe/Athens` to follow daylight saving. A wrong
  offset shifts every candle; check that the first candles of a day look right.
- **History depth.** Brokers only provide a limited history. If the terminal has not downloaded older bars, MT5
  returns fewer than requested; open the chart and scroll back first.
- **Broker-specific prices.** Spreads and candle values differ between brokers, so results on one demo broker
  are not guaranteed to match another.
- **Holidays** appear as missing-candle warnings, not errors, unless they exceed 2%.
- **Synthetic data** is random and has no edge. Never judge a strategy on it.

## Feature engine

`compute_features(ohlcv, FeatureParams)` returns the input plus: `ret`, `log_ret`, `sma`, `ema_fast`, `ema_slow`,
`ema_trend`, `rsi`, `atr`, `volatility`, `volume_ratio`, and a `ready` flag that is True once every feature
Strategy V1 needs is valid (after 200 candles with the default EMA 200).

- **Deterministic:** the same input gives bit-identical output (tested).
- **No look-ahead:** the row at time t uses only candles up to t. Tests prove that adding, removing or editing later
  candles never changes earlier features.
- **Warm-up:** values are NaN until enough history exists. EMA, RSI and ATR are seeded from the first value (not an
  SMA seed as TA-Lib does), so early values differ slightly from other platforms and converge afterwards.

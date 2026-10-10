# Strategy V1 — EMA Trend + ATR Stop (rules-level spec)

Deterministic, explainable. Signals only; the strategy never places orders.

**Indicators** (on closed candles only): EMA(20) fast, EMA(50) slow, EMA(200) trend filter, ATR(14).

**Long entry** (on candle close): close > EMA200 and EMA20 crosses above EMA50.
**Short entry**: close < EMA200 and EMA20 crosses below EMA50.
Entry fills at the **next candle's open** (no look-ahead).

**Stop:** entry ∓ 1 × ATR(14) (configurable `atr_stop_mult`; 2 × was too wide for a $100 account at 0.01 lot), measured at the signal candle.
**Exit:** opposite EMA20/EMA50 cross, or stop hit. No take-profit in V1. If the opposite cross also passes the trend filter, the strategy exits and reverses on the same candle.
**One position per symbol/timeframe.** Warm-up: no signals until EMA200 is valid (200 candles).

**Each signal records:** symbol, timeframe, direction, timestamp, strategy name/version, reason, stop price.

All parameters live in `config/default.toml` under `[strategy_v1]`. Defaults are standard starting values, not optimised ones; any tuning happens in Milestone 7 with out-of-sample checks.

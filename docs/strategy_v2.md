# Strategy V2 - Bollinger/RSI mean reversion (rules-level spec)

Added after Strategy V1 failed the research gate. Run it with `--strategy v2`.

**Indicators** (closed candles only): SMA(20) mean, Bollinger bands = mean +/- 2 x population std(20),
RSI(14), ATR(14).
**Long entry:** close below the lower band AND RSI < 30, no open position.
**Short entry:** close above the upper band AND RSI > 70, no open position (if `allow_short`).
**Exit:** close back at/through the mean (long: close >= SMA; short: close <= SMA), or the stop.
**Stop:** entry -/+ 1.5 x ATR(14) at the signal candle. No take-profit, no trend filter.
**Fills:** signal at the close of candle i, executed at the open of candle i+1 (same as V1).
Parameters were fixed before out-of-sample results were examined (config `[strategy_v2]`).

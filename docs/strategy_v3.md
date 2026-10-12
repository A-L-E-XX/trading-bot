# Strategy V3 - volume-price breakout (VPA)

**Our interpretation** of the VPA video (accumulation/distribution, price vs volume). The creator's
exact numbers were not given, so every value below is OUR proposal, fixed before any OOS look.
Volume is MT5 *tick* volume, not exchange volume.

- Range = high/low of the previous 20 candles (the current candle is excluded).
- Long: range tight (width <= 6 ATR), candle closes above the range high, is a up candle, and
  volume >= 1.5 x its 20-candle average. Short is the mirror.
- Stop 1 ATR. Exit on an opposite candle with high volume. (No profit target: the engine exits on
  signals; adding targets is a later change.)
- Run: `tb-research --strategy v3 ...`, `tb-backtest --strategy v3 ...`, `tb-paper run --strategy v3`.

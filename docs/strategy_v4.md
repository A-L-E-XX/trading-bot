# Strategy V4 - opening-range breakout scalper (ORB)

**Our interpretation** of the ORB video. The video shows a higher-timeframe range of the first few
candles after the open, then a 1-minute chart with the range box and a breakout. It does not give
the exact range length, entry trigger, stop or target, so every number below is OUR proposal,
fixed before any out-of-sample look. A retest/pullback entry (as drawn in the video) is NOT built
yet.

- Session: US cash open 09:30 America/New_York (daylight-saving aware, never a fixed UTC offset).
- Range: high/low of the candles starting 09:30-10:00. Hidden until the first candle at/after 10:00.
- Entry: the FIRST candle close above (long) or below (short) the range between 10:00 and 12:00.
  One trade per day. A day is skipped if the open candle is missing or the range < 2 ATR.
- Stop: the opposite side of the range. Exit: flatten at 15:55 local. (No profit target: the
  engine exits on signals; targets are a later change.)
- Run on 5m data: `tb-research --strategy v4 --timeframes 5m --symbols USTEC US30_x10 US500_x100`.

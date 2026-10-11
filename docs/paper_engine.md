# Paper engine (Milestone 9)

`tradingbot.execution.PaperEngine`: one strategy on one symbol/timeframe with fake money. It never
talks to a broker. Feed it CLOSED candles with `on_bar(time, o, h, l, c, volume)`.

Per candle, in this order (same model as the backtester):
1. Execute orders decided at the previous close, at this candle's open (exits first). Entries go
   through the risk engine (Milestone 8); rejections are logged with a reason code.
2. Check the stop against this candle's high/low (gap through the stop fills at the open).
3. Mark equity, update the risk engine; flatten and halt if the kill switch trips.
4. Ask the strategy for signals from the candle that just closed; they wait for the next candle.

Safety properties (all tested)
- **Replay equals backtest:** feeding history reproduces the backtest trade for trade. Verified on
  real data: V2 USDJPY 4h (244 trades), V1 EURUSD 4h (77), V2 EURUSD 1h (852), all identical, with
  both full-history and trailing-1500-candle features.
- **Idempotent:** a candle at or before the last processed time is ignored (`bar_ignored`).
- **Fails closed on bad data:** non-finite/non-positive prices or inconsistent OHLC are refused
  (`bar_rejected`) and not traded; naive timestamps raise.
- **Restart-safe:** state is written atomically after every candle (`state_path`); a restart needs
  history up to the last processed candle, otherwise it refuses to start. State for another
  symbol/timeframe is refused.
- **Audit trail:** append-only JSON-lines event log (`event_log_path`): signal, order_rejected,
  position_opened, position_closed, daily_loss_limit, kill_switch, halted, bar_ignored, bar_rejected.
- `spread_override` lets a real quote's spread replace the snapshot spread (used by Milestone 10).

Not yet: running several symbols against one shared account (the risk engine supports it, the
engine class is single-symbol), a live candle feed, and any broker connection (Milestone 10).

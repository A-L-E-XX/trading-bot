# Backtesting (Milestones 5 and 6)

Run: `uv run tb-backtest` (all symbols and timeframes) or e.g.
`uv run tb-backtest --symbols EURUSD --timeframes 4h --spread-multiplier 2`.
Outputs go to `outputs/backtests/<SYMBOL>_<TF>/` (trades.csv, signals.csv, equity.csv,
summary.json, equity.png) plus `report.md` and `summary.csv`.

## Execution model
- Candle prices are BID. A signal made at the close of candle i is filled at the OPEN of candle i+1.
- Longs buy at ask (bid + spread) + slippage and sell at bid - slippage; shorts the reverse.
- Stops are checked against each candle's high/low (including the entry candle; shorts trigger on
  ask = high + spread). A gap through the stop fills at the open, not at the stop.
- Fixed 0.01 lot, one position at a time, profit converted to USD, equity marked to market.
- If equity reaches 0 the run halts (`account_blown`).
- The per-trade risk limit (5%) is only REPORTED (`trades_over_risk_limit`); Milestone 8 enforces it.
- The strategy may reverse on the same candle: if the opposite cross also passes the trend
  filter, it exits and enters on that same signal candle.

## Costs
Spread comes from one snapshot of the broker's current spread (`config/instrument_specs.json`);
slippage = 0.25 x spread per fill; commission 0. Real spreads vary (wider at rollover and news),
so use `--spread-multiplier` to stress.

## How it is checked
- Unit tests with hand-computed fills, costs, stops, JPY and gold conversions.
- Look-ahead tests: damaging the next candle must not change today's decision.
- An independent re-implementation (shares no code) re-derived every signal and re-priced every
  trade of all 21 real-data runs, and agreed.

## Reading results
In-sample results are NOT evidence of an edge. On a $100 account a fixed 0.01 lot makes gold and
BTC risk far more than 5% per trade, so those profits are not realistic. The research gate
(Milestone 7) decides whether the strategy continues.

## Small accounts and cent accounts (added 2026-10-10)
- `--balance 20` sets the starting balance. `--lot-scale 0.01` models a cent account (one lot is
  worth 1/100; verify with specs exported from a cent demo). `--risk-sizing --risk-pct 3` sizes each
  trade so its stop risks about 3% of the balance (lots rounded DOWN to the step; trades that
  cannot fit under the limit are skipped). `--max-lots` caps the size.
- Example: `uv run tb-research --balance 20 --lot-scale 0.01 --risk-sizing --risk-pct 3`
- Only symbols whose profit currency is USD, or pairs with USD as the base (USDJPY, USDCAD, USDCHF),
  are supported. Cross pairs (EURJPY, GBPJPY, EURGBP...) need a cross-rate conversion that is not
  built yet.

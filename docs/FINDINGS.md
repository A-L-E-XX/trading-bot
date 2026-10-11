# Findings (2026-10-11)

The MVP was stopped after Milestone 7 (research gate) by decision of the project owner.

## What exists and is tested (137 tests)
- Data: MT5 download, validation (gaps, broker breaks, OHLC checks), checksum-verified storage.
- Features: causal EMA/RSI/ATR/Bollinger std; look-ahead protection tests.
- Backtester: bid/ask spread, slippage, stops on candle ranges, gaps, USD conversion, risk-based
  sizing, cent-account lot scale, enforceable per-trade risk limit; independently re-verified
  on all 21 original real-data runs.
- Research gate: out-of-sample, parameter sensitivity, 2x spread stress, Monte Carlo, walk-forward.

## Results
| Strategy | Setup | Candidates | Passed |
|---|---|---|---|
| V1 EMA trend + ATR stop (1x) | $100, 0.01 lot, 3% risk | 21 | 0 |
| V1 | $20 cent account, 3% risk sizing | 33 | 0 |
| V2 Bollinger/RSI mean reversion | $20 cent account, 3% risk sizing | 33 | 0 |

Conclusion: neither simple price-only strategy shows an edge that survives out-of-sample data,
doubled spreads and the 30% drawdown rule. Nothing has earned live money. A 50% per-trade stop on
a $20 account is not survivable under any tested strategy.

## If work resumes
1. Milestones 8-14 (risk engine, paper engine, MT5 adapter, order management, monitoring,
   security, release gate) are not built. The bot must stay demo/paper only.
2. Re-export broker specs on a weekday (`uv run --extra mt5 tb-data specs`): the 4 later symbols
   (NZDUSD, USDCHF, XAGUSD, ETHUSD) have provisional weekend spreads.
3. Verify cent-account lot value (assumed 0.01) with specs from a real cent demo.
4. Any new strategy must go through `tb-research` unchanged; do not loosen the gate to pass it.

# Findings (updated 2026-10-13)

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

| V3 volume-price breakout (VPA) | $20 cent, 3% risk, forex/metals/crypto, 1h/4h/1d | 33 | 0 |
| V1 / V2 / V3 on US indices (USTEC, US30_x10, US500_x100) | $20 cent, 3% risk, 15m/1h/4h | 27 | 1 (V2 USTEC 4h) |
| V4 opening-range breakout scalper (5m, NY open) | $20 cent, 3% risk, 3 indices | 3 | 0 |

**V2 USTEC 4h** passed every check but only just (2 of 4 folds profitable against a 50% minimum,
Monte Carlo breach 21.6% against 25%), the in-sample period lost money (profit factor 0.69), the
neighbouring 15m and 1h timeframes failed, and it is 1 pass out of 27 index candidates. Status:
observation only, forward paper test running (see "Open experiment" below). Not proven.

**V4 scalper** lost money out of sample on all three indices (profit factor 0.73-0.88 with 66-108
trades), so it is closed. A retest-entry variant (as drawn in the video) was NOT tested; testing it
would be one more candidate and raises the chance of a lucky pass.

Earlier conclusion: neither simple price-only strategy shows an edge that survives out-of-sample data,
doubled spreads and the 30% drawdown rule. Nothing has earned live money. A 50% per-trade stop on
a $20 account is not survivable under any tested strategy.

## Open experiment
USTEC 4h V2, paper only: `tb-paper run --strategy v2 --symbol USTEC --timeframe 4h --balance 20
--lot-scale 0.01 --risk-sizing --risk-pct 3 --server-utc-offset 0`. Rules fixed in advance: stop and
rethink if the kill switch fires, or if after 3 months it has at least 8 trades and profit factor
below 0.8. No real money before 6 months, and only with profit factor above about 1.2.

## If work resumes
1. Milestones 11-14 (order management, monitoring, security, release gate) are not built.
   Risk engine, paper engine and the read-only MT5 feed (M8-M10) are done. The bot must stay
   demo/paper only.
2. Re-export all specs on a weekday (`uv run --extra mt5 tb-data specs`): NZDUSD, USDCHF, XAGUSD
   and ETHUSD still carry provisional weekend spreads (the three indices were exported on a weekday).
3. Verify cent-account lot value (assumed 0.01) with specs from a real cent demo.
4. Any new strategy must go through `tb-research` unchanged; do not loosen the gate to pass it.

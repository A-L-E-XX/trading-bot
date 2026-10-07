# Product Requirements Document — Trading Bot MVP (v1)

**Principle:** build the smallest complete trading system that can be tested honestly.

## 1. What Version 1 does

A Python system that, for BTC, XAU and 5 forex majors on 1h / 4h / 1d candles:

1. downloads and validates historical OHLCV data from an MT5 demo account;
2. computes features and runs one deterministic, explainable strategy (Strategy V1);
3. backtests it with realistic costs and no look-ahead;
4. decides via a research gate whether the strategy deserves to continue;
5. wraps it in an independent risk engine;
6. runs it continuously in **paper mode** against a **MT5 demo account**.

**Out of scope:** real-money trading, multiple brokers, ML, dashboard, public users, HFT, Java service.

## 2. Decisions (from you)

| Item | Decision |
|---|---|
| Milestones | 14-milestone MVP only |
| Instruments | BTC, XAU, 5 forex majors |
| Timeframes | 1h, 4h, 1d |
| Strategy | EMA trend-following + ATR stop + trend filter |
| Broker | Any MT5 demo account |
| Language | Python |
| Account / sizing | $100 demo account, fixed 0.01 lot |
| Max drawdown | 30% (= $30) |

## 3. Assumptions I made (please confirm or correct)

- **The 5 forex majors** are EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD.
- **"$100, lot size 0.01, drawdown 30%"** means a $100 account, always trading 0.01 lot, and a 30% maximum drawdown that fails the strategy and triggers the kill switch.
- **Risk numbers** in `config/default.toml` marked PROPOSED (5% risk per trade, max 3 positions, 10% daily loss limit, research-gate thresholds).

## 4. Constraints worth knowing

- **$100 is very small with BTC and XAU.** At 0.01 lot, 1 oz of gold moves $1 per $1. A $30 drawdown can come from a handful of stops, so BTC/XAU results will be noisy. Forex at 0.01 lot is gentler (~$0.10 per pip on most pairs).
- **Short samples on 1d.** Daily candles give few trades, so results there carry less statistical weight. The research gate has a minimum trade count for this reason.
- **MT5 Python package is Windows-only.** The cloud workspace cannot connect to MT5. Data download and the paper-trading bot must run on your Windows machine (or a Windows VPS). I develop and test everything else against synthetic and CSV data here.
- **Broker symbols differ** (suffixes like `EURUSD.m`, `BTCUSD` vs `BTCUSDT`). `symbol_suffix` handles the common case.

## 5. Security requirements

- Demo credentials only, in `.env` (git-ignored). Secrets are `SecretStr` and never logged.
- Live mode is not an allowed setting.

## 6. Success criteria (MVP)

Clean clone installs and tests pass; data regenerates and validates; backtests are reproducible, include costs and have no look-ahead; Strategy V1 passes the research gate (out-of-sample PF ≥ 1.2, ≥ 30 trades, max drawdown ≤ 30%, parameter stability); risk limits cannot be bypassed; the bot runs in paper mode and fails safely.

**Failure rule:** if the research gate fails, we go back to strategy research instead of building more infrastructure around it.

## 7. Milestone map

M1 Spec · M2 Environment · M3 Data · M4 Features · M5 Strategy · M6 Backtest · M7 Research gate · M8 Risk · M9 Paper engine · M10 MT5 adapter · M11 Order management & reconciliation · M12 Monitoring · M13 Security & reliability · M14 Release gate

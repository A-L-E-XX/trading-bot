# Risk engine (Milestone 8)

`tradingbot.risk.RiskEngine` is the single place that decides whether an ENTRY is allowed and how
big it is. Exits are never blocked. Pure logic (no clock, no broker): the caller passes the time, so
backtests and paper trading behave identically. Limits come from `config/default.toml` `[risk]`.

| Rule | Reason code | Effect |
|---|---|---|
| Kill switch (drawdown from peak equity >= `kill_switch_drawdown_pct`, or equity not finite) | `kill_switch_active` | Latched: blocks entries until `reset_kill_switch()` (a deliberate human action). Backtests flatten and stop. |
| Daily loss (equity down >= `daily_loss_limit_pct` since the start of the UTC day) | `daily_loss_limit_reached` | Blocks entries until the next UTC day. |
| Max open positions / one position per symbol | `max_open_positions`, `position_already_open_for_symbol` | Rejects |
| Total exposure (normal-lot equivalents; a cent lot counts 1/100) | `max_total_exposure_lots` | Rejects |
| Invalid / missing stop distance | `invalid_stop_distance` | Rejects |
| Even the minimum lot risks more than `max_risk_per_trade_pct` of equity | `stop_too_wide_for_account` | Rejects |
| Anything unexpected (bad price, bad lot step, exceptions) | `invalid_input` | Rejects (fails closed) |

Sizing: lots = floor(equity x risk% / (stop distance x contract size x lot value x USD value)) to
the lot step, never above the budget. Backtests use it with `--enforce-risk` or `--risk-sizing`;
the research gate always does.

Tested: every rule, the kill switch latching, day rollover, cent-lot exposure, failing closed, and
inside the backtester (rejection counts, gap loss -> kill switch, same-day re-entry blocked).
Mutation-checked: breaking the lot rounding or the kill switch makes the tests fail.

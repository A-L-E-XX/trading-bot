# Demo execution (M11)

Optional: mirrors the paper engine's trades on your **MT5 demo account**. Off unless you pass
`--demo-orders`. The paper engine stays the brain; this only turns its decisions into orders.

## Run (Windows cmd; MT5 open and logged in to the DEMO account)

    uv run --extra mt5 tb-paper run --strategy v2 --symbol USTEC --timeframe 4h ^
        --balance 20 --lot-scale 0.01 --risk-sizing --risk-pct 3 --server-utc-offset 0 ^
        --demo-orders

Panic button (closes only this bot's positions on that symbol, tagged with its magic number):

    uv run --extra mt5 tb-paper flatten --symbol USTEC

## Safety rules (each has a test)
- Only `execution/broker.py` can send orders; a test scans every other file for order functions.
- The account is re-checked as DEMO before every order and every close. A real account is refused.
- No stop-loss, no order. Lots are rounded to the broker step and capped (`--max-demo-lots`, 1.0).
- The loss at the stop is computed by the BROKER (`order_calc_profit`) on the REAL account equity
  and the order is scaled down, or skipped, if it exceeds the risk limit (3%).
- Only positions with our magic number (`--magic`, default 770001) are ever touched.
- A paper-engine halt (kill switch etc.) closes our positions and blocks new ones.
- A position at the broker we do not know about is closed (orphan). A signal is sent once, even
  across restarts. A failed order is logged and not retried. If the broker stops out and the paper
  engine still holds the trade, it is logged (`demo_mismatch`) and not re-entered.

## What it tells you
The broker acts one candle earlier than the paper fill (paper fills at the next open, which it
only sees a candle later). Events in `events.jsonl`: `demo_order_sent` (requested vs fill price,
`risk_at_stop` from the broker vs the paper `paper_risk_usd`), `demo_position_closed`
(`broker_pnl`), `demo_order_failed`, `demo_order_skipped`, `demo_mismatch`, `demo_orphan_closed`.
Comparing `risk_at_stop` with `paper_risk_usd` also checks the cent-account lot-value assumption.

## Not verified yet
Tested only against a fake MT5. First real run: use `--once` on a quiet moment and check the
events log, and your MT5 "Trade" tab, before leaving it running.

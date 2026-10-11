# Live feed (M10) - read-only

Connects to an MT5 **demo** account, pulls candles and quotes, and feeds the paper engine.
It never places orders (a test scans the code for order functions).

## Run (Windows cmd, MT5 terminal open and logged in to the demo)

    uv run --extra mt5 tb-paper run --strategy v2 --symbol USDJPY --timeframe 4h ^
        --balance 20 --lot-scale 0.01 --risk-sizing --risk-pct 3 --server-utc-offset 0 --once

Drop `--once` to keep polling every 60 s. Check progress with:

    uv run tb-paper status --name <name>

Output: `outputs/paper/<name>/state.json` and `events.jsonl`.

## Behaviour
- Only **closed** candles are processed (the forming candle is dropped).
- On start it catches up on candles missed while offline; restarts are idempotent.
- Fetch errors are retried with backoff; after 30 errors in a row the session stops.
- Wrong server offset or duplicate candles raise a feed error rather than guessing.

## Not yet verified
Behaviour against a real terminal (server offset, closed-candle logic, quotes) is only
tested with a fake MT5. Run `--once` first and review the events log.

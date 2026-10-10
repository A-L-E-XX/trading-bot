# Research gate (Milestone 7)

Run: `uv run tb-research` -> `outputs/research/gate_report.md` (+ gate.csv, gate.json).

A candidate (symbol + timeframe) passes only if EVERY check passes:
1. Out-of-sample (last 40% of history) trades >= `min_oos_trades`.
2. OOS profit factor >= `min_oos_profit_factor`.
3. OOS max drawdown <= `max_oos_drawdown_pct`.
4. Parameter sensitivity: each of EMA fast/slow/trend and the ATR multiplier moved +/-20%; the
   profit factor may not drop by more than `max_param_sensitivity_drop_pct`.
5. Cost stress: profit factor at DOUBLE the spread >= `min_stress_profit_factor`.
6. Monte Carlo: the OOS trades are resampled 2000 times; the share of paths reaching the drawdown
   limit must be <= `max_mc_breach_probability_pct`.
7. Walk-forward: the same fixed rules over 4 consecutive time windows; the share with profit
   factor > 1 must be >= `min_profitable_folds_pct`.

Rules that keep it honest: parameters are not tuned on the OOS data; the per-trade risk limit is
enforced; no trades means FAIL; 21 candidates are tested, so a pass is "worth paper trading",
not proof. Thresholds live in `config/default.toml` under `[research_gate]`.

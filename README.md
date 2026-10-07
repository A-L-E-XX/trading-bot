# trading-bot

MVP trading platform: research → backtest → risk → paper trading (MT5 demo). No live trading.

See [docs/PRD.md](docs/PRD.md) and [docs/strategy_v1.md](docs/strategy_v1.md).

## Quick start

```bash
git clone <your-repo-url> trading-bot && cd trading-bot
uv sync                      # installs deps into .venv
cp .env.example .env         # fill in later (demo credentials only)
uv run pytest                # tests pass
uv run ruff check .          # lint
uv run ruff format --check . # formatting
```

On the Windows machine that runs MT5, also run `uv sync --extra mt5`.

## Layout

`src/tradingbot/` holds data, features, strategies, signals, backtesting, risk, execution, portfolio and analytics. Config is in `config/default.toml`; secrets come from `.env`.

## Progress

- [x] M1 Product definition
- [x] M2 Development environment
- [ ] M3 – M14

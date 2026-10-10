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

## Get data

```bash
uv run tb-data download --source synthetic --symbols EURUSD --start 2023-01-01   # fake data, to try it out
uv run tb-data mt5-check          # Windows + MT5 demo: check connection and symbol names
```

See [docs/data.md](docs/data.md) for real MT5 data.

## Layout

`src/tradingbot/` holds data, features, strategies, signals, backtesting, risk, execution, portfolio and analytics. Config is in `config/default.toml`; secrets come from `.env`.

## Progress

- [x] M1 Product definition
- [x] M2 Development environment
- [x] M3 Market data pipeline ([docs/data.md](docs/data.md))
- [x] M4 Feature engine
- [x] M5 Strategy V1 ([docs/strategy_v1.md](docs/strategy_v1.md))
- [x] M6 Backtest engine ([docs/backtesting.md](docs/backtesting.md))
- [ ] M7 – M14

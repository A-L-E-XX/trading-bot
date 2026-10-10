"""Backtest engine: chronological replay with realistic fills, costs and stops.

No look-ahead, by construction
------------------------------
* A signal is produced at the CLOSE of candle i from candles <= i only.
* It is executed at the OPEN of candle i + 1 (never at candle i's price).
* Stops are checked against each candle's high/low after the entry fill, so a stop can trigger on
  the entry candle itself. If the candle opens through the stop (a gap), the fill is at the open.

Prices and costs
----------------
Candle prices are BID prices. Buys fill at ask (= bid + spread), sells at bid, and slippage is added
against us on every fill. Profit is converted to USD (USDJPY / USDCAD profit is in JPY / CAD).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tradingbot.backtesting.costs import CostModel
from tradingbot.backtesting.instruments import InstrumentSpec
from tradingbot.backtesting.metrics import compute_metrics
from tradingbot.signals import Action, Side, Signal
from tradingbot.strategies.base import Strategy

REQUIRED_COLUMNS = ("open", "high", "low", "close", "ready")


@dataclass(frozen=True)
class BacktestConfig:
    initial_balance: float = 100.0
    lots: float = 0.01
    costs: CostModel = field(default_factory=CostModel)
    max_drawdown_limit_pct: float = 30.0  # reported on, not enforced (the risk engine does that)
    max_risk_per_trade_pct: float = 5.0
    # False: only report trades over the limit. True: skip entries whose stop risk exceeds it
    # (what the Milestone 8 risk engine will do live).
    enforce_risk_limit: bool = False


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    trades: pd.DataFrame
    equity: pd.Series
    signals: pd.DataFrame
    metrics: dict
    halted_reason: str | None = None
    ignored_entry_signals: int = 0
    rejected_over_risk: int = 0


@dataclass
class _Position:
    side: Side
    entry_i: int
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    stop_distance: float
    entry_reason: str
    balance_at_entry: float
    bars: int = 0


def run_backtest(
    features: pd.DataFrame,
    strategy: Strategy,
    spec: InstrumentSpec,
    config: BacktestConfig,
    symbol: str,
    timeframe: str,
) -> BacktestResult:
    """Replay ``features`` (OHLC + indicator columns, ``ready`` flag) through ``strategy``."""
    missing = [c for c in (*REQUIRED_COLUMNS, *strategy.required_features) if c not in features]
    if missing:
        raise ValueError(f"features are missing columns: {sorted(set(missing))}")
    if not features.index.is_monotonic_increasing or features.index.has_duplicates:
        raise ValueError("features must be sorted by time with unique timestamps")
    spec.check_lot(config.lots)

    times = features.index
    rows = features.to_dict("records")
    n = len(rows)
    spread = config.costs.spread(spec)
    slip = config.costs.slippage(spec)
    units = config.lots * spec.contract_size
    commission_side = config.costs.commission_per_lot * config.lots

    balance = config.initial_balance
    pos: _Position | None = None
    pending: list[Signal] = []
    trades: list[dict] = []
    equity: list[float] = []
    equity_times: list[pd.Timestamp] = []
    signal_log: list[dict] = []
    ignored = 0
    rejected = 0
    halted: str | None = None

    def close(i: int, exit_price: float, reason: str) -> None:
        nonlocal balance, pos
        assert pos is not None
        move = (
            exit_price - pos.entry_price if pos.side == Side.LONG else pos.entry_price - exit_price
        )
        commission = 2 * commission_side
        pnl = move * units * spec.quote_to_usd(exit_price) - commission
        cost = (spread + 2 * slip) * units * spec.quote_to_usd(pos.entry_price) + commission
        risk = pos.stop_distance * units * spec.quote_to_usd(pos.entry_price)
        balance += pnl
        trades.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "side": pos.side.value,
                "entry_time": pos.entry_time,
                "entry_price": pos.entry_price,
                "exit_time": times[i],
                "exit_price": exit_price,
                "lots": config.lots,
                "stop_price": pos.stop_price,
                "stop_distance": pos.stop_distance,
                "stop_risk_usd": risk,
                "stop_risk_pct_of_balance": risk / pos.balance_at_entry * 100,
                "exit_reason": reason,
                "entry_reason": pos.entry_reason,
                "pnl_usd": pnl,
                "cost_usd": cost,
                "r_multiple": pnl / risk if risk > 0 else None,
                "bars_held": pos.bars,
                "balance_after": balance,
            }
        )
        pos = None

    def unrealized(close_price: float) -> float:
        assert pos is not None
        if pos.side == Side.LONG:
            move = close_price - pos.entry_price
        else:
            move = pos.entry_price - (close_price + spread)  # a short closes at the ask
        return move * units * spec.quote_to_usd(close_price)

    for i in range(n):
        bar = rows[i]
        o, h, lo, c = bar["open"], bar["high"], bar["low"], bar["close"]

        # 1) execute what was decided at the previous close, at this candle's open. Exits first.
        for sig in sorted(pending, key=lambda s: s.action.is_entry):
            if sig.action.is_entry:
                if pos is not None:
                    ignored += 1
                    continue
                side = Side.LONG if sig.action == Action.ENTER_LONG else Side.SHORT
                fill = o + spread + slip if side == Side.LONG else o - slip
                dist = float(sig.stop_distance)
                if config.enforce_risk_limit:
                    risk_pct = dist * units * spec.quote_to_usd(fill) / balance * 100
                    if risk_pct > config.max_risk_per_trade_pct:
                        rejected += 1
                        continue
                stop = fill - dist if side == Side.LONG else fill + dist
                pos = _Position(side, i, times[i], fill, stop, dist, sig.reason, balance)
            elif pos is not None and (
                (sig.action == Action.EXIT_LONG and pos.side == Side.LONG)
                or (sig.action == Action.EXIT_SHORT and pos.side == Side.SHORT)
            ):
                px = o - slip if pos.side == Side.LONG else o + spread + slip
                close(i, px, "signal")
        pending = []

        # 2) stop check against this candle's range (including the entry candle)
        if pos is not None:
            pos.bars += 1
            if pos.side == Side.LONG and lo <= pos.stop_price:
                close(i, min(o, pos.stop_price) - slip, "stop")
            elif pos.side == Side.SHORT and h + spread >= pos.stop_price:
                close(i, max(o + spread, pos.stop_price) + slip, "stop")

        # 3) mark to market at the close
        eq = balance + (unrealized(c) if pos is not None else 0.0)
        if eq <= 0:
            if pos is not None:
                exit_px = c - slip if pos.side == Side.LONG else c + spread + slip
                close(i, exit_px, "account_blown")
            halted = "account_blown"
            equity.append(balance)
            equity_times.append(times[i])
            break
        equity.append(eq)
        equity_times.append(times[i])

        # 4) decide at the close; acted on at the next open
        if i >= 1 and i < n - 1:
            current = pos.side if pos is not None else None
            for sig in strategy.evaluate(symbol, timeframe, times[i], rows[i - 1], bar, current):
                pending.append(sig)
                signal_log.append(
                    {
                        "bar_time": sig.bar_time,
                        "action": sig.action.value,
                        "reason": sig.reason,
                        "reference_price": sig.reference_price,
                        "stop_distance": sig.stop_distance,
                    }
                )

    if pos is not None and halted is None:  # data ended while in a trade: close at the last bid/ask
        last = n - 1
        px = (
            rows[last]["close"] - slip
            if pos.side == Side.LONG
            else rows[last]["close"] + spread + slip
        )
        close(last, px, "end_of_data")
        equity[-1] = balance

    trades_df = pd.DataFrame(trades)
    equity_s = pd.Series(equity, index=pd.DatetimeIndex(equity_times, name="time"), name="equity")
    metrics = compute_metrics(
        trades_df, equity_s, config.initial_balance, config.max_drawdown_limit_pct
    )
    if len(trades_df):
        over = trades_df["stop_risk_pct_of_balance"] > config.max_risk_per_trade_pct
        metrics["trades_over_risk_limit"] = int(over.sum())
        metrics["max_stop_risk_pct"] = float(trades_df["stop_risk_pct_of_balance"].max())
        metrics["total_cost_usd"] = float(trades_df["cost_usd"].sum())
    return BacktestResult(
        symbol=symbol,
        timeframe=timeframe,
        trades=trades_df,
        equity=equity_s,
        signals=pd.DataFrame(signal_log),
        metrics=metrics,
        halted_reason=halted,
        ignored_entry_signals=ignored,
        rejected_over_risk=rejected,
    )

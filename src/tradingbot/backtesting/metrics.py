"""Basic performance metrics (Milestone 6). The research lab (M7) builds on these."""

from __future__ import annotations

import math

import pandas as pd


def max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """(largest peak-to-trough fall in USD, largest fall as a fraction of the peak)."""
    if equity.empty:
        return 0.0, 0.0
    peak = equity.cummax()
    dd = peak - equity
    return float(dd.max()), float((dd / peak.where(peak > 0)).max())


def _ratio(returns: pd.Series, downside_only: bool) -> float | None:
    if len(returns) < 30:
        return None
    denom = returns[returns < 0].std(ddof=1) if downside_only else returns.std(ddof=1)
    if not denom or math.isnan(denom):
        return None
    return float(returns.mean() / denom * math.sqrt(365))


def compute_metrics(
    trades: pd.DataFrame, equity: pd.Series, initial_balance: float, max_dd_limit_pct: float
) -> dict:
    n = len(trades)
    pnl = trades["pnl_usd"] if n else pd.Series(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_profit, gross_loss = float(wins.sum()), float(-losses.sum())
    dd_usd, dd_frac = max_drawdown(equity)
    final_equity = float(equity.iloc[-1]) if len(equity) else initial_balance
    daily = equity.resample("1D").last().dropna().pct_change().dropna() if len(equity) else equity
    return {
        "trades": n,
        "win_rate": float((pnl > 0).mean()) if n else None,
        "net_profit_usd": float(pnl.sum()),
        "final_equity_usd": final_equity,
        "return_pct": (final_equity / initial_balance - 1) * 100,
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
        "expectancy_usd": float(pnl.mean()) if n else None,
        "avg_win_usd": float(wins.mean()) if len(wins) else None,
        "avg_loss_usd": float(losses.mean()) if len(losses) else None,
        "largest_loss_usd": float(pnl.min()) if n else None,
        "largest_loss_pct_of_start": float(-pnl.min() / initial_balance * 100) if n else None,
        "max_drawdown_usd": dd_usd,
        "max_drawdown_pct": dd_frac * 100,
        "breached_max_drawdown": bool(dd_frac * 100 >= max_dd_limit_pct),
        "time_in_market_pct": (
            float(trades["bars_held"].sum() / max(len(equity), 1) * 100) if n else 0.0
        ),
        "sharpe_daily": _ratio(daily, False),
        "sortino_daily": _ratio(daily, True),
        "ruined": bool(len(equity) and equity.min() <= 0),
    }

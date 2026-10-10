"""The research gate: out-of-sample, parameter sensitivity, cost stress, Monte Carlo, folds.

Rules of the gate (so it cannot flatter the strategy):
- Parameters are fixed BEFORE the out-of-sample (OOS) period is looked at; nothing is tuned on it.
- Features are computed on the full history, but trading only happens inside each window.
- The per-trade risk limit is ENFORCED (as the live risk engine will), so trades that are too big
  for the account are skipped, not counted.
- A candidate passes only if EVERY check passes. A missing number (no trades) is a failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradingbot.backtesting.costs import CostModel
from tradingbot.backtesting.engine import BacktestConfig, BacktestResult, run_backtest
from tradingbot.backtesting.instruments import InstrumentSpec
from tradingbot.config import AppConfig, StrategyV1Config
from tradingbot.features import FeatureParams, compute_features
from tradingbot.strategies import EmaTrendAtrStrategy

PARAM_FIELDS = ("ema_fast", "ema_slow", "ema_trend", "atr_stop_mult")
CHECKS = (
    "oos_trades",
    "oos_profit_factor",
    "oos_drawdown",
    "param_sensitivity",
    "cost_stress",
    "monte_carlo",
    "folds",
)


def profit_factor(m: dict) -> float | None:
    """Gross profit / gross loss; infinite if there were wins and no losses; None if no trades."""
    if not m.get("trades"):
        return None
    gl, gp = m["gross_loss_usd"], m["gross_profit_usd"]
    if gl > 0:
        return gp / gl
    return math.inf if gp > 0 else None


@dataclass
class GateResult:
    symbol: str
    timeframe: str
    checks: dict[str, bool]
    details: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.get(c, False) for c in CHECKS)

    def row(self) -> dict:
        out = {"symbol": self.symbol, "timeframe": self.timeframe, "PASS": self.passed}
        out.update({f"ok_{k}": v for k, v in self.checks.items()})
        out.update(self.details)
        return out


def monte_carlo(
    pnl: list[float] | np.ndarray,
    initial_balance: float,
    dd_limit_pct: float,
    sims: int = 2000,
    seed: int = 7,
) -> dict | None:
    """Resample the OOS trades (with replacement, random order) to see how lucky the order was.

    Returns the 95th percentile of the worst drawdown, and the share of paths whose drawdown
    reaches the limit or whose account is wiped out. Needs at least 5 trades.
    """
    pnl = np.asarray(pnl, dtype=float)
    if len(pnl) < 5:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(pnl, size=(sims, len(pnl)), replace=True)
    equity = initial_balance + np.cumsum(draws, axis=1)
    peak = np.maximum.accumulate(
        np.concatenate([np.full((sims, 1), initial_balance), equity], 1), 1
    )[:, 1:]
    dd = np.where(peak > 0, (peak - equity) / peak, 1.0).max(axis=1) * 100
    ruined = (equity.min(axis=1) <= 0).mean() * 100
    return {
        "mc_p95_drawdown_pct": float(np.percentile(dd, 95)),
        "mc_breach_probability_pct": float((dd >= dd_limit_pct).mean() * 100),
        "mc_ruin_probability_pct": float(ruined),
        "mc_median_final_equity": float(np.median(equity[:, -1])),
    }


class _Runner:
    """Runs the strategy on one dataset with arbitrary parameters, windows and costs."""

    def __init__(self, ohlcv: pd.DataFrame, spec: InstrumentSpec, app: AppConfig, symbol, tf):
        self.ohlcv, self.spec, self.app, self.symbol, self.tf = ohlcv, spec, app, symbol, tf
        self._features: dict[tuple, pd.DataFrame] = {}

    def features(self, p: StrategyV1Config) -> pd.DataFrame:
        key = (p.ema_fast, p.ema_slow, p.ema_trend, p.atr_period)
        if key not in self._features:
            self._features[key] = compute_features(self.ohlcv, FeatureParams.from_config(p))
        return self._features[key]

    def run(
        self,
        p: StrategyV1Config,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
        costs: CostModel,
    ) -> BacktestResult:
        window = self.features(p)
        if start is not None:
            window = window[window.index >= start]
        if end is not None:
            window = window[window.index < end]
        cfg = BacktestConfig(
            initial_balance=self.app.account.initial_balance,
            lots=self.app.account.fixed_lot,
            costs=costs,
            max_drawdown_limit_pct=self.app.account.max_drawdown_pct,
            max_risk_per_trade_pct=self.app.risk.max_risk_per_trade_pct,
            enforce_risk_limit=True,
        )
        return run_backtest(window, EmaTrendAtrStrategy(p), self.spec, cfg, self.symbol, self.tf)


def _variants(base: StrategyV1Config, pct: float = 0.2):
    for name in PARAM_FIELDS:
        for sign in (-1, 1):
            value = getattr(base, name) * (1 + sign * pct)
            value = round(value, 4) if name == "atr_stop_mult" else max(2, round(value))
            try:
                yield (
                    f"{name}{'+' if sign > 0 else '-'}{int(pct * 100)}%",
                    base.model_copy(update={name: value}),
                )
            except ValueError:  # e.g. would break ema_fast < ema_slow < ema_trend
                continue


def evaluate_gate(
    ohlcv: pd.DataFrame,
    spec: InstrumentSpec,
    app: AppConfig,
    costs: CostModel,
    symbol: str,
    timeframe: str,
    oos_fraction: float = 0.4,
    folds: int = 4,
    mc_sims: int = 2000,
) -> GateResult:
    gate = app.research_gate
    runner = _Runner(ohlcv, spec, app, symbol, timeframe)
    base = app.strategy_v1
    n = len(ohlcv)
    split = ohlcv.index[int(n * (1 - oos_fraction))]
    d: dict = {"split_time": str(split)}

    ins = runner.run(base, None, split, costs)
    oos = runner.run(base, split, None, costs)
    om = oos.metrics
    base_pf = profit_factor(om)
    d.update(
        is_trades=ins.metrics["trades"],
        is_profit_factor=profit_factor(ins.metrics),
        is_net_usd=ins.metrics["net_profit_usd"],
        oos_trades=om["trades"],
        oos_profit_factor=base_pf,
        oos_net_usd=om["net_profit_usd"],
        oos_drawdown_pct=om["max_drawdown_pct"],
        oos_rejected_over_risk=oos.rejected_over_risk,
    )

    # parameter sensitivity: each parameter +/-20%, judged on the OOS window
    worst = None
    for label, p in _variants(base):
        pf = profit_factor(runner.run(p, split, None, costs).metrics)
        pf = 0.0 if pf is None else pf
        if base_pf and math.isfinite(base_pf):
            drop = (base_pf - pf) / base_pf * 100
            if worst is None or drop > worst[1]:
                worst = (label, drop)
    d["worst_param_drop_pct"] = None if worst is None else worst[1]
    d["worst_param"] = None if worst is None else worst[0]

    # cost stress: spread x2 on the OOS window
    stress_pf = profit_factor(
        runner.run(
            base, split, None, costs.with_overrides(spread_multiplier=costs.spread_multiplier * 2)
        ).metrics
    )
    d["stress2x_profit_factor"] = stress_pf

    # Monte Carlo on the OOS trades
    mc = (
        monte_carlo(
            oos.trades["pnl_usd"].to_numpy(),
            app.account.initial_balance,
            app.account.max_drawdown_pct,
            sims=mc_sims,
        )
        if len(oos.trades)
        else None
    )
    d.update(
        mc
        or dict.fromkeys(
            (
                "mc_p95_drawdown_pct",
                "mc_breach_probability_pct",
                "mc_ruin_probability_pct",
                "mc_median_final_equity",
            ),
            None,
        )
    )

    # walk-forward: the same fixed rules over consecutive equal time windows of the full history
    edges = [ohlcv.index[int(n * k / folds)] for k in range(folds)] + [None]
    fold_pf = []
    for k in range(folds):
        m = runner.run(base, edges[k], edges[k + 1], costs).metrics
        pf = profit_factor(m)
        fold_pf.append(pf)
    profitable = sum(1 for pf in fold_pf if pf is not None and pf > 1.0)
    d["fold_profit_factors"] = [None if f is None else round(f, 2) for f in fold_pf]
    d["profitable_folds_pct"] = profitable / folds * 100

    def ge(value, limit):
        return (
            value is not None
            and not (isinstance(value, float) and math.isnan(value))
            and value >= limit
        )

    checks = {
        "oos_trades": om["trades"] >= gate.min_oos_trades,
        "oos_profit_factor": ge(base_pf, gate.min_oos_profit_factor),
        "oos_drawdown": om["trades"] > 0 and om["max_drawdown_pct"] <= gate.max_oos_drawdown_pct,
        "param_sensitivity": worst is not None and worst[1] <= gate.max_param_sensitivity_drop_pct,
        "cost_stress": ge(stress_pf, gate.min_stress_profit_factor),
        "monte_carlo": mc is not None
        and mc["mc_breach_probability_pct"] <= gate.max_mc_breach_probability_pct,
        "folds": d["profitable_folds_pct"] >= gate.min_profitable_folds_pct,
    }
    return GateResult(symbol, timeframe, checks, d)

import math
from dataclasses import replace

import pandas as pd
import pytest

from bt_helpers import EUR, JPY, XAU, Scripted, bars, cfg
from tradingbot.backtesting.engine import run_backtest
from tradingbot.config import load_config
from tradingbot.risk import Reason, RiskEngine, RiskLimits, size_lots
from tradingbot.signals import Action

T0 = pd.Timestamp("2024-03-04 09:00", tz="UTC")
LIMITS = RiskLimits(
    max_risk_per_trade_pct=3.0,
    max_open_positions=2,
    max_total_exposure_lots=0.5,
    daily_loss_limit_pct=5.0,
    kill_switch_drawdown_pct=30.0,
)


def eng(equity=100.0, limits=LIMITS):
    return RiskEngine(limits, equity, T0)


def entry(e, symbol="EURUSD", stop=0.0005, price=1.10, spec=EUR, **kw):
    return e.evaluate_entry(symbol=symbol, stop_distance=stop, price=price, spec=spec, **kw)


def test_limits_come_from_config():
    lim = RiskLimits.from_config(load_config().risk)
    assert lim.max_risk_per_trade_pct == 3.0 and lim.daily_loss_limit_pct == 5.0
    assert lim.kill_switch_drawdown_pct == 30.0


# ---------------------------------------------------------------- sizing
def test_size_rounds_down_to_the_step_and_never_exceeds_the_budget():
    # EUR, normal lot: stop 0.0005 x 100k units = $50/lot. Budget $3 -> 0.06 lot ($3.00).
    assert size_lots(3.0, 0.0005, 1.10, EUR) == pytest.approx(0.06)
    # budget $2.99 -> 0.05 lot, not 0.0598 and never 0.06
    assert size_lots(2.99, 0.0005, 1.10, EUR) == pytest.approx(0.05)


def test_size_is_zero_when_minimum_lot_is_still_too_big():
    assert size_lots(3.0, 0.0100, 1.10, EUR) == 0.0  # $1000/lot -> 0.003 lot < 0.01 minimum
    assert size_lots(0.0, 0.0005, 1.10, EUR) == 0.0
    assert size_lots(3.0, 0.0, 1.10, EUR) == 0.0
    assert size_lots(3.0, float("nan"), 1.10, EUR) == 0.0


def test_size_converts_jpy_and_gold_to_dollars():
    # USDJPY stop 0.5 yen on 100k units = 50,000 JPY = $333/lot at 150 -> budget $3 -> 0.009 -> 0
    assert size_lots(3.0, 0.5, 150.0, JPY) == 0.0
    assert size_lots(10.0, 0.5, 150.0, JPY) == pytest.approx(0.03)
    # gold, 100 oz per lot, stop $2 -> $200/lot -> budget $3 -> 0.01 lot
    assert size_lots(3.0, 2.0, 2000.0, XAU) == pytest.approx(0.01)


def test_cent_account_scale_and_max_lots_cap():
    assert size_lots(3.0, 0.0005, 1.10, EUR, lot_value_scale=0.01) == pytest.approx(6.0)
    assert size_lots(3.0, 0.0005, 1.10, EUR, lot_value_scale=0.01, max_lots=2.0) == 2.0


# ---------------------------------------------------------------- per-trade decisions
def test_approved_entry_is_sized_by_risk_and_reports_it():
    d = entry(eng())
    assert d.approved and d.reason == Reason.APPROVED
    assert d.lots == pytest.approx(0.06) and d.risk_usd == pytest.approx(3.0)
    assert d.risk_pct == pytest.approx(3.0)


def test_stop_too_wide_is_rejected_with_a_reason():
    d = entry(eng(), stop=0.0100)
    assert not d.approved and d.reason == Reason.STOP_TOO_WIDE and d.lots == 0.0


@pytest.mark.parametrize("stop", [None, 0.0, -0.001, float("nan"), float("inf")])
def test_invalid_stop_is_rejected(stop):
    d = entry(eng(), stop=stop)
    assert not d.approved and d.reason == Reason.INVALID_STOP


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan")])
def test_invalid_price_fails_closed(price):
    d = entry(eng(), price=price)
    assert not d.approved and d.reason == Reason.INVALID_INPUT


def test_requested_fixed_lot_must_fit_the_risk_budget_and_the_lot_step():
    assert entry(eng(), requested_lots=0.06).approved
    d = entry(eng(), requested_lots=0.07)  # $3.50 > $3.00 budget
    assert not d.approved and d.reason == Reason.STOP_TOO_WIDE
    d = entry(eng(), requested_lots=0.015)  # not a multiple of the step: fails closed
    assert not d.approved and d.reason == Reason.INVALID_INPUT


def test_risk_budget_follows_current_equity():
    e = eng(100.0)
    for day, eq in enumerate([97, 94.5, 92, 89.5, 87, 84.5, 82, 80], start=1):  # slow slide
        e.update_equity(eq, T0 + pd.Timedelta(days=day))  # budget now $2.40 -> 0.04 lot
    assert entry(e).lots == pytest.approx(0.04)


# ---------------------------------------------------------------- portfolio limits
def test_one_position_per_symbol_and_max_open_positions():
    e = eng()
    d = entry(e)
    e.register_open("EURUSD", d.lots, d.risk_usd)
    assert entry(e).reason == Reason.DUPLICATE_SYMBOL
    assert entry(e, symbol="GBPUSD").approved
    e.register_open("GBPUSD", 0.06, 3.0)
    assert entry(e, symbol="USDJPY").reason == Reason.MAX_POSITIONS
    e.register_close("EURUSD")  # closing frees a slot
    assert entry(e, symbol="USDJPY").approved


def test_total_exposure_cap():
    e = RiskEngine(replace(LIMITS, max_total_exposure_lots=0.08), 100.0, T0)
    e.register_open("GBPUSD", 0.05, 2.0)
    d = entry(e)  # wants 0.06 more -> 0.11 > 0.08
    assert not d.approved and d.reason == Reason.MAX_EXPOSURE


# ---------------------------------------------------------------- daily loss limit
def test_daily_loss_limit_blocks_entries_until_the_next_utc_day():
    e = eng()
    assert "daily_loss_limit" in e.update_equity(94.9, T0 + pd.Timedelta(hours=1))
    d = entry(e)
    assert not d.approved and d.reason == Reason.DAILY_LOSS
    e.update_equity(94.9, T0 + pd.Timedelta(hours=14))  # still the same UTC day
    assert entry(e).reason == Reason.DAILY_LOSS
    assert "new_day" in e.update_equity(94.9, pd.Timestamp("2024-03-05 00:00", tz="UTC"))
    assert entry(e).approved  # a new day starts from the new equity


def test_daily_loss_is_measured_from_the_start_of_the_day_not_the_peak():
    e = eng()
    e.update_equity(100.0, pd.Timestamp("2024-03-05 00:00", tz="UTC"))  # day starts at $100
    e.update_equity(96.0, pd.Timestamp("2024-03-05 10:00", tz="UTC"))  # -4%: fine
    assert entry(e).approved
    e.update_equity(94.0, pd.Timestamp("2024-03-05 11:00", tz="UTC"))  # -6%: blocked
    assert entry(e).reason == Reason.DAILY_LOSS


# ---------------------------------------------------------------- kill switch
def test_kill_switch_trips_at_the_limit_and_stays_latched():
    e = eng()
    e.update_equity(120.0, T0)  # new peak
    assert "kill_switch" not in e.update_equity(85.0, T0)  # -29.2% from peak
    assert entry(e).approved or entry(e).reason == Reason.DAILY_LOSS
    assert "kill_switch" in e.update_equity(84.0, T0)  # exactly -30%
    for eq in (84.0, 150.0):  # even a full recovery does not re-arm it
        e.update_equity(eq, T0 + pd.Timedelta(days=2))
        d = entry(e)
        assert not d.approved and d.reason == Reason.KILL_SWITCH


def test_kill_switch_reset_is_deliberate_and_rebases_the_peak():
    e = eng()
    e.update_equity(70.0, T0)
    assert e.state.kill_switch
    e.update_equity(75.0, T0 + pd.Timedelta(days=1))
    e.reset_kill_switch()
    assert entry(e).approved
    assert "kill_switch" not in e.update_equity(60.0, T0 + pd.Timedelta(days=1))  # -20% of 75


def test_non_finite_equity_trips_the_kill_switch():
    e = eng()
    assert e.update_equity(math.nan, T0) == ["kill_switch"]
    assert entry(e).reason == Reason.KILL_SWITCH


def test_engine_refuses_to_start_without_equity():
    with pytest.raises(ValueError):
        RiskEngine(LIMITS, 0.0, T0)


# ---------------------------------------------------------------- inside the backtester
def _limits_cfg(**kw):
    return replace(cfg(balance=100.0), risk_limits=LIMITS, risk_sizing=True, **kw)


def test_backtest_sizes_by_risk_and_counts_rejections_by_reason():
    px = [1.10] * 10
    # stop 0.0005 -> 0.06 lot approved; a second signal with a $10 stop is rejected as too wide
    sc = Scripted(
        {
            2: [(Action.ENTER_LONG, 0.0005)],
            4: [(Action.EXIT_LONG, None)],
            5: [(Action.ENTER_LONG, 0.0100)],
        }
    )
    r = run_backtest(bars(px), sc, EUR, _limits_cfg(), "EURUSD", "1h")
    assert len(r.trades) == 1 and r.trades.iloc[0]["lots"] == pytest.approx(0.06)
    assert r.reject_reasons == {"stop_too_wide_for_account": 1}


def test_backtest_kill_switch_halts_the_run_after_a_gap_loss():
    # 0.06 lot long at 1.10, stop 1.0995; the next candle gaps to 1.094: loss ~ $36 (36% of $100)
    opens = [1.10, 1.10, 1.10, 1.094, 1.094, 1.094, 1.094, 1.094]
    sc = Scripted({1: [(Action.ENTER_LONG, 0.0005)], 4: [(Action.ENTER_LONG, 0.0005)]})
    r = run_backtest(bars(opens), sc, EUR, _limits_cfg(), "EURUSD", "1h")
    assert r.halted_reason == "kill_switch"
    assert len(r.trades) == 1 and r.trades.iloc[0]["exit_reason"] == "stop"
    assert r.trades.iloc[0]["pnl_usd"] == pytest.approx(-36.0)
    assert len(r.equity) == 4  # the run ended on the candle that tripped the switch


def test_backtest_kill_switch_flattens_an_open_position():
    # equity falls through the limit while a position is still open: it is closed at the close
    opens = [1.10, 1.10, 1.10, 1.10, 1.10, 1.10]
    closes = [1.10, 1.10, 1.10, 1.10, 1.0935, 1.0935]  # -65 pips on 0.06 lot ~ -$39 mark to market
    lows = [1.10, 1.10, 1.10, 1.10, 1.0996, 1.0935]  # stop (~1.0995) is not touched on candle 4
    sc = Scripted({1: [(Action.ENTER_LONG, 0.0005)]})
    c = replace(_limits_cfg(), risk_limits=replace(LIMITS, daily_loss_limit_pct=99.0))
    r = run_backtest(bars(opens, lows=lows, closes=closes), sc, EUR, c, "EURUSD", "1h")
    assert r.halted_reason == "kill_switch"
    assert r.trades.iloc[-1]["exit_reason"] == "kill_switch"


def test_backtest_daily_loss_blocks_a_second_entry_the_same_day():
    # first trade stops out for ~ -3% (limit 2%): the same-day re-entry must be refused
    lim = replace(LIMITS, daily_loss_limit_pct=2.0)
    opens = [1.10, 1.10, 1.10, 1.0994, 1.0994, 1.0994, 1.0994, 1.0994]
    lows = [1.10, 1.10, 1.10, 1.0994, 1.0994, 1.0994, 1.0994, 1.0994]
    sc = Scripted({1: [(Action.ENTER_LONG, 0.0005)], 4: [(Action.ENTER_LONG, 0.0005)]})
    c = replace(_limits_cfg(), risk_limits=lim)
    r = run_backtest(bars(opens, lows=lows), sc, EUR, c, "EURUSD", "1h")
    assert len(r.trades) == 1
    assert r.reject_reasons.get("daily_loss_limit_reached") == 1


def test_exposure_cap_counts_cent_lots_as_one_hundredth():
    e = RiskEngine(replace(LIMITS, max_total_exposure_lots=0.05), 20.0, T0, lot_value_scale=0.01)
    d = entry(e, stop=0.0005)  # 6.0 cent lots = 0.06 normal... budget $0.60 -> 1.2 cent lots
    assert d.approved and d.lots == pytest.approx(1.2)  # 1.2 x 0.01 = 0.012 normal lots <= 0.05
    e.register_open("GBPUSD", 4.0, 1.0)  # 0.04 normal lots already open
    assert entry(e, stop=0.0005).reason == Reason.MAX_EXPOSURE  # 0.04 + 0.012 > 0.05

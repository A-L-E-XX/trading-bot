import pandas as pd
import pytest

from tradingbot.config import load_config
from tradingbot.features import FeatureParams, compute_features
from tradingbot.signals import Action, Side
from tradingbot.strategies import OrbScalperStrategy, get_strategy_def

T = pd.Timestamp("2024-03-04 15:00", tz="UTC")
FP = FeatureParams.from_v4(load_config().strategy_v4)


def day(date="2024-07-01", tz="America/New_York", closes=None, minutes=("09:00", "16:30")):
    """Flat 5m candles (high 101, low 99) for one local day; ``closes`` sets closes by HH:MM."""
    idx = pd.date_range(
        f"{date} {minutes[0]}", f"{date} {minutes[1]}", freq="5min", tz=tz
    ).tz_convert("UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}, index=idx
    )
    for hhmm, c in (closes or {}).items():
        t = pd.Timestamp(f"{date} {hhmm}", tz=tz).tz_convert("UTC")
        df.loc[t, "close"] = c
        df.loc[t, "high"] = max(df.loc[t, "high"], c)
        df.loc[t, "low"] = min(df.loc[t, "low"], c)
    return df


def feats(df):
    return compute_features(df, FP)


def at(f, hhmm, date="2024-07-01", tz="America/New_York"):
    return f.loc[pd.Timestamp(f"{date} {hhmm}", tz=tz).tz_convert("UTC")]


def test_range_is_the_first_30_minutes_and_hidden_until_complete():
    f = feats(day(closes={"09:45": 100.0}))
    assert pd.isna(at(f, "09:55")["orb_high"])  # range still forming
    assert at(f, "10:00")["orb_high"] == 101.0 and at(f, "10:00")["orb_low"] == 99.0


def test_range_ignores_candles_after_the_open_window():
    df = day()
    t = pd.Timestamp("2024-07-01 10:05", tz="America/New_York").tz_convert("UTC")
    df.loc[t, "high"] = 150.0  # a spike after the range
    f = feats(df)
    assert at(f, "10:10")["orb_high"] == 101.0


def test_works_across_daylight_saving():
    # The range is complete at 10:00 New York = 14:00 UTC in July, 15:00 UTC in January.
    for date, utc_ready in (("2024-07-01", "14:00"), ("2024-01-02", "15:00")):
        f = feats(day(date=date, closes={"10:00": 105.0}))
        assert at(f, "10:00", date=date)["orb_signal"] == 1
        assert f.index[f["orb_high"].notna()][0].strftime("%H:%M") == utc_ready


def test_first_breakout_only_and_direction():
    f = feats(day(closes={"10:15": 103.0, "10:30": 95.0}))
    assert at(f, "10:15")["orb_signal"] == 1
    assert at(f, "10:30")["orb_signal"] == 0  # later break on the same day is ignored
    g = feats(day(closes={"10:20": 97.0}))
    assert at(g, "10:20")["orb_signal"] == -1


def test_no_signal_after_the_entry_window_or_inside_the_range():
    assert (feats(day(closes={"09:50": 120.0}))["orb_signal"] == 0).all()  # inside the range
    assert (feats(day(closes={"12:30": 120.0}))["orb_signal"] == 0).all()  # window closed


def test_missing_open_candle_means_no_range_that_day():
    df = day(closes={"10:15": 103.0})
    t = pd.Timestamp("2024-07-01 09:30", tz="America/New_York").tz_convert("UTC")
    f = feats(df.drop(t))
    assert f["orb_high"].isna().all() and (f["orb_signal"] == 0).all()


def test_breakout_is_decided_from_the_signal_candle_only():
    base = day(closes={"10:15": 103.0})
    dmg = base.copy()
    t = pd.Timestamp("2024-07-01 10:20", tz="America/New_York").tz_convert("UTC")
    dmg.loc[t, ["close", "high", "low"]] = [50.0, 101.0, 50.0]  # wreck the NEXT candle
    a, b = feats(base), feats(dmg)
    cols = ["orb_high", "orb_low", "orb_signal", "orb_flatten"]
    cut = pd.Timestamp("2024-07-01 10:15", tz="America/New_York").tz_convert("UTC")
    pd.testing.assert_frame_equal(a.loc[:cut, cols], b.loc[:cut, cols])


@pytest.fixture
def strat():
    return OrbScalperStrategy(load_config().strategy_v4)


def row(sig=0, hi=101.0, lo=99.0, close=102.0, atr=0.5, flat=False):
    return {
        "close": close, "orb_high": hi, "orb_low": lo, "orb_signal": sig,
        "orb_flatten": flat, "atr": atr,
    }  # fmt: skip


def ev(strat, cur, position=None):
    return strat.evaluate("USTEC", "5m", T, row(), cur, position)


def test_enters_with_the_opposite_side_of_the_range_as_stop(strat):
    (sig,) = ev(strat, row(sig=1, close=102.0))
    assert sig.action == Action.ENTER_LONG and sig.stop_distance == pytest.approx(3.0)
    (sig,) = ev(strat, row(sig=-1, close=98.0))
    assert sig.action == Action.ENTER_SHORT and sig.stop_distance == pytest.approx(3.0)


def test_skips_narrow_ranges_and_no_signal_candles(strat):
    assert ev(strat, row(sig=1, hi=100.5, lo=100.0)) == []  # 0.5 wide < 2 ATR (1.0)
    assert ev(strat, row(sig=0)) == []


def test_flattens_at_the_end_of_the_session_and_does_not_stack(strat):
    (x,) = ev(strat, row(flat=True), Side.LONG)
    assert x.action == Action.EXIT_LONG
    (x,) = ev(strat, row(flat=True), Side.SHORT)
    assert x.action == Action.EXIT_SHORT
    assert ev(strat, row(sig=1), Side.LONG) == []
    assert ev(strat, row(sig=1, flat=True)) == []  # never opens a trade at the flatten time


def test_no_shorts_when_disallowed():
    params = load_config().strategy_v4.model_copy(update={"allow_short": False})
    assert ev(OrbScalperStrategy(params), row(sig=-1, close=98.0)) == []


def test_registered():
    sdef = get_strategy_def("v4")
    assert sdef.config_attr == "strategy_v4"
    assert FeatureParams.from_v4(load_config().strategy_v4).orb is True

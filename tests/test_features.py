import numpy as np
import pandas as pd
import pytest

from tradingbot.config import load_config
from tradingbot.features import FeatureParams, compute_features
from tradingbot.features import indicators as ind

# ------------------------------------------------------------------ indicators vs hand calculations


def test_sma_matches_manual():
    s = pd.Series([1.0, 2, 3, 4, 5, 6])
    out = ind.sma(s, 3)
    assert out.isna().tolist() == [True, True, False, False, False, False]
    assert out.iloc[2:].tolist() == [2.0, 3.0, 4.0, 5.0]


def test_ema_matches_recursion_and_warmup():
    s = pd.Series([10.0, 11, 12, 11, 13, 14, 13, 15])
    n = 4
    alpha = 2 / (n + 1)
    manual = [s.iloc[0]]
    for x in s.iloc[1:]:
        manual.append(alpha * x + (1 - alpha) * manual[-1])
    out = ind.ema(s, n)
    assert out.iloc[: n - 1].isna().all()
    np.testing.assert_allclose(out.iloc[n - 1 :], manual[n - 1 :], rtol=1e-12)


def test_rsi_extremes_and_bounds(ohlcv):
    rising = pd.Series(np.arange(1.0, 60.0))
    falling = pd.Series(np.arange(60.0, 1.0, -1.0))
    flat = pd.Series(np.full(60, 5.0))
    assert ind.rsi(rising, 14).dropna().eq(100.0).all()
    assert ind.rsi(falling, 14).dropna().eq(0.0).all()
    assert ind.rsi(flat, 14).dropna().eq(50.0).all()
    r = ind.rsi(ohlcv["close"], 14).dropna()
    assert r.between(0, 100).all()
    assert ind.rsi(ohlcv["close"], 14).iloc[:14].isna().all()  # first valid value at row 14


def test_rsi_matches_manual_wilder():
    close = pd.Series(
        [
            44.0,
            44.3,
            44.1,
            43.6,
            44.3,
            44.8,
            45.1,
            45.4,
            45.8,
            46.1,
            45.9,
            46.0,
            46.4,
            46.2,
            45.6,
            46.0,
        ]
    )
    n = 5
    d = close.diff()
    gain, loss = d.clip(lower=0), (-d).clip(lower=0)
    ag, al = gain.iloc[1], loss.iloc[1]
    manual = {}
    for i in range(2, len(close)):
        ag = (1 / n) * gain.iloc[i] + (1 - 1 / n) * ag
        al = (1 / n) * loss.iloc[i] + (1 - 1 / n) * al
        manual[i] = 100 - 100 / (1 + ag / al)
    out = ind.rsi(close, n)
    for i in range(n, len(close)):
        assert out.iloc[i] == pytest.approx(manual[i], rel=1e-10)


def test_true_range_and_atr_match_manual(ohlcv):
    h, lo, c = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    tr = ind.true_range(h, lo, c)
    assert tr.iloc[0] == pytest.approx(h.iloc[0] - lo.iloc[0])
    i = 10
    expected = max(
        h.iloc[i] - lo.iloc[i], abs(h.iloc[i] - c.iloc[i - 1]), abs(lo.iloc[i] - c.iloc[i - 1])
    )
    assert tr.iloc[i] == pytest.approx(expected)
    n = 14
    manual = tr.iloc[0]
    atr_manual = [manual]
    for x in tr.iloc[1:]:
        manual = x / n + (1 - 1 / n) * manual
        atr_manual.append(manual)
    out = ind.atr(h, lo, c, n)
    assert out.iloc[: n - 1].isna().all()
    np.testing.assert_allclose(out.iloc[n - 1 :], atr_manual[n - 1 :], rtol=1e-10)
    assert (out.dropna() > 0).all()


def test_volatility_and_volume_ratio():
    close = pd.Series([100.0, 101, 99, 102, 100, 103, 101, 104])
    lr = ind.log_returns(close)
    vol = ind.volatility(lr, 4)
    assert vol.iloc[:4].isna().all()
    assert vol.iloc[4] == pytest.approx(np.std(lr.iloc[1:5], ddof=1))
    v = pd.Series([10.0, 10, 10, 40])
    assert ind.volume_ratio(v, 4).iloc[3] == pytest.approx(40 / 17.5)
    assert ind.volume_ratio(pd.Series([0.0] * 5), 3).isna().all()


def test_returns():
    s = pd.Series([100.0, 110.0, 99.0])
    assert ind.simple_returns(s).iloc[1] == pytest.approx(0.10)
    assert ind.log_returns(s).iloc[1] == pytest.approx(np.log(1.1))


# ------------------------------------------------------------------ engine: M4 acceptance


def test_features_are_reproducible(ohlcv):
    """Identical input data -> bit-identical output."""
    a = compute_features(ohlcv)
    b = compute_features(ohlcv.copy())
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_no_look_ahead(ohlcv):
    """Features at time t must not change when later candles are added or removed."""
    full = compute_features(ohlcv)
    for k in (250, 400, 777):
        partial = compute_features(ohlcv.iloc[:k])
        pd.testing.assert_frame_equal(partial, full.iloc[:k], check_exact=True)


def test_future_edit_does_not_change_past_features(ohlcv):
    base = compute_features(ohlcv)
    edited = ohlcv.copy()
    edited.iloc[600:, edited.columns.get_loc("close")] *= 1.5
    after = compute_features(edited)
    pd.testing.assert_frame_equal(base.iloc[:600], after.iloc[:600], check_exact=True)


def test_warmup_and_ready_flag(ohlcv):
    params = FeatureParams()
    out = compute_features(ohlcv, params)
    assert not out["ready"].iloc[: params.ema_trend - 1].any()
    assert out["ready"].iloc[params.ema_trend - 1 :].all()
    assert params.strategy_warmup_rows == 200
    assert out.loc[out["ready"], ["ema_fast", "ema_slow", "ema_trend", "atr"]].notna().all().all()


def test_engine_does_not_modify_input_and_checks_it(ohlcv):
    before = ohlcv.copy()
    compute_features(ohlcv)
    pd.testing.assert_frame_equal(before, ohlcv)
    with pytest.raises(ValueError, match="missing columns"):
        compute_features(ohlcv.drop(columns=["volume"]))
    with pytest.raises(ValueError, match="sorted"):
        compute_features(ohlcv.iloc[::-1])


def test_params_from_config():
    p = FeatureParams.from_config(load_config().strategy_v1)
    assert (p.ema_fast, p.ema_slow, p.ema_trend, p.atr_period) == (20, 50, 200, 14)


def test_end_to_end_stored_data_to_features(tmp_path, start, end):
    from tradingbot.data import DatasetStore, build_dataset
    from tradingbot.data.collectors import SyntheticCollector

    store = DatasetStore(tmp_path)
    build_dataset(SyntheticCollector(), "EURUSD", "1h", start, end, store)
    feats = compute_features(store.load("EURUSD", "1h"))
    assert feats["ready"].any() and len(feats) == store.metadata("EURUSD", "1h")["rows"]

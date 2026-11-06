"""Tests ทั้งหมด (ข้อมูลจำลอง SYNTHETIC ไม่ใช้เครือข่าย) — รันด้วย: pytest -q test_all.py"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FREQ = {"4h": "4h", "1d": "1D", "1w": "7D"}


def synth_ohlcv(n: int = 800, tf: str = "1d", seed: int = 1, drift: float = 0.0008, vol: float = 0.03,
                start_price: float = 100.0, end: str = "2026-11-06") -> pd.DataFrame:
    """Random walk แบบ GBM — SYNTHETIC"""
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = start_price * np.exp(np.cumsum(r))
    open_ = np.r_[start_price, close[:-1]]
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    qv = rng.lognormal(16, 0.4, n)
    idx = pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=n, freq=FREQ[tf])
    step = pd.Timedelta(FREQ[tf])
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": qv / close, "qvol": qv,
                         "taker_q": qv / 2, "close_time": idx + step - pd.Timedelta(milliseconds=1)}, index=idx)


class FakeMarket:
    """แทน MarketData โดยไม่ใช้เครือข่าย"""
    host = "synthetic"

    def __init__(self, symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT")):
        self.symbols = symbols

    def tickers(self):
        return pd.DataFrame({"lastPrice": 1.0, "priceChangePercent": 0.0, "quoteVolume": 1e9, "count": 1e6,
                             "base": [s[:-4] for s in self.symbols]}, index=list(self.symbols))

    def ohlcv(self, symbol, tf, limit=None):
        seed = sum(map(ord, symbol)) + {"4h": 1, "1d": 2, "1w": 3}[tf]
        n = {"4h": 500, "1d": 900, "1w": 160}[tf]
        return synth_ohlcv(n, tf, seed=seed)


@pytest.fixture
def daily():
    return synth_ohlcv()


@pytest.fixture
def fake_md():
    return FakeMarket()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """ปิดการเรียก API ภายนอกทุกตัวในการทดสอบ"""
    import pipeline as pl
    import sources as fu
    empty = {"cg": {}, "glob": {}, "llama": {"protocols": [], "chains": [], "fees": []}, "rss": [], "det": {}}
    monkeypatch.setattr(pl, "external_bundle", lambda *a, **k: empty)
    monkeypatch.setattr(pl, "fear_greed", lambda: None)
    monkeypatch.setattr(pl, "cg_global", lambda *a, **k: {})
    monkeypatch.setattr(fu, "fear_greed", lambda: None)



from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import analysis as quant
import market as indicators
from analysis import analyze_coin
from market import DataError, closed_only, parse_klines
from signals import build_signal


# ───────────── Market data ─────────────
def test_parse_klines_validates():
    ms = 1_700_000_000_000
    raw = [[ms + i * 60_000, "1", "2", "0.5", "1.5", "10", ms + i * 60_000 + 59_999, "15", 1, "5", "7", "0"]
           for i in range(5)]
    raw.append(raw[-1])                                           # ซ้ำ
    raw.append([ms + 9 * 60_000, "1", "0.4", "0.5", "1", "1", ms + 9 * 60_000 + 59_999, "1", 1, "1", "1", "0"])  # high<low
    df = parse_klines(raw)
    assert len(df) == 5 and df.index.is_monotonic_increasing
    with pytest.raises(DataError):
        parse_klines([])


def test_closed_only_drops_open_bar(daily):
    now = daily["close_time"].iloc[-1] - pd.Timedelta(hours=1)  # แท่งสุดท้ายยังไม่ปิด
    assert len(closed_only(daily, now)) == len(daily) - 1


# ───────────── Indicators ─────────────
def test_indicators_basic(daily):
    d = indicators.add_indicators(daily)
    for col in ("ema9", "ema20", "ema50", "ema200", "rsi", "macd_h", "atr", "adx", "bb_up", "vwap20", "vol_ratio"):
        assert col in d
    assert d["ema200"].iloc[:199].isna().all() and d["ema200"].iloc[199:].notna().all()
    assert d["rsi"].between(0, 100).all()
    assert (d["bb_up"].dropna() >= d["bb_lo"].dropna()).all()
    flat = daily.copy()
    flat[["open", "high", "low", "close"]] = 10.0
    assert indicators.add_indicators(flat)["rsi"].iloc[-1] == 50  # ไม่มีการเปลี่ยนแปลง → 50


def test_indicator_registry_extensible(daily):
    @indicator_fn
    def _x(d):
        d["x_test"] = 1.0
    assert "x_test" in indicators.add_indicators(daily)
    indicators._REGISTRY.pop("x_test")


def indicator_fn(fn):
    return indicators.indicator("x_test")(fn)


# ───────────── Backtest ─────────────
def test_backtest_no_lookahead(daily):
    """ผลของแท่งก่อนหน้าต้องไม่เปลี่ยนเมื่อแก้ข้อมูล 'อนาคต' หลังจุดตัด"""
    d = indicators.add_indicators(daily)
    cut = 600
    a = quant.backtest(d.iloc[:cut].copy(), 2, 4, 30)
    fut = daily.copy()
    fut.iloc[cut:, :4] *= 3  # เปลี่ยนอนาคตแรงๆ
    b = quant.backtest(indicators.add_indicators(fut).iloc[:cut].copy(), 2, 4, 30)
    assert a["n"] == b["n"] and a["expectancy"] == pytest.approx(b["expectancy"])


def test_backtest_costs_reduce_expectancy(daily, monkeypatch):
    d = indicators.add_indicators(daily)
    base = quant.backtest(d, 2, 4, 60)
    monkeypatch.setattr(quant, "FEE", 0.01)
    worse = quant.backtest(d, 2, 4, 60)
    assert base["n"] == 0 or worse["expectancy"] < base["expectancy"]
    assert 0 <= base["win_rate"] <= 1 and base["max_dd_R"] <= 0
    assert base["is_n"] + base["oos_n"] == base["n"]


def test_metrics_known_values():
    m = quant._metrics([(2.0, 1, True, 0), (-1.0, 1, False, 1), (-1.0, 1, False, 2), (2.0, 1, True, 3)])
    assert m["n"] == 4 and m["win_rate_raw"] == 0.5
    assert m["expectancy"] == pytest.approx(0.5)
    assert m["profit_factor"] == pytest.approx(2.0)
    assert m["max_dd_R"] == pytest.approx(-2.0)


def test_independent():
    assert list(quant.independent(np.array([0, 1, 5, 6, 11]), 5)) == [0, 5, 11]


# ───────────── Full coin analysis ─────────────
@pytest.fixture
def coin(fake_md):
    btc = closed_only(indicators.add_indicators(fake_md.ohlcv("BTCUSDT", "1d")))
    fr = {tf: fake_md.ohlcv("ETHUSDT", tf) for tf in ("4h", "1d", "1w")}
    macro = {"regime": "Neutral", "notes": [], "btc_price": 1.0, "points": 2}
    return analyze_coin("ETHUSDT", fr, btc, macro, {}), macro


def test_analyze_coin_consistent(coin):
    c, _ = coin
    m, l = c["med"], c["long"]
    assert m["sl"] < m["zone"][0] <= m["zone"][1] < m["tp1"] < m["tp2"]
    assert m["rr1"] >= 1.5 - 1e-9
    assert 0 < m["be_win"] < 100 and 5 <= m["win_sys"] <= 95
    assert l["tiers"][0] >= l["tiers"][1] >= l["tiers"][2] > l["stop"]
    for v in c["fc"].values():
        assert v["reach70"] <= v["reach50"] <= v["reach30"]
        assert v["end25"] <= v["end50"] <= v["end75"]
    assert 0 <= c["score"] <= 100


def test_quality_gate_and_signal(coin):
    c, macro = coin
    c["med"]["status"] = "🟢"
    c["cf"]["regime"] = "UP"
    c["bt"].update(n=20, expectancy=0.3, oos_n=5, oos_expectancy=0.2)
    c["med"]["ft1"] = {"p_tp": 0.6, "p_sl": 0.3, "n_eff": 40, "days_tp": 10}
    c["med"]["rel_win"] = 0.6 / 0.9
    c["med"]["be_win"] = 40.0
    c["last_closed"] = pd.Timestamp("2026-11-05", tz="UTC")
    now = datetime(2026, 11, 6, tzinfo=timezone.utc)
    sig, fails = build_signal(c, macro, None, False, now)
    assert sig is not None and not fails, fails
    assert 0 <= sig.confidence <= 100
    assert sig.win_prob == pytest.approx(66.7, abs=0.1)
    _, fails = build_signal(c, macro, None, True, now)
    assert "ข่าวลบรุนแรง" in fails
    c["med"]["status"] = "🟡"
    sig, fails = build_signal(c, macro, None, False, now)
    assert sig is None and fails


# ───────────── Pipeline (end-to-end, no network) ─────────────
def test_pipeline_end_to_end(fake_md):
    from pipeline import RunConfig, run_pipeline
    res = run_pipeline(fake_md, RunConfig(watch=["ETH", "SOL", "XYZ"], n_extra=0, use_news=False))
    assert {c["ticker"] for c in res["coins"]} == {"ETH", "SOL"}
    assert res["missing"] == ["XYZ"]
    assert res["port"]["mode"].startswith("rule-based")
    assert 0 <= res["port"]["cash"] <= 100
    for s in res["signals"]:
        assert s.sl < s.entry_hi < s.tp1


def test_report_renders(fake_md):
    from pipeline import RunConfig, run_pipeline
    from pipeline import deep_dive_md, summary_text, table_backtest, table_forecast, table_long, table_medium
    res = run_pipeline(fake_md, RunConfig(watch=["ETH", "SOL"], n_extra=0, use_news=False))
    coins = res["coins"]
    for fn in (table_medium, table_long):
        assert len(fn(coins, {}, {})) == len(coins)
    assert len(table_backtest(coins)) == len(coins) and len(table_forecast(coins)) > 0
    md = deep_dive_md(coins[0], {}, {}, {}, 10_000, 1.0, None)
    assert "แผนระยะกลาง" in md and "nan" not in md.lower().replace("financ", "")
    assert "Crypto Spot Brief" in summary_text(res, coins, res["port"])


def test_portfolio(fake_md):
    from pipeline import run_portfolio
    out = run_portfolio(fake_md, [{"ticker": "ETH", "cost": 50.0, "qty": 2.0}], None, None, "m", "", False)
    h = out["holds"][0]
    assert h["act"] in ("CUT", "REDUCE", "TRIM", "HOLD", "HOLD_LONG", "ADD")
    assert h["ladder"][0] < h["ladder"][1] < h["ladder"][2]






import importlib
import sys
import types

import pytest


class _Stub:
    """แทน widget/container ของ Streamlit: เรียกอะไรก็ได้ คืนค่า default ที่สมเหตุสมผล"""

    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        def fn(*a, **k):
            self.calls.append(name)
            if name in ("columns",):
                n = a[0] if isinstance(a[0], int) else len(a[0])
                return [_Stub(self.calls) for _ in range(n)]
            if name == "tabs":
                return [_Stub(self.calls) for _ in a[0]]
            if name in ("button", "toggle", "checkbox"):
                return k.get("value", False)
            if name == "selectbox":
                return a[1][k.get("index", 0)]
            if name in ("slider", "number_input"):
                vals = [x for x in a[1:] if isinstance(x, (int, float))]
                return k.get("value", vals[2] if len(vals) >= 3 else (vals[0] if vals else 0.0))
            if name == "text_input":
                return a[1] if len(a) > 1 else ""
            if name in ("file_uploader",):
                return None
            if name == "data_editor":
                return a[0]
            return _Stub(self.calls)
        return fn


class _SS(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


def make_streamlit(calls):
    st = types.ModuleType("streamlit")
    root = _Stub(calls)
    st.__getattr__ = lambda name: getattr(root, name)  # type: ignore[attr-defined]
    st.session_state = _SS()
    st.secrets = {}
    st.sidebar = _Stub(calls)
    passthru = lambda f=None, **k: f if f else (lambda g: g)  # noqa: E731
    st.cache_resource = passthru
    st.cache_data = passthru
    st.cache_resource.clear = lambda: None  # type: ignore[attr-defined]

    class _StopError(Exception):
        pass
    st.stop = lambda: (_ for _ in ()).throw(_StopError())
    st.column_config = _Stub(calls)
    return st


@pytest.fixture
def app_mod(monkeypatch, fake_md, tmp_path):
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "streamlit", make_streamlit(calls))
    sys.modules.pop("app", None)
    app = importlib.import_module("app")
    monkeypatch.setattr(app, "market", lambda: fake_md)
    return app, calls


def test_app_home_renders(app_mod):
    app, calls = app_mod
    app.main()
    assert "data_editor" in calls


def test_app_all_tabs_render(app_mod, fake_md):
    app, calls = app_mod
    from pipeline import RunConfig, run_pipeline
    app.st.session_state["res"] = run_pipeline(fake_md, RunConfig(watch=["ETH", "SOL"], n_extra=0, use_news=False))
    app.main()
    assert calls.count("dataframe") >= 6



# ───────────── Chart patterns / Opportunity ─────────────
def _shape(closes, vol=None):
    """สร้างแท่งเทียนจากเส้นราคาที่กำหนด (SYNTHETIC)"""
    c = np.asarray(closes, dtype=float)
    idx = pd.date_range(end="2026-11-06", periods=len(c), freq="1D", tz="UTC")
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.005, "low": np.minimum(o, c) * 0.995, "close": c,
                       "qvol": vol if vol is not None else np.full(len(c), 1e6)}, index=idx)
    df["volume"] = df["qvol"] / c
    return indicators.add_indicators(df)


def test_double_bottom_detected():
    path = np.r_[np.linspace(100, 80, 30), np.linspace(80, 95, 15), np.linspace(95, 80.5, 15), np.linspace(80.5, 97, 15)]
    pats = quant.chart_patterns(_shape(np.r_[np.full(60, 100.0), path]))
    db = [p for p in pats if p["name"] == "Double Bottom"]
    assert db and db[0]["bias"] == "bull" and db[0]["target"] > db[0]["level"]


def test_range_breakout_detected():
    rng = np.random.default_rng(3)
    base = 100 + rng.normal(0, 0.8, 80)
    vol = np.r_[np.full(80, 1e6), [5e6]]
    pats = quant.chart_patterns(_shape(np.r_[base, 110.0], vol))
    assert any(p["name"] == "Range Breakout 30D" and p["bias"] == "bull" for p in pats)


def test_pattern_points_bounded():
    pats = [{"bias": "bull", "state": "ยืนยัน"}] * 5
    assert quant.pattern_points(pats) == 8
    assert quant.pattern_points([{"bias": "bear", "state": "กำลังก่อตัว"}]) == -2


def test_opportunity_rank(fake_md):
    from pipeline import RunConfig, opportunity_rank, run_pipeline, table_patterns
    res = run_pipeline(fake_md, RunConfig(watch=["ETH", "SOL"], n_extra=0, use_news=False))
    for h in (7, 30, 180):
        df = opportunity_rank(res["coins"], h)
        assert len(df) == 2 and df["คะแนนโอกาส"].is_monotonic_decreasing
        assert df["คะแนนโอกาส"].between(0, 100).all()
    table_patterns(res["coins"])
    for c in res["coins"]:
        for p in c["patterns"]:
            assert p["target"] > 0 and (p["p_target_60"] is None or 0 <= p["p_target_60"] <= 1)

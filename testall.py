"""Tests ทั้งหมด (ข้อมูลจำลอง SYNTHETIC ไม่ใช้เครือข่าย) — รันด้วย: pytest -q test_all.py"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FREQ = {"1h": "1h", "4h": "4h", "1d": "1D", "1w": "7D"}


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
        seed = sum(map(ord, symbol)) + {"1h": 4, "4h": 1, "1d": 2, "1w": 3}[tf]
        n = {"1h": 1000, "4h": 500, "1d": 900, "1w": 160}[tf]
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
    res = run_pipeline(fake_md, RunConfig(watch=["ETH", "SOL", "XYZ"], n_extra=0, use_news=False, max_watch=3))
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
    res = run_pipeline(fake_md, RunConfig(watch=["ETH", "SOL"], n_extra=0, use_news=False, max_watch=3))
    for h in (7, 30, 180):
        df = opportunity_rank(res["coins"], h)
        assert len(df) == 2 and df["คะแนนโอกาส"].is_monotonic_decreasing
        assert df["คะแนนโอกาส"].between(0, 100).all()
    table_patterns(res["coins"])
    for c in res["coins"]:
        for p in c["patterns"]:
            assert p["target"] > 0 and (p["p_target_60"] is None or 0 <= p["p_target_60"] <= 1)


# ───────────── v5.1: Research-based trend / momentum / scanner ─────────────
class FakeMarketWide(FakeMarket):
    """ตลาดจำลอง 14 เหรียญ + bid/ask (เหรียญ WIDE มี spread กว้าง)"""

    def __init__(self):
        super().__init__(tuple(f"{b}USDT" for b in ("BTC", "ETH", "SOL", "XRP", "ADA", "NEAR", "ENA", "HYPE",
                                                     "AAVE", "ZEC", "BNB", "UNI", "HBAR", "WIDE")))

    def tickers(self):
        df = super().tickers()
        df["bidPrice"], df["askPrice"] = 1.0, 1.0005
        df.loc["WIDEUSDT", "askPrice"] = 1.05
        mid = (df["bidPrice"] + df["askPrice"]) / 2
        df["spread"] = (df["askPrice"] - df["bidPrice"]) / mid
        return df


def test_new_indicators(daily):
    d = indicators.add_indicators(daily)
    assert {"bb_z", "taker_ratio7", "taker_ratio30"} <= set(d.columns)
    assert d["taker_ratio7"].dropna().between(0, 1).all()


def test_donchian_no_lookahead(daily):
    d = indicators.add_indicators(daily)
    full = quant.donchian_trend(d)
    k = 500
    cut = quant.donchian_trend(d.iloc[: k + 1])
    assert np.allclose(full.iloc[: k + 1].to_numpy(), cut.to_numpy(), equal_nan=True)
    assert full.dropna().between(0, 1).all()


def test_donchian_direction():
    up = _shape(50 * 1.01 ** np.arange(300))       # ขึ้น 1%/วัน
    dn = _shape(150 * 0.99 ** np.arange(300))      # ลง 1%/วัน
    assert quant.donchian_trend(up).iloc[-1] == 1.0
    assert quant.donchian_trend(dn).iloc[-1] == 0.0


def test_momentum_profile(daily):
    p = quant.momentum_profile(indicators.add_indicators(daily))
    assert set(p["ret"]) == {30, 90} and p["vol"] > 0 and p["risk_adj"] is not None
    assert 0 <= p["trend"] <= 1


def test_xs_backtest_detects_persistent_momentum():
    rng = np.random.default_rng(7)
    n, k = 700, 12
    drifts = np.linspace(-0.002, 0.003, k)  # เหรียญที่ drift สูงยังคงขึ้นต่อ → การจัดอันดับควรชนะตลาด
    r = rng.normal(drifts, 0.02, (n, k))
    px = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), columns=[f"C{i}" for i in range(k)],
                      index=pd.date_range(end="2026-11-06", periods=n, freq="1D", tz="UTC"))
    bt = quant.xs_momentum_backtest(px)
    assert bt["n"] >= 12 and bt["excess"] > 0 and 0 <= bt["hit"] <= 1
    noise = quant.xs_momentum_backtest(px.iloc[:150])
    assert noise["excess"] is None  # ข้อมูลน้อย → ไม่สรุป


def test_scan_universe_filters_and_ranks():
    from pipeline import scan_universe
    md = FakeMarketWide()
    out = scan_universe(md, {"ETH", "SOL", "BTC"}, 5, 1e6)
    picks = [s[:-4] for s in out["picks"]]
    assert len(picks) == 5 and not ({"ETH", "SOL", "BTC", "WIDE"} & set(picks))
    assert out["dropped_spread"] == 1 and out["universe"] == 13
    assert all(0 < v["xs_pct"] <= 1 for v in out["info"].values())
    assert out["xs_bt"] is not None and "verdict" in out["xs_bt"]


def test_watchlist_capped_to_three():
    from pipeline import RunConfig, run_pipeline, table_market_picks
    md = FakeMarketWide()
    res = run_pipeline(md, RunConfig(watch=["ETH", "SOL", "XRP", "ADA", "NEAR"], n_extra=4, use_news=False, max_watch=3))
    base = {c["ticker"] for c in res["coins"] if not c["is_pick"]}
    picks = {c["ticker"] for c in res["coins"] if c["is_pick"]}
    assert base == {"ETH", "SOL", "XRP"} and res["watch_dropped"] == ["ADA", "NEAR"]
    assert len(picks) == 4 and not (picks & base)
    assert all(c["xs_pct"] is None or 0 <= c["xs_pct"] <= 1 for c in res["coins"])
    mp = table_market_picks(res["coins"])
    assert len(mp) == 4 and "Momentum เทียบตลาด" in mp.columns


# ───────────── v5.2: คะแนนควรซื้อ / เทคนิค / Roadmap / Export ─────────────
def test_technical_votes(coin):
    c, _ = coin
    t = quant.technical_votes(c)
    assert len(t["votes"]) >= 12 and t["bull"] + t["bear"] + t["neutral"] == len(t["votes"])
    assert 0 <= t["score"] <= 100 and all(v["vote"] in (-1, 0, 1) and v["detail"] for v in t["votes"])


def test_parse_event_date():
    from pipeline import parse_event_date
    assert parse_event_date("2026-12-10") == (pd.Timestamp("2026-12-10"), False)
    assert parse_event_date("2027-02") == (pd.Timestamp("2027-02-01"), True)
    assert parse_event_date("2027-Q2")[0] == pd.Timestamp("2027-04-01")
    assert parse_event_date("soon")[0] is None


def test_roadmap_and_event_net():
    from pipeline import build_roadmap, event_net, roadmap_df
    ev = lambda d, dr, imp="HIGH": {"date": d, "type": "UNLOCK", "title": f"ev {d} {dr}", "impact": imp,  # noqa: E731
                                    "certainty": "CONFIRMED", "source_url": "https://x.y", "direction": dr, "why": "w"}
    research = {"SOL": {"verified": True, "data": {"events": [ev("2026-12-01", "POSITIVE"), ev("2025-01-01", "POSITIVE")]}},
                "ENA": {"verified": True, "data": {"events": [ev("2026-11-20", "NEGATIVE")]}}}
    cal = {"verified": True, "events": [{"date": "2026-12-10", "title": "FOMC", "category": "FED", "coins": [],
                                         "direction": "MIXED", "impact": "HIGH", "certainty": "CONFIRMED",
                                         "why": "ลดดอกเบี้ย = +", "source_url": "https://fed.gov"}]}
    rm = build_roadmap(research, cal, ["SOL", "ENA"], today=pd.Timestamp("2026-11-09"))
    assert [r["dt"] for r in rm] == sorted(r["dt"] for r in rm) and len(rm) == 3  # ตัดอีเวนต์ปี 2025 ออก
    assert event_net(rm, "SOL") > 0 > event_net(rm, "ENA")
    assert event_net(rm, "BTC") == 0  # มีแค่ MIXED ทั้งตลาด
    assert len(roadmap_df(rm)) == 3


def test_buy_score_bounds():
    from ai import enforce_buy_score
    assert enforce_buy_score(95, 50) == 65 and enforce_buy_score(5, 50) == 35
    assert enforce_buy_score(None, 42) == 42 and enforce_buy_score(55, 50) == 55


def test_decision_plan_export_and_evaluate(fake_md):
    from pipeline import (RunConfig, analysis_export_df, decision_table, evaluate_exports, run_pipeline,
                          trade_plan_table)
    res = run_pipeline(fake_md, RunConfig(watch=["ETH", "SOL"], n_extra=0, use_news=False, horizon=30, max_watch=3))
    for c in res["coins"]:
        assert 0 <= c["buy_sys"] <= 100 and c["buy_final"] == c["buy_sys"] and c["buy_ai"] is None
        assert c["p_up_h"] is None or 0 <= c["p_up_h"] <= 1
    assert len(decision_table(res["coins"], {}, 30)) == 2
    tp = trade_plan_table(res["coins"], res["ts"])
    assert len(tp) == 2 and {"🛑 Stop Loss", "✅ TP1 ขาย 50%", "ทำอะไร"} <= set(tp.columns)
    ex = analysis_export_df(res, 30)
    assert len(ex) == 2 and {"entry_hi", "sl", "tp1", "buy_final", "p_up_h", "tech_pos"} <= set(ex.columns)
    # ย้อนเวลาวิเคราะห์ไปกลางข้อมูลจำลอง แล้วตรวจกับราคาหลังจากนั้น
    d = fake_md.ohlcv("ETHUSDT", "1d")
    t0 = d.index[-200]
    ex["run_ts_utc"] = (t0 + pd.Timedelta(hours=23, minutes=59, seconds=59)).isoformat()
    ex["price"] = [float(fake_md.ohlcv(f"{t}USDT", "1d")["close"].loc[t0]) for t in ex["ticker"]]
    ex["entry_hi"], ex["sl"], ex["tp1"] = ex["price"] * 1.01, ex["price"] * 0.9, ex["price"] * 1.1
    ex["fill_days"], ex["time_stop_days"] = 5, 30
    out = evaluate_exports(ex, fake_md, now=pd.Timestamp("2026-11-07", tz="UTC"))
    assert set(out["table"]["outcome"]) <= {"✅ WIN", "❌ LOSS", "✅ WIN (time stop)", "❌ LOSS (time stop)"}
    assert out["summary"]["closed"] == 2 and out["summary"]["n_h"] == 2


# ───────────── v5.3: Deep dive 1 เหรียญ ─────────────
def test_default_single_focus_and_deep():
    from deep import mtf_table, peer_table, strategy_table
    from pipeline import RunConfig, run_pipeline
    md = FakeMarketWide()
    res = run_pipeline(md, RunConfig(watch=["SOL", "ETH"], n_extra=3, use_news=False))
    base = [c["ticker"] for c in res["coins"] if not c["is_pick"]]
    assert base == ["SOL"] and res["watch_dropped"] == ["ETH"]
    dp = res["deep"]
    assert dp["focus"] == "SOL" and dp["data"] is not None, dp.get("errors")
    d = dp["data"]
    assert [s["tf"] for s in d["snaps"]] == ["1h", "4h", "1d", "1w"]
    names = [(r["tf"], r["name"]) for r in d["lab"]["rows"]]
    assert sum(tf == "1d" for tf, _ in names) == 6 and sum(tf == "1h" for tf, _ in names) == 3
    assert d["peers"]["sector"] == "Layer 1" and d["peers"]["rank"] is not None
    assert len(mtf_table(d["snaps"])) == 4 and len(strategy_table(d["lab"])) == 9 and len(peer_table(d["peers"], "SOL"))
    from pipeline import analysis_export_df
    ex = analysis_export_df(res, 30)
    assert ex.loc[ex["ticker"] == "SOL", "is_focus"].item() is True


def test_pos_backtest_no_lookahead_and_costs(daily):
    from deep import pos_backtest, strategy_positions
    d = indicators.add_indicators(daily)
    bh = pos_backtest(d, np.ones(len(d)), 365)
    c = d["close"].to_numpy()
    r = c[1:] / c[:-1] - 1
    expected = (1 + r[0] - 0.0015) * np.prod(1 + r[1:]) - 1  # เข้าครั้งเดียวที่แท่งแรก เสียต้นทุนฝั่งเดียว
    assert bh["ret"] == pytest.approx(expected, rel=1e-6)
    pos = strategy_positions(d)["MA20/50 Crossover"]
    full = pos_backtest(d, pos, 365)
    k = 600
    part = pos_backtest(d.iloc[:k], pos[:k], 365)
    cut = pos_backtest(d, np.r_[pos[:k], np.zeros(len(d) - k)], 365)
    assert part["trades"] <= cut["trades"] <= full["trades"] + 1
    for r in (full, part):
        assert -1 <= r["mdd"] <= 0 and 0 <= r["exposure"] <= 1


# ───────────── v5.4: ทิศทางรายช่วงเวลา / สัญญาณที่กำลังจะเกิด / โปรเจกต์ ─────────────
def test_horizons_and_signals():
    from deep import horizon_table, signals_table
    from pipeline import RunConfig, analysis_export_df, run_pipeline
    res = run_pipeline(FakeMarketWide(), RunConfig(watch=["SOL"], n_extra=0, use_news=False))
    d = res["deep"]["data"]
    hz = d["horizons"]
    assert [h["label"] for h in hz] == ["1 วัน", "1 สัปดาห์", "1 เดือน", "3 เดือน", "1 ปี"]
    for h in hz:
        assert 0 <= h["score"] <= 100 and h["confidence"] in ("สูง", "กลาง", "ต่ำ")
        assert h["p_up"] is None or 0 <= h["p_up"] <= 1
        if h["lo"] is not None:
            assert h["lo"] <= h["mid"] <= h["hi"]
    assert hz[-1]["confidence"] == "ต่ำ"  # 1 ปี ตัวอย่างอิสระน้อยเสมอ
    assert len(horizon_table(hz)) == 5
    for x in d["signals"]:
        assert x["bias"] in (-1, 0, 1) and x["signal"]
    signals_table(d["signals"])
    ex = analysis_export_df(res, 30)
    assert "hz_365d_score" in ex.columns


def test_cross_eta():
    from deep import _cross_eta
    i = pd.RangeIndex(30)
    a = pd.Series(np.linspace(90, 99, 30), index=i)   # เส้นเร็ววิ่งเข้าหาเส้นช้า
    b = pd.Series(np.full(30, 100.0), index=i)
    gap, eta, crossed = _cross_eta(a, b)
    assert gap < 0 and eta is not None and 0 < eta < 5 and not crossed


def test_horizon_view_enforced_and_year_parse():
    from ai import enforce_horizon_views
    from pipeline import parse_event_date
    sys_ = [{"label": "1 ปี", "score": 50.0}]
    v = enforce_horizon_views([{"horizon": "1 ปี", "direction": "UP", "score": 95, "reason": "x"}], sys_)
    assert v["1 ปี"]["score"] == 60
    assert parse_event_date("2027") == (pd.Timestamp("2027-07-01"), True)


# ───────────── v6.0: ทดสอบคะแนนย้อนหลัง + คำตอบเดียว ─────────────
def test_hist_score_no_lookahead(daily):
    from deep import hist_tech_score
    d = indicators.add_indicators(daily)
    full = hist_tech_score(d)
    k = 500
    part = hist_tech_score(indicators.add_indicators(daily.iloc[: k + 1]))
    assert np.allclose(full.iloc[60: k + 1].to_numpy(), part.iloc[60:].to_numpy(), equal_nan=True)
    assert full.dropna().between(0, 100).all()


def test_score_check_random_walk_has_no_skill(daily):
    from deep import check_table, score_check
    d = indicators.add_indicators(daily)
    for H in (7, 30):
        ck = score_check(d, H)
        assert ck is not None and not ck["ok"]  # ข้อมูลสุ่ม → ต้องไม่อ้างว่าทำนายได้
        assert 0 <= ck["p_today"] <= 1 and ck["n_test"] > 50
    check_table([score_check(d, 7)])


def test_score_check_detects_real_edge():
    from deep import score_check
    # ตลาดจำลองที่ "เทรนด์ต่อเนื่องจริง" (autocorrelation บวก) → คะแนนเทคนิคควรมีพลังทำนาย
    rng = np.random.default_rng(11)
    n, r = 1200, np.zeros(1200)
    for i in range(1, n):
        r[i] = 0.25 * r[i - 1] + rng.normal(0, 0.02) + 0.004 * np.sign(np.sin(i / 60))
    close = 100 * np.exp(np.cumsum(r))
    idx = pd.date_range(end="2026-11-06", periods=n, freq="1D", tz="UTC")
    o = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, close) * 1.01, "low": np.minimum(o, close) * 0.99,
                       "close": close, "qvol": 1e6, "volume": 1e6 / close, "taker_q": 5e5}, index=idx)
    ck = score_check(indicators.add_indicators(df), 7)
    assert ck["skill"] > 0 and ck["spread"] > 0


def test_final_verdict_and_home(fake_md):
    from pipeline import RunConfig, final_verdict, run_pipeline
    res = run_pipeline(FakeMarketWide(), RunConfig(watch=["SOL"], n_extra=0, use_news=False))
    c = [x for x in res["coins"] if x["ticker"] == "SOL"][0]
    v = final_verdict(c, res["deep"]["data"])
    assert v["action"][:1] in ("✅", "⏳", "⛔") and v["reasons"] and 0 <= v["score"] <= 100
    assert c["score_check"] is not None


def test_plotly_charts_have_unique_keys():
    """กราฟที่อาจเหมือนกัน 2 ที่ในหน้าเดียว → Streamlit ขึ้น DuplicateElementId ถ้าไม่มี key"""
    import re
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"), encoding="utf-8").read()
    calls = re.findall(r"st\.plotly_chart\((.*)\)", src)
    keys = [re.search(r'key="([^"]+)"', c) for c in calls]
    assert calls and all(keys), "ทุก st.plotly_chart ต้องมี key"
    names = [k.group(1) for k in keys]
    assert len(names) == len(set(names))


# ───────────── v6.1: แผนเมื่อยังไม่แนะนำซื้อ + ข้อมูลพื้นฐาน ─────────────
def test_downside_plan(fake_md):
    from pipeline import RunConfig, downside_plan, final_verdict, run_pipeline
    res = run_pipeline(FakeMarketWide(), RunConfig(watch=["SOL"], n_extra=0, use_news=False))
    c = [x for x in res["coins"] if x["ticker"] == "SOL"][0]
    dn = downside_plan(c, 30)
    assert dn["dip30"] <= dn["dip50"] < c["price"]
    assert dn["buy_zone"][0] < dn["buy_zone"][1] < c["price"] * 1.01
    assert all(x["level"] < c["price"] for x in dn["supports"]) and dn["flip"]
    v = final_verdict(c, res["deep"]["data"])
    assert v["recommend"] == v["action"].startswith("✅")


def test_project_schema_backers_optional():
    from ai import ProjectDeep
    old = {"summary": "s", "category": "c", "roadmap": [], "tokenomics": "", "adoption": "", "competitors": [],
           "growth_drivers": [], "risks": [], "growth_view": "LOW", "growth_reason": ""}
    assert ProjectDeep.model_validate(old).backers == []  # cache เก่าที่ไม่มีนายทุนยังใช้ได้
    new = ProjectDeep.model_validate({**old, "backers": [{"name": "a16z", "kind": "VC", "detail": "Seed 2018",
                                                           "source_url": "https://x"}]})
    assert new.backers[0].kind == "VC"


# ───────────── v6.2: อ่านสถานการณ์ (เฟส · แรงซื้อขาย · จะจบเมื่อไหร่) ─────────────
def test_market_read_all_coins():
    from pipeline import RunConfig, run_pipeline
    res = run_pipeline(FakeMarketWide(), RunConfig(watch=["SOL"], n_extra=4, use_news=False))
    ranks = sorted(res["coins"], key=lambda c: -c["buy_final"])
    for c in res["coins"]:
        rd = c["read"]
        assert rd["phase"] and rd["story"] and all(e["sign"] in (-1, 1) for e in rd["evidence"])
        if rd["timing"]:
            t = rd["timing"]
            assert 1 <= t["p25"] <= t["p50"] <= t["p75"] <= 60
        if not rd["verdict"]["recommend"]:
            assert "ยังไม่แนะนำ" in rd["story"] and rd["flip"]
    top = ranks[0]["read"]
    if not top["verdict"]["recommend"]:
        assert top["story"].startswith("อันดับ 1 ในตาราง แต่")


def test_phase_downtrend_and_selling_pressure():
    from pipeline import market_phase, pressure_evidence
    path = np.r_[np.linspace(100, 160, 200), np.linspace(160, 80, 200)]  # ขึ้นแล้วลงแรง
    d = _shape(path, vol=np.r_[np.full(390, 1e6), np.full(10, 4e6)])
    d["taker_q"] = d["qvol"] * 0.45
    d = indicators.add_indicators(d)
    c = {"d1": d, "cf": {"regime": "DOWN"}, "mom": {"trend": 0.0, "orderflow": -0.05}}
    ev = pressure_evidence(c)
    assert any(e["sign"] < 0 for e in ev)
    phase, _ = market_phase(c, ev)
    assert "ขาลง" in phase or "Capitulation" in phase

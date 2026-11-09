"""🔬 Deep Dive — วิเคราะห์เจาะลึก 1 เหรียญ (v5.3)

1) หลายกรอบเวลา 1H / 4H / 1D / 1W — แต่ละกรอบเป็นบวกหรือลบ เพราะเทคนิคอะไร
2) Backtest 6 กลยุทธ์จากงานวิจัยบนเหรียญนี้ (หักค่าธรรมเนียม+slippage · ตัดสินที่ close แล้วถือแท่งถัดไป · มี Out-of-sample)
   - Trend ensemble Donchian (Zarattini et al. 2025) · MA20/50 crossover และ Bollinger mean reversion (Ekström 2025)
   - ย่อในเทรนด์ (เทรนด์รายวัน + mean reversion) · Setup ของระบบ · Buy & Hold เป็นเกณฑ์
   - 1H: Bollinger mean reversion / MA crossover (Ekström พบ BTC รายชั่วโมงเด้งกลับ) — ข้อมูลสั้น ใช้ประกอบเท่านั้น
3) เทียบกับเหรียญกลุ่มเดียวกัน (sector) + BTC/ETH
ราคา/ผลทุกตัวคำนวณจากข้อมูลจริง ไม่ใช่ AI
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import COST, donchian_trend, momentum_profile, regime_arr, setup_mask
from config import MAJORS, OOS_FRACTION, PEERS_MAX, SECTORS
from market import add_indicators, closed_only

TF_LABEL = {"1h": "1 ชั่วโมง", "4h": "4 ชั่วโมง", "1d": "รายวัน", "1w": "รายสัปดาห์"}
BARS_PER_YEAR = {"1h": 8760, "4h": 2190, "1d": 365, "1w": 52}


# ════════════════════ 1) Multi-timeframe ════════════════════
def tf_snapshot(d: pd.DataFrame, tf: str) -> dict:
    """สรุปกรอบเวลาเดียว: เทคนิคแต่ละตัว +1/0/-1 และภาพรวม"""
    x, p = d.iloc[-1], d.iloc[-2]
    votes = []
    e_long = x.ema200 if np.isfinite(x.ema200) else x.ema50
    trend = 1 if x.close > e_long and x.ema20 > x.ema50 else (-1 if x.close < e_long and x.ema20 < x.ema50 else 0)
    votes.append(("EMA", trend, f"ราคา{'เหนือ' if x.close > e_long else 'ใต้'} EMA{'200' if np.isfinite(x.ema200) else '50'}"))
    rsi = float(x.rsi)
    votes.append(("RSI", 1 if 52 <= rsi <= 72 else (-1 if rsi < 42 or rsi > 78 else 0), f"{rsi:.0f}"))
    mh = float(x.macd_h)
    votes.append(("MACD", 1 if mh > 0 and mh >= p.macd_h else (-1 if mh < 0 and mh <= p.macd_h else 0),
                  "บวก" if mh > 0 else "ลบ"))
    adx = float(x.adx) if np.isfinite(x.adx) else 0.0
    votes.append(("ADX/DI", (1 if x.pdi > x.mdi else -1) if adx >= 20 else 0, f"{adx:.0f}"))
    dt = donchian_trend(d)
    dtv = float(dt.iloc[-1]) if np.isfinite(dt.iloc[-1]) else None
    if dtv is not None:
        votes.append(("Donchian", 1 if dtv >= 0.6 else (-1 if dtv <= 0.2 else 0), f"{dtv * 5:.0f}/5"))
    z = float(x.bb_z) if np.isfinite(x.bb_z) else 0.0
    votes.append(("Bollinger Z", -1 if z > 2.2 else (1 if z < -1.5 and trend >= 0 else 0), f"{z:+.1f}"))
    net = sum(v[1] for v in votes)
    view = "🟢 บวก" if net >= 2 else ("🔴 ลบ" if net <= -2 else "⚪ กลาง")
    return {"tf": tf, "label": TF_LABEL[tf], "close": float(x.close), "view": view, "net": net, "votes": votes,
            "bars": len(d), "last": d.index[-1]}


def mtf_table(snaps: list[dict]) -> pd.DataFrame:
    rows = []
    for s in snaps:
        row = {"กรอบเวลา": s["label"], "ภาพรวม": s["view"], "คะแนน": f"{s['net']:+d}"}
        for name, v, det in s["votes"]:
            row[name] = f"{'🟢' if v > 0 else '🔴' if v < 0 else '⚪'} {det}"
        rows.append(row)
    return pd.DataFrame(rows)


def mtf_alignment(snaps: list[dict]) -> str:
    """กรอบใหญ่ (1D/1W) กำหนดทิศ · กรอบเล็ก (1H/4H) ใช้หาจังหวะ"""
    big = [s["net"] for s in snaps if s["tf"] in ("1d", "1w")]
    small = [s["net"] for s in snaps if s["tf"] in ("1h", "4h")]
    if big and min(big) >= 1 and small and min(small) >= 1:
        return "🟢 ทุกกรอบเวลาไปทางเดียวกัน (ขาขึ้น)"
    if big and max(big) <= -1 and small and max(small) <= -1:
        return "🔴 ทุกกรอบเวลาเป็นขาลง — หลีกเลี่ยงการซื้อ"
    if big and min(big) >= 1 and small and min(small) <= -1:
        return "🟡 กรอบใหญ่ขาขึ้น แต่กรอบเล็กกำลังย่อ → จังหวะ 'ย่อในเทรนด์' รอสัญญาณกลับตัว 1H/4H"
    if big and max(big) <= -1 and small and max(small) >= 1:
        return "🟠 กรอบเล็กเด้ง แต่กรอบใหญ่ยังขาลง → เด้งสั้น ระวังโดนขายซ้ำ"
    return "⚪ สัญญาณแต่ละกรอบเวลาขัดกัน — รอให้ชัดก่อน"


# ════════════════════ 2) Strategy backtests ════════════════════
def _stateful(enter: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    pos, on = np.zeros(len(enter)), False
    for i in range(len(enter)):
        if not on and enter[i]:
            on = True
        elif on and exit_[i]:
            on = False
        pos[i] = on
    return pos


def strategy_positions(d: pd.DataFrame) -> dict[str, np.ndarray]:
    """ตำแหน่งที่ตัดสินที่ close ของแต่ละแท่ง (0..1) — ยังไม่ shift"""
    c = d["close"]
    sma20, sma50 = c.rolling(20).mean(), c.rolling(50).mean()
    reg = regime_arr(d)
    z = d["bb_z"].to_numpy()
    out = {
        "Trend ensemble (Donchian)": np.nan_to_num(donchian_trend(d).to_numpy()),
        "MA20/50 Crossover": (sma20 > sma50).to_numpy(float),
        "Bollinger Mean Reversion": _stateful((c < d["bb_lo"]).to_numpy(), (c >= d["bb_mid"]).to_numpy()),
        "ย่อในเทรนด์ (Trend + MR)": _stateful((reg == "UP") & (np.nan_to_num(z) <= -1.5),
                                              (c >= d["bb_mid"]).to_numpy() | (reg == "DOWN")),
        "Setup ระบบ (EMA/MACD/RSI)": setup_mask(d).astype(float),
        "Buy & Hold": np.ones(len(d)),
    }
    return out


def pos_backtest(d: pd.DataFrame, pos: np.ndarray, bars_per_year: int) -> dict:
    """ถือตาม pos[t] ในแท่ง t+1 (ไม่มี look-ahead) · หักต้นทุนตามการเปลี่ยนตำแหน่ง · Out-of-sample = 30% ท้าย"""
    c = d["close"].to_numpy(float)
    r = np.r_[0.0, c[1:] / c[:-1] - 1]
    held = np.r_[0.0, np.nan_to_num(np.asarray(pos, float))[:-1]]
    turn = np.abs(np.diff(np.r_[0.0, held]))
    net = held * r - turn * COST
    eq = np.cumprod(1 + net)
    n = len(net)

    def stats(x: np.ndarray) -> dict:
        if len(x) < 10:
            return {"ret": None, "sharpe": None}
        sd = x.std()
        return {"ret": float(np.prod(1 + x) - 1), "sharpe": float(x.mean() / sd * np.sqrt(bars_per_year)) if sd > 0 else 0.0}

    trades, start = [], None
    for i in range(n):
        if held[i] > 0 and start is None:
            start = i
        if start is not None and (held[i] == 0 or i == n - 1):
            end = i if held[i] == 0 else i + 1
            trades.append(float(np.prod(1 + net[start:end]) - 1))
            start = None
    tr = np.array(trades)
    wins, losses = tr[tr > 0], tr[tr <= 0]
    split = int(n * (1 - OOS_FRACTION))
    years = max(n / bars_per_year, 1e-9)
    full, oos = stats(net), stats(net[split:])
    cagr = None if years < 0.5 else (float(eq[-1] ** (1 / years) - 1) if eq[-1] > 0 else -1.0)  # ข้อมูลสั้น → ไม่คิดต่อปี
    return {"ret": float(eq[-1] - 1), "cagr": cagr,
            "sharpe": full["sharpe"], "mdd": float((eq / np.maximum.accumulate(eq) - 1).min()),
            "exposure": float((held > 0).mean()), "trades": len(tr),
            "win": float(len(wins) / len(tr)) if len(tr) else None,
            "pf": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else None,
            "oos_ret": oos["ret"], "oos_sharpe": oos["sharpe"], "now": float(np.nan_to_num(pos[-1])) > 0,
            "years": n / bars_per_year}


def strategy_lab(d1: pd.DataFrame, h1: pd.DataFrame | None) -> dict:
    """ทดสอบทุกกลยุทธ์บน 1D (+ 1H ถ้ามี) แล้วเลือกกลยุทธ์ที่เหมาะกับเหรียญนี้"""
    rows = []
    for name, pos in strategy_positions(d1).items():
        rows.append({"tf": "1d", "name": name, **pos_backtest(d1, pos, 365)})
    if h1 is not None and len(h1) >= 300:
        hp = strategy_positions(h1)
        for name in ("Bollinger Mean Reversion", "MA20/50 Crossover", "Buy & Hold"):
            rows.append({"tf": "1h", "name": name, **pos_backtest(h1, hp[name], 8760)})
    bh = next(r for r in rows if r["tf"] == "1d" and r["name"] == "Buy & Hold")
    cands = [r for r in rows if r["tf"] == "1d" and r["name"] != "Buy & Hold" and r["trades"] >= 5
             and r["ret"] > 0 and (r["oos_sharpe"] or 0) > 0]
    best = max(cands, key=lambda r: (r["sharpe"] or 0) + (r["oos_sharpe"] or 0), default=None)
    if best is None:
        verdict = "🔴 ยังไม่มีกลยุทธ์ไหนได้ผลบวกทั้งช่วงเต็มและช่วง Out-of-sample บนเหรียญนี้ — ลดขนาดไม้ / รอ"
    elif (best["sharpe"] or 0) > (bh["sharpe"] or 0) or best["mdd"] > bh["mdd"] * 0.7:
        verdict = (f"✅ กลยุทธ์ที่เหมาะกับเหรียญนี้: **{best['name']}** (Sharpe {best['sharpe']:.2f} vs ถือเฉยๆ "
                   f"{bh['sharpe']:.2f} · DD {best['mdd'] * 100:.0f}% vs {bh['mdd'] * 100:.0f}%)")
    else:
        verdict = f"🟡 {best['name']} ได้ผลบวก แต่ยังไม่ดีกว่าถือเฉยๆ ชัดเจน — ใช้เพื่อลดความเสี่ยงมากกว่าเพิ่มกำไร"
    return {"rows": rows, "best": best, "buy_hold": bh, "verdict": verdict}


def strategy_table(lab: dict) -> pd.DataFrame:
    def f(x, pct=True, d=0):
        return "-" if x is None else (f"{x * 100:+.{d}f}%" if pct else f"{x:.2f}")
    return pd.DataFrame([{
        "กลยุทธ์": ("⭐ " if lab["best"] is r else "") + r["name"], "กรอบ": TF_LABEL[r["tf"]],
        "สถานะตอนนี้": "🟢 ถือ" if r["now"] else "⚪ เงินสด", "ผลตอบแทนรวม": f(r["ret"]), "ต่อปี": f(r["cagr"]),
        "Sharpe": f(r["sharpe"], False), "Max DD": f(r["mdd"]), "ไม้": r["trades"], "ชนะ": f"{r['win'] * 100:.0f}%" if r["win"] is not None else "-",
        "Profit Factor": f(r["pf"], False), "Out-of-sample": f(r["oos_ret"]), "Sharpe OOS": f(r["oos_sharpe"], False),
        "เวลาในตลาด": f"{r['exposure'] * 100:.0f}%", "ช่วงข้อมูล": f"{r['years']:.1f} ปี" if r["years"] >= 0.5 else f"{r['years'] * 365:.0f} วัน",
    } for r in lab["rows"]])


# ════════════════════ 3) Peers ════════════════════
def sector_of(ticker: str) -> str | None:
    return next((s for s, xs in SECTORS.items() if ticker in xs), None)


def peers_for(ticker: str) -> list[str]:
    sec = sector_of(ticker)
    peers = [t for t in SECTORS.get(sec, []) if t != ticker][:PEERS_MAX] if sec else []
    return list(dict.fromkeys(peers + [t for t in ("BTC", "ETH") if t != ticker]))


def peer_compare(ticker: str, focus_d1: pd.DataFrame, peer_d1: dict[str, pd.DataFrame]) -> dict:
    """เทียบ momentum/เทรนด์/ความผันผวน กับเหรียญกลุ่มเดียวกัน · คืนตาราง + อันดับของเหรียญที่เจาะลึก"""
    rows = {ticker: (momentum_profile(focus_d1), focus_d1["close"])}
    for t, d in peer_d1.items():
        if d is None or len(d) < 120 or t == ticker:
            continue
        rows[t] = (momentum_profile(d), d["close"])
    fr = focus_d1["close"].pct_change().iloc[-90:]
    out = []
    for t, (p, cl) in rows.items():
        j = pd.concat([fr, cl.pct_change().iloc[-90:]], axis=1, join="inner").dropna()
        out.append({"t": t, "ret30": p["ret"].get(30), "ret90": p["ret"].get(90), "vol": p["vol"],
                    "risk_adj": p["risk_adj"], "trend": p["trend"],
                    "corr": float(j.iloc[:, 0].corr(j.iloc[:, 1])) if len(j) > 30 and t != ticker else 1.0})
    ranked = sorted([r for r in out if r["risk_adj"] is not None], key=lambda r: -r["risk_adj"])
    rank = next((i + 1 for i, r in enumerate(ranked) if r["t"] == ticker), None)
    return {"sector": sector_of(ticker) or "ไม่ระบุกลุ่ม", "rows": ranked, "rank": rank, "n": len(ranked)}


def peer_table(pc: dict, ticker: str) -> pd.DataFrame:
    def f(x):
        return "-" if x is None or not np.isfinite(x) else f"{x * 100:+.0f}%"
    return pd.DataFrame([{
        "อันดับ": i + 1, "เหรียญ": ("🔬 " if r["t"] == ticker else ("🅱️ " if r["t"] in MAJORS else "")) + r["t"],
        "30 วัน": f(r["ret30"]), "90 วัน": f(r["ret90"]), "Momentum ปรับความผันผวน": f"{r['risk_adj']:+.2f}",
        "Trend": f"{r['trend'] * 5:.0f}/5" if r["trend"] is not None else "-",
        "ความผันผวน/ปี": f"{r['vol'] * 100:.0f}%" if r["vol"] and np.isfinite(r["vol"]) else "-",
        "Corr กับเหรียญนี้": f"{r['corr']:.2f}"} for i, r in enumerate(pc["rows"])])


# ════════════════════ รวม ════════════════════
def deep_analysis(c: dict, raw: dict[str, pd.DataFrame], peer_d1: dict[str, pd.DataFrame]) -> dict:
    """raw = แท่งเทียนดิบของเหรียญนี้ {tf: df} · peer_d1 = 1D (ปิดแล้ว + indicators) ของเหรียญคู่แข่ง"""
    frames = {}
    for tf in ("1h", "4h", "1d", "1w"):
        if tf in raw and raw[tf] is not None and len(raw[tf]) > 60:
            frames[tf] = closed_only(add_indicators(raw[tf]))
    snaps = [tf_snapshot(frames[tf], tf) for tf in ("1h", "4h", "1d", "1w") if tf in frames and len(frames[tf]) > 30]
    lab = strategy_lab(c["d1"], frames.get("1h"))
    pc = peer_compare(c["ticker"], c["d1"], peer_d1)
    return {"ticker": c["ticker"], "snaps": snaps, "alignment": mtf_alignment(snaps), "lab": lab, "peers": pc}


def deep_brief(deep: dict) -> dict:
    """ข้อมูลย่อสำหรับส่งให้ AI (ไม่ส่งข้อมูลดิบทั้งหมด)"""
    lab = deep["lab"]
    return {
        "timeframes": [{"tf": s["tf"], "view": s["view"], "net": s["net"],
                        "votes": [f"{n}:{'+' if v > 0 else '-' if v < 0 else '0'}({d})" for n, v, d in s["votes"]]}
                       for s in deep["snaps"]],
        "alignment": deep["alignment"],
        "strategies": [{"name": r["name"], "tf": r["tf"], "ret": round(r["ret"], 3), "sharpe": r["sharpe"] and round(r["sharpe"], 2),
                        "mdd": round(r["mdd"], 3), "oos_sharpe": r["oos_sharpe"] and round(r["oos_sharpe"], 2),
                        "trades": r["trades"], "in_position_now": r["now"]} for r in lab["rows"]],
        "best_strategy": lab["best"]["name"] if lab["best"] else None, "strategy_verdict": lab["verdict"],
        "peers": {"sector": deep["peers"]["sector"], "rank": deep["peers"]["rank"], "of": deep["peers"]["n"],
                  "table": [{"t": r["t"], "ret30": r["ret30"] and round(r["ret30"], 3), "risk_adj": round(r["risk_adj"], 2)}
                            for r in deep["peers"]["rows"]]},
    }

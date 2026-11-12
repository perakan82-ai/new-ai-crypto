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

from analysis import COST, _fc_samples, donchian_trend, independent, laplace, momentum_profile, regime_arr, setup_mask
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


# ════════════════════ 4) สัญญาณที่กำลังจะเกิด (v5.4) ════════════════════
def _cross_eta(a: pd.Series, b: pd.Series, look: int = 5) -> tuple[float, float | None, bool]:
    """ช่องห่าง (a-b)/b ตอนนี้ · จำนวนแท่งโดยประมาณจนตัดกันถ้าแนวโน้มช่องห่างเดิมต่อเนื่อง · เพิ่งตัดกันใน look แท่ง"""
    g = ((a - b) / b.abs().replace(0, np.nan)).dropna()
    if len(g) < look + 2:
        return 0.0, None, False
    now, prev = float(g.iloc[-1]), float(g.iloc[-1 - look])
    rate = (now - prev) / look
    eta = (-now / rate) if rate != 0 and now * rate < 0 else None
    crossed = bool((np.sign(g.iloc[-look - 1:]).diff().fillna(0) != 0).any())
    return now, eta, crossed


def upcoming_signals(c: dict, frames: dict[str, pd.DataFrame]) -> list[dict]:
    """สัญญาณอินดิเคเตอร์ที่ 'เพิ่งเกิด' หรือ 'กำลังจะเกิด' บน 1D และ 1W (ประมาณการเชิงเส้น ถ้าแนวโน้มปัจจุบันต่อเนื่อง)"""
    out: list[dict] = []

    def add(tf, sig, bias, when, trigger, detail):
        out.append({"tf": tf, "signal": sig, "bias": bias, "when": when, "trigger": trigger, "detail": detail})

    for tf, unit, mx in (("1d", "วัน", 20), ("1w", "สัปดาห์", 8)):
        d = frames.get(tf)
        if d is None or len(d) < 60:
            continue
        x = d.iloc[-1]
        lab = TF_LABEL[tf]
        for fast, slow, up_name, dn_name in (("ema20", "ema50", "EMA20 ตัดขึ้น EMA50", "EMA20 ตัดลง EMA50"),
                                             ("ema50", "ema200", "Golden Cross (EMA50 ตัดขึ้น EMA200)",
                                              "Death Cross (EMA50 ตัดลง EMA200)")):
            if d[slow].isna().iloc[-1]:
                continue
            gap, eta, crossed = _cross_eta(d[fast], d[slow])
            if crossed:
                add(lab, up_name if gap > 0 else dn_name, 1 if gap > 0 else -1, "✅ เพิ่งเกิด", f"{x[slow]:.6g}",
                    f"ห่าง {gap * 100:+.1f}%")
            elif eta is not None and eta <= mx:
                add(lab, dn_name if gap > 0 else up_name, -1 if gap > 0 else 1,
                    "⏳ ภายใน 1–2 " + unit if eta < 1.5 else f"⏳ ~{eta:.0f} {unit}",
                    f"{x[slow]:.6g}", f"ช่องห่างแคบลง ตอนนี้ {gap * 100:+.1f}% (ถ้าแนวโน้มเดิมต่อเนื่อง)")
        gap, eta, crossed = _cross_eta(d["macd"], d["macd_sig"])
        if crossed:
            add(lab, "MACD ตัดขึ้น Signal" if gap > 0 else "MACD ตัดลง Signal", 1 if x.macd > x.macd_sig else -1,
                "✅ เพิ่งเกิด", "-", f"Histogram {x.macd_h:+.4g}")
        elif eta is not None and eta <= mx / 2:
            add(lab, "MACD กำลังจะตัดขึ้น" if x.macd < x.macd_sig else "MACD กำลังจะตัดลง",
                1 if x.macd < x.macd_sig else -1, f"⏳ ~{eta:.0f} {unit}", "-", "Histogram เข้าใกล้ 0")
        rsi, rsi5 = float(x.rsi), float(d["rsi"].iloc[-6])
        if 60 <= rsi < 70 and rsi > rsi5:
            add(lab, "RSI กำลังเข้าเขต Overbought (70)", 0, f"⏳ ใกล้ (RSI {rsi:.0f})", "-", "แรงซื้อแรง แต่เสี่ยงพักตัวเมื่อเกิน 70")
        elif 30 < rsi <= 40 and rsi < rsi5:
            add(lab, "RSI กำลังเข้าเขต Oversold (30)", 0, f"⏳ ใกล้ (RSI {rsi:.0f})", "-", "แรงขายแรง มักเด้งเมื่อต่ำกว่า 30")
        elif rsi >= 70:
            add(lab, "RSI Overbought", -1, f"✅ ตอนนี้ RSI {rsi:.0f}", "-", "ร้อนแรงเกิน ระวังย่อ")
        elif rsi <= 30:
            add(lab, "RSI Oversold", 1, f"✅ ตอนนี้ RSI {rsi:.0f}", "-", "ขายมากเกิน มีโอกาสเด้ง")
        elif (rsi - 50) * (rsi5 - 50) < 0:
            add(lab, "RSI ตัดเส้น 50 " + ("ขึ้น" if rsi > 50 else "ลง"), 1 if rsi > 50 else -1, "✅ เพิ่งเกิด", "-", f"RSI {rsi5:.0f} → {rsi:.0f}")
        w = d["bb_width"].iloc[-120:].dropna()
        if len(w) > 40 and (w.iloc[-1] <= w.quantile(0.15)):
            add(lab, "Bollinger บีบตัว (Squeeze)", 0, "⏳ ใกล้เลือกทาง", f"ขึ้น > {x.bb_up:.6g} / ลง < {x.bb_lo:.6g}",
                f"ความกว้างแคบที่สุด 15% ในรอบ {len(w)} แท่ง → มักตามด้วยการเคลื่อนไหวแรง")
        hi20 = float(d["high"].iloc[-21:-1].max())
        lo20 = float(d["low"].iloc[-21:-1].min())
        if x.close > hi20:
            add(lab, "Breakout 20 แท่ง", 1, "✅ เพิ่งเกิด", f"{hi20:.6g}", "ปิดเหนือ High 20 แท่ง (สัญญาณเข้า Donchian)")
        elif hi20 / x.close - 1 <= 0.03:
            add(lab, "ใกล้ Breakout 20 แท่ง", 1, "⏳ ห่าง " + f"{(hi20 / x.close - 1) * 100:.1f}%", f"{hi20:.6g}",
                "ถ้าปิดเหนือระดับนี้ = เข้าเทรนด์ขาขึ้นระยะสั้น")
        if x.close < lo20:
            add(lab, "หลุด Low 20 แท่ง", -1, "✅ เพิ่งเกิด", f"{lo20:.6g}", "สัญญาณอ่อนแรง")
        elif 1 - lo20 / x.close <= 0.03:
            add(lab, "ใกล้หลุด Low 20 แท่ง", -1, f"⏳ ห่าง {(1 - lo20 / x.close) * 100:.1f}%", f"{lo20:.6g}",
                "ถ้าปิดต่ำกว่า = แนวรับระยะสั้นแตก")
        adx, adx5 = float(x.adx), float(d["adx"].iloc[-6])
        if adx >= 20 and adx > adx5 + 3:
            add(lab, "ADX เทรนด์กำลังแรงขึ้น", 1 if x.pdi > x.mdi else -1, f"✅ ADX {adx5:.0f} → {adx:.0f}", "-",
                "+DI นำ (ขาขึ้น)" if x.pdi > x.mdi else "-DI นำ (ขาลง)")
    price = c["price"]
    above = sorted((L for L in c["levels"] if L["level"] > price * 1.003), key=lambda L: L["level"])[:1]
    below = sorted((L for L in c["levels"] if L["level"] < price * 0.997), key=lambda L: -L["level"])[:1]
    for L in above:
        add("แนวสำคัญ", f"แนวต้านถัดไป ({L['src']})", 0, f"ห่าง {(L['level'] / price - 1) * 100:+.1f}%", f"{L['level']:.6g}",
            f"ความแข็งแรง {L['strength']:.0f} · ทะลุได้ = เปิดทางขึ้นต่อ")
    for L in below:
        add("แนวสำคัญ", f"แนวรับถัดไป ({L['src']})", 0, f"ห่าง {(L['level'] / price - 1) * 100:+.1f}%", f"{L['level']:.6g}",
            f"ความแข็งแรง {L['strength']:.0f} · หลุด = เสี่ยงลงต่อ")
    for q in c.get("patterns", []):
        add("รูปแบบกราฟ 1D", f"{q['name']} ({q['state']})", {"bull": 1, "bear": -1}.get(q["bias"], 0),
            "✅ ยืนยันแล้ว" if not q["state"].startswith(("กำลัง", "บีบ")) else "⏳ รอยืนยัน",
            f"{q['level']:.6g}", f"เป้าตามรูปแบบ {q['target']:.6g} · โอกาสแตะใน 60 วัน "
            + (f"{q['p_target_60'] * 100:.0f}%" if q.get("p_target_60") is not None else "-"))
    return out


def signals_table(sigs: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"กรอบ": s["tf"], "สัญญาณ": s["signal"],
                          "ผล": "🟢 บวก" if s["bias"] > 0 else "🔴 ลบ" if s["bias"] < 0 else "⚪ จับตา",
                          "เมื่อไหร่": s["when"], "ระดับราคา": s["trigger"], "รายละเอียด": s["detail"]} for s in sigs])


# ════════════════════ 5) ทิศทางแต่ละช่วงเวลา (v5.4) ════════════════════
HZ = (("1 วัน", 1, ("1h", "4h")), ("1 สัปดาห์", 7, ("4h", "1d")), ("1 เดือน", 30, ("1d",)),
      ("3 เดือน", 90, ("1d", "1w")), ("1 ปี", 365, ("1w",)))


def horizon_outlook(c: dict, snaps: list[dict], news_short: float, news_long: float,
                    checks: dict[int, dict] | None = None) -> list[dict]:
    """ทิศทาง + คะแนน 0-100 + กรอบราคา + ความมั่นใจ ของแต่ละช่วงเวลา

    คะแนน = โอกาสขึ้นจากสถิติ 50% + เทคนิคของกรอบเวลาที่เกี่ยวข้อง 35% + ข่าว/โรดแมป 15%
    ความมั่นใจ = จำนวนตัวอย่างอิสระ (n) + สถิติ/เทคนิค/ข่าวชี้ทางเดียวกันหรือไม่ — ไม่ใช่การรับประกัน
    """
    tech_by = {s["tf"]: 50 + 50 * s["net"] / max(len(s["votes"]), 1) for s in snaps}
    out = []
    for label, H, tfs in HZ:
        S = _fc_samples(c, H)
        p_up, n_eff, q, mv_up, mv_dn = None, 0, None, None, None
        if S is not None:
            n_eff = S["n_eff"]
            end = S["end"]
            p_up = laplace(float((end > 0).mean()), n_eff)
            q = np.quantile(end, [0.25, 0.5, 0.75])
            mv_up = float(np.median(end[end > 0])) if (end > 0).sum() >= 5 else None  # ถ้าขึ้น ปกติขึ้นเท่าไร
            mv_dn = float(np.median(end[end < 0])) if (end < 0).sum() >= 5 else None  # ถ้าลง ปกติลงเท่าไร
        ts = [tech_by[t] for t in tfs if t in tech_by]
        tech = float(np.mean(ts)) if ts else 50.0
        news = news_short if H <= 30 else (0.5 * news_short + 0.5 * news_long if H <= 90 else news_long)
        ck = (checks or {}).get(H)
        w_tech = 0.35
        if ck is not None:  # ปรับด้วยผลทดสอบย้อนหลังของเหรียญนี้
            if ck["ok"]:
                p_up = 0.5 * (p_up if p_up is not None else 0.5) + 0.5 * ck["p_today"]
            else:
                w_tech = 0.10  # เทคนิคไม่เคยทำนายเหรียญนี้ได้ → ลดน้ำหนัก
        w_stat = 0.85 - w_tech
        score = w_stat * (p_up * 100 if p_up is not None else 50) + w_tech * tech + 0.15 * (50 + 50 * news)
        votes = [np.sign((p_up or 0.5) - 0.5), np.sign(tech - 50), np.sign(news)]
        nz = [v for v in votes if v != 0]
        agree = len(nz) >= 2 and (all(v > 0 for v in nz) or all(v < 0 for v in nz))  # สถิติ/เทคนิค/ข่าว ชี้ทางเดียวกัน
        if score >= 60 and (p_up or 0) >= 0.55:
            direction = "🟢 ขึ้น"
        elif score <= 40 and (p_up or 1) <= 0.45:
            direction = "🔴 ลง"
        else:
            direction = "⚪ ไซด์เวย์/ไม่ชัด"
        conf = "สูง" if n_eff >= 30 and agree else ("กลาง" if n_eff >= 12 and agree else "ต่ำ")
        rel, rel_lv = reliability(ck, n_eff)
        ev = (p_up * mv_up + (1 - p_up) * mv_dn) if (p_up is not None and mv_up is not None and mv_dn is not None) else None
        out.append({"label": label, "H": H, "score": float(np.clip(score, 0, 100)), "p_up": p_up, "tech": tech,
                    "tested": (ck["verdict"].split(" ")[0] if ck else "—"), "w_tech": w_tech,
                    "news": news, "direction": direction, "confidence": conf, "n_eff": n_eff,
                    "lo": c["price"] * (1 + q[0]) if q is not None else None,
                    "mid": c["price"] * (1 + q[1]) if q is not None else None,
                    "hi": c["price"] * (1 + q[2]) if q is not None else None,
                    "p_dn": (1 - p_up) if p_up is not None else None,
                    "ret_lo": float(q[0]) if q is not None else None, "ret_mid": float(q[1]) if q is not None else None,
                    "ret_hi": float(q[2]) if q is not None else None, "move_up": mv_up, "move_dn": mv_dn, "ev": ev,
                    "reliability": rel, "rel_level": rel_lv})
    return out


def reliability(ck: dict | None, n_eff: int) -> tuple[str, str]:
    """ป้ายความน่าเชื่อถือของตัวเลขโอกาส: ok / warn / bad / info (บอกตรงๆ ว่าผ่านการทดสอบย้อนหลังหรือยัง)"""
    if ck is not None:
        if ck["ok"]:
            if n_eff < 30:
                return f"✅ ทดสอบผ่าน แต่ตัวอย่างน้อย (n={n_eff})", "warn"
            return f"✅ ทดสอบย้อนหลังผ่าน (n={n_eff})", "ok"
        if ck["verdict"].startswith("🔴"):
            return f"🔴 ทดสอบแล้วทำนายไม่ได้ ใช้สถิติพื้นฐาน (n={n_eff})", "bad"
        return f"🟡 ทดสอบแล้วทำนายได้อ่อน (n={n_eff})", "warn"
    if n_eff < 30:
        return f"⚠️ ตัวอย่างน้อย ใช้ประกอบเท่านั้น (n={n_eff})", "warn"
    return f"⚪ สถิติย้อนหลัง ยังไม่ได้ทดสอบ (n={n_eff})", "info"


def horizon_table(hz: list[dict], ai_views: dict | None = None) -> pd.DataFrame:
    from config import fp  # noqa: PLC0415
    rows = []
    for h in hz:
        v = (ai_views or {}).get(h["label"]) or {}
        rows.append({"ช่วงเวลา": h["label"], "ทิศทาง": h["direction"], "คะแนน (ระบบ)": f"{h['score']:.0f}/100",
                     "คะแนน (AI)": f"{v['score']:.0f}/100" if v else "-",
                     "โอกาสขึ้น (สถิติ)": f"{h['p_up'] * 100:.0f}%" if h["p_up"] is not None else "-",
                     "เทคนิค": f"{h['tech']:.0f}/100", "ทดสอบย้อนหลัง": h.get("tested", "—"),
                     "ข่าว/โรดแมป": f"{h['news']:+.2f}",
                     "กรอบราคา (50% กลาง)": f"{fp(h['lo'])} – {fp(h['hi'])}" if h["lo"] else "-",
                     "ถ้าขึ้น / ถ้าลง ปกติ": (f"{h['move_up'] * 100:+.1f}% / {h['move_dn'] * 100:+.1f}%"
                                            if h.get("move_up") is not None and h.get("move_dn") is not None else "-"),
                     "ความน่าเชื่อถือ": h.get("reliability", "-"),
                     "ความมั่นใจ": f"{h['confidence']} (n={h['n_eff']})", "AI ให้เหตุผล": v.get("reason", "-")})
    return pd.DataFrame(rows)


# ════════════════════ 6) ทดสอบคะแนนย้อนหลัง + ปรับเทียบ (v6.0) ════════════════════
def hist_tech_score(d: pd.DataFrame) -> pd.Series:
    """คะแนนเทคนิค 0-100 ของทุกวันในอดีต (7 เทคนิคแบบเดียวกับที่ใช้วันนี้ · ใช้ข้อมูลถึงวันนั้นเท่านั้น)"""
    c = d["close"]
    e_long = d["ema200"].fillna(d["ema50"])
    trend = np.where((c > e_long) & (d["ema20"] > d["ema50"]), 1, np.where((c < e_long) & (d["ema20"] < d["ema50"]), -1, 0))
    rsi = d["rsi"]
    mh, mh0 = d["macd_h"], d["macd_h"].shift()
    adx = d["adx"].fillna(0)
    dt = donchian_trend(d)
    z = d["bb_z"]
    obv_up = d["obv"] > d["obv"].shift(20)
    votes = np.vstack([
        trend,
        np.where((rsi >= 52) & (rsi <= 72), 1, np.where((rsi < 42) | (rsi > 78), -1, 0)),
        np.where((mh > 0) & (mh >= mh0), 1, np.where((mh < 0) & (mh <= mh0), -1, 0)),
        np.where(adx >= 20, np.where(d["pdi"] > d["mdi"], 1, -1), 0),
        np.where(dt >= 0.6, 1, np.where(dt <= 0.2, -1, 0)),
        np.where(z > 2.2, -1, np.where((z < -1.5) & (trend >= 0), 1, 0)),
        np.where(obv_up & (c > d["ema20"]), 1, np.where(~obv_up & (c < d["ema20"]), -1, 0)),
    ])
    sc = pd.Series(50 + 50 * votes.mean(axis=0), index=d.index)
    sc.iloc[:60] = np.nan  # warm-up
    return sc


def _cal_p(score: np.ndarray, up: np.ndarray, known: np.ndarray, x: float, H: int, band: float = 10.0) -> tuple[float, int]:
    """โอกาสขึ้นจากวันในอดีตที่คะแนนใกล้ x (±band) · ใช้เฉพาะวันที่รู้ผลแล้ว · ดึงเข้าหา 50% ตามจำนวนตัวอย่างอิสระ"""
    m = known & np.isfinite(score) & (np.abs(score - x) <= band)
    idx = np.where(m)[0]
    if len(idx) == 0:
        return 0.5, 0
    n_ind = len(independent(idx, H))
    return laplace(float(up[idx].mean()), n_ind), n_ind


def score_check(d: pd.DataFrame, H: int, start: int = 250, macro: np.ndarray | None = None) -> dict | None:
    """Backtest คะแนนเทคนิคของเหรียญนี้ที่กรอบเวลา H วัน (walk-forward ไม่มี look-ahead)

    - ทุกวัน t: โอกาสขึ้นที่ปรับเทียบแล้ว = อัตราขึ้นจริงของวันก่อนหน้าที่คะแนนใกล้กัน และ 'รู้ผลแล้ว' ณ วัน t (j + H ≤ t)
    - วัดผลช่วง 30% ท้าย (out-of-sample): Brier skill เทียบการเดาด้วยอัตราขึ้นเฉลี่ยในอดีต · ความแม่นของทิศทาง
    - วันนี้: โอกาสขึ้นของวันในอดีตที่คะแนนใกล้วันนี้
    - macro (สภาพ BTC รายวัน UP/TRANS/DOWN): ถ้ามี จะเทียบเฉพาะวันที่ BTC อยู่ในสภาพเดียวกัน (ตัวอย่างไม่พอ → ไม่กรอง)
    """
    c = d["close"].to_numpy(float)
    n = len(c)
    if n < start + H + 60:
        return None
    sc = hist_tech_score(d).to_numpy()
    mac = np.asarray(macro, dtype=object) if macro is not None and len(macro) == n else None

    def cal(known: np.ndarray, t: int) -> tuple[float, int, bool]:
        if mac is not None:
            p, k = _cal_p(sc, up, known & (mac == mac[t]), sc[t], H)
            if k >= 8:
                return p, k, True
        p, k = _cal_p(sc, up, known, sc[t], H)
        return p, k, False
    fwd = np.full(n, np.nan)
    fwd[: n - H] = c[H:] / c[: n - H] - 1
    up = fwd > 0
    preds, base, actual, ts = [], [], [], []
    for t in range(start, n - H):
        if not np.isfinite(sc[t]):
            continue
        known = np.zeros(n, bool)
        known[: t - H + 1] = True  # วัน j รู้ผล ณ วัน t ก็ต่อเมื่อ j + H ≤ t
        known &= np.isfinite(fwd)
        p, _, _ = cal(known, t)
        preds.append(p)
        base.append(float(up[known].mean()) if known.any() else 0.5)
        actual.append(float(up[t]))
        ts.append(t)
    if len(preds) < 60:
        return None
    preds, base, actual = np.array(preds), np.array(base), np.array(actual)
    k = int(len(preds) * 0.7)
    bs = float(np.mean((preds[k:] - actual[k:]) ** 2))
    bb = float(np.mean((base[k:] - actual[k:]) ** 2))
    skill = 1 - bs / bb if bb > 0 else 0.0
    sure = np.abs(preds[k:] - 0.5) >= 0.05
    hit = float(((preds[k:][sure] > 0.5) == (actual[k:][sure] > 0.5)).mean()) if sure.any() else None
    known_all = np.isfinite(fwd)
    p_today, n_today, macro_used = cal(known_all, n - 1) if np.isfinite(sc[-1]) else (0.5, 0, False)
    macro_up = None
    if mac is not None:
        mm = known_all & (mac == mac[-1])
        if mm.sum() >= 30:
            macro_up = {"state": str(mac[-1]), "up": float(up[mm].mean()), "n_ind": len(independent(np.where(mm)[0], H))}
    hi = known_all & (sc >= 60)
    lo = known_all & (sc <= 40)
    spread = float(fwd[hi].mean() - fwd[lo].mean()) if hi.sum() >= 20 and lo.sum() >= 20 else None
    buckets = []
    for a, b, lab in ((-1, 40, "≤40 (ลบ)"), (40, 60, "40-60 (กลาง)"), (60, 101, "≥60 (บวก)")):
        m = known_all & (sc > a) & (sc <= b) if a >= 0 else known_all & (sc <= b)
        if m.sum():
            buckets.append({"label": lab, "n": int(m.sum()), "n_ind": len(independent(np.where(m)[0], H)),
                            "up": float(up[m].mean()), "ret": float(np.median(fwd[m]))})
    if skill > 0.02 and (spread or 0) > 0:
        verdict, ok = "✅ คะแนนเทคนิคทำนายเหรียญนี้ได้ดีกว่าการเดา", True
    elif skill > -0.02:
        verdict, ok = "🟡 คะแนนเทคนิคทำนายได้อ่อน ใช้ประกอบเท่านั้น", False
    else:
        verdict, ok = "🔴 คะแนนเทคนิคไม่มีพลังทำนายเหรียญนี้ → ระบบลดน้ำหนักเทคนิค", False
    return {"H": H, "skill": skill, "hit": hit, "n_test": int(len(preds) - k), "p_today": p_today, "n_today": n_today,
            "score_today": float(sc[-1]) if np.isfinite(sc[-1]) else None, "spread": spread, "buckets": buckets,
            "verdict": verdict, "ok": ok, "macro_used": macro_used, "macro_up": macro_up}


REG_NAME = {"UP": "ขาขึ้น", "TRANS": "เปลี่ยนผ่าน", "DOWN": "ขาลง"}


def check_table(checks: list[dict]) -> pd.DataFrame:
    rows = []
    for ck in checks:
        bk = {b["label"]: b for b in ck["buckets"]}
        mu = ck.get("macro_up")
        row = {"กรอบเวลา": f"{ck['H']} วัน", "ผลทดสอบ": ck["verdict"],
               "เมื่อ BTC สภาพเดียวกับวันนี้": (f"BTC {REG_NAME.get(mu['state'], mu['state'])}: ขึ้นจริง {mu['up'] * 100:.0f}% "
                                              f"(n≈{mu['n_ind']})" if mu else "-"),
               "ทายทิศถูก (ช่วงทดสอบ)": f"{ck['hit'] * 100:.0f}%" if ck["hit"] is not None else "-",
               "Skill vs เดา": f"{ck['skill'] * 100:+.1f}%", "วันทดสอบ": ck["n_test"]}
        for lab in ("≥60 (บวก)", "40-60 (กลาง)", "≤40 (ลบ)"):
            b = bk.get(lab)
            row[f"คะแนน {lab}: ขึ้นจริง"] = f"{b['up'] * 100:.0f}% (n≈{b['n_ind']})" if b else "-"
        rows.append(row)
    return pd.DataFrame(rows)


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
    sigs = upcoming_signals(c, frames)
    checks = {H: ck for H in (7, 30, 90) if (ck := score_check(c["d1"], H, macro=c.get("macro_arr"))) is not None}
    return {"ticker": c["ticker"], "snaps": snaps, "alignment": mtf_alignment(snaps), "lab": lab, "peers": pc,
            "signals": sigs, "checks": checks,
            "horizons": horizon_outlook(c, snaps, c.get("event_net", 0.0), 0.0, checks)}


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
        "horizons": [{"horizon": h["label"], "system_score": round(h["score"]), "direction": h["direction"],
                      "p_up": h["p_up"] and round(h["p_up"], 2), "tech": round(h["tech"]), "confidence": h["confidence"],
                      "n": h["n_eff"], "range_p25_p75": [h["lo"] and round(h["lo"], 6), h["hi"] and round(h["hi"], 6)],
                      "typical_move_if_up": h.get("move_up") and round(h["move_up"], 3),
                      "typical_move_if_down": h.get("move_dn") and round(h["move_dn"], 3),
                      "expected_move": h.get("ev") and round(h["ev"], 3), "reliability": h.get("reliability")}
                     for h in deep.get("horizons", [])],
        "score_backtest": {f"{H}d": {"verdict": ck["verdict"], "skill": round(ck["skill"], 3), "hit": ck["hit"],
                                     "p_up_calibrated": round(ck["p_today"], 2), "n_similar": ck["n_today"]}
                           for H, ck in deep.get("checks", {}).items()},
        "upcoming_signals": [f"[{x['tf']}] {x['signal']} · {x['when']} · ระดับ {x['trigger']} · "
                             f"{'+' if x['bias'] > 0 else '-' if x['bias'] < 0 else '0'}" for x in deep.get("signals", [])],
        "peers": {"sector": deep["peers"]["sector"], "rank": deep["peers"]["rank"], "of": deep["peers"]["n"],
                  "table": [{"t": r["t"], "ret30": r["ret30"] and round(r["ret30"], 3), "risk_adj": round(r["risk_adj"], 2)}
                            for r in deep["peers"]["rows"]]},
    }


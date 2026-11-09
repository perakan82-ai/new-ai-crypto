"""Module 3 + Orchestration — Scanner, Portfolio Agent, รายงาน และ pipeline รวมทุก Agent"""
from __future__ import annotations

import json
import logging
import re
import numpy as np
import pandas as pd
import sources as na
import time
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel
from typing import Callable
from typing import Literal
from ai import (_r, enforce_buy_score, fallback_strategy, gemini_json, llm_fn, macro_calendar_agent, research_agent,
                strategist_and_risk)
from config import (FEE, MAX_SPREAD, MAX_WATCH, QUOTE, REG_TH, SCAN_DEFAULT_EXTRA, SCAN_TOP, SIGNAL_EXPIRY_H, STABLES,
                    STATUS_TXT, TZ_TH, __version__, dist, fbig, fp, fpct, fprob)
from analysis import _fc_samples, technical_votes
from ai import coin_brief, enforce_horizon_views, focus_ai, project_deep_agent
from deep import deep_analysis, deep_brief, horizon_outlook, peers_for, score_check
from analysis import (BIAS_TH, analyze_coin, cap_for, chart_patterns, corr_matrix, first_touch, macro_agent,
                      momentum_profile, touch_prob, xs_momentum_backtest, xs_rank)
from sources import cg_global, external_bundle, fear_greed, fundamentals_for, rss_for
from market import MarketData, add_indicators, closed_only, fetch_frames
from signals import build_signal


# ════════════════════ portfolio ════════════════════
PF_ACT = {"CUT": "✂️ คัดลอส", "REDUCE": "🟠 ลดน้ำหนัก", "TRIM": "🟡 ทยอยขายทำกำไร", "HOLD": "🔵 ถือต่อ",
          "HOLD_LONG": "🏦 ถือยาว", "ADD": "🟢 ถัวเพิ่ม"}
PF_SEV = {"CUT": 0, "REDUCE": 1, "TRIM": 2, "HOLD": 3, "HOLD_LONG": 4, "ADD": 5}
PF_SLOTS = 10


class HoldingView(BaseModel):
    ticker: str
    action: Literal["CUT", "REDUCE", "TRIM", "HOLD", "HOLD_LONG", "ADD"]
    verdict: str
    reasons: list[str]
    plan: str


class PortfolioReview(BaseModel):
    overall: str
    holdings: list[HoldingView]
    rebalance: list[str]


def _sev_to_act(sev: int) -> str:
    return next(k for k, x in PF_SEV.items() if x == sev)


def holding_plan(c: dict, qty: float, cost: float, news_c: dict | None) -> dict:
    price, atr_ = c["price"], float(c["d1"]["atr"].iloc[-1])
    m, l, cf = c["med"], c["long"], c["cf"]
    rsi = float(c["d1"]["rsi"].iloc[-1])
    pnl = price / cost - 1
    stop_d = m["support"] - 1.2 * atr_
    trailed = False
    if pnl > 0.15:
        new = max(stop_d, cost * 1.02, price - 2.5 * atr_)
        trailed, stop_d = new > stop_d, new
    stop_w = min(l["stop"], stop_d * 0.97)
    cands = {m["tp1"], m["tp2"], l["tp1"], l["tp2"]} | {L["level"] for L in c["levels"] if L["strength"] >= 2}
    ladder: list[float] = []
    for x in sorted(cands):
        if x > price * 1.03 and (not ladder or x >= ladder[-1] * 1.06):
            ladder.append(x)
        if len(ladder) == 3:
            break
    while len(ladder) < 3:
        ladder.append((ladder[-1] if ladder else price) * 1.25)
    probs = [{H: touch_prob(c, x, H) for H in (30, 90, 180)} for x in ladder]
    p_be = {H: (touch_prob(c, cost, H) if cost > price else 1.0) for H in (30, 90, 180)}
    p_stop = {H: touch_prob(c, stop_d, H) for H in (30, 90)}
    ft = first_touch(c, ladder[0], stop_d, 90, price) or {}
    ev = (ft.get("p_tp") or 0) * (ladder[0] / price - 1) - (ft.get("p_sl") or 0) * (1 - stop_d / price)

    p180 = p_be[180] or 0
    hard = price <= stop_d
    if hard:
        act, why = "CUT", f"ราคาหลุดจุดป้องกัน {fp(stop_d)} แล้ว"
    elif pnl < 0:
        if cf["regime"] == "DOWN" and l["w_down"] and p180 < 0.35:
            act, why = "CUT", f"ขาลงทั้งรายวันและรายสัปดาห์ + โอกาสกลับไปทุนใน 6 เดือนแค่ {fprob(p180)}"
        elif cf["regime"] == "DOWN" or l["w_down"]:
            act, why = "REDUCE", "แนวโน้มยังเป็นขาลง → ลดครึ่งหนึ่ง ที่เหลือรอขายตามบันได"
        elif (m["status"] == "🟢" or l["status"] == "🟢") and pnl > -0.25:
            act, why = "ADD", "ราคาอยู่โซนแนวรับ + โครงสร้างยังดี → ถัวลดต้นทุนได้"
        elif l["status"] != "🔴" and p180 >= 0.45:
            act, why = "HOLD_LONG", f"โครงสร้างระยะยาวยังไม่เสีย โอกาสกลับไปทุนใน 6 เดือน {fprob(p180)}"
        else:
            act, why = "HOLD", "ยังไม่หลุดจุดป้องกัน รอสัญญาณชัดขึ้น"
    else:
        if pnl > 0.25 and (rsi > 70 or price >= 0.97 * ladder[0]):
            act, why = "TRIM", f"กำไร {fpct(pnl, 0)} + {'RSI สูง' if rsi > 70 else 'ใกล้แนวต้านหลัก'} → ทยอยขาย"
        elif cf["regime"] == "DOWN" or l["w_down"]:
            act, why = "REDUCE", "มีกำไรแต่แนวโน้มเริ่มเสีย → ล็อกกำไรบางส่วน"
        elif l["w_up"]:
            act, why = "HOLD_LONG", "แนวโน้มรายสัปดาห์ขาขึ้น → ถือยาว ขายตามบันได"
        else:
            act, why = "HOLD", "โครงสร้างยังไม่เสีย"
    news_note = ""
    if news_c and news_c["score"] <= -0.45 and news_c["confidence"] >= 0.45 and act != "CUT":
        act = _sev_to_act(max(PF_SEV[act] - 1, PF_SEV["REDUCE"]))
        news_note = f"ข่าวลบแรง ({news_c['score']:+.2f}) → ลดระดับคำแนะนำ"
    portions = {"TRIM": (40, 30, 30), "HOLD_LONG": (25, 35, 40), "REDUCE": (50, 25, 25)}.get(act, (33, 33, 34))
    return {"ticker": c["ticker"], "qty": qty, "cost": cost, "price": price, "pnl": pnl, "pnl_usd": qty * (price - cost),
            "value": qty * price, "act_sys": act, "why": why, "hard": hard, "news_note": news_note,
            "stop_d": stop_d, "stop_w": stop_w, "trailed": trailed, "ladder": ladder, "probs": probs,
            "portions": portions, "p_be": p_be, "p_stop": p_stop, "ft": ft, "ev": ev, "rsi": rsi,
            "regime": cf["regime"], "med_status": m["status"], "long_status": l["status"], "news": news_c}


def _pf_brief(h: dict, c: dict) -> dict:
    return {"ticker": h["ticker"], "qty": h["qty"], "avg_cost": _r(h["cost"], 6), "price": _r(h["price"], 6),
            "pnl_pct": _r(h["pnl"] * 100, 1), "system_action": h["act_sys"], "system_reason": h["why"],
            "regime": REG_TH[h["regime"]], "structure_1d": c["struct_d"]["label"], "structure_1w": c["struct_w"]["label"],
            "rsi": _r(h["rsi"], 0), "medium_status": h["med_status"], "long_status": h["long_status"],
            "stop_daily": _r(h["stop_d"], 6), "stop_weekly": _r(h["stop_w"], 6),
            "sell_ladder": [{"price": _r(x, 6), "portion_pct": pc, "p_touch_30d": _r(p[30]), "p_touch_90d": _r(p[90]),
                             "p_touch_180d": _r(p[180])} for x, pc, p in zip(h["ladder"], h["portions"], h["probs"])],
            "p_back_to_cost": {f"{k}d": _r(v) for k, v in h["p_be"].items()},
            "p_ladder1_before_stop_90d": _r(h["ft"].get("p_tp")), "expected_value_pct": _r(h["ev"] * 100, 1),
            "fund": {"rank": c["fund"].get("rank"), "ath_dist": _r(c["fund"].get("ath_dist"))},
            "news": {k: (h["news"] or {}).get(k) for k in ("score", "label", "summary", "drivers")}}


def portfolio_ai(client, model: str, holds: list[dict], coins: dict, macro: dict, corr: pd.DataFrame):
    prompt = f"""คุณคือ Portfolio Advisor Agent (Spot) ระดับมืออาชีพ ประเมินพอร์ตจริงของผู้ใช้
MACRO: {json.dumps({k: macro[k] for k in ('regime', 'notes')}, ensure_ascii=False)}
HOLDINGS (ตัวเลขทั้งหมดระบบคำนวณแล้ว): {json.dumps([_pf_brief(h, coins[h['ticker']]) for h in holds], ensure_ascii=False, default=str)}
Correlation 90D ระหว่างเหรียญในพอร์ต: {corr.to_json() if len(corr) else "{}"}

สำหรับทุกเหรียญ ให้:
- action: CUT / REDUCE / TRIM / HOLD / HOLD_LONG / ADD (ระบบยอมให้ต่างจาก system_action ได้ไม่เกิน 1 ระดับ)
- verdict: ข้อสรุป 1 ประโยคที่ชัดเจน
- reasons: 3 เหตุผล (เทคนิค/สถิติ/ข่าว-พื้นฐาน) อ้างตัวเลขจากข้อมูล
- plan: แผนปฏิบัติ 1-2 ประโยค (ขายกี่ % ที่ราคาไหน, ถ้าขาดทุน: โอกาสคืนทุน vs ค่าเสียโอกาส)
และ overall = ภาพรวมพอร์ต 2 ประโยค, rebalance = คำแนะนำปรับพอร์ต 2-4 ข้อ
กฎ: ใช้เฉพาะราคาที่อยู่ในข้อมูล ห้ามคิดราคาใหม่ · ภาษาไทยกระชับ"""
    out, _, _, errs = gemini_json(client, model, prompt, PortfolioReview, search=False)
    if not out:
        return None, errs
    for v in out.holdings:
        v.ticker = v.ticker.upper().replace(QUOTE, "").strip()
    return out, errs


def enforce_holding_ai(h: dict, v: HoldingView | None) -> str:
    """AI เปลี่ยนคำแนะนำได้ไม่เกิน 1 ระดับ และเปลี่ยนไม่ได้เมื่อหลุดจุดคัดลอสแล้ว"""
    if v is None or h["hard"]:
        return h["act_sys"]
    s, a = PF_SEV[h["act_sys"]], PF_SEV[v.action]
    if h["news_note"] and a > s:
        return h["act_sys"]
    if abs(a - s) <= 1:
        return v.action
    return _sev_to_act(s + (1 if a > s else -1))


def holding_md(h: dict) -> str:
    v = h.get("view") or {}
    pr, cost, p = h["probs"], h["cost"], h["price"]
    ai_txt = f" · AI: {PF_ACT[h['act_ai']]}" if h.get("act_ai") and h["act_ai"] != h["act"] else ""
    rows = [f"| 🛑 คัดลอส (ปิดรายวัน) | {fp(h['stop_d'])}{' ⬆️trail' if h['trailed'] else ''} | {fpct(h['stop_d'] / cost - 1)} | "
            f"{fpct(h['stop_d'] / p - 1)} | {fprob(h['p_stop'][30])} / {fprob(h['p_stop'][90])} / – | ขายทั้งหมด |",
            f"| 🧱 ยอมแพ้ระยะยาว (ปิดรายสัปดาห์) | {fp(h['stop_w'])} | {fpct(h['stop_w'] / cost - 1)} | {fpct(h['stop_w'] / p - 1)} | – | ขายทั้งหมด |"]
    if cost > p:
        rows.append(f"| ↩️ คืนทุน | {fp(cost)} | 0% | {fpct(cost / p - 1)} | {fprob(h['p_be'][30])} / {fprob(h['p_be'][90])} / "
                    f"{fprob(h['p_be'][180])} | – |")
    for i, (x, pc, pb) in enumerate(zip(h["ladder"], h["portions"], pr)):
        rows.append(f"| 🎯 ขายไม้ {i + 1} | {fp(x)} | {fpct(x / cost - 1)} | {fpct(x / p - 1)} | "
                    f"{fprob(pb[30])} / {fprob(pb[90])} / {fprob(pb[180])} | ขาย {pc}% |")
    reasons = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(v.get("reasons", []))) or f"- {h['why']}"
    nw = h.get("news")
    news_line = f"\n**📰 ข่าว:** {nw['emoji']} {nw['label']} ({nw['score']:+.2f}) — {nw['summary']}" if nw else ""
    plan = f"\n**แผน:** {v['plan']}" if v.get("plan") else ""
    note = f"\n⚠️ {h['news_note']}" if h["news_note"] else ""
    sign = "+" if h["pnl_usd"] >= 0 else "-"
    return f"""### {PF_ACT[h['act']].split()[0]} {h['ticker']} · ถือ {h['qty']:g} @ {fp(cost)} → {fp(p)} ({fpct(h['pnl'])} · {sign}${abs(h['pnl_usd']):,.0f})
**คำแนะนำ: {PF_ACT[h['act']]}** (ระบบ: {PF_ACT[h['act_sys']]}{ai_txt}) · Regime {REG_TH[h['regime']]} · ระยะกลาง {h['med_status']} · ระยะยาว {h['long_status']}

> 🧠 {v.get('verdict') or h['why']}

**ทำไม:**
{reasons}
{plan}{note}{news_line}

| ระดับ | ราคา | จากทุน | จากราคาตอนนี้ | โอกาสแตะ 30 / 90 / 180 วัน | ทำอะไร |
|---|---|---|---|---|---|
{chr(10).join(rows)}

โอกาสถึงไม้ 1 ก่อนโดนคัดลอส (90 วัน): **{fprob(h['ft'].get('p_tp'))}** · โดนคัดลอสก่อน: {fprob(h['ft'].get('p_sl'))} · ค่าคาดหวัง {h['ev'] * 100:+.1f}%

---
"""


# ════════════════════ report ════════════════════
# ───────────────────────── Position sizing ─────────────────────────
def size_medium(c: dict, capital: float, risk_pct: float, alloc_pct: float | None) -> dict:
    """ขนาดไม้จากความเสี่ยงต่อไม้ (% ทุน) แต่ไม่เกิน allocation/cap"""
    m = c["med"]
    per_unit = m["entry"] - m["sl"]
    qty = capital * risk_pct / 100 / per_unit
    cap = min(capital * (alloc_pct if alloc_pct else cap_for(c)) / 100, capital)
    if qty * m["entry"] > cap:
        qty = cap / m["entry"]
    loss = qty * per_unit + qty * m["entry"] * 2 * FEE
    return {"qty": qty, "notional": qty * m["entry"], "loss": loss, "loss_pct": loss / capital * 100}


def size_long(c: dict, capital: float, alloc_pct: float | None) -> dict:
    l = c["long"]
    budget = capital * (alloc_pct if alloc_pct else min(10.0, cap_for(c))) / 100
    tiers = [{"price": p, "usd": budget * w, "qty": budget * w / p, "w": w} for p, w in zip(l["tiers"], l["weights"])]
    qty = sum(t["qty"] for t in tiers)
    return {"budget": budget, "tiers": tiers, "qty": qty, "avg": budget / qty, "loss_at_stop": budget - qty * l["stop"]}


# ───────────────────────── Text helpers ─────────────────────────
def key_catalyst(c: dict, research: dict) -> str:
    rd = ((research.get(c["ticker"]) or {}).get("data")) or {}
    ev = rd.get("events") or []
    if ev:
        warn = "" if ev[0].get("source_url", "").startswith("http") else "⚠️ "
        return f"{warn}{ev[0]['title'][:40]} ({ev[0]['date']})"
    if rd.get("strengths"):
        return rd["strengths"][0][:50]
    return c["struct_d"]["label"]


def news_badge(c: dict) -> str:
    n = c.get("news")
    return f"{n['emoji']} {n['score']:+.2f} ({n['n']})" if n else "-"


def rule_verdict(c: dict) -> str:
    m, l = c["med"], c["long"]
    f30 = c["fc"].get(30, {})
    if m["status"] == "🟢":
        s = f"ซื้อได้ในโซน {fp(m['zone'][0])}–{fp(m['zone'][1])} เป้า {fp(m['tp1'])}"
    elif m["status"] == "🟡":
        s = f"ตั้งรอซื้อ {fp(m['zone'][0])}–{fp(m['zone'][1])} (โอกาสได้ของใน 14 วัน {fprob(m['p_fill'])}) เป้า {fp(m['tp1'])}"
    else:
        s = f"เลี่ยงระยะกลาง · ระยะยาว {STATUS_TXT[l['status']]} DCA เริ่ม {fp(l['tiers'][0])}"
    if m["ft1"] and m["status"] != "🔴":
        s += f" · โอกาสถึงก่อน SL {fprob(m['ft1']['p_tp'])}"
    if f30:
        s += f" · 30 วัน: โอกาส 50% แตะ {fp(f30['reach50'])}"
    return s


# ───────────────────────── Tables ─────────────────────────
def table_medium(coins, research, final_win) -> pd.DataFrame:
    rows = []
    for c in coins:
        m = c["med"]
        ft = m["ft1"] or {}
        rows.append({
            "Ticker": ("⭐" if c.get("is_pick") else "") + c["ticker"], "ราคา": fp(c["price"]),
            "คำแนะนำ": f"{m['status']} {STATUS_TXT[m['status']]}",
            "Entry Zone": f"{fp(m['zone'][0])} – {fp(m['zone'][1])}", "Stop Loss": fp(m["sl"]),
            "TP1 / TP2": f"{fp(m['tp1'])} / {fp(m['tp2'])}", "R:R": f"1:{m['rr1']:.1f} / 1:{m['rr2']:.1f}",
            "Win Rate": f"{final_win.get((c['ticker'], 'MEDIUM'), m['win_sys']):.0f}% (คุ้มทุน {m['be_win']:.0f}%)",
            "TP1 ก่อน SL": (f"{fprob(ft.get('p_tp'))} · ~{ft['days_tp']:.0f} วัน" if ft.get("days_tp")
                           else fprob(ft.get("p_tp"))),
            "Time stop": f"{m['tstop']} วัน", "ข่าว 7D": news_badge(c), "Key Catalyst": key_catalyst(c, research),
        })
    return pd.DataFrame(rows)


def table_long(coins, research, final_win) -> pd.DataFrame:
    return pd.DataFrame([{
        "Ticker": ("⭐" if c.get("is_pick") else "") + c["ticker"], "ราคา": fp(c["price"]),
        "คำแนะนำ": f"{c['long']['status']} {STATUS_TXT[c['long']['status']]}", "กรอบเวลา": c["long"]["hold"],
        "Win Rate": f"{final_win.get((c['ticker'], 'LONG'), c['long']['win_sys']):.0f}%",
        "DCA T1/T2/T3": " / ".join(fp(t) for t in c["long"]["tiers"]),
        "น้ำหนัก": "/".join(f"{int(w * 100)}" for w in c["long"]["weights"]),
        "ต้นทุนเฉลี่ย": fp(c["long"]["avg"]), "TP1 / TP2": f"{fp(c['long']['tp1'])} / {fp(c['long']['tp2'])}",
        "โอกาสแตะ TP1 ใน 180D": fprob(c["long"]["p_tp1_180"]), "ข่าว 7D": news_badge(c),
        "ตัดขาดทุน (W close)": fp(c["long"]["stop"]), "Key Catalyst": key_catalyst(c, research),
    } for c in coins])


def table_forecast(coins) -> pd.DataFrame:
    rows = []
    for c in coins:
        for H in (14, 30, 90, 180):
            v = c["fc"].get(H)
            if not v:
                continue
            rows.append({"Ticker": c["ticker"], "ภายใน": f"{H} วัน", "ราคาตอนนี้": fp(c["price"]),
                         "70% แตะ ≥": f"{fp(v['reach70'])} ({dist(c, v['reach70'])})",
                         "50% แตะ ≥": f"{fp(v['reach50'])} ({dist(c, v['reach50'])})",
                         "30% แตะ ≥": f"{fp(v['reach30'])} ({dist(c, v['reach30'])})",
                         "ย่อลึก (50%)": f"{fp(v['dip50'])} ({dist(c, v['dip50'])})",
                         "กรอบปลายงวด": f"{fp(v['end25'])} – {fp(v['end75'])}",
                         "โอกาสปิดสูงกว่าวันนี้": fprob(v["p_up"]), "ตัวอย่างอิสระ": v["n_eff"]})
    return pd.DataFrame(rows)


def table_backtest(coins) -> pd.DataFrame:
    """ผล Backtest ของแผนระยะกลาง (เข้าแท่งถัดไป · หักค่าธรรมเนียม+slippage)"""
    rows = []
    for c in coins:
        b = c["bt"]
        rows.append({"Ticker": c["ticker"], "ไม้": b["n"], "Win (ดิบ)": fprob(b["win_rate_raw"]) if b["n"] else "-",
                     "Expectancy": f"{b['expectancy']:+.2f}R",
                     "Profit Factor": f"{b['profit_factor']:.2f}" if b["profit_factor"] else "-",
                     "Max DD": f"{b['max_dd_R']:.1f}R",
                     "In-sample": f"{b['is_n']} ไม้ · {b['is_expectancy']:+.2f}R",
                     "Out-of-sample": f"{b['oos_n']} ไม้ · {b['oos_expectancy']:+.2f}R" if b["oos_n"] else "-",
                     "SL/TP (ATR)": f"{b['sl_mult']:.1f} / {b['tp_mult']:.1f}"})
    return pd.DataFrame(rows)


def forecast_md(c: dict) -> str:
    lines = ["| ภายใน | โอกาส 70% แตะ ≥ | โอกาส 50% แตะ ≥ | โอกาส 30% แตะ ≥ | ย่อลึก (50%) | กรอบปลายงวด (50% กลาง) | ปิดสูงกว่าวันนี้ |",
             "|---|---|---|---|---|---|---|"]
    for H, v in c["fc"].items():
        lines.append(f"| {H} วัน | {fp(v['reach70'])} ({dist(c, v['reach70'])}) | {fp(v['reach50'])} ({dist(c, v['reach50'])}) | "
                     f"{fp(v['reach30'])} ({dist(c, v['reach30'])}) | {fp(v['dip50'])} ({dist(c, v['dip50'])}) | "
                     f"{fp(v['end25'])}–{fp(v['end75'])} | {fprob(v['p_up'])} |")
    return "\n".join(lines)


def deep_dive_md(c: dict, research: dict, views: dict, pick_by: dict, capital: float, risk_pct: float,
                 news_ctx: dict | None = None) -> str:
    m, l, r, f, cf = c["med"], c["long"], c["risk"], c["fund"], c["cf"]
    d = c["d1"].iloc[-1]
    rr = research.get(c["ticker"]) or {}
    rd, verified = rr.get("data") or {}, rr.get("verified", True)
    v = views.get(c["ticker"]) or {}
    pm, pl = pick_by.get((c["ticker"], "MEDIUM")), pick_by.get((c["ticker"], "LONG"))
    win_m = pm["win_final"] if pm else m["win_sys"]
    win_l = pl["win_final"] if pl else l["win_sys"]
    sm = size_medium(c, capital, risk_pct, pm["alloc_final"] if pm else None)
    sl_ = size_long(c, capital, pl["alloc_final"] if pl else None)
    ft1, ft2 = m["ft1"] or {}, m["ft2"] or {}
    bt = c["bt"]
    reasons = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(v.get("reasons", []))) or \
        "\n".join(f"- {x}" for x in m["reasons"])
    ev = "\n".join(f"- `{e['date']}` **{e['type']}** {e['title']} · {e['impact']} · {e['certainty']}"
                   + (f" ([ที่มา]({e['source_url']}))" if e.get("source_url", "").startswith("http") else " ⚠️ ไม่มีลิงก์อ้างอิง")
                   for e in rd.get("events", [])) or "- ไม่พบข้อมูลยืนยัน"
    news = "\n".join(f"- {n}" for n in rd.get("news_30d", [])[:5]) or "- (เปิด AI Research เพื่อดูข่าว)"
    if rd and not verified:
        news = "⚠️ *ข้อมูลจาก AI ที่ไม่ได้ค้นเว็บ อาจไม่อัปเดต*\n" + news
    tiers = "\n".join(f"| T{i + 1} | {fp(t['price'])} | {dist(c, t['price'])} | {int(t['w'] * 100)}% | ${t['usd']:,.0f} | {t['qty']:.4g} | {fprob(l['tier_fill'][i])} |"
                      for i, t in enumerate(sl_["tiers"]))
    pf = f"{f['pf']:.1f}x" if f.get("pf") else "-"
    tp1_txt = f"{fprob(ft1.get('p_tp'))} ภายใน {m['tstop']} วัน" + (f" (~{ft1['days_tp']:.0f} วัน)" if ft1.get("days_tp") else "")
    news_sec = (na.news_md(news_ctx["report"], news_ctx["items"], c["ticker"], 5)
                if (news_ctx and news_ctx.get("items")) else "- ปิด News Agent")
    watch = f"**👀 จับตา:** {v['watch']}" if v.get("watch") else ""
    pat_md = "\n".join(f"- **{p['name']}** {BIAS_TH[p['bias']]} · {p['state']} · ระดับ {fp(p['level'])} → เป้า {fp(p['target'])} "
                       f"({dist(c, p['target'])}, โอกาสแตะใน 60 วัน {fprob(p.get('p_target_60'))}) · {p['note']}"
                       for p in c.get("patterns", [])) or f"- ไม่พบรูปแบบเด่น · แท่งล่าสุด: {c['pattern']}"
    strat = f"> 🧭 Strategist: {pm['rationale']} · ⚠️ {pm['key_risk']}" if pm else ""
    invalid = v.get("invalidation") or f"ราคาปิดต่ำกว่า {fp(m['sl'])} (ระยะกลาง) / Weekly ปิดต่ำกว่า {fp(l['stop'])} (ระยะยาว)"
    oos = f" · Out-of-sample {bt['oos_n']} ไม้ {bt['oos_expectancy']:+.2f}R" if bt["oos_n"] else ""
    obv = "↑ สะสม" if c["d1"]["obv"].iloc[-1] > c["d1"]["obv"].iloc[-21] else "↓ แจกจ่าย"
    fdvm = f"{f['fdv_mcap']:.2f}" if f.get("fdv_mcap") else "-"
    proj = f"- **โปรเจกต์:** {rd.get('project', '')} · Tokenomics: {rd.get('tokenomics', '-')}" if rd else ""
    bb = f"- 🐂 {rd.get('bull_case', '-')} · 🐻 {rd.get('bear_case', '-')} · Catalyst {rd.get('catalyst_score')}" if rd else ""
    tech = c.get("tech") or {"votes": [], "score": 50, "bull": 0, "bear": 0}
    tech_rows = "\n".join(f"| {t['name']} | {'🟢 บวก' if t['vote'] > 0 else '🔴 ลบ' if t['vote'] < 0 else '⚪ กลาง'} | {t['detail']} |"
                          for t in tech["votes"])
    ai_s = f"**{c['buy_final']:.0f}/100**" if c.get("buy_ai") is not None else "— (ไม่มี AI)"
    score_sec = (f"คะแนนควรซื้อ AI {ai_s} · ระบบ **{c.get('buy_sys', 0):.0f}/100** · มุมมอง **{outlook_label(c) if 'buy_sys' in c else '-'}** · "
                 f"โอกาสขึ้น (สถิติ) **{fprob(c.get('p_up_h'))}** · เทคนิคเห็นตรงกัน {tech['score']:.0f}/100 "
                 f"(บวก {tech['bull']} · ลบ {tech['bear']}) · ข่าวข้างหน้า {c.get('event_net', 0):+.2f}")
    return f"""### 🪙 {f.get('name') or c['ticker']} ({c['ticker']}) · {fp(c['price'])}
**⏱ ระยะกลาง:** {m['status']} {STATUS_TXT[m['status']]} · **🏦 ระยะยาว:** {l['status']} {STATUS_TXT[l['status']]} · **Regime:** {REG_TH[cf['regime']]} · **Score:** {c['score']:.0f}/100

> 🧠 **สรุป:** {v.get('verdict') or rule_verdict(c)}

**ทำไม:**
{reasons}

**❌ มุมมองผิดเมื่อ:** {invalid}
{watch}

#### 🧮 คะแนน & เทคนิคที่ใช้ตัดสิน
{score_sec}

| เทคนิค | ผล | รายละเอียด |
|---|---|---|
{tech_rows}

*คะแนนควรซื้อ ≠ โอกาสชนะ · โอกาสขึ้นมาจากสถิติย้อนหลังของเหรียญนี้ในสภาพตลาดคล้ายกัน ไม่รับประกันอนาคต*

#### 🎯 แผนระยะกลาง (Swing {m['hold']})
| จุด | ราคา | ห่างจากตอนนี้ | โอกาส / เวลา | ทำอะไร |
|---|---|---|---|---|
| 🟦 ซื้อ | {fp(m['zone'][0])} – {fp(m['zone'][1])} | {dist(c, m['zone'][1])} | ได้ของใน 14 วัน {fprob(m['p_fill'])} | ตั้ง Limit แบ่ง 2 ไม้ |
| 🛑 Stop Loss | {fp(m['sl'])} | {dist(c, m['sl'])} | โดนก่อน TP1 {fprob(ft1.get('p_sl'))} | ขายทั้งหมดทันที |
| ✅ TP1 | {fp(m['tp1'])} | {dist(c, m['tp1'])} | {tp1_txt} | ขาย 50% + เลื่อน SL ไปที่ทุน |
| 🚀 TP2 | {fp(m['tp2'])} | {dist(c, m['tp2'])} | {fprob(ft2.get('p_tp'))} ภายใน {min(90, m['tstop'] * 2)} วัน | ขายส่วนที่เหลือ |
| ⏳ Time stop | {m['tstop']} วัน | | | ไม่ถึง TP1 → ขายออก |

R:R **1:{m['rr1']:.1f} / 1:{m['rr2']:.1f}** · Win rate **{win_m:.0f}%** (คุ้มทุนที่ {m['be_win']:.0f}%) · Backtest {bt['n']} ไม้ expectancy {bt['expectancy']:+.2f}R{oos} (เข้าแท่งถัดไป หักค่าธรรมเนียม+slippage)
→ ขนาดไม้: **{sm['qty']:.4g} {c['ticker']} ≈ ${sm['notional']:,.0f}** · ขาดทุนถึง SL ${sm['loss']:,.0f} ({sm['loss_pct']:.2f}% ของทุน)
{strat}

#### 🏦 แผนระยะยาว (DCA · ถือ {l['hold']})
| Tier | ราคา | ห่างจากตอนนี้ | น้ำหนัก | เงิน | จำนวน | ได้ของใน 90 วัน |
|---|---|---|---|---|---|---|
{tiers}

ต้นทุนเฉลี่ย **{fp(sl_['avg'])}** · TP1 **{fp(l['tp1'])}** ({dist(c, l['tp1'])}, แตะใน 180 วัน {fprob(l['p_tp1_180'])}) ขาย 30% · TP2 **{fp(l['tp2'])}** ({dist(c, l['tp2'])}, {fprob(l['p_tp2_180'])}) ขายที่เหลือ
· 🛑 ตัดขาดทุนเมื่อ **Weekly ปิด < {fp(l['stop'])}** (ขาดทุนสูงสุด ~${sl_['loss_at_stop']:,.0f}) · Win rate {win_l:.0f}%

#### 📰 ข่าว & Sentiment 7 วัน
{news_sec}

#### 📐 รูปแบบกราฟ (1D)
{pat_md}

#### 📈 คาดการณ์ราคา (สถิติของเหรียญนี้ ปรับตามความผันผวนปัจจุบัน)
{forecast_md(c)}

<details><summary>🔬 รายละเอียดเทคนิค / พื้นฐาน / ข่าว</summary>

- **โครงสร้าง:** 1D {c['struct_d']['label']} {c['struct_d']['bos']} · 1W {c['struct_w']['label']} · EMA9/21/50/200 = {fp(d.ema9)} / {fp(d.ema21)} / {fp(d.ema50)} / {fp(d.ema200)} · RSI {d.rsi:.0f} · ADX {d.adx:.0f} · แท่งล่าสุด: {c['pattern']} · 4H trigger {'✅' if m['trig4h'] else '—'}
- **Bollinger/VWAP:** BB {fp(d.bb_lo)}–{fp(d.bb_up)} (กว้าง {d.bb_width * 100:.1f}%) · VWAP20 {fp(d.vwap20)}
- **Volume Profile 180D:** POC {fp(c['vp']['poc'])} · Value Area {fp(c['vp']['val'])}–{fp(c['vp']['vah'])} · Vol ratio {d.vol_ratio:.2f}x · OBV 20D {obv}
- **Fib 52W:** 0.382 {fp(c['fib']['ret'][0.382])} · 0.618 {fp(c['fib']['ret'][0.618])} · 0.786 {fp(c['fib']['ret'][0.786])} · High {fp(c['fib']['hi52'])}
- **Risk:** Vol {r['vol_ann'] * 100:.0f}%/ปี (cap {cap_for(c):.0f}%) · MDD 1Y {r['mdd_1y'] * 100:.0f}% · Beta {r['beta']:.2f} · RS90 vs BTC {fpct(r['rs90'])}
- **พื้นฐาน:** Rank #{f.get('rank', '-')} · MCap {fbig(f.get('mcap'))} · FDV/MCap {fdvm} · ห่าง ATH {fpct(f.get('ath_dist'), 0)} · TVL {fbig(f.get('tvl'))} · Fees 30D {fbig(f.get('fees30d'))} · P/F {pf}
{proj}
{bb}

**ข่าว 30 วัน**
{news}

**กำหนดการ 6 เดือน**
{ev}
</details>

---
"""


def summary_text(res: dict, coins: list[dict], port: dict) -> str:
    m = res["macro"]
    fng = res["fng"]
    reg = {"Bull": "🟢 Bull", "Bear": "🔴 Bear", "Neutral": "🟡 Neutral"}[m["regime"]]
    L = [f"🧠 Crypto Spot Brief {res['ts']:%d/%m %H:%M}",
         f"Macro {reg} | BTC {fp(m['btc_price'])}" + (f" | F&G {fng['value']}" if fng else ""), ""]
    cb = {c["ticker"]: c for c in coins}
    sigs = res.get("signals") or []
    if sigs:
        L.append("🚦 สัญญาณ (ผ่านเกณฑ์คุณภาพ)")
        for s in sigs:
            L.append(f"• {s.ticker} ซื้อ {fp(s.entry_lo)}–{fp(s.entry_hi)} | SL {fp(s.sl)} | TP {fp(s.tp1)} | "
                     f"คะแนน {s.confidence:.0f} | TP ก่อน SL {s.win_prob or 0:.0f}%")
    med = [p for p in port["picks"] if p["horizon"] == "MEDIUM"]
    lng = [p for p in port["picks"] if p["horizon"] == "LONG"]
    if med:
        L.append("⏱ ระยะกลาง")
        for p in med:
            x = cb[p["ticker"]]["med"]
            ft = x["ft1"] or {}
            L.append(f"• {p['ticker']} {x['status']}{(cb[p['ticker']].get('news') or {}).get('emoji', '')} "
                     f"ซื้อ {fp(x['zone'][0])}–{fp(x['zone'][1])} | SL {fp(x['sl'])} | "
                     f"TP {fp(x['tp1'])}/{fp(x['tp2'])} | ถึงก่อน SL {fprob(ft.get('p_tp'))} | {p['alloc_final']:.0f}%")
    if lng:
        L.append("🏦 ระยะยาว (DCA)")
        for p in lng:
            x = cb[p["ticker"]]["long"]
            L.append(f"• {p['ticker']} {x['status']} " + "/".join(fp(t) for t in x["tiers"]) +
                     f" | TP {fp(x['tp1'])}/{fp(x['tp2'])} | Stop W<{fp(x['stop'])} | {p['alloc_final']:.0f}%")
    others = [c for c in coins if c["ticker"] not in {p["ticker"] for p in port["picks"]}]
    if others:
        L.append("👀 เฝ้าดู")
        for c in others[:6]:
            L.append(f"• {c['ticker']} {c['med']['status']}/{c['long']['status']} {fp(c['price'])} — "
                     f"{' · '.join(rule_verdict(c).split(' · ')[:2])}")
    L += ["", f"💵 เงินสด {port['cash']:.0f}%", "⚠️ ไม่ใช่คำแนะนำการลงทุน"]
    return "\n".join(L)


def full_report_md(res: dict, coins, research, views, pick_by, final_win, capital, risk_pct, md_table) -> str:
    macro, port = res["macro"], res["port"]
    return (f"# Crypto AI Lab Report — {res['ts']:%d %b %Y %H:%M} (UTC+7)\n\n"
            f"Macro {macro['regime']} · BTC {fp(macro['btc_price'])} · Cash {port['cash']:.0f}%\n\n"
            "## 1. 📊 ตารางสรุป\n\n### ⏱ ระยะกลาง\n\n" + md_table(table_medium(coins, research, final_win)) +
            "\n\n### 🏦 ระยะยาว\n\n" + md_table(table_long(coins, research, final_win)) +
            "\n\n### 🧪 Backtest\n\n" + md_table(table_backtest(coins)) +
            "\n\n## 2. 🔍 บทวิเคราะห์รายเหรียญ\n\n" +
            "\n".join(deep_dive_md(c, research, views, pick_by, capital, risk_pct, res.get("news")) for c in coins) +
            "\n\n> ⚠️ เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน")


def symbol_of(ticker: str) -> str:
    return f"{ticker}{QUOTE}"


def ts_th(ts: datetime) -> str:
    return ts.strftime("%d/%m %H:%M")


# ════════════════════ pipeline ════════════════════
log = logging.getLogger(__name__)
Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


@dataclass
class RunConfig:
    watch: list[str]         # เหรียญที่ผู้ใช้สนใจ (ใช้แค่ max_watch ตัวแรก · ตัวแรก = เจาะลึก)
    n_extra: int = SCAN_DEFAULT_EXTRA
    max_watch: int = MAX_WATCH
    deep: bool = True        # วิเคราะห์เจาะลึกเหรียญตัวแรก (1H-1W · 6 กลยุทธ์ · คู่แข่ง)
    news_items: list = field(default_factory=list)  # ข่าวที่เก็บได้ในรอบนี้ (ใช้ใน Deep dive)
    horizon: int = 30        # กรอบเวลาที่สนใจ (วัน) ใช้จัดอันดับโอกาสและให้ AI คัดเหรียญ
    min_qvol: float = 20e6
    cg_key: str = ""
    client: object | None = None
    model: str = "gemini-2.5-flash"
    model_research: str = "gemini-2.5-flash"
    n_research: int = 8
    n_med: int = 3
    n_long: int = 3
    use_news: bool = True
    news_w: float = 8.0
    holdings: list[str] = field(default_factory=list)
    research_cache: dict = field(default_factory=dict)
    ai_agents: bool = True   # Research + Strategist + Risk Manager (ปิดใน scan_job เพื่อประหยัดโควต้า)
    news_ai: bool = True     # ใช้ AI วิเคราะห์ Sentiment (ปิด = keyword)


# ───────────────────────── Market Scanner ─────────────────────────
def detect_setups(d: pd.DataFrame) -> list[str]:
    """ตรวจจับ Volume Spike / Breakout / Trend Reversal / Momentum จากแท่งปิดล่าสุด"""
    if len(d) < 60:
        return []
    x, p = d.iloc[-1], d.iloc[-2]
    tags = []
    if x.vol_ratio >= 2.5:
        tags.append(f"Volume Spike {x.vol_ratio:.1f}x")
    hh20 = d["high"].iloc[-21:-1].max()
    if x.close > hh20 and x.vol_ratio >= 1.3:
        tags.append("Breakout 20 แท่ง")
    if p.ema20 <= p.ema50 and x.ema20 > x.ema50:
        tags.append("EMA20 ตัดขึ้น EMA50")
    if p.macd_h <= 0 < x.macd_h and x.rsi < 55:
        tags.append("MACD กลับตัวขึ้น")
    if 55 <= x.rsi <= 70 and x.adx >= 25 and x.pdi > x.mdi:
        tags.append(f"Momentum (ADX {x.adx:.0f})")
    return tags


def scan_universe(md: MarketData, exclude: set[str], n: int, min_qvol: float, top: int = SCAN_TOP,
                  progress: Callable[[float, str], None] | None = None) -> dict:
    """AI Market Scanner — คัดเหรียญที่น่าสนใจจากตลาด ณ ตอนนี้ (Spot USDT สภาพคล่องสูง)

    1) กรอง: ไม่ใช่ stablecoin/leveraged token · Volume ≥ min_qvol · spread ≤ MAX_SPREAD
    2) คำนวณทุกเหรียญ (รวม Watchlist เพื่อใช้เป็นตลาดอ้างอิง): Momentum ปรับความผันผวนเทียบตลาด (percentile),
       Trend ensemble (Donchian 5 lookback), สัญญาณเทคนิค, รูปแบบกราฟ, order flow
    3) Backtest การจัดอันดับย้อนหลัง → บอกว่าอันดับนี้ "เคยมี edge" หรือไม่
    คืน {"picks", "tags", "info", "xs_bt", "universe", "dropped_spread", "ref_risk_adj"}
    """
    out = {"picks": [], "tags": {}, "info": {}, "xs_bt": None, "universe": 0, "dropped_spread": 0, "ref_risk_adj": []}
    t = md.tickers()
    t = t[~t["base"].isin(STABLES) & ~t["base"].str.endswith(("UP", "DOWN", "BULL", "BEAR"))]
    t = t[(t["quoteVolume"] >= min_qvol) & (t["count"] > 20_000)]
    if "spread" in t:
        wide = t["spread"].notna() & (t["spread"] > MAX_SPREAD)
        out["dropped_spread"] = int(wide.sum())
        t = t[~wide]
    t = t.sort_values("quoteVolume", ascending=False).head(top)
    if t.empty:
        return out
    fr = fetch_frames(md, list(t.index), ("1d",), progress)
    rows, prof, closes = [], {}, {}
    for s, f in fr.items():
        if "1d" not in f or len(f["1d"]) < 120:
            continue
        d = closed_only(add_indicators(f["1d"]))
        closes[s] = d["close"]
        prof[s] = momentum_profile(d)
        last = d.iloc[-1]
        vr = t.loc[s, "quoteVolume"] / max(d["qvol"].iloc[-20:].mean(), 1)
        sc = 2 * min(vr, 3) + (45 <= last.rsi <= 70) + (last.close > last.ema50) + (last.macd_h > 0)
        sc += bool(not np.isnan(last.ema200) and last.close > last.ema200)
        tg = detect_setups(d)
        pats = chart_patterns(d)
        tg += [f"{p['name']} ({p['state']})" for p in pats if p["bias"] == "bull"]
        sc += 0.5 * len(tg) - sum(p["bias"] == "bear" for p in pats)
        tr = prof[s]["trend"]
        if tr is not None:
            sc += 2 * tr - (2 if tr == 0 else 0)  # ไม่มี lookback ไหนอยู่ในเทรนด์ = ลงโทษ
            if tr >= 0.6:
                tg.append(f"Trend {tr * 5:.0f}/5")
        of = prof[s]["orderflow"]
        if of is not None and of > 0.02:
            sc += 0.5
            tg.append(f"แรงซื้อเชิงรุกเพิ่ม {of * 100:+.1f}%")
        out["tags"][s] = tg
        rows.append((s, sc))
    pct = xs_rank(prof)
    for s, sc in rows:
        out["info"][s] = {**prof[s], "xs_pct": pct.get(s), "scan_score": sc}
    # Momentum เทียบตลาด: percentile สูง = แข็งกว่าเหรียญส่วนใหญ่ในตลาด (cross-sectional)
    ranked = sorted(((s, sc + 3 * (pct.get(s) or 0)) for s, sc in rows), key=lambda x: x[1], reverse=True)
    out["picks"] = [s for s, _ in ranked if s[: -len(QUOTE)] not in exclude][:n]
    out["universe"] = len(rows)
    out["ref_risk_adj"] = [v["risk_adj"] for v in prof.values() if v["risk_adj"] is not None]
    if len(closes) >= 10:
        out["xs_bt"] = xs_momentum_backtest(pd.DataFrame(closes))
    return out


def market_pct(x: float | None, ref: list[float]) -> float | None:
    """percentile ของค่า x เทียบกับตลาดอ้างอิง"""
    if x is None or not ref:
        return None
    return float(np.mean(np.asarray(ref) <= x))


def opportunity_rank(coins: list[dict], horizon: int, research: dict | None = None) -> pd.DataFrame:
    """จัดอันดับเหรียญที่มีโอกาสขึ้นในกรอบเวลา horizon วัน (ตัวเลขจากสถิติ + เหตุผลประกอบ)

    คะแนนโอกาส = 35% โอกาสปิดสูงกว่าวันนี้ + 20% upside ที่มีโอกาส 50% แตะ (ปรับด้วยความผันผวน)
                 + 15% สถานะจุดเข้า + 10% รูปแบบกราฟ + 10% ข่าว/อีเวนต์
                 + 10% Momentum เทียบตลาด/Trend ensemble — ไม่ใช่การรับประกัน
    """
    H = min((h for h in (7, 14, 30, 60, 90, 180) if h >= horizon), default=180)
    rows = []
    for c in coins:
        v = c["fc"].get(H)
        if not v:
            continue
        up50 = v["reach50"] / c["price"] - 1
        dn50 = v["dip50"] / c["price"] - 1
        rr = up50 / max(-dn50, 1e-6)
        news = (c.get("news") or {})
        rd = ((research or {}).get(c["ticker"]) or {}).get("data") or {}
        cat = (rd.get("catalyst_score") or 0) / 100
        pattern_score = sum({"bull": 1, "bear": -1}.get(p["bias"], 0) for p in c.get("patterns", []))
        mo = c.get("mom") or {}
        xs, tr = c.get("xs_pct"), mo.get("trend")
        s = 35 * v["p_up"] + 20 * float(np.clip(rr / 2, 0, 1)) + 15 * {"🟢": 1, "🟡": 0.5, "🔴": 0}[c["med"]["status"]]
        s += 10 * float(np.clip(0.5 + pattern_score / 4, 0, 1))
        s += 10 * float(np.clip(0.5 + news.get("score", 0) * news.get("confidence", 0) + cat / 2, 0, 1))
        s += 10 * (0.5 * (xs if xs is not None else 0.5) + 0.5 * (tr if tr is not None else 0.5))
        why = [f"{p['name']} {p['state']}" for p in c.get("patterns", []) if p["bias"] == "bull"]
        if xs is not None and xs >= 0.7:
            why.append(f"Momentum แข็งกว่าตลาด {xs * 100:.0f}%")
        if c["med"].get("mr4h"):
            why.append("ย่อในเทรนด์ขาขึ้น (4H)")
        why += c.get("scan_tags", [])[:2]
        if news.get("score", 0) >= 0.12:
            why.append(f"ข่าวบวก {news['score']:+.2f}")
        if cat > 0.2:
            why.append(f"อีเวนต์โปรเจกต์ +{cat * 100:.0f}")
        if c["med"]["status"] == "🟢":
            why.append("อยู่ในโซนซื้อ")
        warn = [f"{p['name']} ({BIAS_TH['bear']})" for p in c.get("patterns", []) if p["bias"] == "bear"]
        if c["cf"]["regime"] == "DOWN":
            warn.append("Regime ขาลง")
        if xs is not None and xs <= 0.3:
            warn.append(f"Momentum อ่อนกว่าตลาด ({xs * 100:.0f}%)")
        if news.get("score", 0) <= -0.12:
            warn.append(f"ข่าวลบ {news['score']:+.2f}")
        rows.append({"Ticker": ("⭐" if c.get("is_pick") else "") + c["ticker"], "คะแนนโอกาส": round(float(s)),
                     f"โอกาสปิดสูงกว่าวันนี้ใน {H} วัน": fprob(v["p_up"]),
                     "Momentum เทียบตลาด": fprob(xs), "Trend": f"{tr * 5:.0f}/5" if tr is not None else "-",
                     "50% แตะ ≥": f"{fp(v['reach50'])} ({fpct(up50)})", "30% แตะ ≥": f"{fp(v['reach30'])} ({dist(c, v['reach30'])})",
                     "ย่อลึก (50%)": f"{fp(v['dip50'])} ({fpct(dn50)})",
                     "จุดเข้า": f"{c['med']['status']} {fp(c['med']['zone'][0])}–{fp(c['med']['zone'][1])}",
                     "เหตุผล": " · ".join(why) or "-", "ข้อควรระวัง": " · ".join(warn) or "-",
                     "ตัวอย่างอิสระ": v["n_eff"]})
    df = pd.DataFrame(rows)
    return df.sort_values("คะแนนโอกาส", ascending=False).reset_index(drop=True) if len(df) else df


def table_market_picks(coins: list[dict]) -> pd.DataFrame:
    """เหรียญที่ AI Market Scanner คัดจากตลาด (⭐) พร้อมเหตุผลเชิงตัวเลข และสถานะผ่านเกณฑ์จุดเข้า"""
    rows = []
    for c in sorted([c for c in coins if c.get("is_pick")], key=lambda c: -c["score"]):
        mo = c.get("mom") or {}
        r = mo.get("ret") or {}
        fails = c.get("gate_fails") or []
        rows.append({"เหรียญ": c["ticker"], "ราคา": fp(c["price"]), "Momentum เทียบตลาด": fprob(c.get("xs_pct")),
                     "Trend (Donchian)": f"{mo['trend'] * 5:.0f}/5" if mo.get("trend") is not None else "-",
                     "30D / 90D": f"{fpct(r.get(30), 0)} / {fpct(r.get(90), 0)}",
                     "ความผันผวน/ปี": f"{mo['vol'] * 100:.0f}%" if mo.get("vol") and np.isfinite(mo["vol"]) else "-",
                     "สัญญาณ Scanner": " · ".join(c.get("scan_tags", [])[:4]) or "-",
                     "จุดเข้า": "✅ ผ่านเกณฑ์" if not fails else f"{c['med']['status']} ยังไม่ยืนยัน",
                     "โซนซื้อ": f"{fp(c['med']['zone'][0])}–{fp(c['med']['zone'][1])}",
                     "ยังไม่ผ่านเพราะ": " · ".join(fails[:2]) or "-"})
    return pd.DataFrame(rows)


# ───────────────────────── v5.2: คะแนนควรซื้อ / Roadmap / แผนเข้า-ออก / Export ─────────────────────────
DIR_EMO = {"POSITIVE": "🟢 +", "NEGATIVE": "🔴 −", "MIXED": "⚪ ±"}
_IMPACT_W = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}
_CERT_W = {"CONFIRMED": 1.0, "EXPECTED": 0.7, "RUMOR": 0.3}
_DIR_V = {"POSITIVE": 1, "NEGATIVE": -1, "MIXED": 0}


def horizon_bucket(horizon: int) -> int:
    return min((h for h in (7, 14, 30, 60, 90, 180) if h >= horizon), default=180)


def parse_event_date(s: str) -> tuple[pd.Timestamp | None, bool]:
    """'2026-12-10' / '2026-12' / '2026-Q1' → (วันที่, ประมาณการหรือไม่)"""
    s = (s or "").strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return pd.Timestamp(s), False
        if re.fullmatch(r"\d{4}-\d{2}", s):
            return pd.Timestamp(s + "-01"), True
        q = re.fullmatch(r"(\d{4})-?Q([1-4])", s, re.I)
        if q:
            return pd.Timestamp(f"{q.group(1)}-{3 * int(q.group(2)) - 2:02d}-01"), True
        if re.fullmatch(r"\d{4}", s):  # ระบุแค่ปี → ประมาณกลางปี
            return pd.Timestamp(f"{s}-07-01"), True
    except ValueError:
        pass
    return None, True


def build_roadmap(research: dict, macro_cal: dict | None, tickers: list[str], today: pd.Timestamp | None = None) -> list[dict]:
    """รวมอีเวนต์ในอนาคตจาก Research รายเหรียญ + ปฏิทินมหภาค → เรียงตามวันที่ (ตัดอีเวนต์ที่ผ่านไปแล้ว)"""
    now = today if today is not None else pd.Timestamp.now(tz=TZ_TH)
    today = (now.tz_localize(None) if now.tzinfo else now).normalize()
    rows = []
    for t, r in (research or {}).items():
        for e in ((r or {}).get("data") or {}).get("events", []):
            rows.append({"coins": [t], "scope": t, **{k: e.get(k) for k in ("date", "title", "impact", "certainty",
                                                                             "source_url", "why")},
                         "type": e.get("type", "OTHER"), "direction": e.get("direction") or "MIXED",
                         "verified": (r or {}).get("verified", False)})
    for e in (macro_cal or {}).get("events", []):
        cs = [x.upper().replace(QUOTE, "") for x in e.get("coins") or []]
        rows.append({"coins": cs, "scope": ", ".join(cs) if cs else "🌐 ตลาดรวม", "type": e.get("category", "OTHER"),
                     **{k: e.get(k) for k in ("date", "title", "impact", "certainty", "source_url", "why", "direction")},
                     "verified": (macro_cal or {}).get("verified", False)})
    out, seen = [], set()
    for r in rows:
        dt, approx = parse_event_date(r.get("date"))
        if dt is None or dt + pd.Timedelta(days=31 if approx else 0) < today:
            continue
        key = (r["scope"], (r.get("title") or "")[:40].lower())
        if key in seen:
            continue
        seen.add(key)
        r.update(dt=dt, approx=approx, days=int((dt - today).days),
                 has_src=str(r.get("source_url") or "").startswith("http"))
        out.append(r)
    return sorted(out, key=lambda r: r["dt"])


def event_net(roadmap: list[dict], ticker: str, horizon_days: int = 90) -> float:
    """ผลรวมข่าว/อีเวนต์ข้างหน้าของเหรียญ -1..1 (ทิศทาง × ผลกระทบ × ความแน่นอน · ข่าวทั้งตลาดน้ำหนักครึ่ง)"""
    s = 0.0
    for r in roadmap:
        own = ticker in r["coins"]
        if not own and r["coins"]:
            continue
        w = _IMPACT_W.get(r.get("impact"), 0.3) * _CERT_W.get(r.get("certainty"), 0.3) * (1.0 if own else 0.5)
        w *= 1.0 if r["days"] <= horizon_days else 0.5
        w *= 1.0 if r.get("has_src") else 0.5  # ไม่มีลิงก์อ้างอิง → ลดน้ำหนัก
        s += _DIR_V.get(r.get("direction"), 0) * w
    return float(np.tanh(s / 2))


def entry_points(c: dict) -> float:
    if not c.get("gate_fails"):
        return 100.0
    return {"🟢": 70.0, "🟡": 40.0, "🔴": 0.0}[c["med"]["status"]]


def buy_score_sys(c: dict) -> float:
    """คะแนนควรซื้อของระบบ 0-100 = เทคนิค 35% + โอกาสขึ้น 35% + สถานะจุดเข้า 15% + ข่าว/อีเวนต์ 15%
    ถ้าทดสอบย้อนหลังแล้วเทคนิคทำนายเหรียญนี้ไม่ได้ → เทคนิค 15% / โอกาสขึ้น 55%"""
    n = c.get("news") or {}
    news = 0.5 * n.get("score", 0) * n.get("confidence", 0) + 0.5 * c.get("event_net", 0.0)
    p_up = c.get("p_up_h")
    ck = c.get("score_check")
    w_tech = 0.15 if (ck is not None and not ck["ok"]) else 0.35  # เทคนิคทำนายเหรียญนี้ไม่ได้ → ลดน้ำหนัก
    s = w_tech * c["tech"]["score"] + (0.70 - w_tech) * (p_up * 100 if p_up is not None else 50) + 0.15 * entry_points(c)
    s += 0.15 * (50 + 50 * float(np.clip(news, -1, 1)))
    return float(np.clip(s, 0, 100))


def outlook_label(c: dict) -> str:
    s, p = c.get("buy_final", c.get("buy_sys", 50)), c.get("p_up_h") or 0.5
    t = c["tech"]
    if s >= 62 and p >= 0.55 and t["bull"] > t["bear"]:
        return "🟢 บวก"
    if s <= 42 or (t["bear"] > t["bull"] and p < 0.5):
        return "🔴 ลบ"
    return "⚪ กลาง"


def downside_plan(c: dict, horizon: int = 30) -> dict:
    """กรณียังไม่แนะนำซื้อ: ราคาอาจร่วงถึงไหน · แนวรับ · โซนที่ควรพิจารณาซื้อ · เงื่อนไขที่จะเปลี่ยนเป็นน่าซื้อ
    ทุกตัวเลขมาจากสถิติของเหรียญนี้ (ปรับตามความผันผวนปัจจุบัน) + แนวรับจริง — ไม่ใช่การรับประกัน"""
    H = horizon_bucket(max(horizon, 14))
    price = c["price"]
    fc = c["fc"].get(H) or {}
    d = c["d1"].iloc[-1]
    atr_ = float(d.atr)
    sup = sorted([L for L in c["levels"] if L["level"] < price * 0.99 and L["strength"] >= 2],
                 key=lambda L: -L["level"])[:3]
    supports = [{"level": L["level"], "src": L["src"], "strength": L["strength"],
                 "p": touch_prob(c, L["level"], H)} for L in sup]
    dip50, dip30 = fc.get("dip50"), fc.get("dip30")
    # โซนน่าซื้อ = แนวรับแข็งแรงที่ใกล้จุดย่อ 50% ที่สุด (ถ้าไม่มี ใช้ DCA ไม้แรกของแผนระยะยาว)
    anchor = dip50 if dip50 else price * 0.9
    if supports:
        best = min(supports, key=lambda x: abs(x["level"] - anchor))
        zone = (best["level"] - 0.3 * atr_, best["level"] + 0.2 * atr_)
        why = f"แนวรับ {best['src']} (ความแข็งแรง {best['strength']:.0f}) ใกล้จุดที่สถิติบอกว่ามีโอกาสย่อถึง"
    else:
        t1 = c["long"]["tiers"][0] if c["long"]["tiers"][0] < price else c["long"]["tiers"][1]
        zone = (t1 - 0.3 * atr_, t1 + 0.2 * atr_)
        why = "DCA ไม้แรกของแผนระยะยาว (แนวรับ/Fib/EMA รายสัปดาห์)"
    # เงื่อนไขกลับมาน่าซื้อ: ยืนเหนือ EMA50 + ทะลุ swing high ล่าสุด (หรือ High 20 วัน)
    hi20 = float(c["d1"]["high"].iloc[-20:].max())
    sh = c["struct_d"].get("last_high") or hi20
    flip = []
    if np.isfinite(d.ema50) and price < d.ema50:
        flip.append(f"ปิดรายวันกลับเหนือ EMA50 ({fp(d.ema50)})")
    if c["cf"]["regime"] == "DOWN" and np.isfinite(d.ema200):
        flip.append(f"ยืนเหนือ EMA200 ({fp(d.ema200)}) = จบขาลงหลัก")
    flip.append(f"ทะลุ High ล่าสุด {fp(min(sh, hi20) if sh > price else hi20)} พร้อม Volume")
    flip.append("คะแนนควรซื้อ ≥ 55 และผ่านเกณฑ์จุดเข้าในแท็บนี้")
    return {"H": H, "dip50": dip50, "dip30": dip30, "p_dip50": 0.5 if dip50 else None, "supports": supports,
            "buy_zone": zone, "buy_zone_p": touch_prob(c, zone[1], H), "buy_zone_why": why, "flip": flip,
            "long_stop": c["long"]["stop"]}


def pressure_evidence(c: dict) -> list[dict]:
    """หลักฐานแรงซื้อ/แรงขายจาก Volume, Order flow, OBV, Divergence (แท่ง 1D ที่ปิดแล้ว)"""
    d = c["d1"]
    x = d.iloc[-10:]
    out = []

    def add(sign: int, text: str):
        out.append({"sign": sign, "text": text})

    red = x["close"] < x["open"]
    dv = float(x.loc[red, "qvol"].sum() / max(x["qvol"].sum(), 1e-9))
    if dv >= 0.6:
        add(-1, f"แรงขายครอง: Volume ของวันแดง {dv * 100:.0f}% ใน 10 วันล่าสุด")
    elif dv <= 0.4:
        add(1, f"แรงซื้อครอง: Volume ของวันเขียว {(1 - dv) * 100:.0f}% ใน 10 วันล่าสุด")
    last20 = d.iloc[-20:]
    dist_days = int(((last20["close"] < last20["close"].shift() * 0.99) & (last20["vol_ratio"] > 1.2)).sum())
    if dist_days >= 4:
        add(-1, f"วันเทขาย (ลง >1% พร้อม Volume สูง) {dist_days} วันใน 20 วัน")
    of = (c.get("mom") or {}).get("orderflow")
    if of is not None and abs(of) >= 0.02:
        add(1 if of > 0 else -1, f"แรงซื้อเชิงรุก (taker buy) 7 วัน {'สูง' if of > 0 else 'ต่ำ'}กว่าค่าเฉลี่ย 30 วัน {of * 100:+.1f}%")
    obv_up = d["obv"].iloc[-1] > d["obv"].iloc[-21]
    px_up = d["close"].iloc[-1] > d["close"].iloc[-21]
    if px_up and not obv_up:
        add(-1, "ราคาขึ้นแต่ OBV ลง (Divergence) — การขึ้นไม่มี Volume สนับสนุน")
    elif not px_up and obv_up:
        add(1, "ราคาลงแต่ OBV ขึ้น — มีการสะสมระหว่างราคาลง")
    a, b = d.iloc[-10:], d.iloc[-30:-10]
    if len(b) > 5:
        if a["high"].max() > b["high"].max() and a["rsi"].max() < b["rsi"].max() - 3:
            add(-1, f"Bearish divergence: ราคาทำ High ใหม่ แต่ RSI ต่ำกว่าเดิม ({a['rsi'].max():.0f} < {b['rsi'].max():.0f})")
        if a["low"].min() < b["low"].min() and a["rsi"].min() > b["rsi"].min() + 3:
            add(1, f"Bullish divergence: ราคาทำ Low ใหม่ แต่ RSI สูงกว่าเดิม ({a['rsi'].min():.0f} > {b['rsi'].min():.0f})")
    lb = d.iloc[-1]
    if lb.close < lb.open and lb.vol_ratio >= 2:
        add(-1, f"แท่งล่าสุดลงพร้อม Volume {lb.vol_ratio:.1f} เท่าของปกติ")
    if lb.macd_h < d["macd_h"].iloc[-2] < 0:
        add(-1, "MACD Histogram ติดลบและยังลดลง — แรงลงยังไม่หมด")
    elif d["macd_h"].iloc[-2] < lb.macd_h < 0:
        add(1, "MACD Histogram ติดลบแต่เริ่มหดตัว — แรงขายเริ่มอ่อน")
    return out


def market_phase(c: dict, ev: list[dict]) -> tuple[str, str]:
    """เหรียญอยู่ในเหตุการณ์แบบไหน (กฎง่ายๆ จากเทรนด์ + RSI + Bollinger + หลักฐานแรงขาย)"""
    d = c["d1"].iloc[-1]
    reg = c["cf"]["regime"]
    tr = (c.get("mom") or {}).get("trend") or 0
    rsi, z = float(d.rsi), float(d.bb_z) if np.isfinite(d.bb_z) else 0.0
    sell = sum(e["sign"] < 0 for e in ev)
    ret10 = float(c["d1"]["close"].iloc[-1] / c["d1"]["close"].iloc[-11] - 1)
    if reg == "DOWN" and rsi < 32 and ret10 < -0.12:
        return "🩸 ขายตื่นตระหนก (Capitulation)", "ร่วงแรงในเวลาสั้น RSI ต่ำมาก — มักใกล้จุดต่ำ แต่ต้องรอแท่งกลับตัวยืนยันก่อน"
    if reg == "DOWN":
        return "🔻 ขาลง (Downtrend)", "ราคาอยู่ใต้เส้นค่าเฉลี่ยหลัก การซื้อสวนเทรนด์มีโอกาสติดดอยสูง"
    if reg == "UP" and (sell >= 2 or rsi > 72):
        return "⚠️ แรงขายทำกำไร (Distribution)", "แนวโน้มยังขึ้น แต่เริ่มมีสัญญาณแรงขายออก — มักย่อก่อนไปต่อ"
    if reg == "UP" and z < 0:
        return "📉 ย่อในเทรนด์ขาขึ้น (Pullback)", "แนวโน้มหลักยังดี ราคากำลังพักตัว — รอให้ย่อถึงแนวรับแล้วกลับตัว"
    if reg == "UP" and tr >= 0.6:
        return "🚀 เทรนด์ขาขึ้นแข็งแรง", "ราคาไปต่อได้ แต่ไม่ควรไล่ราคา รอย่อเข้าโซน"
    if float(c["d1"]["bb_width"].iloc[-1]) <= float(c["d1"]["bb_width"].iloc[-120:].quantile(0.2)):
        return "⏸ สะสมพลัง / ไซด์เวย์แคบ (Base)", "ราคาบีบตัวแคบ — รอเลือกทาง ทะลุขึ้นค่อยตาม"
    return "↔️ ไซด์เวย์ / ทิศทางไม่ชัด", "ยังไม่มีฝั่งไหนชนะชัดเจน"


def dip_timing(c: dict, H: int = 30, min_dip: float = 0.03) -> dict | None:
    """ในอดีตที่สภาพตลาดคล้ายวันนี้ ถ้าราคาย่อ ≥3% → จุดต่ำสุดมักเกิดหลังจากนี้กี่วัน (ใช้ประมาณว่าการย่อจะจบเมื่อไหร่)"""
    S = _fc_samples(c, H)
    if S is None:
        return None
    rl = S["rel_l"]
    dipped = rl.min(axis=1) <= -min_dip
    if dipped.sum() < 10:
        return None
    days = rl[dipped].argmin(axis=1) + 1
    return {"p25": float(np.percentile(days, 25)), "p50": float(np.median(days)), "p75": float(np.percentile(days, 75)),
            "n": int(dipped.sum()), "share": float(dipped.mean())}


def market_read(c: dict, horizon: int = 30, ts: datetime | None = None, rank: int | None = None) -> dict:
    """อ่านสถานการณ์ 1 เหรียญ: ควรซื้อไหม · เหตุการณ์ที่กำลังเกิด · แรงซื้อ/ขาย · อาจร่วงถึงไหน · จะจบเมื่อไหร่ · กลับมาดูเมื่อไหร่"""
    ts = ts or datetime.now(TZ_TH)
    v = final_verdict(c)
    ev = pressure_evidence(c)
    phase, phase_why = market_phase(c, ev)
    dn = downside_plan(c, horizon)
    tm = dip_timing(c, min(dn["H"], 60))
    z = dn["buy_zone"]
    head = f"อันดับ {rank} ในตาราง แต่ " if rank == 1 and not v["recommend"] else (f"อันดับ {rank} · " if rank else "")
    fails = c.get("gate_fails") or []
    if v["recommend"]:
        story = (f"{head}✅ แนะนำให้ซื้อได้ในโซน {fp(c['med']['zone'][0])}–{fp(c['med']['zone'][1])} · "
                 f"ถ้าผิดทาง อาจย่อถึง {fp(dn['dip50'])} (โอกาส 50%) — SL {fp(c['med']['sl'])}")
    else:
        story = (f"{head}❌ ยังไม่แนะนำให้ซื้อตอนนี้ · กำลังอยู่ในช่วง {phase.split(' ', 1)[1]} · "
                 f"มีโอกาส 50% ที่ราคาจะย่อถึง {fp(dn['dip50'])} ({dist(c, dn['dip50'])}) ภายใน {dn['H']} วัน"
                 + (f" · ถ้าแย่กว่าปกติ {fp(dn['dip30'])}" if dn["dip30"] else "")
                 + f" · โซนที่ควรพิจารณาซื้อ {fp(z[0])}–{fp(z[1])}"
                 + (f" · ยังไม่ผ่าน: {' · '.join(fails[:2])}" if fails else ""))
    when = None
    if tm and not v["recommend"]:
        d1, d2 = ts + pd.Timedelta(days=round(tm["p25"])), ts + pd.Timedelta(days=round(tm["p75"]))
        when = (f"ในอดีตที่สภาพคล้ายกัน ถ้าราคาย่อ จุดต่ำมักเกิดภายใน ~{tm['p50']:.0f} วัน "
                f"(ช่วง {d1:%d/%m} – {d2:%d/%m}) · กลับมาดูอีกครั้งราว {ts + pd.Timedelta(days=round(tm['p50'])):%d/%m/%Y} "
                f"หรือทันทีที่ราคาเข้าโซนซื้อ (n={tm['n']})")
    return {"ticker": c["ticker"], "verdict": v, "phase": phase, "phase_why": phase_why, "evidence": ev,
            "downside": dn, "timing": tm, "when": when, "story": story, "flip": dn["flip"]}


def final_verdict(c: dict, deep: dict | None = None) -> dict:
    """คำตอบเดียวสำหรับหน้าแรก: ซื้อ / รอ / เลี่ยง + ราคา + เหตุผล 3 ข้อ (ทุกตัวเลขมาจากระบบ)"""
    m = c["med"]
    fails = c.get("gate_fails") or []
    score = c.get("buy_final", c.get("buy_sys", 50))
    if not fails and score >= 55:
        act, color = "✅ ซื้อได้", "green"
        line = f"ตั้ง Limit {fp(m['zone'][0])} – {fp(m['zone'][1])} · SL {fp(m['sl'])} · TP1 {fp(m['tp1'])} · TP2 {fp(m['tp2'])}"
    elif m["status"] == "🔴" or score < 40:
        act, color = "⛔ ยังไม่แนะนำให้ซื้อ", "red"
        line = f"แนวโน้มยังไม่เอื้อ · ถ้าจะสะสมระยะยาว DCA เริ่ม {fp(c['long']['tiers'][0])} ตัดขาดทุน Weekly < {fp(c['long']['stop'])}"
    else:
        act, color = "⏳ ยังไม่แนะนำให้ซื้อตอนนี้ — รอจังหวะ", "orange"
        line = f"ตั้งรอซื้อ {fp(m['zone'][0])} – {fp(m['zone'][1])} เมื่อผ่านเงื่อนไข: " + (" · ".join(fails[:2]) or "4H กลับตัว")
    pos = [x for x in c["tech"]["votes"] if x["vote"] > 0]
    neg = [x for x in c["tech"]["votes"] if x["vote"] < 0]
    reasons = []
    if pos:
        reasons.append("🟢 " + " · ".join(f"{x['name']} ({x['detail']})" for x in pos[:3]))
    if neg:
        reasons.append("🔴 " + " · ".join(f"{x['name']} ({x['detail']})" for x in neg[:3]))
    sig = next((x for x in (deep or {}).get("signals", []) if x["bias"] != 0), None)
    if sig:
        reasons.append(f"🔮 {sig['signal']} · {sig['when']}" + (f" · ระดับ {sig['trigger']}" if sig["trigger"] != "-" else ""))
    ck = c.get("score_check")
    acc = None
    if ck:
        acc = (f"{ck['verdict']} · ทดสอบ {ck['n_test']} วันล่าสุด ทายทิศถูก "
               + (f"{ck['hit'] * 100:.0f}%" if ck["hit"] is not None else "-")
               + f" · วันที่คะแนนใกล้วันนี้ ขึ้นจริง {ck['p_today'] * 100:.0f}% (n≈{ck['n_today']})")
    return {"action": act, "color": color, "line": line, "reasons": reasons, "score": score,
            "p_up": c.get("p_up_h"), "accuracy": acc, "recommend": act.startswith("✅")}


def decision_table(coins: list[dict], views: dict, horizon: int) -> pd.DataFrame:
    """ตารางตัดสินใจ: คะแนนควรซื้อ (AI/ระบบ) · โอกาสขึ้น · เทคนิคไหนบวก/ลบ · ข่าวข้างหน้า · มุมมอง +/−"""
    H = horizon_bucket(horizon)
    rows = []
    for c in sorted(coins, key=lambda c: -c.get("buy_final", 0)):
        t, v = c["tech"], views.get(c["ticker"]) or {}
        pos = [x["name"] for x in t["votes"] if x["vote"] > 0]
        neg = [x["name"] for x in t["votes"] if x["vote"] < 0]
        ft = c["med"]["ft1"] or {}
        rows.append({
            "เหรียญ": ("⭐" if c.get("is_pick") else "") + c["ticker"],
            "มุมมอง": outlook_label(c),
            "คะแนนควรซื้อ (AI)": f"{c['buy_final']:.0f}/100" if c.get("buy_ai") is not None else "-",
            "คะแนนระบบ": f"{c['buy_sys']:.0f}/100",
            f"โอกาสขึ้นใน {H} วัน (สถิติ)": fprob(c.get("p_up_h")),
            "TP1 ก่อน SL": fprob(ft.get("p_tp")),
            "เทคนิค บวก/ลบ/กลาง": f"{t['bull']} / {t['bear']} / {t['neutral']}",
            "✅ เทคนิคที่บวก": " · ".join(pos) or "-", "❌ เทคนิคที่ลบ": " · ".join(neg) or "-",
            "อาจร่วงถึง (50%)": fp((c["fc"].get(H) or {}).get("dip50")),
            "โซนน่าซื้อ": (lambda z: f"{fp(z[0])}–{fp(z[1])}")(downside_plan(c, horizon)["buy_zone"]),
            "ข่าวข้างหน้า": f"{c.get('event_net', 0):+.2f}",
            "ทดสอบคะแนนย้อนหลัง": (c.get("score_check") or {}).get("verdict", "—").split(" ")[0],
            "AI สรุปจากเทคนิค": " · ".join(v.get("key_techniques") or []) or "-"})
    return pd.DataFrame(rows)


def roadmap_df(roadmap: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "วันที่": r["dt"].strftime("%Y-%m-%d") + (" (ประมาณ)" if r["approx"] else ""), "อีก (วัน)": r["days"],
        "เหรียญ": r["scope"], "ประเภท": r["type"], "เหตุการณ์": r.get("title") or "-",
        "ผลที่คาด": DIR_EMO.get(r.get("direction"), "⚪ ±"), "ผลกระทบ": r.get("impact"), "ความแน่นอน": r.get("certainty"),
        "เหตุผล": r.get("why") or "-", "ที่มา": r.get("source_url") if r.get("has_src") else "⚠️ ไม่มีลิงก์"}
        for r in roadmap])


def _fill_days(c: dict) -> int:
    if not c.get("gate_fails") or c["med"]["status"] == "🟢":
        return max(1, SIGNAL_EXPIRY_H // 24)
    return 14 if c["med"]["status"] == "🟡" else 0


def trade_plan_table(coins: list[dict], ts: datetime) -> pd.DataFrame:
    """แผนเข้า-ออกที่ชัดเจน: ทำอะไร ที่ราคาไหน จนถึงวันไหน"""
    rows = []
    for c in coins:
        m = c["med"]
        fails = c.get("gate_fails") or []
        if not fails:
            act, cond = "✅ เข้าได้", "ผ่านเกณฑ์ครบ ตั้ง Limit ได้เลย"
        elif m["status"] == "🔴":
            act, cond = "⛔ ห้ามเข้า", " · ".join(m["reasons"][:2]) or "แนวโน้มขาลง"
        else:
            act, cond = "⏳ ตั้งรอ", "เข้าเมื่อ: ราคาลงมาในโซน + " + (" · ".join(fails[:2]) if fails else "")
        fd = _fill_days(c)
        rows.append({
            "เหรียญ": ("⭐" if c.get("is_pick") else "") + c["ticker"], "ราคา": fp(c["price"]), "ทำอะไร": act,
            "เข้าไม้ 1 (50%)": f"Limit {fp(m['zone'][1])}" if fd else "-",
            "เข้าไม้ 2 (50%)": f"Limit {fp(m['zone'][0])}" if fd else "-",
            "เงื่อนไข": cond, "ตั้งรอถึง": (ts + pd.Timedelta(days=fd)).strftime("%d/%m/%Y") if fd else "-",
            "🛑 Stop Loss": f"{fp(m['sl'])} ({fpct(m['sl'] / m['entry'] - 1)})",
            "✅ TP1 ขาย 50%": f"{fp(m['tp1'])} ({fpct(m['tp1'] / m['entry'] - 1)})",
            "🚀 TP2 ขายที่เหลือ": f"{fp(m['tp2'])} ({fpct(m['tp2'] / m['entry'] - 1)})",
            "หลัง TP1": f"เลื่อน SL → {fp(m['entry'])} (ทุน)",
            "⏳ ไม่ถึง TP1 ขายออกวันที่": (ts + pd.Timedelta(days=fd + m["tstop"])).strftime("%d/%m/%Y") if fd else "-",
            "R:R": f"1:{m['rr1']:.1f}"})
    return pd.DataFrame(rows)


def analysis_export_df(res: dict, horizon: int) -> pd.DataFrame:
    """ไฟล์บันทึกผลวิเคราะห์ (1 แถว/เหรียญ) สำหรับเก็บไว้ทำ Backtest/Forward test ภายหลัง"""
    ts = res["ts"]
    H = horizon_bucket(horizon)
    views = (res.get("port") or {}).get("views") or {}
    dp = res.get("deep") or {}
    dd = dp.get("data") or {}
    best = ((dd.get("lab") or {}).get("best") or {})
    rows = []
    for c in res["coins"]:
        m, l = c["med"], c["long"]
        is_focus = c["ticker"] == dp.get("focus")
        n = c.get("news") or {}
        rows.append({
            "run_ts_utc": pd.Timestamp(ts).tz_convert("UTC").isoformat(), "run_date_th": ts.strftime("%Y-%m-%d"),
            "app_version": __version__, "ticker": c["ticker"], "source": "scanner" if c.get("is_pick") else "watchlist",
            "price": c["price"], "macro": res["macro"]["regime"], "regime": c["cf"]["regime"],
            "med_status": m["status"], "passed_gate": not c.get("gate_fails"), "gate_fails": " | ".join(c.get("gate_fails") or []),
            "entry_lo": m["zone"][0], "entry_hi": m["zone"][1], "entry_mid": m["entry"], "sl": m["sl"],
            "tp1": m["tp1"], "tp2": m["tp2"], "rr1": m["rr1"], "fill_days": _fill_days(c), "time_stop_days": m["tstop"],
            "horizon_days": H, "p_up_h": c.get("p_up_h"), "p_tp_first": (m["ft1"] or {}).get("p_tp"),
            "be_win": m["be_win"], "win_sys": m["win_sys"],
            "buy_sys": round(c.get("buy_sys", 0), 1), "buy_ai": c.get("buy_ai"), "buy_final": round(c.get("buy_final", 0), 1),
            "outlook": outlook_label(c), "ai_outlook": (views.get(c["ticker"]) or {}).get("outlook"),
            "tech_score": round(c["tech"]["score"], 1), "tech_bull": c["tech"]["bull"], "tech_bear": c["tech"]["bear"],
            "tech_pos": " | ".join(x["name"] for x in c["tech"]["votes"] if x["vote"] > 0),
            "tech_neg": " | ".join(x["name"] for x in c["tech"]["votes"] if x["vote"] < 0),
            "xs_pct": c.get("xs_pct"), "trend": (c.get("mom") or {}).get("trend"),
            "news_score": n.get("score"), "event_net": c.get("event_net"),
            "bt_n": c["bt"]["n"], "bt_expectancy": c["bt"]["expectancy"], "bt_oos_expectancy": c["bt"]["oos_expectancy"],
            "long_status": l["status"], "dca1": l["tiers"][0], "dca2": l["tiers"][1], "dca3": l["tiers"][2],
            "long_stop": l["stop"], "long_tp1": l["tp1"], "is_focus": is_focus,
            "score_test": (c.get("score_check") or {}).get("verdict"),
            "dip50_h": (c["fc"].get(H) or {}).get("dip50"), "dip30_h": (c["fc"].get(H) or {}).get("dip30"),
            "buy_zone_lo": downside_plan(c, horizon)["buy_zone"][0], "buy_zone_hi": downside_plan(c, horizon)["buy_zone"][1],
            "score_test_skill": (c.get("score_check") or {}).get("skill"),
            "score_test_p_today": (c.get("score_check") or {}).get("p_today"),
            "mtf_alignment": dd.get("alignment") if is_focus else None,
            "best_strategy": best.get("name") if is_focus else None,
            "best_strategy_in_pos": best.get("now") if is_focus and best else None,
            "peer_rank": f"{dd['peers']['rank']}/{dd['peers']['n']}" if is_focus and dd.get("peers") else None,
            **({f"hz_{h['H']}d_score": round(h["score"], 1) for h in dd.get("horizons", [])} if is_focus else {}),
            **({f"hz_{h['H']}d_p_up": h["p_up"] for h in dd.get("horizons", [])} if is_focus else {}),
            **({f"hz_{h['H']}d_ai": (dp.get("horizon_ai") or {}).get(h["label"], {}).get("score")
                for h in dd.get("horizons", [])} if is_focus else {})})
    return pd.DataFrame(rows)


def evaluate_exports(df: pd.DataFrame, md: MarketData, now: pd.Timestamp | None = None) -> dict:
    """ตรวจไฟล์ผลวิเคราะห์เก่ากับราคาจริงหลังจากนั้น (Forward test · ใช้แท่ง 1D ที่เปิดหลังเวลาวิเคราะห์)

    แผนระยะกลาง: ได้ของเมื่อ Low ≤ entry_hi ภายใน fill_days → จากนั้น SL ก่อน TP1 (แท่งเดียวกันนับ SL) / time stop
    โอกาสขึ้น: ราคาปิดหลัง horizon_days เทียบราคาตอนวิเคราะห์ → เทียบกับ p_up_h ที่ประเมินไว้ (calibration)
    """
    now = now or pd.Timestamp.now(tz="UTC")
    df = df.copy()
    syms = [f"{t}{QUOTE}" for t in df["ticker"].unique()]
    frames = fetch_frames(md, syms, ("1d",))
    out = []
    for _, p in df.iterrows():
        d = (frames.get(f"{p['ticker']}{QUOTE}") or {}).get("1d")
        rec = p.to_dict()
        rec.update(outcome="ไม่มีข้อมูล", R=None, ret_h=None, up_h=None, filled=False)
        if d is None:
            out.append(rec)
            continue
        d = closed_only(d, now)
        x = d[d.index > pd.Timestamp(p["run_ts_utc"])]
        H = int(p.get("horizon_days") or 30)
        if len(x) >= H:
            rec["ret_h"] = float(x["close"].iloc[H - 1] / p["price"] - 1)
            rec["up_h"] = rec["ret_h"] > 0
        fd, ts_ = int(p.get("fill_days") or 0), int(p.get("time_stop_days") or 30)
        if fd <= 0 or pd.isna(p.get("entry_hi")):
            rec["outcome"] = "ไม่ได้แนะนำให้เข้า"
            out.append(rec)
            continue
        res_, filled_at, entry = "⏳ รอได้ของ", None, None
        for k in range(len(x)):
            bar = x.iloc[k]
            if filled_at is None:
                if bar.low <= p["entry_hi"]:
                    filled_at, entry = k, min(p["entry_hi"], bar.open)
                elif k + 1 >= fd:
                    res_ = "หมดเวลา (ไม่ได้ของ)"
                    break
                else:
                    continue
            if bar.low <= p["sl"]:
                res_, rec["R"] = "❌ LOSS", -1.0
                break
            if bar.high >= p["tp1"]:
                res_, rec["R"] = "✅ WIN", (p["tp1"] - entry) / (entry - p["sl"])
                break
            if k - filled_at + 1 >= ts_:
                rec["R"] = (bar.close - entry) / (entry - p["sl"])
                res_ = "✅ WIN (time stop)" if rec["R"] > 0 else "❌ LOSS (time stop)"
                break
        else:
            if filled_at is not None:
                res_ = "⏳ ถืออยู่"
        rec["filled"] = filled_at is not None
        rec["outcome"] = res_
        out.append(rec)
    ev = pd.DataFrame(out)
    done = ev[ev["outcome"].str.contains("WIN|LOSS", na=False)]
    summ = {"rows": len(ev), "closed": len(done),
            "win_rate": float(done["outcome"].str.contains("WIN").mean()) if len(done) else None,
            "avg_R": float(done["R"].mean()) if len(done) else None,
            "est_tp_first": float(done["p_tp_first"].mean()) if len(done) and done["p_tp_first"].notna().any() else None}
    hz = ev[ev["up_h"].notna()]
    summ["n_h"] = len(hz)
    summ["up_rate"] = float(hz["up_h"].astype(bool).mean()) if len(hz) else None
    summ["est_p_up"] = float(hz["p_up_h"].mean()) if len(hz) and hz["p_up_h"].notna().any() else None
    buckets = []
    if len(hz):
        b = pd.cut(hz["buy_final"].astype(float), [-1, 40, 55, 70, 101], labels=["<40", "40-55", "55-70", "≥70"])
        for lab, g in hz.groupby(b, observed=True):
            buckets.append({"คะแนนควรซื้อ": str(lab), "จำนวน": len(g), "ขึ้นจริง": fprob(g["up_h"].astype(bool).mean()),
                            "ผลตอบแทนเฉลี่ย": fpct(g["ret_h"].mean()), "โอกาสที่ประเมินไว้": fprob(g["p_up_h"].mean())})
    summ["buckets"] = pd.DataFrame(buckets)
    return {"table": ev, "summary": summ}


def table_patterns(coins: list[dict]) -> pd.DataFrame:
    rows = [{"Ticker": c["ticker"], "รูปแบบ": p["name"], "ทิศทาง": BIAS_TH[p["bias"]], "สถานะ": p["state"],
             "ระดับสำคัญ": fp(p["level"]), "เป้าตามรูปแบบ": f"{fp(p['target'])} ({dist(c, p['target'])})",
             "โอกาสแตะเป้าใน 60 วัน (สถิติ)": fprob(p.get("p_target_60")), "รายละเอียด": p["note"]}
            for c in coins for p in c.get("patterns", [])]
    return pd.DataFrame(rows)


# ───────────────────────── News ─────────────────────────
def apply_news(coins: list[dict], report: dict, weight: float) -> None:
    """Sentiment (score × confidence) ปรับคะแนน ±weight · ข่าวลบแรง → ห้ามระยะกลาง"""
    smap = na.news_score_map(report)
    for c in coins:
        c["news"] = report["coins"].get(c["ticker"])
        adj = float(np.clip(smap.get(c["ticker"], 0.0) * weight, -weight, weight))
        c["score_base"], c["news_adj"] = c["score"], adj
        c["score"] = float(np.clip(c["score"] + adj, 0, 100))
        c["news_block"] = na.is_strong_negative(report, c["ticker"])


def btc_context(md: MarketData, cg_key: str, frames: dict | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """คืน (btc_d แท่งปิดแล้ว, macro, glob)"""
    b = (frames or {}).get(f"BTC{QUOTE}") or fetch_frames(md, [f"BTC{QUOTE}"], ("1d", "1w"))[f"BTC{QUOTE}"]
    btc_d = closed_only(add_indicators(b["1d"]))
    btc_w = closed_only(add_indicators(b["1w"]))
    glob = cg_global(cg_key)
    macro = macro_agent(btc_d, btc_w, glob, fear_greed())
    macro["btc_price"] = float(b["1d"]["close"].iloc[-1])
    return btc_d, macro, glob


# ───────────────────────── Deep dive (1 เหรียญ) ─────────────────────────
def run_deep(md: MarketData, cfg: RunConfig, c: dict, raw: dict, research: dict, roadmap: list[dict]) -> dict:
    """เจาะลึกเหรียญเดียว: ดึง 1H เพิ่ม + 1D ของคู่แข่ง → deep_analysis → (AI) บทวิเคราะห์ + สถานการณ์"""
    t = c["ticker"]
    raw = dict(raw)
    peers = peers_for(t)
    extra = fetch_frames(md, [c["symbol"]] + [f"{p}{QUOTE}" for p in peers], ("1d",))
    h1 = fetch_frames(md, [c["symbol"]], ("1h",)).get(c["symbol"], {}).get("1h")
    if h1 is not None:
        raw["1h"] = h1
    peer_d1 = {}
    for p in peers:
        d = (extra.get(f"{p}{QUOTE}") or {}).get("1d")
        if d is not None and len(d) >= 120:
            peer_d1[p] = closed_only(add_indicators(d))
    data = deep_analysis(c, raw, peer_d1)
    report, errs, project, proj_rows = None, [], None, []
    if cfg.client and cfg.ai_agents:
        heads = na.headlines(cfg.news_items, t, 8) if cfg.news_items else []
        project = project_deep_agent(cfg.client, cfg.model_research, c, cfg.research_cache, heads)
        pdat = (project or {}).get("data") or {}
        cert = {"DONE": "CONFIRMED", "IN_PROGRESS": "EXPECTED", "PLANNED": "EXPECTED", "RUMOR": "RUMOR"}
        ev = [{"date": x["period"], "type": x["kind"], "title": x["title"], "impact": x["impact"],
               "certainty": cert.get(x["status"], "EXPECTED"), "source_url": x["source_url"], "direction": x["direction"],
               "why": x["why"]} for x in pdat.get("roadmap", [])]
        proj_rows = build_roadmap({t: {"verified": (project or {}).get("verified"), "data": {"events": ev}}}, None, [t])
        g = {"HIGH": 0.2, "MEDIUM": 0.0, "LOW": -0.2}.get(pdat.get("growth_view"), 0.0)
        news_long = float(np.clip(event_net(proj_rows, t, 365) + g, -1, 1))
        data["horizons"] = horizon_outlook(c, data["snaps"], c.get("event_net", 0.0), news_long, data.get("checks"))
        rows = [{k: r.get(k) for k in ("date", "scope", "title", "direction", "impact", "certainty", "why")}
                for r in roadmap + proj_rows if t in r["coins"] or not r["coins"]]
        report, errs = focus_ai(cfg.client, cfg.model, coin_brief(c, research.get(t)), deep_brief(data), rows, pdat)
        errs = (project or {}).get("errors", []) + errs
    ai = report.model_dump() if report else None
    hviews = enforce_horizon_views(ai.get("horizon_views", []), data["horizons"]) if ai else {}
    return {"focus": t, "data": data, "ai": ai, "errors": errs, "project": project, "project_rows": proj_rows,
            "horizon_ai": hviews}


# ───────────────────────── Full pipeline ─────────────────────────
def run_pipeline(md: MarketData, cfg: RunConfig, progress: Progress = _noop,
                 bar: Callable[[float, str], None] | None = None) -> dict:
    t0 = time.perf_counter()
    progress("📡 Market Data Agent: ตรวจรายชื่อคู่เทรด…")
    valid = set(md.tickers().index)
    req = list(dict.fromkeys(cfg.watch))
    watch_dropped = req[cfg.max_watch:]      # ใส่เกินจำนวน → ใช้ตัวแรก ที่เหลือให้ Scanner ตัดสิน
    req = req[:cfg.max_watch]
    watch = [w for w in req if f"{w}{QUOTE}" in valid]
    missing = [w for w in req if f"{w}{QUOTE}" not in valid]
    scan = {"picks": [], "tags": {}, "info": {}, "xs_bt": None, "universe": 0, "dropped_spread": 0, "ref_risk_adj": []}
    if cfg.n_extra:
        progress("🔎 AI Market Scanner: จัดอันดับ Momentum เทียบตลาด + Trend + สัญญาณเทคนิค…")
        scan = scan_universe(md, set(watch) | {"BTC"}, cfg.n_extra, cfg.min_qvol, progress=bar)
    extra, scan_tags = scan["picks"], scan["tags"]
    syms = list(dict.fromkeys([f"BTC{QUOTE}", *[f"{w}{QUOTE}" for w in watch], *extra]))
    progress("📡 Market Data Agent: ดึง 4H/1D/1W แบบขนาน…")
    frames = fetch_frames(md, syms, ("4h", "1d", "1w"), bar)
    if "1d" not in frames.get(f"BTC{QUOTE}", {}) or "1w" not in frames.get(f"BTC{QUOTE}", {}):
        raise RuntimeError("ดึงข้อมูล BTC ไม่ได้ — ระงับการวิเคราะห์ (ข้อมูลไม่เพียงพอ)")

    progress("🏛 Fundamental + Macro Agent…")
    tickers = tuple(s[: -len(QUOTE)] for s in syms)
    ext = external_bundle(tickers, cfg.cg_key)
    cg, glob, llama, rss, det = ext["cg"], ext["glob"], ext["llama"], ext["rss"], ext["det"]
    fng = fear_greed()
    btc_d = closed_only(add_indicators(frames[f"BTC{QUOTE}"]["1d"]))
    btc_w = closed_only(add_indicators(frames[f"BTC{QUOTE}"]["1w"]))
    macro = macro_agent(btc_d, btc_w, glob, fng)
    macro["btc_price"] = float(frames[f"BTC{QUOTE}"]["1d"]["close"].iloc[-1])

    progress("📐 Structure + Quant + Forecast Agent…")
    coins, skipped = [], []
    for s in syms:
        t = s[: -len(QUOTE)]
        fund = fundamentals_for(t, cg.get(t), det.get((cg.get(t) or {}).get("id")), llama)
        try:
            c = analyze_coin(s, frames.get(s, {}), btc_d, macro, fund)
        except Exception as e:  # noqa: BLE001 — เหรียญเดียวพังไม่ล้มทั้งระบบ
            log.exception("analyze %s", s)
            skipped.append(f"{t}: {e}")
            continue
        if c:
            c["is_pick"] = s in extra
            c["scan_tags"] = scan_tags.get(s, [])
            info = scan["info"].get(s) or {}
            c["xs_pct"] = info.get("xs_pct") if info else market_pct(c["mom"]["risk_adj"], scan["ref_risk_adj"])
            coins.append(c)
        else:
            skipped.append(f"{t}: ข้อมูลไม่พอ")

    news = {"items": [], "src": {}, "report": {"overall": "", "themes": [], "coins": {}}, "mode": "off", "focus": []}
    if cfg.use_news and coins:
        progress("📰 News Agent: ดึงข่าว 6 เว็บ + Google News รายเหรียญ…")
        try:
            hold = [h for h in cfg.holdings if h]
            names = {c["ticker"]: c["fund"].get("name") for c in coins if c["fund"].get("name")}
            sym_all = list(dict.fromkeys([c["ticker"] for c in coins] + hold))
            focus = list(dict.fromkeys([c["ticker"] for c in coins if c["is_pick"]] + watch + hold))
            items, src = na.collect_news(sym_all, names, focus, days=7)
            progress("📰 News Agent: วิเคราะห์ Sentiment…")
            report, nmode = na.analyze_news(items, sym_all, llm=llm_fn(cfg.client, cfg.model_research) if cfg.news_ai else None,
                                             focus=focus)
            news = {"items": items, "src": src, "report": report, "mode": nmode, "focus": focus}
            apply_news(coins, report, cfg.news_w)
            cfg.news_items = items
        except Exception as e:  # noqa: BLE001
            news["error"] = str(e)[:200]
    d1s = {c["symbol"]: c["d1"] for c in coins}
    corr = corr_matrix(d1s)
    base = [c for c in coins if not c["is_pick"] and (c["ticker"] != "BTC" or "BTC" in cfg.watch)]
    analyzed = base + sorted([c for c in coins if c["is_pick"]], key=lambda c: c["score"], reverse=True)

    H = horizon_bucket(cfg.horizon)
    for c in analyzed:  # เทคนิครายตัว (หลังได้ Momentum เทียบตลาด) + โอกาสขึ้นตามกรอบเวลาที่เลือก
        c["tech"] = technical_votes(c)
        c["p_up_h"] = (c["fc"].get(H) or {}).get("p_up")
        ck = score_check(c["d1"], min((7, 30, 90), key=lambda x: abs(x - H)))  # ทดสอบคะแนนย้อนหลังของเหรียญนี้
        c["score_check"] = ck
        if ck is not None and ck["ok"] and c["p_up_h"] is not None:
            c["p_up_h"] = 0.5 * c["p_up_h"] + 0.5 * ck["p_today"]  # ปรับเทียบด้วยผลจริงในอดีต

    signals, watch_only = [], []
    for c in analyzed:
        sig, fails = build_signal(c, macro, c.get("news"), bool(c.get("news_block")))
        c["gate_fails"] = fails
        if sig:
            signals.append(sig)
        elif c["med"]["status"] != "🔴":
            watch_only.append(c["ticker"])

    research = {"by_ticker": {}, "log": []}
    macro_cal = None
    if cfg.client and cfg.ai_agents:
        focus_c = [c for c in analyzed if watch and c["ticker"] == watch[0]]  # เหรียญที่เจาะลึกได้ Research เสมอ
        short = focus_c + [c for c in sorted(analyzed, key=lambda c: c["score"], reverse=True) if c not in focus_c]
        short = short[: max(cfg.n_research, len(focus_c))]
        research = research_agent(cfg.client, cfg.model_research, short, cfg.research_cache, progress,
                                  news["items"], na.headlines,
                                  lambda c: [f"{x['date']} {x['src']}: {x['title']}"
                                             for x in rss_for(c["ticker"], c["fund"].get("name"), rss)])
        progress("🗓 Event Calendar Agent: ค้นข่าว/อีเวนต์ 90 วันข้างหน้า…")
        try:
            macro_cal = macro_calendar_agent(cfg.client, cfg.model_research, [c["ticker"] for c in analyzed],
                                             cfg.research_cache)
        except Exception as e:  # noqa: BLE001 — ปฏิทินล่มไม่ทำให้ทั้งระบบล่ม
            macro_cal = {"events": [], "errors": [str(e)[:200]]}
    roadmap = build_roadmap(research["by_ticker"], macro_cal, [c["ticker"] for c in analyzed])
    for c in analyzed:
        c["event_net"] = event_net(roadmap, c["ticker"], max(H, 30))
        c["buy_sys"] = buy_score_sys(c)
        c["read"] = market_read(c, cfg.horizon)  # ให้ AI ใช้หลักฐานเดียวกัน

    if cfg.client and cfg.ai_agents:
        port = strategist_and_risk(cfg.client, cfg.model, analyzed, research["by_ticker"], macro, corr,
                                   cfg.n_med, cfg.n_long, progress, cfg.horizon, scan.get("xs_bt"))
    else:
        picks = fallback_strategy(analyzed, cfg.n_med, cfg.n_long)
        port = {"picks": picks, "cash": 100 - sum(p["alloc_final"] for p in picks), "strategist": None, "views": {},
                "risk": None, "errors": [], "mode": "rule-based (ไม่มี API key)"}
    views = port.get("views") or {}
    for c in analyzed:  # คะแนน AI ถูกจำกัดให้อยู่ภายใน ±15 ของคะแนนระบบ
        v = views.get(c["ticker"]) or {}
        c["buy_ai"] = v.get("buy_score") if v else None
        c["buy_final"] = enforce_buy_score(c["buy_ai"], c["buy_sys"])
    for i, c in enumerate(sorted(analyzed, key=lambda c: -c["buy_final"])):  # อ่านสถานการณ์อีกครั้งด้วยคะแนนสุดท้าย
        c["read"] = market_read(c, cfg.horizon, rank=i + 1)
        c["read"]["ai_wait"] = (views.get(c["ticker"]) or {}).get("wait_plan", "")

    deep = None
    fc_ = next((c for c in analyzed if watch and c["ticker"] == watch[0]), None)
    if cfg.deep and fc_ is not None:
        progress(f"🔬 Deep Dive {fc_['ticker']}: 1H-1W · Backtest 6 กลยุทธ์ · เทียบคู่แข่ง…")
        try:
            deep = run_deep(md, cfg, fc_, frames.get(fc_["symbol"], {}), research["by_ticker"], roadmap)
        except Exception as e:  # noqa: BLE001 — Deep dive ล่มไม่ทำให้ผลหลักหาย
            log.exception("deep dive")
            deep = {"focus": fc_["ticker"], "data": None, "ai": None, "errors": [str(e)[:200]]}

    return {"coins": analyzed, "macro": macro, "glob": glob, "fng": fng, "corr": corr, "research": research,
            "port": port, "missing": missing, "skipped": skipped, "rss": rss, "news": news,
            "signals": signals, "watch_only": watch_only, "watch_dropped": watch_dropped,
            "scan": {k: scan[k] for k in ("xs_bt", "universe", "dropped_spread")},
            "roadmap": roadmap, "macro_cal": macro_cal, "horizon": cfg.horizon, "deep": deep,
            "elapsed": time.perf_counter() - t0, "btc_d": btc_d, "ts": datetime.now(TZ_TH), "host": md.host}


def run_portfolio(md: MarketData, holds_in: list[dict], res: dict | None, client, model: str, cg_key: str,
                  use_news: bool) -> dict:
    if res:
        btc_d, macro = res["btc_d"], res["macro"]
    else:
        btc_d, macro, _ = btc_context(md, cg_key)
    coins = {c["ticker"]: c for c in (res or {}).get("coins", [])}
    tick = [h["ticker"] for h in holds_in]
    need = [t for t in tick if t not in coins]
    errors: list[str] = []
    if need:
        fr = fetch_frames(md, [f"{t}{QUOTE}" for t in need], ("4h", "1d", "1w"))
        ext = external_bundle(tuple(need), cg_key)
        for t in need:
            cg = ext["cg"].get(t)
            fund = fundamentals_for(t, cg, ext["det"].get((cg or {}).get("id")), ext["llama"])
            try:
                cc = analyze_coin(f"{t}{QUOTE}", fr.get(f"{t}{QUOTE}", {}), btc_d, macro, fund)
            except Exception as e:  # noqa: BLE001
                cc = None
                errors.append(f"{t}: {e}")
            if cc:
                coins[t] = cc
            else:
                errors.append(f"{t}: ไม่พบคู่ {QUOTE} บน Binance หรือข้อมูลไม่พอ")
    news = (res or {}).get("news") or {}
    report = news.get("report") or {"coins": {}}
    miss_news = [t for t in tick if t in coins and t not in report.get("coins", {})]
    if use_news and miss_news:
        try:
            names = {t: coins[t]["fund"].get("name") for t in miss_news if coins[t]["fund"].get("name")}
            items, _ = na.collect_news(miss_news, names, miss_news, days=7)
            rep2, _ = na.analyze_news(items, miss_news, llm=llm_fn(client, model), focus=miss_news)
            report = {**report, "coins": {**report.get("coins", {}), **rep2["coins"]}}
            news = {**news, "items": (news.get("items") or []) + items, "report": report}
        except Exception as e:  # noqa: BLE001
            errors.append(f"ข่าว: {str(e)[:120]}")
    holds = []
    for h in holds_in:
        c = coins.get(h["ticker"])
        if c:
            holds.append(holding_plan(c, h["qty"], h["cost"], report.get("coins", {}).get(h["ticker"])))
            c.pop("_fc", None)
    corr = corr_matrix({coins[h["ticker"]]["symbol"]: coins[h["ticker"]]["d1"] for h in holds}) if len(holds) > 1 \
        else pd.DataFrame()
    review, ai_err = (portfolio_ai(client, model, holds, coins, macro, corr) if (client and holds) else (None, []))
    vmap = {v.ticker: v for v in (review.holdings if review else [])}
    for h in holds:
        v = vmap.get(h["ticker"])
        h["act_ai"] = v.action if v else None
        h["act"] = enforce_holding_ai(h, v)
        h["view"] = v.model_dump() if v else None
    return {"holds": holds, "review": review, "corr": corr, "errors": errors + ai_err, "news": news,
            "macro": macro, "ts": datetime.now(TZ_TH)}

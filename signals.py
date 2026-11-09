"""Signal Engine — จุดเข้า Spot ที่ผ่านเกณฑ์คุณภาพ

- สัญญาณ (ผ่านเกณฑ์ทุกข้อ) แยกจาก "ยังไม่ยืนยัน"
- confidence = คะแนนคุณภาพ 0-100 (ไม่ใช่ความน่าจะเป็นชนะ)
- win_prob   = โอกาส TP1 ก่อน SL จากสถิติย้อนหลังของเหรียญนั้น (แสดงคู่กับจำนวนตัวอย่างอิสระ n_eff)
- Spot LONG เท่านั้น · ไม่ส่งคำสั่งซื้อขาย
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from config import MIN_BT_TRADES, MIN_EDGE_PTS, MIN_N_EFF, MIN_RR, SIGNAL_EXPIRY_H


class Signal(BaseModel):
    id: str
    symbol: str
    ticker: str
    direction: Literal["LONG"] = "LONG"
    created_at: datetime
    expires_at: datetime
    price_at_signal: float
    entry_lo: float
    entry_hi: float
    sl: float
    tp1: float
    tp2: float
    rr1: float
    confidence: float = Field(ge=0, le=100, description="คะแนนคุณภาพ ไม่ใช่โอกาสชนะ")
    win_prob: float | None = Field(default=None, description="โอกาส TP1 ก่อน SL (สถิติย้อนหลัง) 0-100")
    be_win: float
    n_eff: int = 0
    time_stop_days: int
    reasons: list[str] = []
    risks: list[str] = []


def risk_factors(c: dict, macro: dict, news: dict | None) -> list[str]:
    r, m, bt = c["risk"], c["med"], c["bt"]
    out = []
    if macro["regime"] == "Bear":
        out.append("Macro Bear — ตลาดรวมยังเป็นขาลง")
    if r["vol_ann"] > 1.0:
        out.append(f"ผันผวนสูง {r['vol_ann'] * 100:.0f}%/ปี")
    if not np.isnan(r["beta"]) and r["beta"] > 1.5:
        out.append(f"Beta {r['beta']:.1f} — เคลื่อนแรงกว่า BTC")
    if bt["oos_n"] >= 3 and bt["oos_expectancy"] < 0:
        out.append("ผล Backtest ช่วงหลัง (out-of-sample) ติดลบ")
    if m["ft1"] and m["ft1"]["n_eff"] < 15:
        out.append(f"ตัวอย่างสถิติน้อย (n={m['ft1']['n_eff']})")
    if news and news.get("score", 0) < -0.15:
        out.append(f"ข่าวเชิงลบ ({news['score']:+.2f})")
    mo = c.get("mom") or {}
    r30 = (mo.get("ret") or {}).get(30)
    if r30 is not None and r30 > 0.5 and r["vol_ann"] > 0.9:  # Grobys et al. 2025: momentum คริปโต crash แรง
        out.append(f"ขึ้นมาแรง {r30 * 100:.0f}% ใน 30 วัน + ผันผวนสูง → เสี่ยงย่อแรง ควรลดขนาดไม้")
    if mo.get("trend") is not None and mo["trend"] < 0.4:
        out.append(f"Trend ensemble อ่อน ({mo['trend'] * 5:.0f}/5 lookback อยู่ในเทรนด์)")
    fd = (c["fund"] or {}).get("fdv_mcap")
    if fd and fd > 2:
        out.append(f"FDV/MCap {fd:.1f}x — มี supply รอปลดล็อกมาก")
    return out


def quality_gate(c: dict, macro: dict, news_block: bool = False, now: datetime | None = None) -> list[str]:
    """คืนรายการเหตุผลที่ 'ไม่ผ่าน' (ว่าง = ผ่าน)"""
    m, bt = c["med"], c["bt"]
    ft = m["ft1"] or {}
    fails = []
    if m["status"] != "🟢":
        fails.append(f"สถานะระยะกลาง {m['status']}")
    if c["cf"]["regime"] == "DOWN":
        fails.append("Regime ขาลง")
    if m["rr1"] < MIN_RR:
        fails.append(f"R:R {m['rr1']:.1f} < {MIN_RR}")
    if not ft or ft.get("n_eff", 0) < MIN_N_EFF:
        fails.append("สถิติไม่พอ")
    elif (m["rel_win"] or 0) * 100 < m["be_win"] + MIN_EDGE_PTS:
        fails.append("โอกาส TP ก่อน SL ไม่เกินจุดคุ้มทุนพอ")
    if bt["n"] < MIN_BT_TRADES:
        fails.append(f"Backtest {bt['n']} ไม้ (น้อยไป)")
    elif bt["expectancy"] <= 0:
        fails.append("Backtest expectancy ≤ 0")
    if news_block:
        fails.append("ข่าวลบรุนแรง")
    now = now or datetime.now(timezone.utc)
    last = c.get("last_closed")
    if last is not None and (pd.Timestamp(now) - pd.Timestamp(last)) > pd.Timedelta(days=3):
        fails.append("ข้อมูลไม่อัปเดต")
    return fails


def confidence_score(c: dict, macro: dict, news: dict | None) -> float:
    """คะแนนคุณภาพ 0-100 = ความสอดคล้องของหลักฐาน (ไม่ใช่ความน่าจะเป็น)"""
    m, bt = c["med"], c["bt"]
    edge = ((m["rel_win"] or 0) * 100 - m["be_win"])  # จุดที่เกินคุ้มทุน
    s = 35 * np.clip(edge / 20, 0, 1)                                # 35: edge เชิงสถิติ
    s += 15 * np.clip(bt["expectancy"] / 0.5, 0, 1)                    # 15: backtest
    s += 10 * (1.0 if bt["oos_n"] >= 3 and bt["oos_expectancy"] > 0 else 0.4 if bt["oos_n"] < 3 else 0)  # 10: OOS
    s += 10 * np.clip((m["rr1"] - 1.5) / 1.5, 0, 1)                    # 10: R:R
    s += 10 * {"Bull": 1.0, "Neutral": 0.5, "Bear": 0.0}[macro["regime"]]  # 10: macro
    s += 10 * (1.0 if c["struct_d"]["code"] == "UP" else 0.5 if c["struct_d"]["code"] in ("SQUEEZE", "EXPAND") else 0)
    s += 5 * (1.0 if c["long"]["w_up"] else 0)                          # 5: weekly
    if news:
        s += 5 * np.clip(news["score"] * news["confidence"] * 2, -1, 1)  # ±5: ข่าว
    n_eff = (m["ft1"] or {}).get("n_eff", 0)
    s *= float(np.clip(n_eff / 30, 0.6, 1.0))                           # ตัวอย่างน้อย → ลดคะแนน
    return float(np.clip(s, 0, 100))


def build_signal(c: dict, macro: dict, news: dict | None = None, news_block: bool = False,
                 now: datetime | None = None) -> tuple[Signal | None, list[str]]:
    """สร้าง Signal ถ้าผ่าน quality gate ไม่งั้นคืน (None, เหตุผลที่ไม่ผ่าน)"""
    fails = quality_gate(c, macro, news_block, now)
    if fails:
        return None, fails
    now = now or datetime.now(timezone.utc)
    m = c["med"]
    ft = m["ft1"] or {}
    reasons = list(m["reasons"]) + [
        f"โครงสร้าง 1D: {c['struct_d']['label']} {c['struct_d']['bos']}".strip(),
        f"Backtest {c['bt']['n']} ไม้ · expectancy {c['bt']['expectancy']:+.2f}R · PF "
        f"{c['bt']['profit_factor']:.2f}" if c["bt"]["profit_factor"] else f"Backtest {c['bt']['n']} ไม้",
        f"RSI {c['d1']['rsi'].iloc[-1]:.0f} · ADX {c['d1']['adx'].iloc[-1]:.0f} · แท่งล่าสุด {c['pattern']}",
    ]
    sig = Signal(
        id=f"{c['symbol']}-{now:%Y%m%d%H%M}", symbol=c["symbol"], ticker=c["ticker"], created_at=now,
        expires_at=now + timedelta(hours=SIGNAL_EXPIRY_H), price_at_signal=c["price"],
        entry_lo=m["zone"][0], entry_hi=m["zone"][1], sl=m["sl"], tp1=m["tp1"], tp2=m["tp2"], rr1=m["rr1"],
        confidence=round(confidence_score(c, macro, news), 1),
        win_prob=round(m["rel_win"] * 100, 1) if m["rel_win"] is not None else None,
        be_win=round(m["be_win"], 1), n_eff=int(ft.get("n_eff", 0)), time_stop_days=m["tstop"],
        reasons=reasons, risks=risk_factors(c, macro, news))
    return sig, []

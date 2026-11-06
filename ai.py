"""Module 4 — AI Analysis Engine (Gemini)"""
from __future__ import annotations

import json
import logging
import numpy as np
import pandas as pd
import re
import time
from datetime import datetime
from pydantic import BaseModel, ValidationError
from typing import Callable, Literal
from config import MAJORS, MODEL_FALLBACK, QUOTE, REG_TH, RESEARCH_TTL, TZ_TH
from analysis import cap_for, rel_win


# ════════════════════ ai ════════════════════
try:
    from google import genai
    from google.genai import types as gtypes
except ImportError:  # แอปยังทำงานได้แบบ rule-based
    genai = gtypes = None

log = logging.getLogger(__name__)
Progress = Callable[[str], None]


# ───────────────────────── Schemas ─────────────────────────
class Event(BaseModel):
    date: str
    type: Literal["UNLOCK", "UPGRADE", "MAINNET", "ETF", "LISTING", "PARTNERSHIP", "GOVERNANCE",
                  "CONFERENCE", "REGULATION", "BURN", "OTHER"]
    title: str
    impact: Literal["HIGH", "MEDIUM", "LOW"]
    certainty: Literal["CONFIRMED", "EXPECTED", "RUMOR"]
    source_url: str


class CoinResearch(BaseModel):
    ticker: str
    project: str
    strengths: list[str]
    tokenomics: str
    news_30d: list[str]
    bull_case: str
    bear_case: str
    catalyst_score: int
    events: list[Event]


class ResearchBatch(BaseModel):
    coins: list[CoinResearch]


class Pick(BaseModel):
    ticker: str
    horizon: Literal["MEDIUM", "LONG"]
    allocation_pct: float
    win_rate: float
    rationale: str
    key_risk: str


class CoinView(BaseModel):
    ticker: str
    verdict: str
    reasons: list[str]
    invalidation: str
    watch: str


class StrategistOut(BaseModel):
    market_view: str
    picks: list[Pick]
    cash_pct: float
    summary: str
    coin_views: list[CoinView]


class RiskAdj(BaseModel):
    ticker: str
    horizon: Literal["MEDIUM", "LONG"]
    allocation_pct: float
    win_rate: float
    note: str


class RiskOut(BaseModel):
    verdict: Literal["APPROVE", "CAUTION", "REJECT"]
    issues: list[str]
    adjustments: list[RiskAdj]
    final_note: str


class SignalNote(BaseModel):
    ticker: str
    note: str


class SignalNotes(BaseModel):
    notes: list[SignalNote]


# ───────────────────────── Gemini helpers ─────────────────────────
_SKIP_MODEL = ("404", "NOT_FOUND", "not found", "is not supported", "deprecated", "RESOURCE_EXHAUSTED", "quota")


def make_client(api_key: str):
    """คืน client หรือ None (ไม่มี key / ไม่มีไลบรารี)"""
    if not api_key or genai is None:
        return None
    return genai.Client(api_key=api_key)


def _retry(fn, tries: int = 3):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            transient = any(k in msg for k in ("503", "UNAVAILABLE", "overloaded")) or \
                ("429" in msg and "quota" not in msg.lower())
            if i < tries - 1 and transient:
                time.sleep(min(20, 4 * 2 ** i))
                continue
            raise


def _parse(resp, schema: type[BaseModel]):
    p = getattr(resp, "parsed", None)
    if isinstance(p, schema):
        return p
    if isinstance(p, dict):
        return schema.model_validate(p)
    txt = re.sub(r"```(?:json)?", "", resp.text or "")
    s, e = txt.find("{"), txt.rfind("}")
    if s < 0 or e <= s:
        raise ValueError("ไม่พบ JSON")
    return schema.model_validate(json.loads(txt[s: e + 1]))


def _sources(resp) -> list[dict]:
    out = []
    try:
        for ch in resp.candidates[0].grounding_metadata.grounding_chunks or []:
            if ch.web and ch.web.uri:
                out.append({"title": ch.web.title or ch.web.uri, "uri": ch.web.uri})
    except Exception:  # noqa: BLE001
        pass
    return out


def model_chain(primary: str) -> list[str]:
    return list(dict.fromkeys([primary, *MODEL_FALLBACK]))


def gemini_json(client, model: str, prompt: str, schema: type[BaseModel], search: bool):
    """Model fallback × Mode fallback (schema+search → search+parse → schema) · คืน (obj|None, sources, mode, errors)"""
    if client is None or gtypes is None:
        return None, [], "no-client", []
    prompt += "\n\nตอบเป็น JSON ตาม schema นี้เท่านั้น:\n" + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    tool = [gtypes.Tool(google_search=gtypes.GoogleSearch())]
    modes = []
    if search:
        modes += [("schema+search", dict(tools=tool, response_mime_type="application/json", response_schema=schema)),
                  ("search+parse", dict(tools=tool))]
    modes.append(("schema", dict(response_mime_type="application/json", response_schema=schema)))
    errs: list[str] = []
    for m in model_chain(model):
        for name, kw in modes:
            try:
                cfg = gtypes.GenerateContentConfig(temperature=0.2, **kw)
                resp = _retry(lambda: client.models.generate_content(model=m, contents=prompt, config=cfg))
                return _parse(resp, schema), _sources(resp), f"{m}:{name}", errs
            except (ValidationError, ValueError, json.JSONDecodeError) as e:
                errs.append(f"{m}:{name}: {str(e)[:110]}")
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                errs.append(f"{m}:{name}: {msg[:110]}")
                if "API key" in msg or "PERMISSION_DENIED" in msg:
                    return None, [], "failed", errs
                if any(k in msg for k in _SKIP_MODEL):
                    break
    return None, [], "failed", errs


def llm_fn(client, model: str):
    """callable(prompt, schema) สำหรับ news_agent"""
    if client is None:
        return None
    return lambda p, s: gemini_json(client, model, p, s, search=False)[0]


def _r(x, d=2):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


# ───────────────────────── Research Agent ─────────────────────────
def research_agent(client, model: str, coins: list[dict], cache: dict, progress: Progress | None = None,
                   news_items: list | None = None, headlines_fn=None, rss_fn=None) -> dict:
    """ค้นเว็บทีละ 4 เหรียญ · cache 6 ชม. ใน dict ที่ส่งเข้ามา"""
    now = time.time()
    todo = [c for c in coins if not ((k := cache.get((model, c["ticker"]))) and now - k["ts"] < RESEARCH_TTL)]
    today = datetime.now(TZ_TH).strftime("%Y-%m-%d")
    log_ = []
    for i in range(0, len(todo), 4):
        batch = todo[i: i + 4]
        if progress:
            progress(f"📰 Research Agent: {', '.join(c['ticker'] for c in batch)}…")
        ctx = []
        for c in batch:
            f = c["fund"]
            heads = (headlines_fn(news_items, c["ticker"], 8) if (headlines_fn and news_items) else
                     (rss_fn(c) if rss_fn else []))
            ctx.append({"ticker": c["ticker"], "name": f.get("name"), "rank": f.get("rank"),
                        "categories": f.get("categories"), "price": round(c["price"], 6), "headlines": heads})
        prompt = f"""วันนี้ {today}. คุณคือ Research Agent วิเคราะห์คริปโต ใช้ Google Search หาข้อมูลล่าสุดของเหรียญต่อไปนี้:
{json.dumps(ctx, ensure_ascii=False)}
สำหรับแต่ละเหรียญ ให้: project (โปรเจกต์ทำอะไร 1-2 ประโยค), strengths (จุดแข็ง 2-4 ข้อ), tokenomics (supply/inflation/unlock สำคัญ),
news_30d (ข่าวสำคัญ 30 วันล่าสุดพร้อมวันที่ 3-5 ข้อ), bull_case, bear_case, catalyst_score (-100 ถึง 100 = แรงส่งเชิงข่าว/อีเวนต์ 1-6 เดือน),
events (อีเวนต์ 6 เดือนข้างหน้า: date เป็น YYYY-MM-DD หรือ YYYY-Qn, type, title, impact, certainty, source_url ต้องเป็นลิงก์จริงที่ค้นเจอ)
กฎ: ใช้ข้อมูลที่ค้นเจอจริงเท่านั้น ห้ามแต่งวันที่/ตัวเลข ถ้าไม่พบให้ใส่ list ว่าง เขียนภาษาไทย (ชื่อเฉพาะอังกฤษได้) ไม่ต้องใส่ราคาเป้าหมาย"""
        out, srcs, mode, errs = gemini_json(client, model, prompt, ResearchBatch, search=True)
        log_.append({"batch": [c["ticker"] for c in batch], "mode": mode, "errors": errs})
        if out:
            verified = "search" in mode
            names = [c["ticker"] for c in batch]
            for j, r in enumerate(out.coins):
                t = r.ticker.upper().replace(QUOTE, "").strip()
                if t not in names and j < len(names):
                    t = names[j]
                r.ticker = t
                cs = int(np.clip(r.catalyst_score, -100, 100))
                r.catalyst_score = cs if verified else cs // 2  # ไม่ได้ค้นเว็บ → ลดน้ำหนัก
                cache[(model, t)] = {"ts": now, "data": r.model_dump(), "sources": srcs, "mode": mode,
                                     "verified": verified}
    res = {c["ticker"]: cache.get((model, c["ticker"])) for c in coins}
    return {"by_ticker": {k: v for k, v in res.items() if v}, "log": log_}


# ───────────────────────── Strategist + Risk Manager ─────────────────────────
def coin_brief(c: dict, research: dict | None) -> dict:
    m, l, r, f, cf, fc = c["med"], c["long"], c["risk"], c["fund"], c["cf"], c["fc"]
    rd = (research or {}).get("data") or {}
    fcs = {f"{H}d": {"reach50": _r(v["reach50"], 6), "reach30": _r(v["reach30"], 6), "dip50": _r(v["dip50"], 6),
                     "range_p25_p75": [_r(v["end25"], 6), _r(v["end75"], 6)], "p_up": _r(v["p_up"])}
           for H, v in fc.items() if H in (14, 30, 90, 180)}
    return {
        "ticker": c["ticker"], "price": _r(c["price"], 6), "is_major": c["ticker"] in MAJORS,
        "regime": REG_TH[cf["regime"]], "structure_1d": c["struct_d"]["label"], "structure_1w": c["struct_w"]["label"],
        "rsi": _r(c["d1"]["rsi"].iloc[-1], 0), "adx": _r(c["d1"]["adx"].iloc[-1], 0),
        "medium": {"status": m["status"], "reasons": m["reasons"], "win_rate_sys": _r(m["win_sys"], 1),
                   "breakeven_win": _r(m["be_win"], 1), "buy_zone": [_r(m["zone"][0], 6), _r(m["zone"][1], 6)],
                   "sl": _r(m["sl"], 6), "tp1": _r(m["tp1"], 6), "tp2": _r(m["tp2"], 6), "rr1": _r(m["rr1"]),
                   "p_fill_14d": _r(m["p_fill"]), "p_tp1_before_sl": _r((m["ft1"] or {}).get("p_tp")),
                   "p_sl_first": _r((m["ft1"] or {}).get("p_sl")), "days_to_tp1": (m["ft1"] or {}).get("days_tp"),
                   "time_stop_days": m["tstop"], "backtest_n": c["bt"]["n"], "expectancy_R": _r(c["bt"]["expectancy"]),
                   "oos_expectancy_R": _r(c["bt"]["oos_expectancy"]), "profit_factor": _r(c["bt"]["profit_factor"])},
        "long": {"status": l["status"], "win_rate_sys": _r(l["win_sys"], 1), "hold": l["hold"],
                 "dca": [_r(t, 6) for t in l["tiers"]], "tp1": _r(l["tp1"], 6), "tp2": _r(l["tp2"], 6),
                 "p_tp1_180d": _r(l["p_tp1_180"]), "stop_weekly": _r(l["stop"], 6)},
        "forecast": fcs,
        "risk": {"vol_ann": _r(r["vol_ann"]), "mdd_1y": _r(r["mdd_1y"]), "beta": _r(r["beta"]),
                 "rs90_vs_btc": _r(r["rs90"], 3)},
        "fund": {"rank": f.get("rank"), "fdv_mcap": _r(f.get("fdv_mcap")), "pf": _r(f.get("pf"), 1),
                 "ath_dist": _r(f.get("ath_dist")), "tvl": f.get("tvl")},
        "news_7d": {k: (c.get("news") or {}).get(k) for k in ("score", "label", "confidence", "n", "summary", "drivers")},
        "score": {"total": _r(c["score"], 0), "news_adj": _r(c.get("news_adj", 0), 1)},
        "chart_patterns": [{"name": p["name"], "bias": p["bias"], "state": p["state"], "target": _r(p["target"], 6),
                            "p_target_60d": _r(p.get("p_target_60"))} for p in c.get("patterns", [])],
        "scanner_tags": c.get("scan_tags", []),
        "research": {"verified": (research or {}).get("verified"), "catalyst_score": rd.get("catalyst_score"),
                     "bull": rd.get("bull_case"), "bear": rd.get("bear_case"),
                     "events": [f"{e['date']} {e['type']} {e['title']} ({e['impact']})" for e in rd.get("events", [])[:4]]},
    }


def med_edge_ok(c: dict) -> bool:
    rw = rel_win(c["med"]["ft1"])
    return True if rw is None else rw * 100 >= c["med"]["be_win"] - 5


def enforce_strategy(out: StrategistOut, coins_by: dict, n_med: int, n_long: int) -> list[dict]:
    picks, seen, cnt = [], set(), {"MEDIUM": 0, "LONG": 0}
    for p in out.picks:
        t = p.ticker.upper().replace(QUOTE, "")
        if t not in coins_by or (t, p.horizon) in seen or cnt[p.horizon] >= (n_med if p.horizon == "MEDIUM" else n_long):
            continue
        c = coins_by[t]
        if p.horizon == "MEDIUM" and (c["med"]["status"] == "🔴" or c["cf"]["regime"] == "DOWN" or c.get("news_block")
                                      or not med_edge_ok(c)):
            continue
        sw = c["med"]["win_sys"] if p.horizon == "MEDIUM" else c["long"]["win_sys"]
        picks.append({"ticker": t, "horizon": p.horizon, "alloc": float(np.clip(p.allocation_pct, 0, cap_for(c))),
                      "win_sys": sw, "win_strat": float(np.clip(p.win_rate, sw - 10, sw + 10)),
                      "rationale": p.rationale, "key_risk": p.key_risk})
        seen.add((t, p.horizon))
        cnt[p.horizon] += 1
    for t in {x["ticker"] for x in picks}:
        xs = [x for x in picks if x["ticker"] == t]
        tot, cap = sum(x["alloc"] for x in xs), cap_for(coins_by[t])
        if tot > cap:
            for x in xs:
                x["alloc"] *= cap / tot
    tot = sum(x["alloc"] for x in picks)
    if tot > 100:
        for x in picks:
            x["alloc"] *= 100 / tot
    return picks


def enforce_risk(rk: RiskOut, picks: list[dict]) -> list[dict]:
    adj = {(a.ticker.upper(), a.horizon): a for a in rk.adjustments}
    for x in picks:
        a = adj.get((x["ticker"], x["horizon"]))
        x["alloc_final"] = min(x["alloc"], max(0.0, a.allocation_pct)) if a else x["alloc"]
        w = a.win_rate if a else x["win_strat"]
        x["win_final"] = float(np.clip(w, x["win_sys"] - 10, x["win_sys"] + 10))
        x["risk_note"] = a.note if a else ""
    return picks


def fallback_strategy(coins: list[dict], n_med: int, n_long: int) -> list[dict]:
    """จัดพอร์ตแบบ rule-based เมื่อไม่มี AI"""
    med = sorted([c for c in coins if c["med"]["status"] != "🔴" and c["cf"]["regime"] != "DOWN"
                   and not c.get("news_block") and med_edge_ok(c)],
                 key=lambda c: (c["med"]["status"] == "🟢", c["score"]), reverse=True)[:n_med]
    lng = sorted([c for c in coins if c["long"]["status"] != "🔴"], key=lambda c: c["score"], reverse=True)[:n_long]
    picks = [{"ticker": c["ticker"], "horizon": "MEDIUM", "alloc": min(10.0, cap_for(c)),
              "win_sys": c["med"]["win_sys"], "rationale": "; ".join(c["med"]["reasons"]), "key_risk": ""} for c in med]
    picks += [{"ticker": c["ticker"], "horizon": "LONG", "alloc": min(20.0 if c["ticker"] in MAJORS else 12.0, cap_for(c)),
               "win_sys": c["long"]["win_sys"], "rationale": "DCA ตามสถานะ Weekly/แนวรับ", "key_risk": ""} for c in lng]
    tot = sum(p["alloc"] for p in picks)
    if tot > 80:
        for p in picks:
            p["alloc"] *= 80 / tot
    for p in picks:
        p.update(win_strat=p["win_sys"], alloc_final=p["alloc"], win_final=p["win_sys"], risk_note="")
    return picks


def strategist_and_risk(client, model, coins, research, macro, corr, n_med, n_long,
                        progress: Progress | None = None, horizon: int = 30) -> dict:
    coins_by = {c["ticker"]: c for c in coins}
    briefs = [coin_brief(c, research.get(c["ticker"])) for c in coins]
    bmap = {b["ticker"]: b for b in briefs}
    if progress:
        progress("🧭 Strategist Agent: วิเคราะห์ + จัดพอร์ต…")
    sp = f"""คุณคือ Chief Strategist ระดับสถาบัน (Spot only) ทำงานร่วมกับ Agent อื่นที่คำนวณตัวเลขมาแล้ว
MACRO: {json.dumps({k: macro[k] for k in ('regime', 'notes')}, ensure_ascii=False)}
COINS: {json.dumps(briefs, ensure_ascii=False, default=str)}

ความหมายข้อมูล: forecast.reach50 = ราคาที่มีโอกาส 50% จะ "แตะ" ในกรอบเวลานั้น, reach30 = โอกาส 30%, dip50 = จุดย่อที่มีโอกาส 50%,
range_p25_p75 = กรอบราคาปลายงวด 50% กลาง, p_tp1_before_sl = โอกาสที่ราคาถึง TP1 ก่อนโดน SL, breakeven_win = win rate ขั้นต่ำที่คุ้มทุน
oos_expectancy_R = ผล backtest ช่วงข้อมูลล่าสุดที่ไม่ได้ใช้ปรับแต่ง (ติดลบ = ระวัง)
chart_patterns = รูปแบบกราฟที่ระบบตรวจพบ (p_target_60d = โอกาสแตะเป้าใน 60 วันจากสถิติ), scanner_tags = สัญญาณจาก Scanner
ผู้ใช้สนใจกรอบเวลา {horizon} วัน: ให้น้ำหนักเหรียญที่มีโอกาสขึ้นในช่วงนี้ (forecast, รูปแบบกราฟ, ข่าว, อีเวนต์โปรเจกต์ที่สอดคล้องกัน)

งาน 1 (picks): คัด MEDIUM ≤ {n_med} และ LONG ≤ {n_long} เหรียญ พร้อม allocation_pct และ cash_pct รวม = 100
กฎ: altcoin ≤ 15% (เหรียญผันผวนสูงระบบจะลดเหลือ 10%/6%), BTC/ETH ≤ 35% · ห้าม MEDIUM ที่ status 🔴 หรือ regime ขาลง
· เลือก MEDIUM เฉพาะที่ p_tp1_before_sl สูงกว่า breakeven_win อย่างมีนัย · win_rate ปรับจาก win_rate_sys ได้ ±10 เท่านั้น
· Macro Bear → เงินสดมากขึ้น · ห้ามคิดราคาใหม่ ใช้เฉพาะตัวเลขที่ให้มา
· news_7d = Sentiment ข่าว 7 วัน (-1..1): ข่าวลบแรงห้ามเลือก MEDIUM, ข่าวบวกที่สอดคล้องกับเทคนิคให้น้ำหนักเพิ่ม
งาน 2 (coin_views): ทุกเหรียญใน COINS ต้องมี 1 รายการ:
  verdict = ข้อสรุป 1 ประโยค (ซื้อ/รอ/เลี่ยง + ที่ราคาไหน + เป้าเท่าไหร่ภายในกี่วัน โดยอ้างตัวเลขจากข้อมูล)
  reasons = 3 เหตุผลหลัก (เทคนิค + สถิติ + พื้นฐาน/ข่าว) ข้อละ 1 ประโยค
  invalidation = เงื่อนไขที่ทำให้มุมมองผิด (อ้างระดับราคา SL/โครงสร้าง)
  watch = สิ่งที่ต้องจับตา/อีเวนต์ถัดไป
เขียนภาษาไทย กระชับ ตรงประเด็น"""
    so, _, mode_s, err_s = gemini_json(client, model, sp, StrategistOut, search=False)
    if not so:
        picks = fallback_strategy(coins, n_med, n_long)
        return {"picks": picks, "cash": 100 - sum(p["alloc_final"] for p in picks), "strategist": None,
                "views": {}, "risk": None, "errors": err_s, "mode": "fallback (rule-based)"}
    picks = enforce_strategy(so, coins_by, n_med, n_long)
    views = {v.ticker.upper().replace(QUOTE, ""): v.model_dump() for v in so.coin_views}

    if progress:
        progress("🛡️ Risk Manager Agent: ตรวจพอร์ต…")
    tick = [p["ticker"] for p in picks]
    sub = (corr.loc[[t for t in tick if t in corr.index], [t for t in tick if t in corr.columns]]
           if len(corr) else pd.DataFrame())
    rp = f"""คุณคือ Risk Manager Agent ตรวจพอร์ต Spot นี้: {json.dumps(picks, ensure_ascii=False, default=str)}
Correlation 90D: {sub.to_json() if len(sub) else "{}"}
Risk ต่อเหรียญ: {json.dumps({t: bmap[t]['risk'] for t in set(tick)}, default=str)}
สถิติแผนระยะกลาง: {json.dumps({t: {k: bmap[t]['medium'][k] for k in ('p_tp1_before_sl', 'breakeven_win', 'expectancy_R', 'oos_expectancy_R')} for t in set(tick)}, default=str)}
MACRO regime: {macro['regime']}
ตรวจการกระจุกตัว, correlation > 0.8, beta สูง, เหรียญที่ edge ต่ำ แล้วให้ verdict APPROVE/CAUTION/REJECT
adjustments: ใส่ทุกรายการ (ticker, horizon) allocation_pct ต้อง ≤ ค่าเดิม (ลดได้อย่างเดียว) win_rate ปรับได้ ±10 จาก win_sys · เขียนภาษาไทย"""
    ro, _, mode_r, err_r = gemini_json(client, model, rp, RiskOut, search=False)
    if ro:
        picks = enforce_risk(ro, picks)
    else:
        for p in picks:
            p.update(alloc_final=p["alloc"], win_final=p["win_strat"], risk_note="")
    return {"picks": picks, "cash": max(0.0, 100 - sum(p["alloc_final"] for p in picks)), "strategist": so,
            "views": views, "risk": ro, "errors": err_s + err_r, "mode": f"{mode_s} / {mode_r if ro else 'fail'}"}


def explain_signals(client, model: str, items: list[tuple[dict, object]]) -> dict[str, str]:
    """ให้ AI สรุปเหตุผลของสัญญาณใหม่ 1-2 ประโยค (ส่งเฉพาะ brief ของเหรียญที่ผ่าน quality gate)"""
    if client is None or not items:
        return {}
    data = [{"ticker": c["ticker"], "signal": {"entry": [s.entry_lo, s.entry_hi], "sl": s.sl, "tp1": s.tp1,
                                               "confidence": s.confidence, "win_prob": s.win_prob, "risks": s.risks},
             "brief": coin_brief(c, None)} for c, s in items]
    prompt = f"""คุณคือ Signal Analyst ตรวจความสอดคล้องของ Indicators สำหรับสัญญาณ Spot ต่อไปนี้:
{json.dumps(data, ensure_ascii=False, default=str)}
สำหรับแต่ละเหรียญ เขียน note ภาษาไทย 1-2 ประโยค: ทำไมสัญญาณนี้น่าสนใจ + ความเสี่ยงหลักที่ต้องระวัง
ห้ามคิดราคาใหม่ ห้ามรับประกันผล"""
    out, _, _, _ = gemini_json(client, model, prompt, SignalNotes, search=False)
    return {n.ticker.upper().replace(QUOTE, ""): n.note for n in (out.notes if out else [])}

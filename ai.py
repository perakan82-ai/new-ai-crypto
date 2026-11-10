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
from config import (AI_BUDGET_S, AI_MAX_MODELS, AI_TIMEOUT_S, AUTO_MODEL, MAJORS, MODEL_FALLBACK, QUOTE, REG_TH,
                    RESEARCH_TTL, TZ_TH)
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
    direction: Literal["POSITIVE", "NEGATIVE", "MIXED"] = "MIXED"  # คาดว่าผลต่อราคาเป็น + / − / ไม่แน่นอน
    why: str = ""


class MacroEvent(BaseModel):
    date: str
    title: str
    category: Literal["FED", "CPI", "JOBS", "ETF", "REGULATION", "UNLOCK", "UPGRADE", "EXCHANGE", "OTHER"]
    coins: list[str]          # ว่าง = กระทบทั้งตลาด
    direction: Literal["POSITIVE", "NEGATIVE", "MIXED"]
    impact: Literal["HIGH", "MEDIUM", "LOW"]
    certainty: Literal["CONFIRMED", "EXPECTED", "RUMOR"]
    why: str
    source_url: str


class MacroCalendar(BaseModel):
    events: list[MacroEvent]


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
    buy_score: int = 50                       # ความเห็น AI: ควรซื้อแค่ไหน 0-100 (ระบบจำกัดให้ห่างคะแนนระบบ ≤ ±15)
    outlook: Literal["BULLISH", "BEARISH", "NEUTRAL"] = "NEUTRAL"
    key_techniques: list[str] = []            # เทคนิคหลักที่ทำให้สรุปแบบนี้
    wait_plan: str = ""                       # ถ้ายังไม่แนะนำ: เหตุการณ์ที่กำลังเกิด · อาจร่วงถึง · จะจบเมื่อไหร่ · กลับมาซื้อเมื่อไหร่


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


_DISCOVERED: list[str] = []
_SKIP_NAME = ("image", "tts", "live", "embedding", "audio", "robotics", "computer-use", "nano", "gemma", "aqa", "learnlm",
              "veo", "imagen", "omni")


def _model_rank(name: str) -> tuple:
    """เรียงโมเดล: เวอร์ชันใหม่ก่อน · stable ก่อน preview · flash > pro > lite (เร็ว/ถูก/โควต้าเยอะ)"""
    v = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    ver = float(v.group(1)) if v else 0.0
    tier = 0 if ("flash" in name and "lite" not in name) else (1 if "pro" in name else 2)
    return (-ver, "preview" in name or "exp" in name, tier, name)


def discover_models(client) -> list[str]:
    """ถาม Gemini API ว่าตอนนี้มีโมเดลไหนที่ใช้ generateContent ได้ (กันโมเดลถูกปิดแล้วแอปพัง)"""
    out = []
    try:
        for m in client.models.list():
            n = (getattr(m, "name", "") or "").split("/")[-1]
            acts = getattr(m, "supported_actions", None) or []
            if "gemini" not in n or any(x in n for x in _SKIP_NAME) or (acts and "generateContent" not in acts):
                continue
            out.append(n)
    except Exception as e:  # noqa: BLE001
        log.warning("list models failed: %s", e)
    return sorted(set(out), key=_model_rank)


def make_client(api_key: str):
    """คืน client หรือ None (ไม่มี key / ไม่มีไลบรารี) · ค้นหาโมเดลที่ใช้ได้ครั้งเดียวต่อโปรเซส"""
    if not api_key or genai is None:
        return None
    try:  # จำกัดเวลารอต่อครั้ง — กันแอปค้างเมื่อ Gemini ตอบช้า (timeout หน่วย ms)
        client = genai.Client(api_key=api_key, http_options=gtypes.HttpOptions(timeout=AI_TIMEOUT_S * 1000))
    except Exception:  # noqa: BLE001 — google-genai รุ่นเก่าที่ไม่รองรับ http_options
        client = genai.Client(api_key=api_key)
    if not _DISCOVERED:
        _DISCOVERED.extend(discover_models(client))
    return client


_TIMEOUT_MSG = ("timed out", "timeout", "Timeout", "DEADLINE_EXCEEDED", "ReadTimeout")


def _retry(fn, tries: int = 2):
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
    """โมเดลที่เลือก → โมเดลที่ค้นเจอว่าใช้ได้จริง (ใหม่สุดก่อน) → alias -latest"""
    base = [] if (not primary or primary == AUTO_MODEL) else [primary]
    chain = list(dict.fromkeys(base + _DISCOVERED[:4] + MODEL_FALLBACK))
    # จำกัดจำนวนโมเดลที่ลอง (ประหยัดเวลา) แต่เก็บ alias ที่เสถียรที่สุดไว้เป็นตัวสุดท้ายเสมอ
    return list(dict.fromkeys(chain[: AI_MAX_MODELS - 1] + [MODEL_FALLBACK[0]]))


def gemini_json(client, model: str, prompt: str, schema: type[BaseModel], search: bool,
                budget_s: float = AI_BUDGET_S):
    """Model fallback × Mode fallback (schema+search → search+parse → schema) · คืน (obj|None, sources, mode, errors)
    หยุดลองเมื่อใช้เวลารวมเกิน budget_s · ถ้าโมเดลไหน timeout → ข้ามไปโมเดลถัดไปทันที (ไม่ลองโหมดอื่นของโมเดลเดิม)"""
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
    deadline = time.monotonic() + budget_s
    for m in model_chain(model):
        for name, kw in modes:
            if time.monotonic() > deadline:
                errs.append(f"หมดเวลา {budget_s:.0f}s — ใช้ผลแบบ rule-based แทน")
                return None, [], "timeout", errs
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
                if any(k in msg for k in _SKIP_MODEL) or any(k in msg for k in _TIMEOUT_MSG):
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
events (อีเวนต์ 6 เดือนข้างหน้า: date เป็น YYYY-MM-DD หรือ YYYY-MM หรือ YYYY-Qn, type, title, impact, certainty,
source_url ต้องเป็นลิงก์จริงที่ค้นเจอ, direction = POSITIVE/NEGATIVE/MIXED คาดว่าผลต่อราคาเหรียญเป็นบวกหรือลบ
(เช่น token unlock ใหญ่ = NEGATIVE, mainnet/ETF อนุมัติ = POSITIVE, ผลยังไม่แน่ = MIXED), why = เหตุผล 1 ประโยค)
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


MACRO_TTL = 12 * 3600


def macro_calendar_agent(client, model: str, tickers: list[str], cache: dict, days: int = 90) -> dict:
    """ปฏิทินเหตุการณ์ที่จะกระทบตลาดคริปโต N วันข้างหน้า (Google Search) · cache 12 ชม."""
    key = ("macro_cal", model, tuple(sorted(tickers)), days)
    hit = cache.get(key)
    if hit and time.time() - hit["ts"] < MACRO_TTL:
        return hit
    today = datetime.now(TZ_TH).strftime("%Y-%m-%d")
    prompt = f"""วันนี้ {today}. คุณคือ Macro & Event Calendar Agent ใช้ Google Search หาเหตุการณ์ใน {days} วันข้างหน้า
ที่มีผลต่อราคาคริปโต: ประชุม FED/FOMC, CPI, ตัวเลขจ้างงานสหรัฐ, กำหนดตัดสิน ETF คริปโต, กฎหมาย/การกำกับ,
Token unlock ขนาดใหญ่, อัปเกรดเครือข่าย, เหตุการณ์ของ Exchange — โดยเฉพาะที่เกี่ยวกับเหรียญ: {', '.join(tickers)}
สำหรับแต่ละเหตุการณ์: date (YYYY-MM-DD ถ้ารู้วันแน่ ไม่งั้น YYYY-MM), title, category, coins (ticker ที่กระทบโดยตรง ว่าง = ทั้งตลาด),
direction (POSITIVE/NEGATIVE/MIXED = ผลที่คาดต่อราคาคริปโต ถ้าขึ้นกับผลประกาศให้ใช้ MIXED แล้วอธิบายใน why ว่าออกแบบไหนเป็นบวก/ลบ),
impact, certainty, why (ภาษาไทย 1 ประโยค), source_url (ลิงก์จริงที่ค้นเจอ)
กฎ: ใช้ข้อมูลที่ค้นเจอจริงเท่านั้น ห้ามแต่งวันที่ ไม่พบให้ใส่ list ว่าง เรียงตามวันที่ ไม่เกิน 20 รายการ"""
    out, srcs, mode, errs = gemini_json(client, model, prompt, MacroCalendar, search=True)
    res = {"ts": time.time(), "events": [e.model_dump() for e in out.events] if out else [], "sources": srcs,
           "mode": mode, "verified": "search" in mode, "errors": errs}
    if out:
        cache[key] = res
    return res


class Scenario(BaseModel):
    name: Literal["BULL", "BASE", "BEAR"]
    trigger: str        # เงื่อนไขที่ทำให้เกิด (อ้างระดับราคาจากข้อมูลระบบ)
    target: str         # ราคาเป้าหมาย/กรอบราคา (จากข้อมูลระบบเท่านั้น)
    action: str         # ควรทำอะไร


class HorizonView(BaseModel):
    horizon: Literal["1 วัน", "1 สัปดาห์", "1 เดือน", "3 เดือน", "1 ปี"]
    direction: Literal["UP", "DOWN", "SIDEWAYS"]
    score: int                  # 0-100 (ระบบจำกัดให้ห่างคะแนนระบบ ≤ ±10)
    reason: str


class RoadmapItem(BaseModel):
    period: str                 # YYYY-MM-DD / YYYY-MM / YYYY-Qn / YYYY
    title: str
    kind: Literal["UPGRADE", "LAUNCH", "MAINNET", "UNLOCK", "ETF", "PARTNERSHIP", "REGULATION", "GOVERNANCE",
                  "BURN", "LISTING", "OTHER"]
    status: Literal["DONE", "IN_PROGRESS", "PLANNED", "RUMOR"]
    direction: Literal["POSITIVE", "NEGATIVE", "MIXED"]
    impact: Literal["HIGH", "MEDIUM", "LOW"]
    why: str
    source_url: str


class Backer(BaseModel):
    name: str
    kind: Literal["VC", "INSTITUTION", "ETF", "TREASURY_COMPANY", "EXCHANGE", "FOUNDATION", "STRATEGIC", "OTHER"]
    detail: str                 # ลงทุนเมื่อไหร่ รอบไหน จำนวนเท่าไหร่ (ถ้าค้นเจอ)
    source_url: str


class ProjectDeep(BaseModel):
    summary: str                # โปรเจกต์ทำอะไร แก้ปัญหาอะไร
    category: str
    use_cases: list[str] = []   # ใช้ทำอะไรได้จริง
    token_utility: str = ""     # เหรียญใช้ทำอะไรในระบบ (ค่าธรรมเนียม, stake, governance …)
    founded: str = ""           # ปีที่ก่อตั้ง/เปิดตัว
    team: str = ""              # ผู้ก่อตั้ง/บริษัทหลัก
    funding: str = ""           # ระดมทุนรวม/รอบล่าสุด
    backers: list[Backer] = []  # นายทุนใหญ่: VC, สถาบัน, ETF, บริษัทถือเหรียญ
    roadmap: list[RoadmapItem]  # ปีนี้ + ปีหน้า
    tokenomics: str             # supply, inflation, unlock schedule
    adoption: str               # ตัวชี้วัดการใช้งานจริง (TVL, ผู้ใช้, รายได้) ที่ค้นเจอ
    competitors: list[str]
    growth_drivers: list[str]
    risks: list[str]
    growth_view: Literal["HIGH", "MEDIUM", "LOW"]   # ศักยภาพเติบโต 1-2 ปี (เชิงคุณภาพ)
    growth_reason: str


class FocusReport(BaseModel):
    thesis: str                 # ข้อสรุปหลัก 2-3 ประโยค
    timeframe_read: str         # อ่านหลายกรอบเวลาอย่างไร
    best_strategy: str          # กลยุทธ์ที่เหมาะกับเหรียญนี้ตอนนี้ + เหตุผล
    vs_peers: str               # เทียบกับเหรียญกลุ่มเดียวกัน
    scenarios: list[Scenario]
    checklist: list[str]        # สิ่งที่ต้องเช็กก่อนกดซื้อ
    horizon_views: list[HorizonView] = []
    signals_read: str = ""      # สัญญาณที่กำลังจะเกิดหมายความว่าอะไร


PROJECT_TTL = 12 * 3600


def project_deep_agent(client, model: str, c: dict, cache: dict, headlines: list[str] | None = None) -> dict | None:
    """ค้นข้อมูลโปรเจกต์เจาะลึกของเหรียญเดียว (Google Search) · Roadmap ปีนี้ + ปีหน้า · cache 12 ชม."""
    key = ("project_deep", model, c["ticker"])
    hit = cache.get(key)
    if hit and time.time() - hit["ts"] < PROJECT_TTL:
        return hit
    now = datetime.now(TZ_TH)
    y = now.year
    f = c.get("fund") or {}
    prompt = f"""วันนี้ {now:%Y-%m-%d}. คุณคือ Crypto Project Research Analyst ใช้ Google Search ค้นข้อมูลล่าสุดของ
{f.get('name') or c['ticker']} ({c['ticker']}) · หมวด {f.get('categories')} · Rank #{f.get('rank')}
ค้นจาก: เว็บไซต์/บล็อกทางการ, เอกสาร roadmap/whitepaper, GitHub/forum governance, Token unlock trackers, ข่าวจากสื่อคริปโตหลัก
หัวข่าวล่าสุดที่ระบบเก็บได้: {json.dumps((headlines or [])[:8], ensure_ascii=False)}
ให้:
- summary: โปรเจกต์ทำอะไร แก้ปัญหาอะไร (2-3 ประโยค เข้าใจง่ายสำหรับมือใหม่), category
- use_cases: ใช้ทำอะไรได้จริง 2-5 ข้อ · token_utility: ตัวเหรียญใช้ทำอะไรในระบบ
- founded: ปีก่อตั้ง/เปิด mainnet · team: ผู้ก่อตั้ง/บริษัทหลักที่พัฒนา · funding: ระดมทุนรวมและรอบสำคัญ (จำนวน+ปี ถ้าค้นเจอ)
- backers: นายทุนใหญ่ 3-10 ราย เช่น VC ที่ลงทุน (a16z, Paradigm, Multicoin …), สถาบัน/ETF ที่ถือ,
  บริษัทที่ถือเหรียญเป็นทุนสำรอง (treasury company), Exchange/พันธมิตรเชิงกลยุทธ์ · kind, detail, source_url (ลิงก์จริง)
- roadmap: เหตุการณ์/เป้าหมายของปี {y} และ {y + 1} (5-12 รายการ) ทั้งที่เสร็จแล้วล่าสุดและที่วางแผนไว้:
  period (YYYY-MM-DD / YYYY-MM / YYYY-Qn / YYYY), title, kind, status, direction (ผลต่อราคา +/−/±), impact, why, source_url (ลิงก์จริง)
  รวม token unlock ก้อนใหญ่ด้วย (direction NEGATIVE)
- tokenomics: supply หมุนเวียน/สูงสุด, เงินเฟ้อ, ตาราง unlock
- adoption: ตัวเลขการใช้งานจริงที่ค้นเจอ (TVL, ผู้ใช้, ธุรกรรม, รายได้) พร้อมช่วงเวลา
- competitors: คู่แข่งหลัก 2-5 ตัว · growth_drivers: ปัจจัยที่จะทำให้โต 3-5 ข้อ · risks: ความเสี่ยง 3-5 ข้อ
- growth_view: HIGH/MEDIUM/LOW ศักยภาพเติบโต 1-2 ปีเทียบคู่แข่ง + growth_reason (อ้าง adoption, roadmap, นายทุน, tokenomics)
กฎ: ใช้ข้อมูลที่ค้นเจอจริงเท่านั้น ห้ามแต่งวันที่/ตัวเลข ห้ามใส่ราคาเป้าหมาย ไม่พบให้เว้นว่าง ภาษาไทย (ชื่อเฉพาะอังกฤษได้)"""
    out, srcs, mode, errs = gemini_json(client, model, prompt, ProjectDeep, search=True)
    if not out:
        return {"ts": time.time(), "data": None, "sources": [], "mode": mode, "verified": False, "errors": errs}
    res = {"ts": time.time(), "data": out.model_dump(), "sources": srcs, "mode": mode, "verified": "search" in mode,
           "errors": errs}
    cache[key] = res
    return res


def enforce_horizon_views(views: list[dict], system: list[dict], band: float = 10.0) -> dict[str, dict]:
    """คะแนน AI รายช่วงเวลา ถูกจำกัดให้อยู่ใน ±band ของคะแนนระบบ"""
    sys_by = {h["label"]: h["score"] for h in system}
    out = {}
    for v in views:
        if v["horizon"] in sys_by:
            sc = float(np.clip(v["score"], sys_by[v["horizon"]] - band, sys_by[v["horizon"]] + band))
            out[v["horizon"]] = {**v, "score": float(np.clip(sc, 0, 100))}
    return out


def focus_ai(client, model: str, brief: dict, deep_b: dict, roadmap_rows: list[dict],
             project: dict | None = None) -> tuple[FocusReport | None, list[str]]:
    """AI วิเคราะห์เจาะลึก 1 เหรียญ จากตัวเลขที่ระบบคำนวณแล้ว (ห้ามคิดราคาใหม่)"""
    prompt = f"""คุณคือ Senior Crypto Analyst วิเคราะห์เจาะลึกเหรียญ {brief['ticker']} สำหรับนักลงทุน Spot (ไม่ใช้ leverage)
ข้อมูลรายเหรียญ (ระบบคำนวณแล้ว): {json.dumps(brief, ensure_ascii=False, default=str)}
หลายกรอบเวลา + Backtest 6 กลยุทธ์ + เทียบคู่แข่ง: {json.dumps(deep_b, ensure_ascii=False, default=str)}
ข่าว/อีเวนต์ข้างหน้า: {json.dumps(roadmap_rows[:12], ensure_ascii=False, default=str)}
โปรเจกต์ & Roadmap (ค้นเว็บ): {json.dumps(project or {}, ensure_ascii=False, default=str)[:6000]}

เขียน:
- thesis: ข้อสรุปหลัก 2-3 ประโยค (ซื้อ/รอ/เลี่ยง + เพราะอะไร)
- timeframe_read: กรอบใหญ่บอกทิศ กรอบเล็กบอกจังหวะ ตอนนี้เป็นอย่างไร
- best_strategy: เลือกกลยุทธ์จาก strategies ที่เหมาะกับเหรียญนี้ อ้าง Sharpe/DD/OOS จริง ถ้าไม่มีตัวไหนผ่านให้บอกตรงๆ
- vs_peers: แข็งหรืออ่อนกว่าเหรียญกลุ่มเดียวกัน อันดับที่เท่าไหร่
- scenarios: BULL / BASE / BEAR อย่างละ 1 · trigger และ target ต้องใช้ราคาที่อยู่ในข้อมูล (buy_zone, sl, tp1, tp2, forecast, dca, stop_weekly) เท่านั้น
  ห้ามใส่ความน่าจะเป็นที่ไม่มีในข้อมูล
- checklist: 3-5 ข้อที่ต้องเช็กก่อนกดซื้อ
- horizon_views: ครบ 5 ช่วง (1 วัน, 1 สัปดาห์, 1 เดือน, 3 เดือน, 1 ปี) direction + score 0-100 + reason 1 ประโยค
  เริ่มจาก horizons.system_score แล้วปรับได้ไม่เกิน ±10 (ระบบจะตัดส่วนเกิน) · ช่วงสั้นอิงเทคนิค/สัญญาณที่กำลังจะเกิด
  ช่วงยาว (3 เดือน, 1 ปี) อิง roadmap โปรเจกต์ tokenomics/unlock และแนวโน้มรายสัปดาห์ · ถ้า confidence ต่ำให้บอกในเหตุผล
- signals_read: อธิบาย upcoming_signals ที่สำคัญที่สุด 2-3 ตัวว่าถ้าเกิดจริงหมายความว่าอะไร และระดับราคาที่ต้องดู
ภาษาไทย กระชับ ห้ามรับประกันผล"""
    out, _, _, errs = gemini_json(client, model, prompt, FocusReport, search=False)
    return out, errs


def enforce_buy_score(ai_score: float | None, sys_score: float, band: float = 15.0) -> float:
    """AI ให้คะแนนได้ แต่ห่างจากคะแนนระบบได้ไม่เกิน ±band (กัน AI มั่นใจเกินหลักฐาน)"""
    if ai_score is None:
        return float(sys_score)
    return float(np.clip(ai_score, max(0.0, sys_score - band), min(100.0, sys_score + band)))


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
        "source": "scanner" if c.get("is_pick") else "watchlist",
        "momentum": {"vs_market_pct": _r(c.get("xs_pct")), "trend_0_1": _r((c.get("mom") or {}).get("trend")),
                     "ret30": _r(((c.get("mom") or {}).get("ret") or {}).get(30), 3),
                     "ret90": _r(((c.get("mom") or {}).get("ret") or {}).get(90), 3),
                     "risk_adj": _r((c.get("mom") or {}).get("risk_adj")),
                     "orderflow_7v30": _r((c.get("mom") or {}).get("orderflow"), 3),
                     "pullback_in_uptrend_4h": bool(m.get("mr4h"))},
        "entry_gate_fails": c.get("gate_fails", []),
        "techniques": [f"{'+' if t['vote'] > 0 else '−' if t['vote'] < 0 else '0'} {t['name']}: {t['detail']}"
                       for t in (c.get("tech") or {}).get("votes", [])],
        "tech_agreement_0_100": _r((c.get("tech") or {}).get("score"), 0),
        "buy_score_system": _r(c.get("buy_sys"), 0),
        "score_backtest": {k: (c.get("score_check") or {}).get(k) for k in ("verdict", "skill", "hit", "p_today", "n_today")},
        "p_up_horizon": _r(c.get("p_up_h")),
        "upcoming_news_net": _r(c.get("event_net")),
        "market_read": ({"phase": c["read"]["phase"], "phase_why": c["read"]["phase_why"],
                         "pressure": [("+" if e["sign"] > 0 else "−") + " " + e["text"] for e in c["read"]["evidence"]],
                         "dip50": _r(c["read"]["downside"]["dip50"], 6), "dip30": _r(c["read"]["downside"]["dip30"], 6),
                         "buy_zone": [_r(x, 6) for x in c["read"]["downside"]["buy_zone"]],
                         "days_to_low_p50": (c["read"]["timing"] or {}).get("p50"),
                         "days_to_low_p25_p75": [(c["read"]["timing"] or {}).get("p25"), (c["read"]["timing"] or {}).get("p75")],
                         "flip_conditions": c["read"]["flip"]} if c.get("read") else None),
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
                        progress: Progress | None = None, horizon: int = 30, xs_bt: dict | None = None) -> dict:
    coins_by = {c["ticker"]: c for c in coins}
    briefs = [coin_brief(c, research.get(c["ticker"])) for c in coins]
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
source = "watchlist" คือเหรียญที่ผู้ใช้สนใจเอง (≤3), "scanner" คือเหรียญที่ระบบคัดจากตลาดตอนนี้
momentum.vs_market_pct = momentum ปรับความผันผวนเทียบเหรียญอื่นในตลาด (1 = แข็งที่สุด), trend_0_1 = สัดส่วน Donchian 5 lookback ที่อยู่ในเทรนด์,
pullback_in_uptrend_4h = 1D ขาขึ้นแต่ 4H ย่อต่ำกว่าค่าเฉลี่ยมาก (จังหวะเข้าแบบ mean reversion ระยะสั้น), entry_gate_fails = เกณฑ์จุดเข้าที่ยังไม่ผ่าน
ผล Backtest ของการจัดอันดับ momentum เทียบตลาด (ถือ top-5 ทีละ 7 วัน เทียบถือทุกเหรียญ): {json.dumps(xs_bt or {}, ensure_ascii=False, default=str)}
· ถ้าผล backtest นี้ไม่มี edge ห้ามใช้ momentum เป็นเหตุผลหลัก · momentum สูง+ผันผวนสูงเสี่ยง crash ให้ลด allocation
· เหรียญ scanner ต้องอธิบายใน coin_views ว่าทำไมน่าสนใจ "ตอนนี้" และแยกให้ชัดว่าผ่านเกณฑ์จุดเข้าแล้ว หรือยังเป็นแค่ตัวที่ควรเฝ้าดู

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
  buy_score = ความเห็นของคุณ 0-100 ว่าควรซื้อเหรียญนี้ "ตอนนี้" แค่ไหนจากข้อมูลทั้งหมด (เทคนิค สถิติ ข่าว อีเวนต์ข้างหน้า มหภาค)
     เริ่มจาก buy_score_system แล้วปรับได้ไม่เกิน ±15 (ระบบจะตัดส่วนเกิน) · ห้ามให้สูงถ้า entry_gate_fails ยังมีข้อสำคัญ
  outlook = BULLISH/BEARISH/NEUTRAL · key_techniques = เทคนิคหลัก 2-4 ข้อจาก techniques ที่ทำให้สรุปแบบนี้ (ระบุ +/−)
  wait_plan = ถ้ายังไม่ควรซื้อ (แม้จะอันดับดีที่สุดในตาราง): 2-4 ประโยค อธิบายจาก market_read ว่า
     ตอนนี้เกิดเหตุการณ์อะไร (phase) · หลักฐานแรงขาย/Volume/Divergence · ราคาอาจย่อถึงเท่าไหร่ (dip50/dip30)
     · คาดว่าการย่อจะจบในกี่วัน (days_to_low) · กลับมาซื้อได้เมื่อไหร่/ที่ราคาไหน (buy_zone + flip_conditions)
     ใช้เฉพาะตัวเลขที่ให้มา · ถ้าแนะนำซื้อ ให้ใส่ว่าถ้าผิดทางจะเกิดอะไรและออกที่ไหน
p_up_horizon = โอกาสปิดสูงกว่าวันนี้ในกรอบเวลาที่เลือกจากสถิติย้อนหลัง · tech_agreement = เทคนิคเห็นตรงกันแค่ไหน (ไม่ใช่ความน่าจะเป็น)
upcoming_news_net = ผลรวมข่าว/อีเวนต์ข้างหน้า (-1..1) จาก Roadmap
เขียนภาษาไทย กระชับ ตรงประเด็น"""
    so, _, mode_s, err_s = gemini_json(client, model, sp, StrategistOut, search=False)
    if not so:
        picks = fallback_strategy(coins, n_med, n_long)
        return {"picks": picks, "cash": 100 - sum(p["alloc_final"] for p in picks), "strategist": None,
                "views": {}, "risk": None, "errors": err_s, "mode": "fallback (rule-based)"}
    picks = enforce_strategy(so, coins_by, n_med, n_long)
    views = {v.ticker.upper().replace(QUOTE, ""): v.model_dump() for v in so.coin_views}

    picks, ro = risk_rules(picks, coins_by, corr, macro)
    return {"picks": picks, "cash": max(0.0, 100 - sum(p["alloc_final"] for p in picks)), "strategist": so,
            "views": views, "risk": ro, "errors": err_s, "mode": f"{mode_s} / risk: rule-based"}


def risk_rules(picks: list[dict], coins_by: dict, corr: pd.DataFrame, macro: dict) -> tuple[list[dict], RiskOut]:
    """Risk Manager แบบกติกา (แทนการเรียก AI รอบที่ 2 — เร็วกว่า ตรวจสอบได้ ผลเหมือนเดิมทุกครั้ง)

    ลด allocation ได้อย่างเดียว:
      1) คู่ที่ correlation 90D > 0.8 → ตัวที่คะแนนต่ำกว่าลด 30% (เสี่ยงซ้ำซ้อน)
      2) altcoin beta > 1.5 → ลด 25%
      3) ระยะกลางที่ backtest ช่วงล่าสุด (OOS) ติดลบ → ลด 50%
      4) Macro Bear → เงินลงทุนรวมไม่เกิน 60% (ที่เหลือเป็นเงินสด)
    win rate คงตามที่ Strategist ปรับ (อยู่ในกรอบ ±10 ของระบบอยู่แล้ว)
    """
    issues: list[str] = []
    notes: dict[tuple[str, str], list[str]] = {}
    mult = {(p["ticker"], p["horizon"]): 1.0 for p in picks}

    def cut(key, f, why):
        mult[key] *= f
        notes.setdefault(key, []).append(why)

    tick = list(dict.fromkeys(p["ticker"] for p in picks))
    for i, a in enumerate(tick):
        for b in tick[i + 1:]:
            if len(corr) and a in corr.index and b in corr.columns:
                v = corr.loc[a, b]
                if pd.notna(v) and v > 0.8:
                    lo = min((a, b), key=lambda t: coins_by[t]["score"])
                    issues.append(f"{a} กับ {b} เคลื่อนไหวเหมือนกันมาก (corr {v:.2f}) → ลด {lo} 30%")
                    for p in picks:
                        if p["ticker"] == lo:
                            cut((lo, p["horizon"]), 0.7, f"corr กับ {a if lo == b else b} {v:.2f}")
    for p in picks:
        c, key = coins_by[p["ticker"]], (p["ticker"], p["horizon"])
        beta = c["risk"].get("beta")
        if p["ticker"] not in MAJORS and beta is not None and np.isfinite(beta) and beta > 1.5:
            issues.append(f"{p['ticker']} beta {beta:.2f} (แกว่งแรงกว่า BTC) → ลด 25%")
            cut(key, 0.75, f"beta {beta:.2f}")
        oos = c["bt"].get("oos_expectancy")
        if p["horizon"] == "MEDIUM" and c["bt"].get("oos_n", 0) >= 3 and oos is not None and oos < 0:
            issues.append(f"{p['ticker']} backtest ช่วงล่าสุดติดลบ ({oos:+.2f}R) → ลดระยะกลาง 50%")
            cut(key, 0.5, f"OOS {oos:+.2f}R")
    for p in picks:
        key = (p["ticker"], p["horizon"])
        p["alloc_final"] = p["alloc"] * mult[key]
        p["win_final"] = float(np.clip(p["win_strat"], p["win_sys"] - 10, p["win_sys"] + 10))
        p["risk_note"] = " · ".join(notes.get(key, []))
    tot = sum(p["alloc_final"] for p in picks)
    if macro.get("regime") == "Bear" and tot > 60:
        issues.append(f"ตลาดขาลง (Macro Bear) → จำกัดเงินลงทุนรวม 60% จาก {tot:.0f}%")
        for p in picks:
            p["alloc_final"] *= 60 / tot
    verdict = "APPROVE" if not issues else "CAUTION"
    final = ("ผ่านเกณฑ์ความเสี่ยงทั้งหมด" if not issues else
             f"ปรับลดสัดส่วน {len(issues)} จุด · เงินลงทุนรวม {sum(p['alloc_final'] for p in picks):.0f}%")
    return picks, RiskOut(verdict=verdict, issues=issues, adjustments=[], final_note=final)


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

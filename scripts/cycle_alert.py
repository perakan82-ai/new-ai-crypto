"""แจ้งเตือน "ถึงเวลาซื้อ/ขายหรือยัง" ผ่าน Telegram — รันทุก 4 ชม. ผ่าน GitHub Actions (cycle-alert.yml)

ส่วนที่ 1 · Market-cycle dashboard (ภาพรวมตลาด ใช้ปรับขนาดไม้/สัดส่วน altcoin ไม่ใช่จุดเข้ารายเหรียญ)
  1) USDT dominance   CoinGecko /global (ฟรี)            สูง = คนถือเงินสด/กลัว → ฝั่งซื้อ · ต่ำ = ฝั่งขาย
  2) Alt OI vs BTC OI Coinalyze (ฟรี ต้องมี COINALYZE_API_KEY) alt > BTC ต่อเนื่อง = leverage กระจุกใน alt → ฝั่งขาย
  3) Fear & Greed     alternative.me (ฟรี)               ≤20 ฝั่งซื้อ · ≥75 ฝั่งขาย
  4) BTC regime       Binance 1D (ฟรี)                   Bull ฝั่งซื้อ · Bear ฝั่งขาย
  สรุป: 🟢 โซนสะสม / 🟡 ผสม / 🔴 ตลาดร้อน-ลดความเสี่ยง · ส่งเฉพาะตอน "สถานะเปลี่ยน"

ส่วนที่ 2 · Trigger รายเหรียญ จากแผนล่าสุดของสแกนประจำวัน (data/plans.json ที่ daily_scan.py บันทึกไว้)
  เข้าโซนซื้อ · แตะ TP1 / TP2 · หลุด SL · ถึงไม้ DCA ระยะยาว — แต่ละเหตุการณ์ส่งครั้งเดียวต่อแผน

ข้อจำกัดที่ต้องรู้:
  - USDT.D ใช้ percentile จากประวัติที่ระบบเก็บเอง (CoinGecko ฟรีไม่มีข้อมูลย้อนหลัง) ต้องมีอย่างน้อย
    MIN_HISTORY วัน ก่อนหน้านั้นใช้เกณฑ์ชั่วคราว ≥8% / ≤6% และแจ้งในข้อความว่าเป็นเกณฑ์ชั่วคราว
  - Alt OI นับเฉพาะสัญญา perpetual บน Binance (ตัวแทนตลาด ไม่ใช่ทุกกระดาน) และไม่รวม ETH
  - ทุกสัญญาณเป็นข้อมูลประกอบการตัดสินใจ ไม่ใช่คำแนะนำการลงทุน และไม่มีการส่งคำสั่งซื้อขาย

Environment: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (จำเป็น) · COINALYZE_API_KEY, COINGECKO_DEMO_KEY (ไม่บังคับ)
             FORCE_SEND=1 → ส่งสรุปเสมอ (ใช้ตอนกดรันเองเพื่อทดสอบ)
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

TH_TZ = timezone(timedelta(hours=7))
DATA_DIR = ROOT / "data"
STATE_FILE = DATA_DIR / "alert_state.json"
PLANS_FILE = DATA_DIR / "plans.json"

MIN_HISTORY = 90            # วันขั้นต่ำก่อนใช้ percentile ของ USDT.D
HISTORY_KEEP = 730          # เก็บประวัติ 2 ปี
USDT_FALLBACK = (8.0, 6.0)  # (ฝั่งซื้อ ≥, ฝั่งขาย ≤) ใช้ชั่วคราวระหว่างเก็บประวัติ
OI_STREAK_DAYS = 7          # alt OI > BTC OI ติดต่อกันกี่วันถึงนับเป็นฝั่งขาย
OI_FLUSH_DROP = 0.25        # ratio ลดลง ≥25% จากจุดสูงสุด 3 วัน = leverage ถูกล้าง → ฝั่งซื้อ
PLAN_MAX_AGE_D = 7          # แผนเก่ากว่านี้ไม่ใช้แจ้งเตือน


# ═══════════════════════ state ═══════════════════════
def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def upsert_daily(hist: list, day: str, value: float, keep: int = HISTORY_KEEP) -> list:
    """เก็บค่า 1 ค่าต่อวัน (รอบหลังในวันเดียวกันเขียนทับ) เรียงตามวันที่"""
    d = {k: v for k, v in hist}
    d[day] = value
    return [[k, d[k]] for k in sorted(d)][-keep:]


# ═══════════════════════ votes (pure functions — ทดสอบได้) ═══════════════════════
def _v(name: str, vote: str | None, ok: bool, plain: str, act: str = "", tech: str = "") -> dict:
    """ผลของตัวชี้วัด 1 ตัว · plain = อธิบายภาษาง่าย · act = ควรทำอะไร · tech = ตัวเลขเทคนิค (ไว้ใน log)"""
    return {"name": name, "vote": vote, "ok": ok, "plain": plain, "act": act, "text": f"{name}: {tech or plain}"}


def usdt_vote(value: float | None, hist: list) -> dict:
    """USDT dominance สูง = คนขายเหรียญไปถือเงินสด (กลัว) → ฝั่งซื้อ · ต่ำ = เงินลงตลาดหมดแล้ว → ฝั่งขาย"""
    if value is None:
        return _v("USDT.D", None, False, "สัดส่วนเงินสด USDT: ดึงข้อมูลไม่ได้")
    vals = [v for _, v in hist]
    if len(vals) >= MIN_HISTORY:
        pct = sum(v <= value for v in vals) / len(vals)
        vote = "buy" if pct >= 0.8 else ("sell" if pct <= 0.2 else None)
        basis = f"สูงกว่า {pct * 100:.0f}% ของ {len(vals)} วันที่ผ่านมา"
    else:
        vote = "buy" if value >= USDT_FALLBACK[0] else ("sell" if value <= USDT_FALLBACK[1] else None)
        basis = f"ใช้เกณฑ์ชั่วคราว ≥{USDT_FALLBACK[0]:g}% · เก็บประวัติแล้ว {len(vals)}/{MIN_HISTORY} วัน"
    plain = {"buy": f"คนถือเงินสด USDT สูง {value:.2f}% ({basis})",
             "sell": f"คนถือเงินสด USDT ต่ำ {value:.2f}% ({basis})"}.get(vote, f"เงินสด USDT {value:.2f}% ปกติ ({basis})")
    act = {"buy": "คนกลัวและรอซื้อ มักเป็นจังหวะเริ่มสะสม", "sell": "เงินลงตลาดไปเยอะแล้ว ระวังยอด"}.get(vote, "")
    return _v("USDT.D", vote, True, plain, act, f"{value:.2f}% · {basis}")


def oi_vote(ratio: float | None, oi_hist: list) -> dict:
    """ratio = alt OI / BTC OI · oi_hist = [[day, ratio], ...] รวมวันนี้แล้ว"""
    if ratio is None:
        return _v("Alt OI", None, False, "เงินกู้ใน Altcoin เทียบ Bitcoin: ยังไม่มีข้อมูล (ไม่มี COINALYZE_API_KEY หรือดึงไม่สำเร็จ)")
    streak = 0
    for _, r in reversed(oi_hist):
        if r > 1:
            streak += 1
        else:
            break
    recent = [r for _, r in oi_hist[-4:-1]]  # 3 วันก่อนหน้า
    peak = max(recent) if recent else None
    if streak >= OI_STREAK_DAYS:
        return _v("Alt OI", "sell", True, f"คนกู้เงินมาเล่น Altcoin มากกว่า Bitcoin ติดต่อกัน {streak} วัน ({ratio:.2f} เท่า)",
                  "ร้อนเกินไป เสี่ยงร่วงแรง ควรขาย Altcoin บางส่วน", f"{ratio:.2f}x · alt > BTC {streak} วัน")
    if peak and peak > 0 and ratio <= peak * (1 - OI_FLUSH_DROP):
        drop = (1 - ratio / peak) * 100
        return _v("Alt OI", "buy", True, f"เงินกู้ใน Altcoin เพิ่งถูกล้างไป {drop:.0f}% ใน 3 วัน",
                  "แรงขายจากคนติดหนี้หมดไปแล้ว มักเป็นจังหวะเข้า", f"{ratio:.2f}x · ลด {drop:.0f}%")
    why = f"สูงกว่า Bitcoin มา {streak} วัน ยังไม่ถึง {OI_STREAK_DAYS} วัน" if streak else "ต่ำกว่า Bitcoin"
    return _v("Alt OI", None, True, f"เงินกู้ใน Altcoin {ratio:.2f} เท่าของ Bitcoin ({why})", "", f"{ratio:.2f}x · {why}")


def fng_vote(fng: dict | None) -> dict:
    if not fng:
        return _v("F&G", None, False, "ดัชนีความกลัว/โลภ: ดึงข้อมูลไม่ได้")
    v = fng["value"]
    if v <= 20:
        return _v("F&G", "buy", True, f"ดัชนีความกลัว {v} (กลัวสุดขีด)", "คนตื่นตระหนก มักเป็นจังหวะสะสม", f"{v} {fng['label']}")
    if v >= 75:
        return _v("F&G", "sell", True, f"ดัชนีความโลภ {v} (โลภสุดขีด)", "คนไล่ราคา ระวังยอด", f"{v} {fng['label']}")
    return _v("F&G", None, True, f"ดัชนีความกลัว/โลภ {v} (ปกติ)", "", f"{v} {fng['label']}")


def regime_vote(regime: str | None) -> dict:
    if not regime:
        return _v("BTC", None, False, "แนวโน้ม Bitcoin: ดึงข้อมูลไม่ได้")
    if regime == "Bull":
        return _v("BTC", "buy", True, "Bitcoin เป็นขาขึ้น (เหนือเส้นค่าเฉลี่ย 200 วัน)", "ตลาดหลักยังแข็ง", regime)
    if regime == "Bear":
        return _v("BTC", "sell", True, "Bitcoin เป็นขาลง (ต่ำกว่าเส้นค่าเฉลี่ย 200 วัน)", "ตลาดหลักยังอ่อน อย่าลงเงินเยอะ", regime)
    return _v("BTC", None, True, "Bitcoin กำลังเปลี่ยนทิศ (ยังไม่ชัด)", "", regime)


def market_state(votes: list[dict]) -> tuple[str, int, int, int]:
    """🟢 ถ้าฝั่งซื้อ ≥3 หรือ (≥2 และไม่มีฝั่งขาย) · 🔴 กลับกัน · นอกนั้น 🟡 — คืน (state, buy, sell, มีข้อมูล)"""
    b = sum(v["vote"] == "buy" for v in votes)
    s = sum(v["vote"] == "sell" for v in votes)
    n = sum(bool(v.get("ok")) for v in votes)
    if b >= 3 or (b >= 2 and s == 0):
        st = "🟢"
    elif s >= 3 or (s >= 2 and b == 0):
        st = "🔴"
    else:
        st = "🟡"
    return st, b, s, n


def _p(x: float) -> str:
    return f"{x:,.6g}"


# ระดับการซื้อตามสีตลาด (ใช้กับ "เข้าโซนซื้อ" และ "ถึงไม้ DCA")
BUY_VERDICT = {"🟢": ("ซื้อได้เลย ✅", "ผ่านเกณฑ์ครบ: ราคาเข้าโซน + ตลาดรวมเขียว"),
               "🟡": ("ซื้อได้แค่ไม้เล็ก 🟡", "ราคาเข้าโซน แต่ตลาดรวมยังผสม → ใช้เงินแค่ ⅓ ของที่ตั้งใจ"),
               "🔴": ("ยังไม่ควรซื้อ ⛔", "ราคาเข้าโซน แต่ตลาดรวมแดง → รอให้ตลาดดีขึ้นก่อน")}


def coin_events(plans: dict, prices: dict, sent: dict, today: datetime, state: str = "🟡") -> list[tuple[str, str]]:
    """คืน [(key, ข้อความ)] ของเหตุการณ์ใหม่ที่ยังไม่เคยส่ง · key ผูกกับวันที่ของแผน จึงส่งครั้งเดียวต่อแผน
    ข้อความฝั่งซื้อปรับตามสีตลาดรวม (state) — ราคาเข้าโซนอย่างเดียวไม่พอจะบอกว่า "ซื้อได้เลย" """
    out = []
    verdict, why = BUY_VERDICT[state]
    for t, p in plans.items():
        px = prices.get(t)
        if px is None:
            continue
        try:
            age = (today.date() - datetime.fromisoformat(p["date"]).date()).days
        except (KeyError, ValueError):
            continue
        if age > PLAN_MAX_AGE_D:
            continue
        m, lg = p.get("med") or {}, p.get("long") or {}
        checks = []
        if m and m.get("status") != "🔴":
            lo, hi = m["zone"]
            checks += [
                ("sl", px <= m["sl"], f"🪙 {t}: ตัดขาดทุน 🛑\n   ราคา {_p(px)} ต่ำกว่าจุดตัดขาดทุน {_p(m['sl'])}\n   → ขายตามแผน อย่าถัวเพิ่ม"),
                ("tp2", px >= m["tp2"], f"🪙 {t}: ขายส่วนที่เหลือได้แล้ว 🚀\n   ราคา {_p(px)} แตะเป้าที่ 2 ({_p(m['tp2'])})"),
                ("tp1", px >= m["tp1"], f"🪙 {t}: ขายครึ่งหนึ่งได้แล้ว 💰\n   ราคา {_p(px)} แตะเป้าแรก ({_p(m['tp1'])})\n"
                                        f"   → ที่เหลือเลื่อนจุดตัดขาดทุนมาที่ราคาทุน · เป้าถัดไป {_p(m['tp2'])}"),
                ("zone", m["sl"] < px <= hi, f"🪙 {t}: {verdict}\n   ราคา {_p(px)} อยู่ในโซนซื้อ {_p(lo)}–{_p(hi)}\n"
                                             f"   ขายทำกำไร {_p(m['tp1'])} · ตัดขาดทุนถ้าต่ำกว่า {_p(m['sl'])}\n   ({why})"),
            ]
        if lg and lg.get("status") != "🔴":
            for i, tier in enumerate(lg.get("tiers") or []):
                checks.append((f"dca{i + 1}", px <= tier,
                               f"🪙 {t} (ถือยาว): สะสมไม้ที่ {i + 1} — {verdict}\n   ราคา {_p(px)} ถึงจุดซื้อสะสม {_p(tier)}"))
            if lg.get("stop"):
                checks.append(("lstop", px <= lg["stop"],
                               f"🪙 {t} (ถือยาว): ระวัง ⚠️\n   ราคา {_p(px)} ต่ำกว่าจุดยอมแพ้ระยะยาว {_p(lg['stop'])}\n"
                               "   → ถ้าแท่งรายสัปดาห์ปิดต่ำกว่านี้ ให้ขาย"))
        for name, hit, text in checks:
            key = f"{t}:{p['date']}:{name}"
            if hit and key not in sent:
                out.append((key, text))
    return out


STATE_HEAD = {"🟢": "🟢 ดี — ทยอยซื้อตามแผนได้", "🟡": "🟡 ระวัง — ซื้อได้ทีละน้อย", "🔴": "🔴 ร้อน/เสี่ยง — ไม่ซื้อเพิ่ม ทำแผนขาย"}


def todo_list(state: str, votes: list[dict]) -> list[str]:
    alt_hot = any(v["name"] == "Alt OI" and v["vote"] == "sell" for v in votes)
    if state == "🟢":
        out = ["ซื้อตามแผนได้ แบ่ง 2–3 ไม้"]
        out.append("ยังไม่ไล่ซื้อ Altcoin (เงินกู้ใน Alt สูง)" if alt_hot else "Altcoin ซื้อได้เฉพาะตัวที่ราคาเข้าโซน")
    elif state == "🟡":
        out = ["ซื้อ BTC / ETH ได้ แบ่งไม้เล็ก ๆ", "ยังไม่ไล่ซื้อ Altcoin"]
        if alt_hot:
            out.append("ถ้าถือ Altcoin ที่มีกำไร ขายบางส่วนได้")
    else:
        out = ["ไม่ซื้อเพิ่ม", "ขาย Altcoin บางส่วน / ทำตามแผนขาย", "เก็บเงินสดไว้รอจังหวะ"]
    return out


def build_message(state: str, prev: str | None, b: int, s: int, n: int, votes: list[dict],
                  events: list[tuple[str, str]], now: datetime, picks: list[dict] | None = None,
                  mine: list[dict] | None = None) -> str:
    L = [f"🔔 สัญญาณคริปโต {now:%d/%m %H:%M}", "", f"📊 ตลาดรวม: {STATE_HEAD[state]}"]
    if prev and prev != state:
        L.append(f"   (เปลี่ยนจาก {prev})")
    L.append(f"   สัญญาณซื้อ {b} · ขาย {s} · จากที่มีข้อมูล {n}/{len(votes)} ตัว")
    groups = (("🔴 สัญญาณขาย", "sell"), ("🟢 สัญญาณซื้อ", "buy"))
    for title, side in groups:
        vs = [v for v in votes if v["vote"] == side]
        if vs:
            L += ["", title]
            for v in vs:
                L.append(f"• {v['plain']}")
                if v["act"]:
                    L.append(f"  → {v['act']}")
    neutral = [v for v in votes if v["ok"] and v["vote"] is None]
    if neutral:
        L += ["", "⚪ ปกติ (ไม่นับ)"] + [f"• {v['plain']}" for v in neutral]
    missing = [v for v in votes if not v["ok"]]
    if missing:
        L += ["", "❔ ไม่มีข้อมูล"] + [f"• {v['plain']}" for v in missing]
    if b and s:
        L += ["", "ℹ️ ทำไมสัญญาณขัดกัน: แต่ละตัววัดคนละเรื่อง (เงินกู้ = ร้อนเกินไปไหม · เงินสด/ความกลัว = เย็นเกินไปไหม"
                  " · Bitcoin = แนวโน้ม) มักเกิดคนละจังหวะ ระบบนับเสียงข้างมากเป็นสีตลาดรวม → ทำตาม 👉 ด้านล่าง"]
    L += ["", "👉 ควรทำอะไร"] + [f"• {x}" for x in todo_list(state, votes)]
    if events:
        L += ["", "━━━━━━━━━━", "🎯 รายเหรียญ (จากแผนสแกนประจำวัน)"]
        for _, txt in events:
            L += ["", txt]
    L += picks_block(mine or [], state, "⭐ เหรียญของคุณ", "")
    L += picks_block(picks or [], state)
    L += ["", "⚠️ ไม่มีใครทำนายตลาดได้แน่นอน แบ่งเงินซื้อ อย่าลงก้อนเดียว · ไม่ใช่คำแนะนำการลงทุน"]
    return "\n".join(L)


# ═══════════════════════ เหรียญแนะนำ (คัดจากตลาดด้วย indicator ของแอป) ═══════════════════════
N_PICKS = 5          # แสดงสูงสุดกี่เหรียญ
N_CANDIDATES = 10    # คัดจาก Scanner กี่เหรียญมาวิเคราะห์ละเอียด


def pick_verdict(passed: bool, state: str) -> str:
    if not passed:
        return "รอ ⏳"
    return BUY_VERDICT[state][0]


def rank_picks(rows: list[dict], n: int = N_PICKS) -> list[dict]:
    """ผ่านเกณฑ์คุณภาพขึ้นก่อน (เรียงตามคะแนนคุณภาพ) ตามด้วยตัวที่ยังไม่ผ่าน (เรียงตามคะแนนระบบ)"""
    return sorted(rows, key=lambda r: (not r["passed"], -(r["confidence"] if r["passed"] else r["score"])))[:n]


def scan_picks(md, fng: dict | None, watch: list[str]) -> list[dict]:
    """Scanner (Volume สูงสุด 80 เหรียญ: Momentum · Trend · Breakout · รูปแบบกราฟ · order flow)
    → วิเคราะห์ละเอียด (โครงสร้าง · S/R · Backtest · สถิติ TP ก่อน SL) → Quality gate เดียวกับสัญญาณในแอป"""
    from analysis import analyze_coin, macro_agent
    from config import QUOTE
    from market import add_indicators, closed_only, fetch_frames
    from pipeline import scan_universe
    from signals import build_signal

    scan = scan_universe(md, {"BTC"}, N_CANDIDATES, 20e6)
    targets = list(dict.fromkeys(scan["picks"] + [f"{w}{QUOTE}" for w in watch]))  # BTC วิเคราะห์ได้ถ้าอยู่ใน watchlist
    syms = list(dict.fromkeys([f"BTC{QUOTE}"] + targets))
    fr = fetch_frames(md, syms, ("4h", "1d", "1w"))
    b = fr.get(f"BTC{QUOTE}", {})
    if "1d" not in b or "1w" not in b:
        return []
    btc_d, btc_w = closed_only(add_indicators(b["1d"])), closed_only(add_indicators(b["1w"]))
    macro = macro_agent(btc_d, btc_w, {}, fng)
    rows = []
    for s in targets:
        try:
            c = analyze_coin(s, fr.get(s, {}), btc_d, macro, {})
            if not c:
                t = s[: -len(QUOTE)]
                if t in watch:  # เหรียญของคุณที่วิเคราะห์ไม่ได้ ต้องบอก ไม่ใช่หายไปเฉย ๆ
                    n1d = len(fr.get(s, {}).get("1d", []))
                    rows.append({"ticker": t, "watch": True, "nodata": True, "passed": False, "score": 0, "confidence": 0,
                                 "why": f"ข้อมูลรายวันบน Binance Spot มี {n1d} แท่ง ไม่พอวิเคราะห์" if n1d
                                 else "ไม่พบคู่ USDT บน Binance Spot"})
                continue
            sig, fails = build_signal(c, macro)
            m = c["med"]
            rows.append({"ticker": c["ticker"], "price": c["price"], "passed": sig is not None,
                         "confidence": sig.confidence if sig else 0.0, "score": c["score"],
                         "win_prob": sig.win_prob if sig else None, "fails": fails[:2],
                         "tags": (scan["tags"].get(s) or [])[:2], "watch": c["ticker"] in watch,
                         "plan": {"date": "", "price": c["price"],
                                  "med": {"status": m["status"], "zone": [float(m["zone"][0]), float(m["zone"][1])],
                                          "sl": float(m["sl"]), "tp1": float(m["tp1"]), "tp2": float(m["tp2"])},
                                  "long": {}}})
        except Exception as e:  # noqa: BLE001 — เหรียญเดียวพังไม่ทำให้ทั้งรอบล้ม
            print(f"วิเคราะห์ {s} ไม่สำเร็จ: {e}", file=sys.stderr)
    return rows


def split_picks(rows: list[dict], watch: list[str]) -> tuple[list[dict], list[dict]]:
    """(เหรียญของคุณ ตามลำดับ watchlist, 5 อันดับจากตลาดที่ไม่ซ้ำกับ watchlist)"""
    mine = sorted([r for r in rows if r["watch"]], key=lambda r: watch.index(r["ticker"]) if r["ticker"] in watch else 99)
    return mine, rank_picks([r for r in rows if not r["watch"]])


def picks_block(picks: list[dict], state: str, title: str = "🏆 เหรียญน่าสนใจจากตลาด",
                note: str = "คัดจาก 80 เหรียญที่ซื้อขายมากสุด ด้วย Momentum · Trend · Breakout · Backtest") -> list[str]:
    if not picks:
        return []
    n_ok = sum(p["passed"] for p in picks)
    L = ["", "━━━━━━━━━━", f"{title} (ผ่านเกณฑ์ {n_ok}/{len(picks)})"] + ([note] if note else [])
    for i, p in enumerate(picks, 1):
        if p.get("nodata"):
            L += ["", f"{i}. {p['ticker']}: วิเคราะห์ไม่ได้ ❔", f"   {p['why']}"]
            continue
        m = p["plan"]["med"]
        lo, hi = m["zone"]
        L += ["", f"{i}. {p['ticker']}: {pick_verdict(p['passed'], state)}"]
        where = "อยู่ในโซน" if m["sl"] < p["price"] <= hi else ("สูงกว่าโซน ตั้งรอซื้อ" if p["price"] > hi else "ต่ำกว่าโซน")
        L.append(f"   ราคา {_p(p['price'])} · โซนซื้อ {_p(lo)}–{_p(hi)} ({where})")
        L.append(f"   ขายทำกำไร {_p(m['tp1'])} · ตัดขาดทุนถ้าต่ำกว่า {_p(m['sl'])}")
        if p["passed"]:
            extra = f" · โอกาสถึงเป้าก่อนตัดขาดทุน {p['win_prob']:.0f}% (สถิติย้อนหลัง)" if p["win_prob"] else ""
            L.append(f"   คะแนนคุณภาพ {p['confidence']:.0f}/100{extra}")
        else:
            L.append(f"   ยังไม่ผ่าน: {' · '.join(p['fails']) or '-'}")
        if p["tags"]:
            L.append(f"   จุดเด่น: {' · '.join(p['tags'])}")
    if n_ok == 0:
        L += ["", "ยังไม่มีตัวไหนผ่านเกณฑ์ครบ → ตั้งรอที่โซนซื้อ ระบบจะแจ้งเมื่อผ่าน"]
    return L



# ═══════════════════════ data fetch (เครือข่าย) ═══════════════════════
def fetch_usdt_dom() -> float | None:
    key = os.environ.get("COINGECKO_DEMO_KEY", "")
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=20,
                         headers={"x-cg-demo-api-key": key} if key else {})
        r.raise_for_status()
        v = r.json()["data"]["market_cap_percentage"].get("usdt")
        return float(v) if v is not None else None
    except Exception as e:  # noqa: BLE001
        print(f"USDT.D ดึงไม่สำเร็จ: {e}", file=sys.stderr)
        return None


def _coinalyze_get(url: str, hdr: dict, params: dict | None = None, tries: int = 4):
    """GET พร้อมรอตาม Retry-After เมื่อโดนจำกัด (429) — ฟรี 40 ครั้ง/นาที/key"""
    import time
    for i in range(tries):
        r = requests.get(url, headers=hdr, params=params, timeout=30)
        if r.status_code == 429 and i < tries - 1:
            time.sleep(min(float(r.headers.get("Retry-After") or 10) + 1, 65))
            continue
        r.raise_for_status()
        return r.json()
    return None


def fetch_oi_ratio() -> float | None:
    """alt OI (ไม่รวม BTC, ETH) / BTC OI จาก Coinalyze — เฉพาะ perpetual USDT บน Binance (exchange code A) หน่วย USD
    ส่งทีละ 20 symbols และเว้น 1.6 วินาที/ครั้ง ให้อยู่ใต้เพดาน 40 ครั้ง/นาที"""
    import time
    key = os.environ.get("COINALYZE_API_KEY", "")
    if not key:
        return None
    base, hdr = "https://api.coinalyze.net/v1", {"api_key": key}
    try:
        markets = _coinalyze_get(f"{base}/future-markets", hdr) or []
        syms = [(m["symbol"], (m.get("base_asset") or "").upper()) for m in markets
                if m.get("is_perpetual") and m.get("exchange") == "A" and (m.get("quote_asset") or "").upper() == "USDT"]
        asset = dict(syms)
        names = [s for s, _ in syms]
        btc = alt = 0.0
        for i in range(0, len(names), 20):
            time.sleep(1.6)
            rows = _coinalyze_get(f"{base}/open-interest", hdr,
                                  {"symbols": ",".join(names[i:i + 20]), "convert_to_usd": "true"}) or []
            for row in rows:
                a, v = asset.get(row.get("symbol"), ""), float(row.get("value") or 0)
                if a == "BTC":
                    btc += v
                elif a != "ETH":
                    alt += v
        print(f"Coinalyze: {len(names)} สัญญา · BTC OI ${btc / 1e9:.2f}b · Alt OI ${alt / 1e9:.2f}b")
        return alt / btc if btc > 0 else None
    except Exception as e:  # noqa: BLE001
        print(f"Coinalyze ดึงไม่สำเร็จ: {e}", file=sys.stderr)
        return None


def fetch_btc_regime(md) -> str | None:
    """Bull = ปิดเหนือ EMA200 และ EMA50 > EMA200 · Bear = กลับกัน · นอกนั้น Neutral (แท่งที่ปิดแล้วเท่านั้น)"""
    from market import closed_only, ema
    try:
        d = closed_only(md.ohlcv("BTCUSDT", "1d"))
        c = d["close"]
        e50, e200 = ema(c, 50).iloc[-1], ema(c, 200).iloc[-1]
        if math.isnan(e200):
            return None
        last = c.iloc[-1]
        if last > e200 and e50 > e200:
            return "Bull"
        if last < e200 and e50 < e200:
            return "Bear"
        return "Neutral"
    except Exception as e:  # noqa: BLE001
        print(f"BTC regime คำนวณไม่สำเร็จ: {e}", file=sys.stderr)
        return None


# ═══════════════════════ main ═══════════════════════
def main() -> None:
    from market import MarketData
    from sources import fear_greed
    from daily_scan import send_telegram  # ใช้ตัวส่งเดียวกับสแกนประจำวัน (มี retry แล้ว)

    tg_token, tg_chat = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    force = os.environ.get("FORCE_SEND", "") in ("1", "true", "yes")
    now = datetime.now(TH_TZ)
    day = now.strftime("%Y-%m-%d")

    state = load_json(STATE_FILE, {})
    md = MarketData()

    usdt = fetch_usdt_dom()
    if usdt is not None:
        state["usdt_hist"] = upsert_daily(state.get("usdt_hist", []), day, usdt)
    ratio = fetch_oi_ratio()
    if ratio is not None:
        state["oi_hist"] = upsert_daily(state.get("oi_hist", []), day, ratio, keep=60)
    hist_before_today = [x for x in state.get("usdt_hist", []) if x[0] != day]
    fng = fear_greed()
    votes = [usdt_vote(usdt, hist_before_today + ([[day, usdt]] if usdt is not None else [])),
             oi_vote(ratio, state.get("oi_hist", [])),
             fng_vote(fng),
             regime_vote(fetch_btc_regime(md))]
    st, b, s, n = market_state(votes)
    prev = state.get("market_state")

    plans = load_json(PLANS_FILE, {})
    sent = state.get("sent", {})
    events = []
    if plans:
        try:
            tk = md.tickers()
            prices = {sym[:-4]: float(tk.loc[sym, "lastPrice"]) for sym in tk.index if sym[:-4] in plans}
            events = coin_events(plans, prices, sent, now, st)
        except Exception as e:  # noqa: BLE001
            print(f"เช็กราคารายเหรียญไม่สำเร็จ: {e}", file=sys.stderr)

    # เหรียญแนะนำจากตลาด — เหรียญที่ "เพิ่งผ่านเกณฑ์" นับเป็นเรื่องใหม่ที่ต้องแจ้ง (ครั้งเดียวต่อวัน)
    watch = [w.strip().upper() for w in os.environ.get("SCAN_WATCH", "").split(",") if w.strip()]
    rows: list[dict] = []
    try:
        rows = scan_picks(md, fng, watch)
    except Exception as e:  # noqa: BLE001 — Scanner พังไม่ทำให้การแจ้งเตือนหลักล้ม
        print(f"คัดเหรียญแนะนำไม่สำเร็จ: {e}", file=sys.stderr)
    mine, picks = split_picks(rows, watch)
    new_picks = [(f"pick:{p['ticker']}:{day}", p) for p in mine + picks
                 if p["passed"] and f"pick:{p['ticker']}:{day}" not in sent]
    for _, p in new_picks:  # ให้ระบบเฝ้าโซน/เป้า/จุดตัดขาดทุนของเหรียญที่ผ่านเกณฑ์ต่อในรอบถัดไป
        plans[p["ticker"]] = {**p["plan"], "date": day}
    if new_picks:
        save_json(PLANS_FILE, plans)

    changed = prev is not None and prev != st
    first_run = prev is None
    print("\n".join(v["text"] for v in votes), f"\nสถานะ {st} (เดิม {prev}) · เหตุการณ์รายเหรียญ {len(events)}",
          f"\nเหรียญของคุณ: {', '.join(p['ticker'] + ('✅' if p['passed'] else '') for p in mine) or '-'}",
          f"\nเหรียญแนะนำ: {', '.join(p['ticker'] + ('✅' if p['passed'] else '') for p in picks) or '-'}")

    if n == 0:
        print("ไม่มีข้อมูลตัวชี้วัดเลย — ไม่ส่งและไม่เปลี่ยนสถานะ", file=sys.stderr)
    elif force or changed or first_run or events or new_picks:
        msg = build_message(st, prev, b, s, n, votes, events, now, picks, mine)
        if not send_telegram(tg_token, tg_chat, msg):
            print("ส่ง Telegram ไม่สำเร็จ — ไม่บันทึกว่าส่งแล้ว", file=sys.stderr)
            save_json(STATE_FILE, state)  # ยังเก็บประวัติตัวชี้วัด แต่ไม่ mark ว่าส่งแล้ว
            sys.exit(1)
        state["market_state"] = st
        for k, _ in events + new_picks:
            sent[k] = day
    else:
        print("สถานะไม่เปลี่ยนและไม่มีเหตุการณ์ใหม่ — ไม่ส่งข้อความ")

    cutoff = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    state["sent"] = {k: v for k, v in sent.items() if v >= cutoff}  # ล้าง key เก่ากันไฟล์โต
    state["updated"] = now.isoformat(timespec="minutes")
    save_json(STATE_FILE, state)


if __name__ == "__main__":
    main()

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
STATE_TXT = {"🟢": "โซนสะสม (ทยอยซื้อได้ตามแผน)", "🟡": "สัญญาณผสม (ทำตามแผนรายเหรียญ ไม่ไล่ราคา)",
             "🔴": "ตลาดร้อน/เสี่ยง (ลดสัดส่วน altcoin · ทำแผนขาย)"}


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
def usdt_vote(value: float | None, hist: list) -> dict:
    if value is None:
        return {"vote": None, "ok": False, "text": "USDT.D: ไม่มีข้อมูล"}
    vals = [v for _, v in hist]
    if len(vals) >= MIN_HISTORY:
        pct = sum(v <= value for v in vals) / len(vals)
        vote = "buy" if pct >= 0.8 else ("sell" if pct <= 0.2 else None)
        basis = f"percentile {pct * 100:.0f}% ของ {len(vals)} วัน"
    else:
        vote = "buy" if value >= USDT_FALLBACK[0] else ("sell" if value <= USDT_FALLBACK[1] else None)
        basis = f"เกณฑ์ชั่วคราว ≥{USDT_FALLBACK[0]}%/≤{USDT_FALLBACK[1]}% (เก็บประวัติแล้ว {len(vals)}/{MIN_HISTORY} วัน)"
    return {"vote": vote, "ok": True, "text": f"USDT.D {value:.2f}% · {basis}"}


def oi_vote(ratio: float | None, oi_hist: list) -> dict:
    """ratio = alt OI / BTC OI · oi_hist = [[day, ratio], ...] รวมวันนี้แล้ว"""
    if ratio is None:
        return {"vote": None, "ok": False, "text": "Alt OI/BTC OI: ไม่มีข้อมูล (ยังไม่ได้ใส่ COINALYZE_API_KEY หรือดึงไม่สำเร็จ)"}
    streak = 0
    for _, r in reversed(oi_hist):
        if r > 1:
            streak += 1
        else:
            break
    recent = [r for _, r in oi_hist[-4:-1]]  # 3 วันก่อนหน้า
    peak = max(recent) if recent else None
    if streak >= OI_STREAK_DAYS:
        vote, why = "sell", f"alt > BTC ติดต่อกัน {streak} วัน"
    elif peak and peak > 0 and ratio <= peak * (1 - OI_FLUSH_DROP):
        vote, why = "buy", f"ลดลง {(1 - ratio / peak) * 100:.0f}% จากจุดสูงสุด 3 วัน (leverage ถูกล้าง)"
    else:
        vote, why = None, (f"alt > BTC มา {streak} วัน" if streak else "alt < BTC")
    return {"vote": vote, "ok": True, "text": f"Alt OI/BTC OI {ratio:.2f}x · {why}"}


def fng_vote(fng: dict | None) -> dict:
    if not fng:
        return {"vote": None, "ok": False, "text": "Fear & Greed: ไม่มีข้อมูล"}
    v = fng["value"]
    vote = "buy" if v <= 20 else ("sell" if v >= 75 else None)
    return {"vote": vote, "ok": True, "text": f"Fear & Greed {v} ({fng['label']})"}


def regime_vote(regime: str | None) -> dict:
    if not regime:
        return {"vote": None, "ok": False, "text": "BTC regime: ไม่มีข้อมูล"}
    vote = {"Bull": "buy", "Bear": "sell"}.get(regime)
    return {"vote": vote, "ok": True, "text": f"BTC regime {regime}"}


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


def coin_events(plans: dict, prices: dict, sent: dict, today: datetime) -> list[tuple[str, str]]:
    """คืน [(key, ข้อความ)] ของเหตุการณ์ใหม่ที่ยังไม่เคยส่ง · key ผูกกับวันที่ของแผน จึงส่งครั้งเดียวต่อแผน"""
    out = []
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
                ("sl", px <= m["sl"], f"🛑 {t} หลุด SL {m['sl']:.6g} (ราคา {px:.6g}) → ทำตามแผนตัดขาดทุน"),
                ("tp2", px >= m["tp2"], f"🚀 {t} แตะ TP2 {m['tp2']:.6g} (ราคา {px:.6g}) → ขายส่วนที่เหลือตามแผน"),
                ("tp1", px >= m["tp1"], f"✅ {t} แตะ TP1 {m['tp1']:.6g} (ราคา {px:.6g}) → ขายบางส่วน + เลื่อน SL มาที่ทุน"),
                ("zone", m["sl"] < px <= hi, f"🟦 {t} เข้าโซนซื้อ {lo:.6g}–{hi:.6g} (ราคา {px:.6g}) · SL {m['sl']:.6g} · TP1 {m['tp1']:.6g}"),
            ]
        if lg and lg.get("status") != "🔴":
            for i, tier in enumerate(lg.get("tiers") or []):
                checks.append((f"dca{i + 1}", px <= tier, f"🏦 {t} ถึงไม้ DCA {i + 1} ที่ {tier:.6g} (ราคา {px:.6g})"))
            if lg.get("stop"):
                checks.append(("lstop", px <= lg["stop"],
                               f"⚠️ {t} ต่ำกว่าจุดตัดขาดทุนระยะยาว {lg['stop']:.6g} — ยืนยันด้วยแท่งปิดรายสัปดาห์ก่อนขาย"))
        for name, hit, text in checks:
            key = f"{t}:{p['date']}:{name}"
            if hit and key not in sent:
                out.append((key, text))
    return out


def build_message(state: str, prev: str | None, b: int, s: int, n: int, votes: list[dict],
                  events: list[tuple[str, str]], now: datetime) -> str:
    L = [f"🔔 Crypto Alert {now:%d/%m %H:%M}"]
    head = f"ตลาดรวม: {state} {STATE_TXT[state]}"
    if prev and prev != state:
        head += f"  (เปลี่ยนจาก {prev})"
    L += [head, f"ฝั่งซื้อ {b} · ฝั่งขาย {s} · จากตัวชี้วัดที่มีข้อมูล {n}/{len(votes)}"]
    L += [f"• {v['text']}" + {"buy": " → ฝั่งซื้อ", "sell": " → ฝั่งขาย"}.get(v["vote"], "") for v in votes]
    if events:
        L += ["", "🎯 รายเหรียญ (แผนจากสแกนประจำวัน)"] + [f"• {txt}" for _, txt in events]
    L += ["", "⚠️ ข้อมูลประกอบการตัดสินใจ ไม่ใช่คำแนะนำการลงทุน"]
    return "\n".join(L)


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


def fetch_oi_ratio() -> float | None:
    """alt OI (ไม่รวม BTC, ETH) / BTC OI จาก Coinalyze — เฉพาะ perpetual บน Binance (exchange code A) หน่วย USD"""
    key = os.environ.get("COINALYZE_API_KEY", "")
    if not key:
        return None
    base, hdr = "https://api.coinalyze.net/v1", {"api_key": key}
    try:
        mk = requests.get(f"{base}/future-markets", headers=hdr, timeout=30)
        mk.raise_for_status()
        syms = [(m["symbol"], (m.get("base_asset") or "").upper()) for m in mk.json()
                if m.get("is_perpetual") and m.get("exchange") == "A"]
        asset = dict(syms)
        btc = alt = 0.0
        names = [s for s, _ in syms]
        for i in range(0, len(names), 20):  # สูงสุด 20 symbols ต่อครั้ง · จำกัด 40 ครั้ง/นาที
            r = requests.get(f"{base}/open-interest", headers=hdr, timeout=30,
                             params={"symbols": ",".join(names[i:i + 20]), "convert_to_usd": "true"})
            r.raise_for_status()
            for row in r.json():
                a, v = asset.get(row.get("symbol"), ""), float(row.get("value") or 0)
                if a == "BTC":
                    btc += v
                elif a != "ETH":
                    alt += v
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
    votes = [usdt_vote(usdt, hist_before_today + ([[day, usdt]] if usdt is not None else [])),
             oi_vote(ratio, state.get("oi_hist", [])),
             fng_vote(fear_greed()),
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
            events = coin_events(plans, prices, sent, now)
        except Exception as e:  # noqa: BLE001
            print(f"เช็กราคารายเหรียญไม่สำเร็จ: {e}", file=sys.stderr)

    changed = prev is not None and prev != st
    first_run = prev is None
    print("\n".join(v["text"] for v in votes), f"\nสถานะ {st} (เดิม {prev}) · เหตุการณ์รายเหรียญ {len(events)}")

    if n == 0:
        print("ไม่มีข้อมูลตัวชี้วัดเลย — ไม่ส่งและไม่เปลี่ยนสถานะ", file=sys.stderr)
    elif force or changed or first_run or events:
        msg = build_message(st, prev, b, s, n, votes, events, now)
        if not send_telegram(tg_token, tg_chat, msg):
            print("ส่ง Telegram ไม่สำเร็จ — ไม่บันทึกว่าส่งแล้ว", file=sys.stderr)
            save_json(STATE_FILE, state)  # ยังเก็บประวัติตัวชี้วัด แต่ไม่ mark ว่าส่งแล้ว
            sys.exit(1)
        state["market_state"] = st
        for k, _ in events:
            sent[k] = day
    else:
        print("สถานะไม่เปลี่ยนและไม่มีเหตุการณ์ใหม่ — ไม่ส่งข้อความ")

    cutoff = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    state["sent"] = {k: v for k, v in sent.items() if v >= cutoff}  # ล้าง key เก่ากันไฟล์โต
    state["updated"] = now.isoformat(timespec="minutes")
    save_json(STATE_FILE, state)


if __name__ == "__main__":
    main()

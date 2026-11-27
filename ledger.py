"""Signal-outcome ledger — บันทึกทุกสัญญาณที่ผ่านเกณฑ์ (quality_gate ใน signals.py) แล้วเช็คย้อนกลับว่า
ราคาจริงไปแตะ TP ก่อน SL ไหม เพื่อวัด "ความแม่นจริง" ของคะแนนคุณภาพ แยกจาก backtest เชิงทฤษฎีที่มีอยู่แล้ว

วิธีทำงาน (ไม่ต้องให้ผู้ใช้ทำอะไรเอง — รันอัตโนมัติผ่าน GitHub Actions ที่มีอยู่แล้ว):
  1. scripts/daily_scan.py เรียก record_signals() ทุกวัน → บันทึกสัญญาณใหม่ (เฉพาะเหรียญที่ยังไม่มีสัญญาณค้างอยู่)
  2. scripts/cycle_alert.py เรียก resolve() ทุก 4 ชม. ตอนเช็กราคาอยู่แล้ว → ปิดสัญญาณเมื่อแตะ SL/TP1/TP2 จริง
  3. expire_stale() ปิดสัญญาณที่ค้างนานเกินไปโดยไม่ถึง TP/SL เป็น "expired" (ตลาดไม่ไปทางไหนชัดเจน ไม่นับแพ้/ชนะ)
  4. app.py อ่านไฟล์เดียวกันมาแสดงอัตราชนะจริงแยกตามช่วงคะแนนคุณภาพ

ไฟล์ data/signal_log.json ถูก commit กลับเข้า repo โดย workflow เดิม (ขั้นตอน "Save state"/"Save plans"
ที่มีอยู่แล้วทำ `git add data/` ครอบคลุมไฟล์นี้โดยอัตโนมัติ ไม่ต้องแก้ .yml)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

LEDGER_FILE = Path(__file__).resolve().parent / "data" / "signal_log.json"
MAX_AGE_DAYS = 45   # สัญญาณที่เปิดค้างนานกว่านี้โดยไม่ถึง TP/SL → ถือว่า "หมดอายุ" ไม่นับแพ้/ชนะ
OUTCOMES_WIN = ("tp1", "tp2")


def load_ledger(path: Path = LEDGER_FILE) -> list[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_ledger(rows: list[dict], path: Path = LEDGER_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def record_signals(ledger: list[dict], signals: list, day: str) -> int:
    """เพิ่มสัญญาณใหม่เข้าทะเบียน (แก้ ledger ในที่ — in place) คืนจำนวนที่เพิ่ม
    ข้ามเหรียญที่มีสัญญาณเปิดค้างอยู่แล้ว กันนับซ้ำทุกวันที่เหรียญเดิมยังอยู่ในโซนเดิม"""
    open_tickers = {r["ticker"] for r in ledger if r.get("outcome") is None}
    added = 0
    for sig in signals:
        if sig.ticker in open_tickers:
            continue
        ledger.append({
            "id": sig.id, "ticker": sig.ticker, "date": day,
            "confidence": sig.confidence, "win_prob": sig.win_prob,
            "price": sig.price_at_signal, "entry_lo": sig.entry_lo, "entry_hi": sig.entry_hi,
            "sl": sig.sl, "tp1": sig.tp1, "tp2": sig.tp2, "rr1": sig.rr1,
            "outcome": None, "outcome_date": None, "outcome_price": None,
        })
        open_tickers.add(sig.ticker)
        added += 1
    return added


def resolve(ledger: list[dict], ticker: str, outcome: str, price: float, day: str) -> bool:
    """ปิดสัญญาณที่เปิดค้างอยู่ล่าสุดของเหรียญนี้ด้วยผลที่เกิดขึ้นจริง (tp1/tp2/sl) — คืน True ถ้าปิดสำเร็จ
    (False = ไม่มีสัญญาณเปิดค้างของเหรียญนี้ในทะเบียน เช่น แผนนี้ไม่เคยผ่านเกณฑ์คุณภาพมาก่อน)"""
    for r in reversed(ledger):
        if r["ticker"] == ticker and r.get("outcome") is None:
            r["outcome"], r["outcome_date"], r["outcome_price"] = outcome, day, price
            return True
    return False


def expire_stale(ledger: list[dict], today: datetime, max_age_days: int = MAX_AGE_DAYS) -> int:
    """ปิดสัญญาณที่เปิดค้างนานเกิน max_age_days โดยไม่เคยแตะ TP/SL ว่า 'expired' — คืนจำนวนที่ปิด"""
    n = 0
    for r in ledger:
        if r.get("outcome") is not None:
            continue
        try:
            age = (today.date() - datetime.fromisoformat(r["date"]).date()).days
        except (KeyError, ValueError):
            continue
        if age > max_age_days:
            r["outcome"], r["outcome_date"] = "expired", today.strftime("%Y-%m-%d")
            n += 1
    return n


BUCKETS = [(0, 60, "<60"), (60, 75, "60-75"), (75, 90, "75-90"), (90, 101, "90+")]


def _r_multiple(r: dict) -> float | None:
    """ผลลัพธ์เป็นหน่วย R (กี่เท่าของความเสี่ยงที่ตั้งไว้) — ใช้เทียบเหรียญราคาต่างกันได้"""
    risk = r["price"] - r["sl"]
    if risk <= 0 or r.get("outcome_price") is None:
        return None
    return (r["outcome_price"] - r["price"]) / risk


def stats_by_bucket(ledger: list[dict]) -> list[dict]:
    """อัตราชนะ (TP ก่อน SL) + R เฉลี่ย แยกตามช่วงคะแนนคุณภาพตอนออกสัญญาณ — นับเฉพาะที่ปิดแล้ว (expired ไม่นับเข้าอัตราชนะ)
    รวม \"คาดไว้ตอนออกสัญญาณ\" (win_prob เฉลี่ยที่ระบบประเมินไว้ ไม่ใช่คะแนนคุณภาพ) ไว้เทียบกับอัตราชนะจริง"""
    out = []
    for lo, hi, label in BUCKETS:
        rows = [r for r in ledger if r.get("outcome") is not None and lo <= r["confidence"] < hi]
        decided = [r for r in rows if r["outcome"] != "expired"]
        wins = [r for r in decided if r["outcome"] in OUTCOMES_WIN]
        r_vals = [v for v in (_r_multiple(r) for r in decided) if v is not None]
        est = [r["win_prob"] for r in decided if r.get("win_prob") is not None]
        out.append({"ช่วงคะแนนคุณภาพ": label, "สัญญาณ": len(rows), "ตัดสินผลแล้ว": len(decided),
                    "หมดอายุ (ไม่นับ)": len(rows) - len(decided),
                    "คาดไว้ตอนออกสัญญาณ": round(sum(est) / len(est), 0) if est else None,
                    "อัตราชนะจริง": round(len(wins) / len(decided) * 100, 0) if decided else None,
                    "R เฉลี่ย": round(sum(r_vals) / len(r_vals), 2) if r_vals else None})
    return out


MIN_SAMPLE_WARN = 15  # ตัวอย่างขั้นต่ำต่อช่วงคะแนนถึงจะเชื่อส่วนต่างได้ — น้อยกว่านี้อาจเป็นแค่ความบังเอิญ
EDGE_GAP_PTS = 10      # ชนะจริงต่ำกว่าที่คาดไว้เกินกี่จุด ถึงจะเตือน (เท่ากับเกณฑ์เดียวกับฟีเจอร์ตรวจ CSV เดิม)


def calibration_warnings(buckets: list[dict], min_sample: int = MIN_SAMPLE_WARN, gap_pts: float = EDGE_GAP_PTS) -> list[str]:
    """เตือน (ข้อความ ไม่ใช่การปรับขนาดไม้อัตโนมัติ) เมื่อช่วงคะแนนคุณภาพช่วงใดมีตัวอย่างพอและชนะจริงต่ำกว่าที่
    ระบบเคยประเมินไว้เกิน gap_pts จุด — เป็นสัญญาณว่าคะแนนช่วงนั้นเริ่มไม่น่าเชื่อถือ ควรพิจารณาลดขนาดไม้เอง
    (ตั้งใจไม่ทำเป็นการลดขนาดไม้อัตโนมัติ เพราะ win rate ที่อิงตัวอย่างน้อยยังผันผวนได้มาก — ดูการวิเคราะห์ในแชต)"""
    out = []
    for b in buckets:
        if b["ตัดสินผลแล้ว"] < min_sample or b["อัตราชนะจริง"] is None or b["คาดไว้ตอนออกสัญญาณ"] is None:
            continue
        gap = b["คาดไว้ตอนออกสัญญาณ"] - b["อัตราชนะจริง"]
        if gap >= gap_pts:
            out.append(f"ช่วงคะแนน {b['ช่วงคะแนนคุณภาพ']}: ชนะจริง {b['อัตราชนะจริง']:.0f}% ต่ำกว่าที่เคยประเมินไว้ "
                       f"{b['คาดไว้ตอนออกสัญญาณ']:.0f}% อยู่ {gap:.0f} จุด (จาก {b['ตัดสินผลแล้ว']} สัญญาณ) — "
                       "ลองพิจารณาลดขนาดไม้สำหรับช่วงคะแนนนี้ลงเอง")
    return out


def overall_stats(ledger: list[dict]) -> dict:
    decided = [r for r in ledger if r.get("outcome") not in (None, "expired")]
    wins = sum(1 for r in decided if r["outcome"] in OUTCOMES_WIN)
    r_vals = [v for v in (_r_multiple(r) for r in decided) if v is not None]
    return {"total": len(ledger), "open": sum(1 for r in ledger if r.get("outcome") is None),
            "decided": len(decided), "win_rate": round(wins / len(decided) * 100, 0) if decided else None,
            "avg_r": round(sum(r_vals) / len(r_vals), 2) if r_vals else None}

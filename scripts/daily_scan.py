"""สแกนประจำวันแบบไม่มีหน้าจอ (headless) — รันผ่าน GitHub Actions แล้วส่งสรุปเข้า Telegram

ใช้ pipeline เดียวกับแอป Streamlit (run_pipeline) แต่ไม่มี UI — อ่านค่าตั้งค่าจาก environment variables:
  GEMINI_API_KEY      จำเป็น (ถ้าไม่มีจะยังรันได้แต่ AI agents จะปิด ใช้ rule-based แทน)
  TELEGRAM_BOT_TOKEN  จำเป็น
  TELEGRAM_CHAT_ID    จำเป็น
  SCAN_WATCH          ไม่จำเป็น (ค่าเริ่มต้น = DEFAULT_WATCH ใน config.py เช่น "SOL")
                       ใส่ได้หลายเหรียญ คั่นด้วยจุลภาค เช่น "SOL,BTC,ETH,HYPE"

v3: pipeline นี้สแกนทั้งตลาดอยู่แล้วในทุกรอบ (ไม่ได้จำกัดแค่เหรียญใน watch list) —
    วนรันทีละเหรียญจะได้รายงานซ้ำเกือบทุกอย่าง เปลืองโควต้า Gemini ฟรีโดยเปล่าประโยชน์
    เปลี่ยนมารันครั้งเดียวต่อวัน แล้ว "หมุนเวียน" เหรียญที่ได้ไปเจาะลึกเป็นโฟกัสทีละตัวตามวันในสัปดาห์
    (เหรียญทุกตัวใน SCAN_WATCH ยังถูกพิจารณาเข้าเกณฑ์คุณภาพเหมือนเดิม ไม่ได้ตกหล่น)
v2: ตรวจ Telegram ก่อนเริ่มสแกนจริง (ping เบา ๆ) และทำให้ job ล้มเหลวชัดเจน (exit code != 0)
    ถ้าส่งเข้า Telegram ไม่สำเร็จเลยสักครั้ง — กัน "สำเร็จ" หลอกแบบที่เจอมาก่อนหน้านี้
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta

# เพิ่ม path โฟลเดอร์หลักของโปรเจกต์ (นอก scripts/) เพื่อให้ import ai.py/config.py/... เจอ
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from ai import make_client
from config import AUTO_MODEL, DEFAULT_WATCH
from market import MarketData
from pipeline import RunConfig, run_pipeline, summary_text

TH_TZ = timezone(timedelta(hours=7))

SECTION_HEADS = ("🧠", "🚦", "⏱", "🏦", "👀")


def format_for_telegram(text: str) -> str:
    """จัดบรรทัดให้อ่านง่ายขึ้นบนมือถือ: หัวข้อตัวหนา + ตัดค่าที่คั่นด้วย '|' ออกเป็นบรรทัดย่อย"""
    import html
    out: list[str] = []
    for raw in text.split("\n"):
        line = html.escape(raw)
        stripped = raw.strip()
        if stripped.startswith(SECTION_HEADS):
            if out and out[-1] != "":
                out.append("")  # เว้นบรรทัดก่อนหัวข้อใหม่ (ยกเว้นถ้าเว้นอยู่แล้ว)
            out.append(f"<b>{line}</b>")
        elif stripped.startswith("•"):
            parts = [p.strip() for p in stripped.split("|")]
            out.append(html.escape(parts[0]))
            for p in parts[1:]:
                out.append(f"    {html.escape(p)}")
        else:
            out.append(line)
    return "\n".join(out)


def _post_with_retry(url: str, data: dict, tries: int = 3, timeout: int = 40):
    """ยิง POST ซ้ำสูงสุด tries ครั้งถ้าเน็ตสะดุด (timeout/connection error) — กัน job แดงเล่นจากเน็ตกระตุกชั่วคราว
    เช่นที่เจอวันนี้ (ส่งถึง Telegram จริงแล้วแต่รอผลตอบกลับไม่ทัน)"""
    last_exc = None
    for attempt in range(1, tries + 1):
        try:
            return requests.post(url, data=data, timeout=timeout)
        except requests.RequestException as e:
            last_exc = e
            print(f"Telegram เรียก API ไม่สำเร็จ (network, ครั้งที่ {attempt}/{tries}): {e}", file=sys.stderr)
            if attempt < tries:
                time.sleep(3 * attempt)
    raise last_exc


def send_telegram(token: str, chat_id: str, text: str, html_mode: bool = False) -> bool:
    """ส่งข้อความเข้า Telegram (ตัดเป็นท่อนละ ≤4000 ตัวอักษร) คืนค่า True ถ้าส่งสำเร็จทุกท่อน"""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    ok_all = True
    for i in range(0, max(len(text), 1), 4000):
        chunk = text[i:i + 4000] or "(ไม่มีเนื้อหา)"
        data = {"chat_id": chat_id, "text": chunk}
        if html_mode:
            data["parse_mode"] = "HTML"
        try:
            r = _post_with_retry(url, data)
        except requests.RequestException as e:
            print(f"Telegram เรียก API ไม่สำเร็จ (network) หลังลองซ้ำแล้ว: {e}", file=sys.stderr)
            ok_all = False
            continue
        if not r.ok and html_mode:
            # กันไว้เผื่อ HTML parse พลาด (อักขระหลุดรูป) — ส่งใหม่แบบ plain text แทนล้มเหลวเงียบ ๆ
            print(f"Telegram HTML ส่งไม่สำเร็จ ({r.status_code}) ลองส่งแบบ plain text แทน: {r.text}", file=sys.stderr)
            try:
                r = _post_with_retry(url, {"chat_id": chat_id, "text": chunk})
            except requests.RequestException as e:
                print(f"Telegram ส่ง plain text แทนก็ไม่สำเร็จ (network): {e}", file=sys.stderr)
                ok_all = False
                continue
        if not r.ok:
            print(f"Telegram ส่งไม่สำเร็จ: HTTP {r.status_code} — {r.text}", file=sys.stderr)
            ok_all = False
    return ok_all


def main() -> None:
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    tg_token = os.environ["TELEGRAM_BOT_TOKEN"]
    tg_chat = os.environ["TELEGRAM_CHAT_ID"]
    watch_raw = os.environ.get("SCAN_WATCH", DEFAULT_WATCH)
    coins = [c.strip().upper() for c in watch_raw.split(",") if c.strip()] or [DEFAULT_WATCH]

    print(f"ตรวจการเชื่อมต่อ Telegram (bot token ลงท้าย ...{tg_token[-6:]}, chat_id={tg_chat})…")
    if not send_telegram(tg_token, tg_chat, "🔧 เริ่มสแกนประจำวัน — ถ้าเห็นข้อความนี้แปลว่า Telegram เชื่อมถูกต้องแล้ว"):
        print("Telegram ส่งข้อความทดสอบไม่สำเร็จ — ตรวจ secret TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID ใหม่อีกรอบ", file=sys.stderr)
        sys.exit(1)

    # หมุนเวียนเหรียญที่จะได้ "โฟกัส" เจาะลึกวันนี้ (ตัวอื่นใน SCAN_WATCH ยังถูกพิจารณาเข้าเกณฑ์ตามปกติ
    # ผ่าน Scanner ที่สแกนทั้งตลาดอยู่แล้ว ไม่ได้ตกหล่นไปจากรายงาน)
    day_idx = datetime.now(TH_TZ).toordinal()
    focus = coins[day_idx % len(coins)]
    print(f"เหรียญโฟกัสวันนี้: {focus} (จากทั้งหมด {coins})")

    md = MarketData()
    client = make_client(gemini_key)
    cfg = RunConfig(watch=[focus], client=client, model=AUTO_MODEL, model_research=AUTO_MODEL)

    try:
        res = run_pipeline(md, cfg, progress=print)
        text = summary_text(res, res["coins"], res["port"])
    except Exception as e:  # noqa: BLE001
        err = f"⚠️ สแกนประจำวันล้มเหลว: {e}\n\n" + traceback.format_exc()[-1500:]
        send_telegram(tg_token, tg_chat, err)
        raise

    pretty = format_for_telegram(text)
    if not send_telegram(tg_token, tg_chat, pretty, html_mode=True):
        print("ส่งรายงานไม่สำเร็จ", file=sys.stderr)
        sys.exit(1)

    print("สแกนประจำวันเสร็จสิ้น")


if __name__ == "__main__":
    main()

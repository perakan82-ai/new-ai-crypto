"""สแกนประจำวันแบบไม่มีหน้าจอ (headless) — รันผ่าน GitHub Actions แล้วส่งสรุปเข้า Telegram

ใช้ pipeline เดียวกับแอป Streamlit (run_pipeline) แต่ไม่มี UI — อ่านค่าตั้งค่าจาก environment variables:
  GEMINI_API_KEY      จำเป็น (ถ้าไม่มีจะยังรันได้แต่ AI agents จะปิด ใช้ rule-based แทน)
  TELEGRAM_BOT_TOKEN  จำเป็น
  TELEGRAM_CHAT_ID    จำเป็น
  SCAN_WATCH          ไม่จำเป็น (ค่าเริ่มต้น = DEFAULT_WATCH ใน config.py เช่น "SOL")
                       ใส่ได้หลายเหรียญ คั่นด้วยจุลภาค เช่น "SOL,BTC,ETH,HYPE"
                       (MAX_WATCH ของ pipeline เจาะลึกได้ครั้งละ 1 เหรียญ จึงวนรันทีละเหรียญแยกกัน)

v2: ตรวจ Telegram ก่อนเริ่มสแกนจริง (ping เบา ๆ) และทำให้ job ล้มเหลวชัดเจน (exit code != 0)
    ถ้าส่งเข้า Telegram ไม่สำเร็จเลยสักครั้ง — กัน "สำเร็จ" หลอกแบบที่เจอมาก่อนหน้านี้
"""
from __future__ import annotations

import os
import sys
import traceback

# เพิ่ม path โฟลเดอร์หลักของโปรเจกต์ (นอก scripts/) เพื่อให้ import ai.py/config.py/... เจอ
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from ai import make_client
from config import AUTO_MODEL, DEFAULT_WATCH
from market import MarketData
from pipeline import RunConfig, run_pipeline, summary_text


def send_telegram(token: str, chat_id: str, text: str) -> bool:
    """ส่งข้อความเข้า Telegram (ตัดเป็นท่อนละ ≤4000 ตัวอักษร) คืนค่า True ถ้าส่งสำเร็จทุกท่อน"""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    ok_all = True
    for i in range(0, max(len(text), 1), 4000):
        chunk = text[i:i + 4000] or "(ไม่มีเนื้อหา)"
        try:
            r = requests.post(url, data={"chat_id": chat_id, "text": chunk}, timeout=20)
        except requests.RequestException as e:
            print(f"Telegram เรียก API ไม่สำเร็จ (network): {e}", file=sys.stderr)
            ok_all = False
            continue
        if not r.ok:
            print(f"Telegram ส่งไม่สำเร็จ: HTTP {r.status_code} — {r.text}", file=sys.stderr)
            ok_all = False
    return ok_all


def scan_one(md: MarketData, client, coin: str) -> str:
    """สแกนเจาะลึก 1 เหรียญ แล้วคืนข้อความสรุป"""
    cfg = RunConfig(watch=[coin], client=client, model=AUTO_MODEL, model_research=AUTO_MODEL)
    res = run_pipeline(md, cfg, progress=print)
    if coin in res.get("missing", []):
        return (f"⚠️ {coin}: ไม่พบคู่เทรด {coin}USDT ในตลาด (อาจสะกดชื่อย่อไม่ตรงกับที่ขึ้นในตลาด "
                f"หรือยังไม่ได้จดทะเบียนคู่ USDT) — ข้ามเหรียญนี้ไป")
    return summary_text(res, res["coins"], res["port"])


def main() -> None:
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    tg_token = os.environ["TELEGRAM_BOT_TOKEN"]
    tg_chat = os.environ["TELEGRAM_CHAT_ID"]
    watch_raw = os.environ.get("SCAN_WATCH", DEFAULT_WATCH)
    coins = [c.strip().upper() for c in watch_raw.split(",") if c.strip()]

    # เช็ก Telegram ก่อนเริ่มสแกนจริง (เร็ว ไม่เปลืองโควต้า Gemini ถ้าตั้งค่าผิด) —
    # ถ้าส่งไม่ได้ตั้งแต่ตรงนี้ ให้ job แดง (exit 1) ทันที จะได้รู้เร็ว ไม่ต้องรอสแกนจบ 2 นาทีแล้วค่อยพบว่าไม่มีอะไรถึงมือเราเลย
    print(f"ตรวจการเชื่อมต่อ Telegram (bot token ลงท้าย ...{tg_token[-6:]}, chat_id={tg_chat})…")
    if not send_telegram(tg_token, tg_chat, "🔧 เริ่มสแกนประจำวัน — ถ้าเห็นข้อความนี้แปลว่า Telegram เชื่อมถูกต้องแล้ว"):
        print("Telegram ส่งข้อความทดสอบไม่สำเร็จ — ตรวจ secret TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID ใหม่อีกรอบ", file=sys.stderr)
        sys.exit(1)

    md = MarketData()
    client = make_client(gemini_key)

    any_sent = False
    all_ok = True
    for coin in coins:
        try:
            text = scan_one(md, client, coin)
        except Exception as e:  # noqa: BLE001 — ให้รู้ผ่าน Telegram ว่าเหรียญนี้ล้มเหลว แล้วไปเหรียญถัดไปต่อ
            err = f"⚠️ สแกน {coin} ล้มเหลว: {e}\n\n" + traceback.format_exc()[-1200:]
            ok = send_telegram(tg_token, tg_chat, err)
            any_sent = any_sent or ok
            all_ok = all_ok and ok
            continue
        ok = send_telegram(tg_token, tg_chat, text)
        any_sent = any_sent or ok
        all_ok = all_ok and ok

    if not any_sent:
        send_telegram(tg_token, tg_chat, "⚠️ ไม่มีเหรียญให้สแกน (ตรวจค่า SCAN_WATCH)")
        sys.exit(1)

    print("สแกนประจำวันเสร็จสิ้น" if all_ok else "สแกนเสร็จ แต่บางเหรียญส่ง Telegram ไม่สำเร็จ")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

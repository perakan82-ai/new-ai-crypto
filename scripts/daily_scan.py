"""สแกนประจำวันแบบไม่มีหน้าจอ (headless) — รันผ่าน GitHub Actions แล้วส่งสรุปเข้า Telegram

ใช้ pipeline เดียวกับแอป Streamlit (run_pipeline) แต่ไม่มี UI — อ่านค่าตั้งค่าจาก environment variables:
  GEMINI_API_KEY      จำเป็น (ถ้าไม่มีจะยังรันได้แต่ AI agents จะปิด ใช้ rule-based แทน)
  TELEGRAM_BOT_TOKEN  จำเป็น
  TELEGRAM_CHAT_ID    จำเป็น
  SCAN_WATCH          ไม่จำเป็น (ค่าเริ่มต้น = DEFAULT_WATCH ใน config.py เช่น "SOL")
"""
from __future__ import annotations

import os
import sys
import traceback

import requests

from ai import make_client
from config import AUTO_MODEL, DEFAULT_WATCH
from market import MarketData
from pipeline import RunConfig, run_pipeline, summary_text


def send_telegram(token: str, chat_id: str, text: str) -> None:
    """ส่งข้อความเข้า Telegram (ตัดเป็นท่อนละ ≤4000 ตัวอักษร ตามขีดจำกัดของ Telegram API)"""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for i in range(0, max(len(text), 1), 4000):
        chunk = text[i:i + 4000] or "(ไม่มีเนื้อหา)"
        r = requests.post(url, data={"chat_id": chat_id, "text": chunk}, timeout=20)
        if not r.ok:
            print(f"Telegram ส่งไม่สำเร็จ: {r.status_code} {r.text}", file=sys.stderr)


def main() -> None:
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    tg_token = os.environ["TELEGRAM_BOT_TOKEN"]
    tg_chat = os.environ["TELEGRAM_CHAT_ID"]
    watch = os.environ.get("SCAN_WATCH", DEFAULT_WATCH)

    md = MarketData()
    client = make_client(gemini_key)
    cfg = RunConfig(watch=[watch], client=client, model=AUTO_MODEL, model_research=AUTO_MODEL)

    try:
        res = run_pipeline(md, cfg, progress=print)
        text = summary_text(res, res["coins"], res["port"])
    except Exception as e:  # noqa: BLE001 — ให้รู้ผ่าน Telegram ว่าสแกนรอบนี้ล้มเหลว ไม่ใช่เงียบหาย
        err = f"⚠️ สแกนประจำวันล้มเหลว: {e}\n\n" + traceback.format_exc()[-1500:]
        send_telegram(tg_token, tg_chat, err)
        raise

    send_telegram(tg_token, tg_chat, text)
    print("ส่งรายงานเข้า Telegram สำเร็จ")


if __name__ == "__main__":
    main()

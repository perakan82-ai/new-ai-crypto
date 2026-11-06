🧠 Crypto AI Intelligence Lab — v5 (เน้นวิเคราะห์)
แอป Streamlit วิเคราะห์คริปโต Spot USDT ด้วยข้อมูล Binance + Gemini ไม่ใช้ Leverage · ไม่ส่งคำสั่งซื้อขาย · เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน
หน้าในแอป
แท็บ
ทำอะไร
🔥 AI คัดเหรียญ
จัดอันดับเหรียญที่มีโอกาสขึ้นใน กรอบเวลาที่เลือก (7–180 วัน) จากสถิติ + รูปแบบกราฟ + ข่าว + อีเวนต์โปรเจกต์ · Scanner คัดเหรียญจากตลาดเพิ่ม (⭐) · AI จัดพอร์ต
🎯 จุดเข้า Spot
โซนซื้อ / SL / TP1-TP2 / R:R · แยก "ผ่านเกณฑ์" กับ "ยังไม่ยืนยัน" · กราฟ + แผนละเอียดรายเหรียญ
💼 พอร์ตของฉัน
ตารางใส่ เหรียญ / จำนวน / ราคาซื้อ (สูงสุด 10) → คัดลอส / ลด / ทยอยขาย / ถือต่อ / ถือยาว / ถัวเพิ่ม + บันไดขาย 3 ไม้ + โอกาสคืนทุน
🔮 Price Prediction
ราคาที่มีโอกาส 70/50/30% แตะ, จุดย่อ, กรอบราคาปลายงวด ใน 7–180 วัน
📐 รูปแบบกราฟ
Double Bottom/Top, Ascending/Descending/Symmetrical Triangle, Bull Flag, Range Breakout + เป้าและโอกาสแตะเป้า
📰 ข่าว & โปรเจกต์
Sentiment ข่าว 7 วัน + AI Research โปรเจกต์ จุดแข็ง Tokenomics อีเวนต์ (Google Search)
⏱ / 🏦 / ⚠️ / 📄
ตารางระยะกลาง-ยาว, Risk & Backtest, รายงานดาวน์โหลด
ไฟล์ (Python 9 ไฟล์ + ชุดทดสอบ วางหน้าแรกของ repo ทั้งหมด)
app.py หน้าแอป · config.py ตั้งค่า · market.py Binance + Indicators · analysis.py โครงสร้าง/รูปแบบกราฟ/Backtest/Prediction/แผน · sources.py พื้นฐาน + ข่าว · ai.py Gemini · signals.py เกณฑ์จุดเข้า · pipeline.py Scanner/พอร์ต/รายงาน · test_all.py ทดสอบ (ไม่ต้องอัปโหลดก็ได้)
requirements.txt
ติดตั้ง
GitHub → Add file → Upload files → เลือกไฟล์ .py ทั้งหมด + requirements.txt → Commit
share.streamlit.io → Create app → Main file app.py → Deploy
App settings → Secrets:
GEMINI_API_KEY = "..."
⚠️ อัปโหลดเป็นไฟล์เสมอ อย่าคัดลอกโค้ดไปวางในหน้าแก้ไข (ย่อหน้าจะเพี้ยน → IndentationError)
หลักการที่ระบบยึด
ราคาทุกตัว (โซนซื้อ, SL, TP, DCA, บันไดขาย, เป้ารูปแบบ) คำนวณจากข้อมูลจริง · AI แค่คัดเลือก อธิบาย และปรับได้ในกรอบที่โค้ดบังคับ
ใช้เฉพาะแท่งที่ปิดแล้ว · Backtest เข้าแท่งถัดไป หักค่าธรรมเนียม + slippage · มี Out-of-sample
คะแนน ≠ ความน่าจะเป็น · โอกาสทุกตัวมาจากสถิติย้อนหลังของเหรียญนั้นและแสดงจำนวนตัวอย่าง
รูปแบบกราฟเป็นการตรวจตามกฎ (เกิดบ่อยแม้ในตลาดสุ่ม) จึงแสดงคู่กับโอกาสแตะเป้าจากสถิติเสมอ
รันในเครื่อง
pip install -r requirements.txt pytest
pytest -q test_all.py
streamlit run app.py

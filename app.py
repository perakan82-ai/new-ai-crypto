"""
🧠 Crypto AI Intelligence Lab v5 — Streamlit (app.py)
=====================================================================
Spot USDT เท่านั้น · ไม่ใช้ Leverage · ไม่ส่งคำสั่งซื้อขาย · เพื่อการศึกษา

หน้าหลัก
  🔥 AI คัดเหรียญ      เหรียญที่มีโอกาสขึ้นในกรอบเวลาที่เลือก (สถิติ + รูปแบบกราฟ + ข่าว + โปรเจกต์)
  🎯 จุดเข้า Spot      โซนซื้อ / SL / TP + เกณฑ์คุณภาพ + กราฟ
  💼 พอร์ตของฉัน      ตารางใส่เหรียญ จำนวน ราคา → ถือต่อ / ถือยาว / ทยอยขาย / คัดลอส + บันไดขาย
  🔮 Price Prediction  ราคาที่มีโอกาสแตะใน 7-180 วัน
  📐 รูปแบบกราฟ        Double Bottom, Triangle, Bull Flag, Breakout …
  📰 ข่าว & โปรเจกต์   Sentiment ข่าว 7 วัน + Research โปรเจกต์/อีเวนต์ (Gemini + Google Search)

Secrets (Streamlit Cloud → App settings → Secrets):
  GEMINI_API_KEY = "..."
  COINGECKO_DEMO_KEY = "..."   # ไม่บังคับ
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

import sources as na
from ai import make_client
from analysis import cap_for
from config import (DEFAULT_WATCH, GEMINI_MODELS, MAJORS, MAX_WATCH, MODEL_FALLBACK, QUOTE, REG_TH,
                    SCAN_DEFAULT_EXTRA, XS_HOLD, XS_TOP, __version__, fp, fpct, fprob)
from market import MarketData
from pipeline import (DIR_EMO, PF_ACT, PF_SEV, PF_SLOTS, RunConfig, analysis_export_df, decision_table, deep_dive_md,
                      evaluate_exports, forecast_md, full_report_md, holding_md, horizon_bucket, opportunity_rank,
                      roadmap_df, rule_verdict, run_pipeline, run_portfolio, size_long, size_medium, summary_text,
                      table_backtest, table_forecast, table_long, table_market_picks, table_medium, table_patterns,
                      trade_plan_table)

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

HORIZONS = {"7 วัน": 7, "14 วัน": 14, "30 วัน": 30, "60 วัน": 60, "90 วัน": 90, "180 วัน": 180}


@st.cache_resource
def market() -> MarketData:
    return MarketData()


def _secret(k: str) -> str:
    try:
        return st.secrets.get(k, "")
    except Exception:  # noqa: BLE001 — ไม่มีไฟล์ secrets
        return ""


def _md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_csv(index=False)


# ═══════════════════════════ chart ═══════════════════════════
def chart(c: dict):
    if go is None:
        return None
    x = c["d1"].iloc[-220:]
    fig = go.Figure(go.Candlestick(x=x.index, open=x.open, high=x.high, low=x.low, close=x.close, name=c["ticker"],
                                   increasing_line_color="#16a34a", decreasing_line_color="#dc2626"))
    for k, col in (("ema21", "#22d3ee"), ("ema50", "#f59e0b"), ("ema200", "#a78bfa")):
        fig.add_trace(go.Scatter(x=x.index, y=x[k], name=k.upper(), line=dict(width=1.2, color=col)))
    m, l = c["med"], c["long"]
    fig.add_hrect(y0=m["zone"][0], y1=m["zone"][1], fillcolor="#22c55e", opacity=0.15, line_width=0)
    lines = [(m["tp1"], "TP1", "#16a34a"), (m["tp2"], "TP2", "#15803d"), (m["sl"], "SL", "#dc2626"),
             (c["vp"]["poc"], "POC", "#94a3b8"), *[(p, f"DCA{i + 1}", "#38bdf8") for i, p in enumerate(l["tiers"])]]
    lines += [(p["target"], f"เป้า {p['name']}", "#eab308") for p in c.get("patterns", [])[:2]]
    for y, t, col in lines:
        fig.add_hline(y=y, line_dash="dot", line_color=col, annotation_text=f"{t} {fp(y)}", annotation_position="right")
    if c["fc"]:
        last = x.index[-1]
        hs = sorted(c["fc"])
        xs = [last] + [last + pd.Timedelta(days=H) for H in hs]
        lo = [c["price"]] + [c["fc"][H]["end25"] for H in hs]
        hi = [c["price"]] + [c["fc"][H]["end75"] for H in hs]
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], fill="toself", fillcolor="rgba(56,189,248,0.12)",
                                 line=dict(width=0), name="กรอบคาดการณ์ 50%"))
        fig.add_trace(go.Scatter(x=xs, y=[c["price"]] + [c["fc"][H]["end50"] for H in hs], name="ค่ากลางคาดการณ์",
                                 line=dict(dash="dash", color="#38bdf8", width=1.2)))
    fig.update_layout(height=480, margin=dict(l=8, r=8, t=28, b=8), xaxis_rangeslider_visible=False,
                      template="plotly_dark", legend=dict(orientation="h", y=1.07))
    return fig


# ═══════════════════════════ 💼 portfolio ═══════════════════════════
def holdings_table() -> list[dict]:
    """ตารางพอร์ต: แก้ไข/เพิ่ม/ลบแถวได้ · โหลด/บันทึก CSV"""
    up = st.file_uploader("📂 โหลดพอร์ตจาก CSV (เหรียญ, จำนวน, ราคาซื้อเฉลี่ย)", type="csv", key="pf_up")
    if up is not None and st.session_state.get("pf_loaded") != f"{up.name}:{up.size}":
        df = pd.read_csv(up)
        st.session_state["pf_df"] = pd.DataFrame({"เหรียญ": df.iloc[:, 0].astype(str).str.upper(),
                                                  "จำนวน": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
                                                  "ราคาซื้อเฉลี่ย (USDT)": pd.to_numeric(df.iloc[:, 2], errors="coerce")})
        st.session_state["pf_loaded"] = f"{up.name}:{up.size}"
    base = st.session_state.get("pf_df")
    if base is None:
        base = pd.DataFrame({"เหรียญ": ["", ""], "จำนวน": [0.0, 0.0], "ราคาซื้อเฉลี่ย (USDT)": [0.0, 0.0]})
    ed = st.data_editor(base, num_rows="dynamic", use_container_width=True, hide_index=True, key="pf_editor",
                        column_config={
                            "เหรียญ": st.column_config.TextColumn(help="เช่น ETH, SOL (ไม่ต้องใส่ USDT)"),
                            "จำนวน": st.column_config.NumberColumn(min_value=0.0, format="%.6f"),
                            "ราคาซื้อเฉลี่ย (USDT)": st.column_config.NumberColumn(min_value=0.0, format="%.6f")})
    out = []
    for _, r in pd.DataFrame(ed).iterrows():
        t = str(r.get("เหรียญ") or "").strip().upper().replace(QUOTE, "").replace("/", "")
        q, p = r.get("จำนวน"), r.get("ราคาซื้อเฉลี่ย (USDT)")
        if t and t != "NAN" and pd.notna(q) and pd.notna(p) and q > 0 and p > 0:
            out.append({"ticker": t, "qty": float(q), "cost": float(p)})
    dedup = {}
    for h in out:  # เหรียญซ้ำ → รวมเป็นต้นทุนเฉลี่ย
        if h["ticker"] in dedup:
            o = dedup[h["ticker"]]
            tot = o["qty"] + h["qty"]
            o["cost"] = (o["cost"] * o["qty"] + h["cost"] * h["qty"]) / tot
            o["qty"] = tot
        else:
            dedup[h["ticker"]] = dict(h)
    holds = list(dedup.values())
    if len(holds) > PF_SLOTS:
        st.warning(f"วิเคราะห์ได้สูงสุด {PF_SLOTS} เหรียญต่อครั้ง — ใช้ {PF_SLOTS} แถวแรก")
        holds = holds[:PF_SLOTS]
    return holds


def render_portfolio(res: dict | None, client, model: str, cg_key: str, use_news: bool):
    st.markdown(f"ใส่เหรียญที่ถือจริง (สูงสุด {PF_SLOTS} เหรียญ) → ระบบบอกว่าควร **✂️ คัดลอส / 🟠 ลดน้ำหนัก / 🟡 ทยอยขาย / "
                "🔵 ถือต่อ / 🏦 ถือยาว / 🟢 ถัวเพิ่ม** พร้อมจุดคัดลอส บันไดขาย 3 ไม้ และโอกาสถึงแต่ละราคา")
    holds = holdings_table()
    b1, b2 = st.columns(2)
    go_btn = b1.button("🧠 วิเคราะห์พอร์ตของฉัน", type="primary", disabled=not holds, use_container_width=True)
    b2.download_button("💾 บันทึกพอร์ต (CSV)", pd.DataFrame(
        [{"เหรียญ": h["ticker"], "จำนวน": h["qty"], "ราคาซื้อเฉลี่ย": h["cost"]} for h in holds]).to_csv(index=False),
        "my_portfolio.csv", "text/csv", use_container_width=True, disabled=not holds)
    if go_btn and holds:
        with st.spinner("💼 Portfolio Agent กำลังวิเคราะห์…"):
            try:
                st.session_state["port_res"] = run_portfolio(market(), holds, res, client, model, cg_key, use_news)
            except Exception as e:  # noqa: BLE001
                st.error(f"วิเคราะห์พอร์ตไม่สำเร็จ: {e}")
    pr = st.session_state.get("port_res")
    if not pr:
        return
    for e in pr["errors"][:6]:
        st.caption(f"⚠️ {e}")
    H = pr["holds"]
    if not H:
        st.warning("ไม่มีเหรียญที่วิเคราะห์ได้")
        return
    tot, cost_tot = sum(h["value"] for h in H), sum(h["qty"] * h["cost"] for h in H)
    k = st.columns(4)
    k[0].metric("มูลค่าพอร์ต", f"${tot:,.0f}")
    k[1].metric("กำไร/ขาดทุน", f"${tot - cost_tot:+,.0f}", fpct(tot / cost_tot - 1) if cost_tot else None)
    k[2].metric("Macro", pr["macro"]["regime"])
    k[3].metric("ต้องคัดลอส/ลด", sum(h["act"] in ("CUT", "REDUCE") for h in H))
    rv = pr["review"]
    if rv:
        st.info(f"**🧠 ภาพรวมพอร์ต:** {rv.overall}")
        if rv.rebalance:
            st.markdown("**ปรับพอร์ต:**\n" + "\n".join(f"- {x}" for x in rv.rebalance))
    st.dataframe(pd.DataFrame([{
        "เหรียญ": h["ticker"], "คำแนะนำ": PF_ACT[h["act"]], "จำนวน": f"{h['qty']:g}", "ทุน": fp(h["cost"]),
        "ราคา": fp(h["price"]), "P/L": f"{fpct(h['pnl'])} ({'+' if h['pnl_usd'] >= 0 else '-'}${abs(h['pnl_usd']):,.0f})",
        "สัดส่วน": f"{h['value'] / tot * 100:.0f}%", "คัดลอส (ปิดรายวัน)": fp(h["stop_d"]),
        "ขาย 1 / 2 / 3": " / ".join(fp(x) for x in h["ladder"]),
        "โอกาสถึงไม้ 1 ใน 90 วัน": fprob(h["probs"][0][90]),
        "คืนทุนใน 180 วัน": "-" if h["pnl"] >= 0 else fprob(h["p_be"][180]),
        "ข่าว": f"{h['news']['emoji']} {h['news']['score']:+.2f}" if h.get("news") else "-",
    } for h in H]), use_container_width=True, hide_index=True)
    heavy = [h["ticker"] for h in H if h["value"] / tot > 0.35 and h["ticker"] not in MAJORS]
    if heavy:
        st.warning(f"กระจุกตัวสูง: {', '.join(heavy)} เกิน 35% ของพอร์ต (altcoin)")
    if len(pr["corr"]) > 1:
        cm = pr["corr"].where(~np.eye(len(pr["corr"]), dtype=bool))
        hi = cm.stack()[lambda s: s > 0.8]
        if len(hi):
            st.warning("เหรียญเคลื่อนไหวเหมือนกันมาก (corr > 0.8): " +
                       ", ".join(sorted({" / ".join(sorted(i)) for i in hi.index})))
    for h in sorted(H, key=lambda h: PF_SEV[h["act"]]):
        st.markdown(holding_md(h))
    st.caption("จุดคัดลอส = ใต้แนวรับ 1.2 ATR (กำไร >15% เลื่อนขึ้นอัตโนมัติ) · โอกาส = สถิติของเหรียญนั้นในสภาพตลาดคล้ายกัน · "
               "AI เปลี่ยนคำแนะนำของระบบได้ไม่เกิน 1 ระดับ")


# ═══════════════════════════ tabs ═══════════════════════════
def render_picks(res: dict, horizon: int, capital: float, risk_pct: float):
    port, coins, research = res["port"], res["coins"], res["research"]["by_ticker"]
    cb = {c["ticker"]: c for c in coins}
    so, ro = port["strategist"], port["risk"]
    if so:
        st.info(f"**🧠 มุมมองตลาด:** {so.market_view}\n\n{so.summary}")
    st.markdown(f"#### 🧭 สรุปการตัดสินใจ: ควรซื้อไหม · โอกาสขึ้น · เทคนิคไหนบวก/ลบ")
    st.dataframe(decision_table(coins, port.get("views") or {}, horizon), use_container_width=True, hide_index=True)
    st.caption("คะแนนควรซื้อ (AI) = ความเห็น AI จากข้อมูลทั้งหมด แต่ระบบจำกัดให้ห่างคะแนนระบบไม่เกิน ±15 · "
               "คะแนนระบบ = เทคนิคเห็นตรงกัน 35% + โอกาสขึ้น 35% + สถานะจุดเข้า 15% + ข่าว/อีเวนต์ 15% · "
               "โอกาสขึ้น = สถิติย้อนหลังของเหรียญนั้นในสภาพตลาดคล้ายวันนี้ ไม่ใช่การรับประกัน")
    st.markdown("#### 🤖 เหรียญที่ AI คัดจากตลาดตอนนี้")
    sc = res.get("scan") or {}
    mp = table_market_picks(coins)
    if len(mp):
        st.caption(f"สแกน {sc.get('universe', 0)} เหรียญ Volume สูง · ตัด spread กว้าง {sc.get('dropped_spread', 0)} เหรียญ · "
                   "จัดอันดับด้วย Momentum ปรับความผันผวนเทียบตลาด + Trend (Donchian 5 lookback) + สัญญาณเทคนิค/รูปแบบกราฟ")
        st.dataframe(mp, use_container_width=True, hide_index=True)
        bt = sc.get("xs_bt") or {}
        if bt.get("excess") is not None:
            k = st.columns(4)
            k[0].metric(f"Top-{XS_TOP} vs ตลาด / {XS_HOLD} วัน", fpct(bt["excess"], 2))
            k[1].metric("ชนะตลาด", fprob(bt["hit"]), f"{bt['n']} รอบ")
            k[2].metric("Out-of-sample", fpct(bt["oos_excess"], 2) if bt.get("oos_excess") is not None else "-")
            k[3].metric("t-stat", f"{bt['t_stat']:.2f}")
            st.caption(f"{bt['verdict']} · Backtest ตั้งแต่ {bt['start']} หักค่าธรรมเนียม+slippage แล้ว · "
                       "⚠️ ใช้รายชื่อเหรียญ Volume สูงของวันนี้ (survivorship bias) ผลจริงอาจแย่กว่านี้")
        elif bt:
            st.caption(f"Backtest การจัดอันดับ: {bt.get('verdict')}")
    else:
        st.caption("ปิด Scanner อยู่ (เลื่อน 'ให้ AI คัดเหรียญจากตลาด' ใน Sidebar)")
    st.markdown(f"#### 🔥 เหรียญที่มีโอกาสขึ้นใน {horizon} วัน (เรียงตามคะแนนโอกาส)")
    st.caption("⭐ = Scanner คัดจากตลาด (Volume สูง + สัญญาณเทคนิค/รูปแบบกราฟ) · ไม่มีดาว = Watchlist ของคุณ")
    st.dataframe(opportunity_rank(coins, horizon, research), use_container_width=True, hide_index=True)
    st.caption("คะแนนโอกาส = โอกาสปิดสูงกว่าวันนี้ 35% + upside เทียบจุดย่อ 20% + สถานะจุดเข้า 15% + รูปแบบกราฟ 10% + "
               "ข่าว/อีเวนต์ 10% + Momentum/Trend 10% · โอกาสมาจากสถิติย้อนหลังของเหรียญนั้น ไม่รับประกันอนาคต")
    st.markdown("#### 🧭 AI คัดเข้าพอร์ต")
    if ro:
        {"APPROVE": st.success, "CAUTION": st.warning, "REJECT": st.error}[ro.verdict](
            f"🛡️ Risk Manager: **{ro.verdict}** — {ro.final_note}")
    rows = []
    for p in port["picks"]:
        c = cb[p["ticker"]]
        if p["horizon"] == "MEDIUM":
            sz = size_medium(c, capital, risk_pct, p["alloc_final"])
            plan = (f"ซื้อ {fp(c['med']['zone'][0])}–{fp(c['med']['zone'][1])} · SL {fp(c['med']['sl'])} · "
                    f"TP {fp(c['med']['tp1'])}/{fp(c['med']['tp2'])}")
            amt, hold = sz["notional"], c["med"]["hold"]
        else:
            sz = size_long(c, capital, p["alloc_final"])
            plan = "DCA " + " / ".join(fp(t) for t in c["long"]["tiers"]) + f" · Stop W<{fp(c['long']['stop'])}"
            amt, hold = sz["budget"], c["long"]["hold"]
        rows.append({"เหรียญ": p["ticker"], "ระยะ": "⏱ กลาง" if p["horizon"] == "MEDIUM" else "🏦 ยาว",
                     "สัดส่วน": f"{p['alloc_final']:.1f}%", "จำนวนเงิน": f"${amt:,.0f}", "ระยะถือ": hold, "แผน": plan,
                     "เหตุผล": p["rationale"], "ความเสี่ยง": p.get("key_risk", "")})
    rows.append({"เหรียญ": "💵 เงินสด", "ระยะ": "-", "สัดส่วน": f"{port['cash']:.1f}%",
                 "จำนวนเงิน": f"${capital * port['cash'] / 100:,.0f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("#### 🧠 ข้อสรุปรายเหรียญ")
    views = port.get("views") or {}
    for c in coins:
        v = views.get(c["ticker"]) or {}
        st.markdown(f"- **{c['ticker']}** {c['med']['status']}/{c['long']['status']} — {v.get('verdict') or rule_verdict(c)}")
    if not so:
        st.caption(f"โหมด: {port['mode']} — ใส่ Gemini API Key เพื่อให้ AI คัดเหรียญและเขียนเหตุผล")


def render_entries(res: dict, capital: float, risk_pct: float):
    coins, port, research = res["coins"], res["port"], res["research"]["by_ticker"]
    sigs = res.get("signals") or []
    st.markdown("#### 📋 แผนเข้า-ออก (ทำอะไร · ที่ราคาไหน · ถึงวันไหน)")
    st.dataframe(trade_plan_table(coins, res["ts"]), use_container_width=True, hide_index=True)
    st.caption("ไม้ 1 = Limit ที่ขอบบนโซน · ไม้ 2 = Limit ที่ขอบล่าง · SL ตั้งเป็น Stop (แตะระหว่างวันก็ออก — ตรงกับ Backtest) · "
               "ถึง TP1 ขาย 50% แล้วเลื่อน SL ไปที่ทุน · ไม่ได้ของภายในวันที่ตั้งรอ = ยกเลิกคำสั่ง")
    st.markdown("#### ✅ จุดเข้าที่ผ่านเกณฑ์คุณภาพทุกข้อ")
    if sigs:
        st.dataframe(pd.DataFrame([{
            "เหรียญ": s.ticker, "ซื้อ (Limit)": f"{fp(s.entry_lo)} – {fp(s.entry_hi)}", "SL": fp(s.sl),
            "TP1 / TP2": f"{fp(s.tp1)} / {fp(s.tp2)}", "R:R": f"1:{s.rr1:.1f}",
            "คะแนนคุณภาพ": f"{s.confidence:.0f}/100", "TP1 ก่อน SL": f"{s.win_prob or 0:.0f}% (n={s.n_eff})",
            "คุ้มทุนที่": f"{s.be_win:.0f}%", "ความเสี่ยง": " · ".join(s.risks) or "-",
        } for s in sorted(sigs, key=lambda s: -s.confidence)]), use_container_width=True, hide_index=True)
        st.caption("คะแนนคุณภาพ ≠ โอกาสชนะ · TP1 ก่อน SL = สถิติย้อนหลังของเหรียญนั้น (n = จำนวนตัวอย่างอิสระ)")
    else:
        st.info("ตอนนี้ยังไม่มีเหรียญที่ผ่านเกณฑ์ครบ — ดูโซนซื้อที่ควรตั้งรอในตารางด้านล่าง")
    with st.expander("👀 ยังไม่ยืนยัน — เหตุผลที่ยังไม่ผ่าน"):
        st.dataframe(pd.DataFrame([{"เหรียญ": c["ticker"], "สถานะ": c["med"]["status"],
                                    "โซนซื้อ": f"{fp(c['med']['zone'][0])} – {fp(c['med']['zone'][1])}",
                                    "ไม่ผ่านเพราะ": " · ".join(c.get("gate_fails") or [])}
                                   for c in coins if c.get("gate_fails")]), use_container_width=True, hide_index=True)
    st.divider()
    views = port.get("views") or {}
    pick_by = {(p["ticker"], p["horizon"]): p for p in port["picks"]}
    sel = st.selectbox("🔍 แผนละเอียดรายเหรียญ", [c["ticker"] for c in coins], key="dd_sel")
    c = {x["ticker"]: x for x in coins}[sel]
    if go is not None:
        st.plotly_chart(chart(c), use_container_width=True)
    st.markdown(deep_dive_md(c, research, views, pick_by, capital, risk_pct, res.get("news")), unsafe_allow_html=True)


def render_prediction(res: dict):
    coins = res["coins"]
    st.markdown("**อ่านอย่างไร:** “50% แตะ ≥ $X ภายใน 30 วัน” = ในอดีตที่สภาพตลาดคล้ายวันนี้ ครึ่งหนึ่งของกรณีราคาขึ้นไปแตะ $X "
                "อย่างน้อยหนึ่งครั้ง (ปรับตามความผันผวนปัจจุบัน) · ตัวอย่างอิสระน้อย = เชื่อถือได้น้อย")
    sel = st.selectbox("เลือกเหรียญ", [c["ticker"] for c in coins], key="fc_sel")
    c = {x["ticker"]: x for x in coins}[sel]
    st.markdown(f"### 🔮 {sel} · ราคาตอนนี้ {fp(c['price'])}\n" + forecast_md(c))
    if c.get("patterns"):
        st.markdown("**เป้าจากรูปแบบกราฟ:** " + " · ".join(
            f"{p['name']} → {fp(p['target'])} (โอกาสแตะใน 60 วัน {fprob(p.get('p_target_60'))})" for p in c["patterns"]))
    with st.expander("ตารางทุกเหรียญ"):
        st.dataframe(table_forecast(coins), use_container_width=True, hide_index=True)


def render_news_project(res: dict):
    research, port = res["research"]["by_ticker"], res["port"]
    if research:
        st.markdown("#### 🏗 โปรเจกต์ & อีเวนต์ (AI Research + Google Search)")
        for t, r in research.items():
            rd = r.get("data") or {}
            with st.expander(f"{t} · Catalyst {rd.get('catalyst_score', '-')}" + ("" if r.get("verified") else " ⚠️ ไม่ได้ค้นเว็บ")):
                st.markdown(f"**โปรเจกต์:** {rd.get('project', '-')}\n\n**จุดแข็ง:** " + " · ".join(rd.get("strengths", [])) +
                            f"\n\n**Tokenomics:** {rd.get('tokenomics', '-')}\n\n🐂 {rd.get('bull_case', '-')}\n\n🐻 {rd.get('bear_case', '-')}")
                if rd.get("news_30d"):
                    st.markdown("**ข่าว 30 วัน**\n" + "\n".join(f"- {n}" for n in rd["news_30d"]))
                evs = rd.get("events") or []
                if evs:
                    st.markdown("**อีเวนต์ข้างหน้า**\n" + "\n".join(
                        f"- `{e['date']}` {e['type']} {e['title']} · {e['impact']} · {e['certainty']}"
                        + (f" ([ที่มา]({e['source_url']}))" if e.get("source_url", "").startswith("http") else " ⚠️ ไม่มีลิงก์")
                        for e in evs))
                srcs = r.get("sources") or []
                if srcs:
                    st.caption("แหล่งข้อมูล: " + " · ".join(f"[{s['title'][:40]}]({s['uri']})" for s in srcs[:6]))
    else:
        st.caption("ใส่ Gemini API Key และเปิด AI Agents เพื่อดูข้อมูลโปรเจกต์และอีเวนต์")
    st.divider()
    nw = res.get("news") or {}
    if nw.get("items"):
        na.render_news_tab(nw["report"], nw["items"], focus=[p["ticker"] for p in port["picks"]] + nw.get("focus", []),
                           src=nw.get("src"), mode=nw.get("mode", "keyword"))
    else:
        st.caption(nw.get("error") or "News Agent ปิดอยู่ หรือยังไม่มีข่าว")


def render_roadmap(res: dict):
    rm = res.get("roadmap") or []
    cal = res.get("macro_cal") or {}
    st.markdown("#### 🗺 Roadmap ข่าว/อีเวนต์ข้างหน้า ที่จะกระทบราคา")
    if not rm:
        st.info("ยังไม่มีข้อมูลอีเวนต์ — ต้องใส่ Gemini API Key และเปิด AI Agents (ค้นด้วย Google Search)")
        if cal.get("errors"):
            st.caption("Log: " + " | ".join(cal["errors"][:3]))
        return
    if go is not None:
        ys = list(dict.fromkeys(r["scope"] for r in rm))
        col = {"POSITIVE": "#16a34a", "NEGATIVE": "#dc2626", "MIXED": "#9ca3af"}
        size = {"HIGH": 22, "MEDIUM": 15, "LOW": 10}
        fig = go.Figure()
        for dname, lab in (("POSITIVE", "บวก +"), ("NEGATIVE", "ลบ −"), ("MIXED", "ไม่แน่นอน ±")):
            xs = [r for r in rm if (r.get("direction") or "MIXED") == dname]
            if not xs:
                continue
            fig.add_trace(go.Scatter(
                x=[r["dt"] for r in xs], y=[r["scope"] for r in xs], mode="markers", name=lab,
                marker=dict(color=col[dname], size=[size.get(r.get("impact"), 10) for r in xs],
                            symbol=["circle" if r.get("certainty") == "CONFIRMED" else "circle-open" for r in xs],
                            line=dict(width=2, color=col[dname])),
                text=[f"{r.get('title')}<br>{r.get('impact')} · {r.get('certainty')}<br>{r.get('why') or ''}" for r in xs],
                hovertemplate="%{x|%d %b %Y}<br>%{text}<extra></extra>"))
        today = pd.Timestamp.now(tz="Asia/Bangkok").tz_localize(None).normalize()
        fig.add_vline(x=today, line_dash="dot", line_color="#38bdf8")
        fig.update_layout(height=max(280, 46 * len(ys) + 120), template="plotly_dark", margin=dict(l=8, r=8, t=30, b=8),
                          yaxis=dict(categoryorder="array", categoryarray=ys[::-1]), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("วงกลมทึบ = ยืนยันแล้ว · วงกลมโปร่ง = คาดการณ์/ข่าวลือ · ขนาด = ผลกระทบ · เส้นฟ้า = วันนี้")
    st.dataframe(roadmap_df(rm), use_container_width=True, hide_index=True,
                 column_config={"ที่มา": st.column_config.LinkColumn()})
    st.caption("ผลที่คาด (+/−) เป็นการประเมินของ AI จากข่าวที่ค้นเจอ · ⚠️ ไม่มีลิงก์ = ควรตรวจสอบเองก่อนใช้ · "
               "อีเวนต์ที่ขึ้นกับผลประกาศ (เช่น FOMC, CPI) ดูคอลัมน์เหตุผลว่าผลแบบไหนเป็นบวก/ลบ")
    st.markdown("**ผลรวมข่าวข้างหน้ารายเหรียญ** (−1 ลบมาก … +1 บวกมาก)")
    st.dataframe(pd.DataFrame([{"เหรียญ": c["ticker"], "ข่าวข้างหน้า": f"{c.get('event_net', 0):+.2f}",
                                "อีเวนต์ของเหรียญนี้": sum(c["ticker"] in r["coins"] for r in rm)}
                               for c in res["coins"]]), use_container_width=True, hide_index=True)


def render_export(res: dict, horizon: int):
    st.markdown("#### 💾 ดาวน์โหลดผลวิเคราะห์ (เก็บไว้ทำ Backtest / Forward test)")
    ex = analysis_export_df(res, horizon)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ ผลวิเคราะห์วันนี้ (.csv)", ex.to_csv(index=False).encode("utf-8-sig"),
                       f"analysis_{res['ts']:%Y%m%d_%H%M}.csv", "text/csv", use_container_width=True)
    full = {"run": ex.to_dict(orient="records"),
            "roadmap": roadmap_df(res.get("roadmap") or []).to_dict(orient="records"),
            "signals": [s.model_dump(mode="json") for s in res.get("signals") or []],
            "macro": {k: res["macro"].get(k) for k in ("regime", "notes", "btc_price")}}
    c2.download_button("⬇️ แบบละเอียด (.json)", json.dumps(full, ensure_ascii=False, default=str, indent=1),
                       f"analysis_{res['ts']:%Y%m%d_%H%M}.json", "application/json", use_container_width=True)
    st.caption(f"1 แถว/เหรียญ: จุดเข้า SL TP คะแนนควรซื้อ โอกาสขึ้น เทคนิคบวก/ลบ ข่าวข้างหน้า ฯลฯ ({len(ex.columns)} คอลัมน์) · "
               "ดาวน์โหลดทุกครั้งที่วิเคราะห์ แล้วอัปโหลดกลับมาด้านล่างเพื่อตรวจกับราคาจริง")
    st.divider()
    st.markdown("#### 🧪 ตรวจผลไฟล์เก่ากับราคาจริง")
    files = st.file_uploader("อัปโหลดไฟล์ .csv ที่เคยดาวน์โหลด (เลือกได้หลายไฟล์)", type="csv", accept_multiple_files=True,
                             key="ev_up")
    if not files:
        return
    try:
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"อ่านไฟล์ไม่ได้: {e}")
        return
    need = {"run_ts_utc", "ticker", "price", "entry_hi", "sl", "tp1"}
    if not need <= set(df.columns):
        st.error(f"ไฟล์ไม่ใช่รูปแบบของแอปนี้ (ขาดคอลัมน์ {', '.join(sorted(need - set(df.columns)))})")
        return
    with st.spinner("ดึงราคาจริงหลังวันที่วิเคราะห์…"):
        ev = evaluate_exports(df, market())
    sm = ev["summary"]
    k = st.columns(5)
    k[0].metric("แผนที่ปิดแล้ว", sm["closed"], f"จาก {sm['rows']} แถว")
    k[1].metric("ชนะจริง", fprob(sm["win_rate"]), (f"ประเมินไว้ {fprob(sm['est_tp_first'])}" if sm["est_tp_first"] else None))
    k[2].metric("R เฉลี่ย", f"{sm['avg_R']:+.2f}R" if sm["avg_R"] is not None else "-")
    k[3].metric("ขึ้นจริงหลังครบกรอบเวลา", fprob(sm["up_rate"]), f"{sm['n_h']} แถว")
    k[4].metric("โอกาสขึ้นที่ประเมินไว้", fprob(sm["est_p_up"]))
    if sm["closed"] >= 10 and sm["win_rate"] is not None and sm["est_tp_first"] and sm["win_rate"] < sm["est_tp_first"] - 0.10:
        st.error("ชนะจริงต่ำกว่าที่ประเมินเกิน 10 จุด → ลดขนาดไม้ลงครึ่งหนึ่งจนกว่าจะกลับมาใกล้เคียง")
    if len(sm["buckets"]):
        st.markdown("**คะแนนควรซื้อสูง ได้ผลดีกว่าจริงไหม?**")
        st.dataframe(sm["buckets"], use_container_width=True, hide_index=True)
    st.dataframe(ev["table"][["run_date_th", "ticker", "outcome", "R", "ret_h", "up_h", "p_up_h", "buy_final",
                              "passed_gate", "entry_hi", "sl", "tp1"]],
                 use_container_width=True, hide_index=True)
    st.caption("ใช้แท่ง 1D ที่เปิดหลังเวลาวิเคราะห์ · แท่งเดียวกันโดนทั้ง SL และ TP นับ SL · ตัวอย่างน้อยกว่า 30 ยังสรุปไม่ได้")


# ═══════════════════════════ main ═══════════════════════════
def main():
    st.set_page_config(page_title="Crypto AI Intelligence Lab", page_icon="🧠", layout="wide")
    st.markdown("<style>.block-container{padding-top:1.2rem;max-width:1500px}"
                "div[data-testid='stMetricValue']{font-size:1.2rem}"
                "blockquote{border-left:4px solid #38bdf8;background:rgba(56,189,248,.07);padding:.4rem .8rem}</style>",
                unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ ตั้งค่า")
        gkey = _secret("GEMINI_API_KEY") or st.text_input("Gemini API Key", type="password")
        cgkey = _secret("COINGECKO_DEMO_KEY") or st.text_input("CoinGecko Demo Key (ไม่บังคับ)", type="password")
        watch_raw = st.text_input(f"เหรียญที่สนใจ (สูงสุด {MAX_WATCH} เหรียญ คั่นด้วย ,)", DEFAULT_WATCH)
        n_watch = len([w for w in watch_raw.split(",") if w.strip()])
        if n_watch > MAX_WATCH:
            st.warning(f"ใส่ {n_watch} เหรียญ → ใช้ {MAX_WATCH} ตัวแรก ที่เหลือให้ AI Scanner ตัดสินจากตลาด")
        hz = st.selectbox("⏳ กรอบเวลาที่สนใจ", list(HORIZONS), index=2)
        n_extra = st.slider("🤖 ให้ AI คัดเหรียญจากตลาดเพิ่ม", 0, 10, SCAN_DEFAULT_EXTRA)
        min_qvol = st.number_input("Volume 24h ขั้นต่ำ (ล้าน USDT)", 5, 2000, 20, 5) * 1e6
        st.divider()
        capital = st.number_input("เงินทุน (USDT)", 100.0, 1e9, 10_000.0, 500.0)
        risk_pct = st.slider("ความเสี่ยงต่อไม้ (% ทุน)", 0.25, 3.0, 1.0, 0.25)
        n_med = st.slider("จำนวนเหรียญระยะกลางสูงสุด", 0, 6, 3)
        n_long = st.slider("จำนวนเหรียญระยะยาวสูงสุด", 0, 6, 3)
        st.divider()
        msel = st.selectbox("🧠 โมเดล AI", GEMINI_MODELS + ["อื่นๆ (พิมพ์เอง)"], index=0)
        model = st.text_input("ชื่อโมเดล", "gemini-2.5-flash") if msel.startswith("อื่น") else msel
        model_research = st.selectbox("📰 โมเดล Research (ค้นเว็บ)", MODEL_FALLBACK, index=0)
        n_research = st.slider("จำนวนเหรียญให้ AI Research โปรเจกต์", 0, 12, 6)
        use_ai = st.toggle("เปิด AI Agents", value=bool(gkey))
        use_news = st.toggle("📰 เปิด News Agent", value=True)
        news_w = st.slider("น้ำหนักข่าวต่อคะแนน (±)", 0, 15, 8)
        run = st.button("🚀 เริ่มวิเคราะห์", type="primary", use_container_width=True)
        if st.button("🗑 ล้าง cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            na.clear_cache()
            for k in ("research_cache", "res", "port_res"):
                st.session_state.pop(k, None)

    horizon = HORIZONS[hz]
    client = make_client(gkey) if use_ai else None
    st.title("🧠 Crypto AI Intelligence Lab")
    st.caption(f"v{__version__} · Spot USDT · ราคาทุกตัวคำนวณจากข้อมูล Binance จริง · AI คัดเลือกและอธิบาย · ใช้เฉพาะแท่งที่ปิดแล้ว")

    if run:
        watch = list(dict.fromkeys(w.strip().upper().replace(QUOTE, "") for w in watch_raw.split(",") if w.strip()))
        cfg = RunConfig(watch=watch, n_extra=n_extra, min_qvol=min_qvol, cg_key=cgkey, client=client, model=model,
                        model_research=model_research, n_research=n_research, n_med=n_med, n_long=n_long,
                        use_news=use_news, news_w=news_w, horizon=horizon,
                        research_cache=st.session_state.setdefault("research_cache", {}))
        with st.status("กำลังประมวลผล Multi-Agent…", expanded=True) as status:
            bar = st.progress(0.0, text="📡 ดึงข้อมูล")
            try:
                st.session_state["res"] = run_pipeline(
                    market(), cfg, progress=lambda s: status.update(label=s),
                    bar=lambda f, t: bar.progress(min(f, 1.0), text=f"📡 ดึงข้อมูล {t}"))
                bar.empty()
                status.update(label="✅ เสร็จสิ้น", state="complete", expanded=False)
            except Exception as e:  # noqa: BLE001
                status.update(label="❌ ผิดพลาด", state="error")
                st.error(f"วิเคราะห์ไม่สำเร็จ: {e}")
                st.stop()

    res = st.session_state.get("res")
    if not res:
        st.info("👈 ตั้งค่าแล้วกด **เริ่มวิเคราะห์** · หรือวิเคราะห์พอร์ตของคุณได้ทันทีด้านล่าง")
        st.subheader("💼 พอร์ตของฉัน")
        render_portfolio(None, client, model, cgkey, use_news)
        return

    coins, macro, port, research = res["coins"], res["macro"], res["port"], res["research"]["by_ticker"]
    pick_by = {(p["ticker"], p["horizon"]): p for p in port["picks"]}
    final_win = {k: v["win_final"] for k, v in pick_by.items()}
    if res["missing"]:
        st.warning(f"ไม่พบคู่ {QUOTE}: {', '.join(res['missing'])}")
    if res.get("watch_dropped"):
        st.caption(f"ไม่ได้วิเคราะห์เป็น Watchlist (เกิน {MAX_WATCH}): {', '.join(res['watch_dropped'])} — "
                   "ถ้าน่าสนใจจริง AI Scanner จะคัดขึ้นมาเอง")
    if res.get("skipped"):
        st.caption("ข้าม: " + " · ".join(res["skipped"][:6]))

    k = st.columns(5)
    k[0].metric("BTC", fp(macro["btc_price"]), fpct(macro["btc_ret30"]) + " 30D")
    k[1].metric("ภาพรวมตลาด", {"Bull": "🟢 Bull", "Bear": "🔴 Bear", "Neutral": "🟡 Neutral"}[macro["regime"]],
                f"{macro['points']}/4")
    k[2].metric("BTC Dominance", f"{res['glob']['btc_dom']:.1f}%" if res["glob"].get("btc_dom") else "-")
    fng = res["fng"]
    k[3].metric("Fear & Greed", f"{fng['value']} {fng['label']}" if fng else "-",
                f"{fng['value'] - fng['d7']:+d} 7D" if fng else None)
    k[4].metric("จุดเข้าผ่านเกณฑ์", len(res["signals"]))
    with st.expander("🌐 ภาพรวมตลาด (Macro)", expanded=False):
        st.markdown("\n".join(f"- {n}" for n in macro["notes"]))
        st.caption(f"Data: {res['host']} · AI: {port['mode']} · ใช้เวลา {res['elapsed']:.1f}s · {res['ts']:%d/%m %H:%M}")

    tabs = st.tabs(["🔥 AI คัดเหรียญ", "🎯 จุดเข้า-ออก", "💼 พอร์ตของฉัน", "🔮 Price Prediction", "📐 รูปแบบกราฟ",
                    "📰 ข่าว & โปรเจกต์", "⏱ ระยะกลาง", "🏦 ระยะยาว", "⚠️ Risk & Backtest", "📄 รายงาน",
                    "🗺 Roadmap ข่าว", "💾 ดาวน์โหลด & ตรวจผล"])
    with tabs[0]:
        render_picks(res, horizon, capital, risk_pct)
    with tabs[1]:
        render_entries(res, capital, risk_pct)
    with tabs[2]:
        render_portfolio(res, client, model, cgkey, use_news)
    with tabs[3]:
        render_prediction(res)
    with tabs[4]:
        tp = table_patterns(coins)
        if len(tp):
            st.dataframe(tp, use_container_width=True, hide_index=True)
        else:
            st.info("ยังไม่พบรูปแบบกราฟเด่นในเหรียญที่วิเคราะห์")
        st.caption("ตรวจจากแท่ง 1D ที่ปิดแล้ว · เป้า = measured move ตามตำรา · โอกาสแตะเป้าคำนวณจากสถิติจริงของเหรียญนั้น "
                   "(รูปแบบกราฟไม่ได้รับประกันผล)")
    with tabs[5]:
        render_news_project(res)
    with tabs[6]:
        st.dataframe(table_medium(coins, research, final_win), use_container_width=True, hide_index=True)
        st.caption("Win rate ระบบ = Backtest 40% + โอกาส TP1 ก่อน SL 40% + Conditional forward 20% · คุ้มทุนรวมค่าธรรมเนียม+slippage")
    with tabs[7]:
        st.dataframe(table_long(coins, research, final_win), use_container_width=True, hide_index=True)
        st.caption("DCA 3 Tier จากแนวรับ/Fib/EMA 1W · ตัดขาดทุนเมื่อ Weekly ปิดต่ำกว่าระดับที่กำหนด")
    with tabs[8]:
        st.dataframe(pd.DataFrame([{
            "เหรียญ": c["ticker"], "Regime": REG_TH[c["cf"]["regime"]], "Vol/ปี": f"{c['risk']['vol_ann'] * 100:.0f}%",
            "Cap สูงสุด": f"{cap_for(c):.0f}%", "MDD 1Y": f"{c['risk']['mdd_1y'] * 100:.0f}%",
            "Beta": f"{c['risk']['beta']:.2f}", "Corr BTC": f"{c['risk']['corr']:.2f}", "RS90 vs BTC": fpct(c["risk"]["rs90"]),
            "Score": round(c["score"]),
        } for c in coins]), use_container_width=True, hide_index=True)
        st.markdown("#### 🧪 Backtest แผนระยะกลาง (เข้าแท่งถัดไป · ค่าธรรมเนียม 0.1% + slippage 0.05% ต่อฝั่ง)")
        st.dataframe(table_backtest(coins), use_container_width=True, hide_index=True)
        st.caption("Out-of-sample = 30% ท้ายของข้อมูล · In-sample ดีแต่ Out-of-sample ติดลบ = เสี่ยง Overfitting")
        if len(res["corr"]) and go is not None:
            cm = res["corr"]
            fig = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.index, zmin=-1, zmax=1, colorscale="RdBu_r",
                                       text=cm.values, texttemplate="%{text:.2f}"))
            fig.update_layout(height=460, title="Correlation 90D", template="plotly_dark", margin=dict(l=8, r=8, t=40, b=8))
            st.plotly_chart(fig, use_container_width=True)
    with tabs[9]:
        st.code(summary_text(res, coins, port), language=None)
        st.caption("กดไอคอนคัดลอกมุมขวาบนเพื่อเก็บสรุปไว้ใช้")
        report = full_report_md(res, coins, research, port.get("views") or {}, pick_by, final_win, capital, risk_pct,
                                _md_table)
        st.download_button("⬇️ ดาวน์โหลดรายงานเต็ม (.md)", report, f"crypto_report_{res['ts']:%Y%m%d_%H%M}.md",
                           "text/markdown")
    with tabs[10]:
        render_roadmap(res)
    with tabs[11]:
        render_export(res, horizon)

    st.caption("⚠️ เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน · ความน่าจะเป็นมาจากสถิติย้อนหลัง ไม่รับประกันอนาคต")


if __name__ == "__main__":
    main()

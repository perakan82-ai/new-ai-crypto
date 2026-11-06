"""Structure + Quant/Backtest/Forecast + Macro + แผนระยะกลาง/ยาว (ราคาทุกตัวคำนวณที่นี่)"""
from __future__ import annotations

import numpy as np
import pandas as pd
from config import FC_H, FEE, LONG_H, MAJORS, MED_H, MIN_BARS_1D, OOS_FRACTION, QUOTE, SHRINK_K, SLIPPAGE, TIER_W
from market import add_indicators, closed_only


# ════════════════════ structure ════════════════════
def regime_arr(d: pd.DataFrame) -> np.ndarray:
    """UP / TRANS / DOWN ต่อแท่ง (ใช้ EMA200 ถ้ามี ไม่งั้นใช้ EMA50/21)"""
    c, e21, e50, e200 = (d[k].to_numpy() for k in ("close", "ema21", "ema50", "ema200"))
    has200 = ~np.isnan(e200)
    up = np.where(has200, (c > e200) & (e50 > e200), (c > e50) & (e21 > e50))
    dn = np.where(has200, (c < e200) & (e50 < e200), (c < e50) & (e21 < e50))
    return np.where(up, "UP", np.where(dn, "DOWN", "TRANS"))


def candle_pattern(d: pd.DataFrame) -> str:
    """Pattern ของแท่งปิดล่าสุด (d ต้องเป็นแท่งที่ปิดแล้วเท่านั้น)"""
    c, p = d.iloc[-1], d.iloc[-2]
    body, rng = abs(c.close - c.open), max(c.high - c.low, 1e-12)
    lower, upper = min(c.open, c.close) - c.low, c.high - max(c.open, c.close)
    if c.close > c.open and p.close < p.open and c.close >= p.open and c.open <= p.close:
        return "Bullish Engulfing"
    if c.close < c.open and p.close > p.open and c.open >= p.close and c.close <= p.open:
        return "Bearish Engulfing"
    if lower >= 2 * body and upper <= body and body / rng < 0.35:
        return "Hammer/Pin bar (แรงซื้อรับ)"
    if upper >= 2 * body and lower <= body and body / rng < 0.35:
        return "Shooting star (แรงขายกด)"
    if body / rng < 0.1:
        return "Doji (ลังเล)"
    return "ไม่มี pattern เด่น"


def swings(d: pd.DataFrame, w: int, lookback: int) -> tuple[pd.Series, pd.Series]:
    """Swing high/low แบบ fractal (ต้องมีแท่งยืนยัน w แท่งทั้งสองข้าง)"""
    x = d.iloc[-lookback:]
    win = 2 * w + 1
    hi = x["high"][x["high"] == x["high"].rolling(win, center=True).max()]
    lo = x["low"][x["low"] == x["low"].rolling(win, center=True).min()]
    return hi, lo


def market_structure(d: pd.DataFrame | None, w: int = 5, lookback: int = 365) -> dict:
    na = {"code": "NA", "label": "ข้อมูลไม่พอ", "bos": ""}
    if d is None or len(d) < 2 * w + 5:
        return na
    hi, lo = swings(d, w, lookback)
    if len(hi) < 2 or len(lo) < 2:
        return na
    hh, hl = hi.iloc[-1] > hi.iloc[-2], lo.iloc[-1] > lo.iloc[-2]
    code, label = {(True, True): ("UP", "HH/HL ขาขึ้น"), (False, False): ("DOWN", "LH/LL ขาลง"),
                   (True, False): ("EXPAND", "HH/LL ขยายตัว"), (False, True): ("SQUEEZE", "LH/HL บีบตัว")}[(hh, hl)]
    close = d["close"].iloc[-1]
    bos = "BOS ขึ้น (ทะลุ swing high)" if close > hi.iloc[-1] else ("BOS ลง (หลุด swing low)" if close < lo.iloc[-1] else "")
    return {"code": code, "label": label, "bos": bos, "last_high": float(hi.iloc[-1]), "last_low": float(lo.iloc[-1])}


def volume_profile(d: pd.DataFrame, days: int = 180, bins: int = 60) -> dict:
    """POC / Value Area 70% / High-Volume Nodes"""
    x = d.iloc[-days:]
    lo, hi = x["low"].min(), x["high"].max()
    if hi <= lo:
        return {"poc": float(lo), "val": float(lo), "vah": float(hi), "hvn": []}
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    L, H = x["low"].to_numpy()[:, None], x["high"].to_numpy()[:, None]
    ov = np.clip(np.minimum(H, edges[1:]) - np.maximum(L, edges[:-1]), 0, None)
    vol = (ov / np.maximum(H - L, 1e-12) * x["qvol"].to_numpy()[:, None]).sum(0)
    poc = int(np.argmax(vol))
    lo_i = hi_i = poc
    acc, total = vol[poc], vol.sum()
    while acc < 0.7 * total and (lo_i > 0 or hi_i < bins - 1):
        down = vol[lo_i - 1] if lo_i > 0 else -1
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1
        if up >= down:
            hi_i += 1
            acc += vol[hi_i]
        else:
            lo_i -= 1
            acc += vol[lo_i]
    thr = vol.mean() + 0.5 * vol.std()
    hvn = [float(centers[i]) for i in range(1, bins - 1)
           if vol[i] >= vol[i - 1] and vol[i] >= vol[i + 1] and vol[i] > thr and i != poc]
    return {"poc": float(centers[poc]), "val": float(edges[lo_i]), "vah": float(edges[hi_i + 1]), "hvn": hvn}


def sr_levels(d1: pd.DataFrame, w1: pd.DataFrame | None, vp: dict, price: float, tol: float = 0.012) -> list[dict]:
    """รวมแนวรับ/ต้านจาก swing 1D/1W + POC/HVN แล้ว cluster ที่ห่างกัน ≤ 1.2%"""
    cand: list[tuple[float, float, str]] = []
    h, l = swings(d1, 5, 365)
    cand += [(float(v), 1.0, "1D") for v in pd.concat([h, l])]
    if w1 is not None and len(w1) > 20:
        h, l = swings(w1, 3, 156)
        cand += [(float(v), 2.0, "1W") for v in pd.concat([h, l])]
    cand.append((vp["poc"], 2.0, "POC"))
    cand += [(v, 1.0, "HVN") for v in vp["hvn"]]
    cand = sorted(c for c in cand if 0.2 * price < c[0] < 5 * price)
    clusters: list[dict] = []
    for p, w, src in cand:
        if clusters and abs(p - clusters[-1]["level"]) / clusters[-1]["level"] <= tol:
            c = clusters[-1]
            c["level"] = (c["level"] * c["strength"] + p * w) / (c["strength"] + w)
            c["strength"] += w
            c["src"].add(src)
        else:
            clusters.append({"level": p, "strength": w, "src": {src}})
    for c in clusters:
        c["src"] = "+".join(sorted(c["src"]))
    return clusters


def fib_levels(d1: pd.DataFrame) -> dict:
    x = d1.iloc[-365:]
    hi, lo = float(x["high"].max()), float(x["low"].min())
    r = {f: hi - f * (hi - lo) for f in (0.382, 0.5, 0.618, 0.786)}
    return {"hi52": hi, "lo52": lo, "ret": r, "ext1618": lo + 1.618 * (hi - lo)}


BIAS_TH = {"bull": "🟢 ขาขึ้น", "bear": "🔴 ขาลง", "neutral": "⚪ รอทิศทาง"}


def chart_patterns(d: pd.DataFrame, lookback: int = 150) -> list[dict]:
    """ตรวจรูปแบบกราฟแบบกฎตายตัว (rule-based) บนแท่ง 1D ที่ปิดแล้ว

    Double Bottom/Top · Ascending/Descending/Symmetrical Triangle · Bull Flag · Range Breakout/Breakdown
    target = measured move ตามตำรา (ไม่ใช่ความน่าจะเป็น — โอกาสแตะคำนวณแยกด้วยสถิติใน analyze_coin)
    """
    out: list[dict] = []
    if len(d) < 60 or "atr" not in d:
        return out
    x = d.iloc[-lookback:]
    n = len(x)
    close, atr = float(x["close"].iloc[-1]), float(x["atr"].iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return out
    tol = max(0.03, atr / close)
    hi, lo = swings(x, 3, n)
    pos = {t: i for i, t in enumerate(x.index)}

    def _collapse(sw: pd.Series, keep_max: bool, gap: int = 4) -> pd.Series:
        """รวม swing ที่อยู่ติดกัน (ราคาเท่ากันหลายแท่ง) ให้เหลือจุดเดียว"""
        keep: list = []
        for t, v in sw.items():
            if keep and pos[t] - pos[keep[-1][0]] < gap:
                if (v > keep[-1][1]) if keep_max else (v < keep[-1][1]):
                    keep[-1] = (t, v)
                continue
            keep.append((t, v))
        return pd.Series([v for _, v in keep], index=[t for t, _ in keep], dtype=float)

    hi, lo = _collapse(hi, True), _collapse(lo, False)

    def add(name, bias, state, level, target, note):
        if target and target > 0:
            out.append({"name": name, "bias": bias, "state": state, "level": float(level), "target": float(target),
                        "note": note})

    if len(lo) >= 2:
        i1, i2 = pos[lo.index[-2]], pos[lo.index[-1]]
        l1, l2 = float(lo.iloc[-2]), float(lo.iloc[-1])
        neck = float(x["high"].iloc[i1:i2 + 1].max())
        if (i2 - i1 >= 8 and i2 >= n - 45 and abs(l2 / l1 - 1) <= tol and neck > max(l1, l2) * (1 + 2 * tol)
                and close > max(l1, l2)):
            low = min(l1, l2)
            add("Double Bottom", "bull", "ยืนยัน (ทะลุ neckline)" if close > neck else "กำลังก่อตัว", neck,
                neck + (neck - low), f"ฐาน {low:.6g} สองครั้ง · neckline {neck:.6g}")
    if len(hi) >= 2:
        i1, i2 = pos[hi.index[-2]], pos[hi.index[-1]]
        h1, h2 = float(hi.iloc[-2]), float(hi.iloc[-1])
        neck = float(x["low"].iloc[i1:i2 + 1].min())
        if (i2 - i1 >= 8 and i2 >= n - 45 and abs(h2 / h1 - 1) <= tol and neck < min(h1, h2) * (1 - 2 * tol)
                and close < min(h1, h2)):
            top = max(h1, h2)
            add("Double Top", "bear", "ยืนยัน (หลุด neckline)" if close < neck else "กำลังก่อตัว", neck,
                neck - (top - neck), f"ยอด {top:.6g} สองครั้ง · neckline {neck:.6g}")

    hs, ls = hi[hi.index >= x.index[-60]], lo[lo.index >= x.index[-60]]
    if len(hs) >= 2 and len(ls) >= 2:
        h_a, h_b, l_a, l_b = (float(v) for v in (hs.iloc[-2], hs.iloc[-1], ls.iloc[-2], ls.iloc[-1]))
        hflat, lflat = abs(h_b / h_a - 1) <= tol, abs(l_b / l_a - 1) <= tol
        hfall, lrise = h_b < h_a * (1 - tol / 2), l_b > l_a * (1 + tol / 2)
        res_, sup = max(h_a, h_b), min(l_a, l_b)
        height = res_ - sup
        if hflat and lrise:
            add("Ascending Triangle", "bull", "ทะลุแนวต้านแล้ว" if close > res_ else "กำลังก่อตัว", res_,
                res_ + height, f"แนวต้านแบน {res_:.6g} · Low ยกตัว")
        elif lflat and hfall:
            add("Descending Triangle", "bear", "หลุดแนวรับแล้ว" if close < sup else "กำลังก่อตัว", sup,
                sup - height, f"แนวรับแบน {sup:.6g} · High ต่ำลง")
        elif hfall and lrise:
            st = "ทะลุขึ้น" if close > h_b else ("หลุดลง" if close < l_b else "บีบตัว รอทิศทาง")
            bias = "bull" if close > h_b else ("bear" if close < l_b else "neutral")
            add("Symmetrical Triangle", bias, st, h_b if bias != "bear" else l_b,
                (h_b + height) if bias != "bear" else (l_b - height), f"กรอบ {l_b:.6g} – {h_b:.6g}")

    if n >= 50:
        pk = n - 30 + int(np.argmax(x["high"].iloc[-30:].to_numpy()))
        peak = float(x["high"].iloc[pk])
        after = n - 1 - pk
        pole_low = float(x["low"].iloc[max(0, pk - 15):pk].min()) if pk > 0 else peak
        pole = peak - pole_low
        if 3 <= after <= 15 and pole_low > 0 and pole / pole_low >= 0.15 and pole >= 3 * atr:
            pull = peak - float(x["low"].iloc[pk + 1:].min())
            v_pole = float(x["qvol"].iloc[max(0, pk - 15):pk + 1].mean())
            v_flag = float(x["qvol"].iloc[pk + 1:].mean())
            if 0.15 * pole <= pull <= 0.5 * pole and v_flag < v_pole:
                add("Bull Flag", "bull", "ทะลุยอดแล้ว" if close > peak else "กำลังพักตัว (Volume ลด)", peak,
                    peak + pole, f"เสาธง +{pole / pole_low * 100:.0f}% · ย่อ {pull / pole * 100:.0f}% ของเสา")

    rng_hi, rng_lo = float(x["high"].iloc[-31:-1].max()), float(x["low"].iloc[-31:-1].min())
    vr = float(x["vol_ratio"].iloc[-1]) if np.isfinite(x["vol_ratio"].iloc[-1]) else 0.0
    if close > rng_hi and vr >= 1.5:
        add("Range Breakout 30D", "bull", f"ทะลุพร้อม Volume {vr:.1f}x", rng_hi, rng_hi + (rng_hi - rng_lo),
            f"กรอบ {rng_lo:.6g} – {rng_hi:.6g}")
    elif close < rng_lo and vr >= 1.5:
        add("Range Breakdown 30D", "bear", f"หลุดพร้อม Volume {vr:.1f}x", rng_lo, rng_lo - (rng_hi - rng_lo),
            f"กรอบ {rng_lo:.6g} – {rng_hi:.6g}")
    return out


def pattern_points(patterns: list[dict]) -> float:
    """คะแนนจากรูปแบบกราฟ: ยืนยันแล้ว ±4 · กำลังก่อตัว ±2 (รวมไม่เกิน ±8)"""
    s = 0.0
    for p in patterns:
        w = 2.0 if p["state"].startswith(("กำลัง", "บีบ")) else 4.0
        s += {"bull": w, "bear": -w}.get(p["bias"], 0.0)
    return float(np.clip(s, -8, 8))


# ════════════════════ quant ════════════════════
COST = FEE + SLIPPAGE  # ต้นทุนต่อฝั่ง


def setup_mask(d: pd.DataFrame) -> np.ndarray:
    """Setup ขาขึ้นที่ใช้ทั้ง backtest และแผนจริง: close>EMA50, MACD hist>0, RSI 40-70, EMA50>EMA200"""
    return ((d["close"] > d["ema50"]) & (d["macd_h"] > 0) & d["rsi"].between(40, 70)
            & (d["ema50"] > d["ema200"])).to_numpy()


def _metrics(trades: list[tuple[float, int, bool, int]]) -> dict:
    rs = np.array([t[0] for t in trades])
    n = len(rs)
    if n == 0:
        return {"n": 0, "wins": 0, "win_rate_raw": 0.0, "expectancy": 0.0, "profit_factor": None, "max_dd_R": 0.0,
                "avg_win_R": None, "avg_loss_R": None}
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    eq = np.cumsum(rs)
    dd = float((eq - np.maximum.accumulate(np.r_[0.0, eq])[1:]).min())
    pf = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else None
    return {"n": n, "wins": int(len(wins)), "win_rate_raw": float(len(wins) / n), "expectancy": float(rs.mean()),
            "profit_factor": pf, "max_dd_R": min(dd, 0.0),
            "avg_win_R": float(wins.mean()) if len(wins) else None,
            "avg_loss_R": float(losses.mean()) if len(losses) else None}


def backtest(d: pd.DataFrame, sl_mult: float = 2.0, tp_mult: float = 4.0, max_hold: int = 60) -> dict:
    """Backtest ตามระยะ SL/TP (หน่วย ATR) ของแผนจริง

    - สัญญาณจากแท่ง i (ปิดแล้ว) → เข้าซื้อที่ open ของแท่ง i+1 + slippage
    - แท่งเดียวกันโดนทั้ง SL และ TP → นับ SL (conservative)
    - ไม่เปิดไม้ใหม่จนกว่าไม้เดิมจะปิด
    """
    o, c, h, l, a = (d[k].to_numpy() for k in ("open", "close", "high", "low", "atr"))
    cond = setup_mask(d)
    n, i, trades = len(d), 0, []
    while i < n - 2:
        if not cond[i] or not np.isfinite(a[i]):
            i += 1
            continue
        entry = o[i + 1] * (1 + SLIPPAGE)
        sl, tp = entry - sl_mult * a[i], entry + tp_mult * a[i]
        if sl <= 0:
            i += 1
            continue
        exit_px, j = None, i + 1
        for j in range(i + 1, min(i + 1 + max_hold, n)):
            if l[j] <= sl:
                exit_px = sl
                break
            if h[j] >= tp:
                exit_px = tp
                break
        if exit_px is None:
            if j - i < max_hold:  # ข้อมูลหมดก่อนครบกำหนด → ไม่นับ (กันผลที่ยังไม่จบ)
                break
            exit_px = c[j]
        ret = exit_px * (1 - SLIPPAGE) / entry - 1 - 2 * FEE
        trades.append((ret / ((entry - sl) / entry), j - i, ret > 0, i))
        i = j + 1
    m = _metrics(trades)
    split = int(n * (1 - OOS_FRACTION))
    ins = _metrics([t for t in trades if t[3] < split])
    oos = _metrics([t for t in trades if t[3] >= split])
    days = np.array([t[1] for t in trades if t[2]])
    m.update({
        "win_rate": (m["wins"] + 2) / (m["n"] + 4),  # Bayesian smoothing → ไม่เชื่อตัวอย่างน้อยเกินไป
        "hold_p25": float(np.percentile(days, 25)) if len(days) >= 3 else None,
        "hold_p75": float(np.percentile(days, 75)) if len(days) >= 3 else None,
        "active": bool(cond[-1]), "sl_mult": sl_mult, "tp_mult": tp_mult,
        "is_n": ins["n"], "is_expectancy": ins["expectancy"],
        "oos_n": oos["n"], "oos_expectancy": oos["expectancy"], "oos_win_raw": oos["win_rate_raw"],
    })
    return m


def independent(idx: np.ndarray, gap: int) -> np.ndarray:
    """เลือกเฉพาะตัวอย่างที่ไม่ทับซ้อน (ห่างกัน ≥ gap แท่ง)"""
    out, last = [], -10 ** 9
    for i in idx:
        if i - last >= gap:
            out.append(i)
            last = i
    return np.asarray(out, dtype=int)


def conditional_forward(d: pd.DataFrame) -> dict:
    """หาวันในอดีตที่ Regime + ช่วง RSI เหมือนวันนี้ → ผลถือต่อ H วัน"""
    c = d["close"].to_numpy()
    reg = regime_arr(d)
    rb = np.digitize(d["rsi"].to_numpy(), [30, 45, 55, 70])
    valid = ~np.isnan(d["ema50"].to_numpy())
    n = len(c)
    cur_reg, cur_b = reg[-1], rb[-1]
    full = (reg == cur_reg) & (rb == cur_b) & valid
    loose = (reg == cur_reg) & valid
    stats: dict[int, dict | None] = {}
    for H in MED_H + LONG_H:
        if n - H <= 0:
            stats[H] = None
            continue
        idx, relaxed = np.where(full[: n - H])[0], False
        if len(idx) < 10:
            idx, relaxed = np.where(loose[: n - H])[0], True
        if len(idx) < 5:
            stats[H] = None
            continue
        r = c[idx + H] / c[idx] - 1
        ind = independent(idx, H)
        ri = c[ind + H] / c[ind] - 1
        stats[H] = {"n": int(len(idx)), "n_ind": int(len(ind)), "win_raw": float((r > 0).mean()),
                    "win": float(((ri > 0).sum() + 0.5 * SHRINK_K) / (len(ind) + SHRINK_K)),
                    "median": float(np.median(r)), "p25": float(np.percentile(r, 25)),
                    "p75": float(np.percentile(r, 75)), "relaxed": relaxed}

    def best(hs):
        cands = [(H, s) for H in hs if (s := stats.get(H))]
        if not cands:
            return None

        def ra(s):
            iqr = max(s["p75"] - s["p25"], 1e-3)
            base = s["median"] / iqr if s["median"] > 0 else s["median"] - 1
            return base + (s["win"] - 0.5) * 2
        return max(cands, key=lambda t: ra(t[1]))[0]

    return {"regime": cur_reg, "rsi_bucket": int(cur_b), "stats": stats,
            "best_med": best(MED_H), "best_long": best(LONG_H)}


# ───────────────────────── Forecast ─────────────────────────
def _fc_samples(c: dict, H: int):
    """หน้าต่างราคาในอดีต H วัน (Regime เดียวกันถ้าตัวอย่างพอ) ปรับขนาดตามความผันผวนปัจจุบัน"""
    cache = c.setdefault("_fc", {})
    if H in cache:
        return cache[H]
    d = c["d1"]
    cl, hi, lo = (d[k].to_numpy() for k in ("close", "high", "low"))
    n = len(cl)
    res = None
    if n > H + 60:
        lr = np.r_[np.nan, np.diff(np.log(cl))]
        vol = pd.Series(lr).rolling(30).std().to_numpy()
        vol_now = vol[-1]
        t = np.arange(30, n - H)
        t = t[np.isfinite(vol[t]) & (vol[t] > 0)]
        reg = regime_arr(d)
        same = t[reg[t] == reg[-1]]
        filtered = len(independent(same, H)) >= 4 and len(same) >= 40
        use = same if filtered else t
        if len(use) >= 30 and np.isfinite(vol_now) and vol_now > 0:
            Hw = np.lib.stride_tricks.sliding_window_view(hi, H)[use + 1]
            Lw = np.lib.stride_tricks.sliding_window_view(lo, H)[use + 1]
            c0 = cl[use]
            s = np.clip(vol_now / vol[use], 0.33, 3.0)
            res = {"rel_h": (Hw / c0[:, None] - 1), "rel_l": (Lw / c0[:, None] - 1), "s": s,
                   "up": (Hw.max(1) / c0 - 1) * s, "dn": (Lw.min(1) / c0 - 1) * s,
                   "end": (cl[use + H] / c0 - 1) * s, "n": int(len(use)),
                   "n_eff": int(max(len(independent(use, H)), 1)), "filtered": filtered}
    cache[H] = res
    return res


def laplace(p: float, n_eff: int) -> float:
    return float((p * n_eff + 1) / (n_eff + 2))


def touch_prob(c: dict, level: float, H: int, base: float | None = None) -> float | None:
    """โอกาสที่ราคาจะ 'แตะ' level ภายใน H วัน"""
    S = _fc_samples(c, H)
    if S is None:
        return None
    base = base or c["price"]
    x = level / base - 1
    p = float((S["up"] >= x).mean()) if x >= 0 else float((S["dn"] <= x).mean())
    return laplace(p, S["n_eff"])


def first_touch(c: dict, tp: float, sl: float, H: int, base: float) -> dict | None:
    """โอกาสที่ TP มาก่อน SL ภายใน H วัน + จำนวนวันมัธยฐานที่ถึง TP"""
    S = _fc_samples(c, H)
    if S is None:
        return None
    up_thr = (tp / base - 1) / S["s"]
    dn_thr = (sl / base - 1) / S["s"]
    hu = S["rel_h"] >= up_thr[:, None]
    hd = S["rel_l"] <= dn_thr[:, None]
    fu = np.where(hu.any(1), hu.argmax(1), H + 1)
    fd = np.where(hd.any(1), hd.argmax(1), H + 1)
    tp_first = fu < fd
    sl_first = (fd <= fu) & (fd <= H)
    p_tp, p_sl = float(tp_first.mean()), float(sl_first.mean())
    return {"p_tp": laplace(p_tp, S["n_eff"]), "p_sl": laplace(p_sl, S["n_eff"]),
            "p_none": max(0.0, 1 - p_tp - p_sl),
            "days_tp": float(np.median(fu[tp_first]) + 1) if tp_first.any() else None,
            "n_eff": S["n_eff"]}


def forecast_table(c: dict) -> dict:
    p, out = c["price"], {}
    for H in FC_H:
        S = _fc_samples(c, H)
        if S is None:
            continue
        up, dn, end = S["up"], S["dn"], S["end"]
        out[H] = {
            "reach70": p * (1 + np.quantile(up, 0.30)), "reach50": p * (1 + np.quantile(up, 0.50)),
            "reach30": p * (1 + np.quantile(up, 0.70)), "dip50": p * (1 + np.quantile(dn, 0.50)),
            "dip30": p * (1 + np.quantile(dn, 0.30)),
            "end25": p * (1 + np.quantile(end, 0.25)), "end50": p * (1 + np.quantile(end, 0.50)),
            "end75": p * (1 + np.quantile(end, 0.75)), "p_up": laplace(float((end > 0).mean()), S["n_eff"]),
            "n_eff": S["n_eff"], "filtered": S["filtered"],
        }
    return out


def risk_metrics(d: pd.DataFrame, btc: pd.DataFrame) -> dict:
    r = d["close"].pct_change().iloc[-365:]
    rb = btc["close"].pct_change()
    j = pd.concat([r, rb], axis=1, join="inner").dropna()
    j.columns = ["c", "b"]
    c1 = d["close"].iloc[-365:]
    mdd = float((c1 / c1.cummax() - 1).min())

    def ret(s, k):
        return float(s.iloc[-1] / s.iloc[-k - 1] - 1) if len(s) > k else np.nan

    beta = float(j["c"].cov(j["b"]) / j["b"].var()) if len(j) > 30 else np.nan
    return {"vol_ann": float(r.std() * np.sqrt(365)), "mdd_1y": mdd, "beta": beta,
            "corr": float(j["c"].corr(j["b"])) if len(j) > 30 else np.nan,
            "rs30": ret(d["close"], 30) - ret(btc["close"], 30),
            "rs90": ret(d["close"], 90) - ret(btc["close"], 90),
            "ret30": ret(d["close"], 30), "ret90": ret(d["close"], 90)}


def corr_matrix(d1_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rets = {s.replace(QUOTE, ""): d["close"].pct_change().iloc[-90:] for s, d in d1_by_symbol.items()}
    if not rets:
        return pd.DataFrame()
    return pd.DataFrame(rets).dropna(how="all").corr().round(2)


# ════════════════════ macro ════════════════════
def macro_agent(btc_d: pd.DataFrame, btc_w: pd.DataFrame, glob: dict, fng: dict | None) -> dict:
    pts, notes = 0, []
    w, d = btc_w.iloc[-1], btc_d.iloc[-1]
    checks = [
        (w.close > w.ema50 if not np.isnan(w.ema50) else w.close > w.ema21, "BTC Weekly ปิดเหนือ EMA50W"),
        (w.ema21 > w.ema50 if not np.isnan(w.ema50) else False, "EMA21W > EMA50W"),
        (d.close > d.ema200 if not np.isnan(d.ema200) else False, "BTC Daily เหนือ EMA200"),
        (d.ema50 > d.ema200 if not np.isnan(d.ema200) else False, "EMA50D > EMA200D"),
    ]
    for ok, txt in checks:
        pts += int(bool(ok))
        notes.append(("✅ " if ok else "❌ ") + txt)
    regime = "Bull" if pts >= 3 else ("Bear" if pts <= 1 else "Neutral")
    if fng:
        if fng["value"] <= 20:
            notes.append(f"Fear & Greed {fng['value']} (Extreme Fear) → สะสมระยะยาวได้เปรียบ แต่ระยะกลางระวัง")
        elif fng["value"] >= 75:
            notes.append(f"Fear & Greed {fng['value']} (Extreme Greed) → ระวังไล่ราคา")
    if glob.get("btc_dom"):
        notes.append(f"BTC Dominance {glob['btc_dom']:.1f}% " +
                     ("(สูง → Altcoin มักอ่อนกว่า BTC)" if glob["btc_dom"] > 58 else "(ไม่กดดัน Altcoin มาก)"))
    ret30 = float(btc_d["close"].iloc[-1] / btc_d["close"].iloc[-31] - 1) if len(btc_d) > 31 else float("nan")
    return {"regime": regime, "points": pts, "notes": notes, "btc_price": float(d.close),
            "btc_rsi": float(d.rsi), "btc_ret30": ret30}


# ════════════════════ plans ════════════════════
COST = FEE + SLIPPAGE


def fmt_days(a: float, b: float) -> str:
    a, b = int(round(a)), int(round(max(a, b)))
    if b <= 21:
        return f"{a}–{b} วัน"
    if b <= 90:
        return f"{max(1, round(a / 7))}–{max(1, round(b / 7))} สัปดาห์ (~{a}–{b} วัน)"
    return f"{max(1, round(a / 30))}–{max(1, round(b / 30))} เดือน (~{a}–{b} วัน)"


def cap_for(c: dict) -> float:
    """สัดส่วนสูงสุดต่อเหรียญตามความผันผวน"""
    if c["ticker"] in MAJORS:
        return 35.0
    v = c["risk"]["vol_ann"]
    return 15.0 if v < 0.6 else (10.0 if v < 1.0 else 6.0)


def rel_win(ft: dict | None) -> float | None:
    """โอกาส TP ก่อน SL เมื่อเทียบกันสองทาง (ตัดกรณีไม่ถึงทั้งคู่ออก)"""
    if not ft:
        return None
    return ft["p_tp"] / max(ft["p_tp"] + ft["p_sl"], 1e-9)


def medium_plan(c: dict, macro: dict) -> dict:
    d1, h4 = c["d1"], c.get("h4")
    row = d1.iloc[-1]
    price, atr_ = c["price"], float(row.atr)
    lv = c["levels"]
    near = [L for L in lv if price - 3 * atr_ <= L["level"] < price * 0.995]
    strong = [L for L in near if L["strength"] >= 2]
    if strong or near:
        S = max(strong or near, key=lambda L: L["level"])["level"]
    else:
        emas = [v for v in (row.ema21, row.ema50) if not np.isnan(v) and price - 3 * atr_ <= v < price]
        S = max(emas) if emas else price - 1.5 * atr_
    zlo, zhi = S, min(price, S + 0.5 * atr_)
    if zhi <= zlo:
        zhi = zlo + 0.25 * atr_
    sl = S - 1.2 * atr_
    entry = (zlo + zhi) / 2
    R = entry - sl
    res = sorted(L["level"] for L in lv if L["level"] > entry * 1.005)
    c1 = [x for x in res if (x - entry) / R >= 1.5]
    tp1 = min(c1[0], entry + 3 * R) if c1 else entry + 2 * R
    rr1 = (tp1 - entry) / R
    c2 = [x for x in res if x > tp1 * 1.01 and (x - entry) / R >= max(rr1 + 0.5, 2.5)]
    tp2 = min(c2[0], entry + 6 * R) if c2 else min(entry + max(4, rr1 + 1.5) * R, entry + 6 * R)

    bt = backtest(d1, R / atr_, (tp1 - entry) / atr_)
    c["bt"] = bt
    sl_pct = R / entry
    win_R = ((tp1 / entry - 1) - 2 * COST) / sl_pct
    loss_R = (sl_pct + 2 * COST) / sl_pct
    be_win = loss_R / (win_R + loss_R)  # win rate ขั้นต่ำที่คุ้มทุน (รวมค่าธรรมเนียม+slippage)

    cf = c["cf"]
    if bt["hold_p75"] is not None:
        a = float(np.clip(bt["hold_p25"], 5, 90))
        b = float(np.clip(bt["hold_p75"], a, 90))
    elif cf["best_med"]:
        a, b = max(5.0, cf["best_med"] * 0.5), float(cf["best_med"])
    else:
        a, b = 10.0, 45.0
    tstop = int(np.clip(round(b * 1.25), 14, 90))
    ft1 = first_touch(c, tp1, sl, tstop, entry)
    ft2 = first_touch(c, tp2, sl, min(90, tstop * 2), entry)
    p_fill = 1.0 if price <= zhi else touch_prob(c, zhi, 14)

    trig4h = False
    if h4 is not None and len(h4) > 30:
        r4 = h4.iloc[-1]
        trig4h = bool(r4.close > r4.ema21 and r4.macd_h > h4["macd_h"].iloc[-2]) or bool(r4.rsi < 40)
    reasons = []
    rw = rel_win(ft1)
    edge = rw is not None and rw > be_win
    if cf["regime"] == "DOWN" or (c["struct_d"]["code"] == "DOWN" and price < row.ema50):
        status = "🔴"
        reasons.append("โครงสร้าง/Regime ขาลง → ไม่เล่นระยะกลาง")
    elif price <= zhi + 0.3 * atr_ and trig4h and bt["expectancy"] >= 0 and edge:
        status = "🟢"
        reasons.append("ราคาอยู่ในโซนแนวรับ + 4H กลับตัว + สถิติได้เปรียบ")
    else:
        status = "🟡"
        if price > zhi + 0.3 * atr_:
            reasons.append(f"ราคายังเหนือโซนซื้อ {((price / zhi) - 1) * 100:.1f}% → ตั้งรอ")
        elif not trig4h:
            reasons.append("รอ 4H ยืนเหนือ EMA21 + MACD กลับตัว")
        if not edge:
            reasons.append("สถิติ TP-ก่อน-SL ยังไม่เกินจุดคุ้มทุน")
    if status == "🟢" and macro["regime"] == "Bear" and c["risk"]["rs30"] <= 0.10:
        status = "🟡"
        reasons.append("Macro Bear → ลดเป็นรอ confirm")

    parts, wts = [bt["win_rate"]], [0.4]
    if rw is not None:
        parts.append(rw)
        wts.append(0.4)
    bm = cf["stats"].get(cf["best_med"]) if cf["best_med"] else None
    if bm:
        parts.append(bm["win"])
        wts.append(0.2)
    win_sys = float(np.dot(parts, wts) / sum(wts) * 100)
    if bt["expectancy"] < 0 and bt["n"] >= 5:
        win_sys -= min(10.0, -bt["expectancy"] * 15)
    return {"status": status, "reasons": reasons, "zone": (zlo, zhi), "entry": entry, "sl": sl,
            "tp1": tp1, "tp2": tp2, "rr1": rr1, "rr2": (tp2 - entry) / R, "risk_pct": sl_pct,
            "be_win": be_win * 100, "win_sys": float(np.clip(win_sys, 5, 95)), "hold": fmt_days(a, b), "tstop": tstop,
            "ft1": ft1, "ft2": ft2, "p_fill": p_fill, "support": S, "trig4h": trig4h, "rel_win": rw}


def long_plan(c: dict) -> dict:
    d1, w1 = c["d1"], c.get("w1")
    price = c["price"]
    fib, vp, lv = c["fib"], c["vp"], c["levels"]
    wr = w1.iloc[-1] if w1 is not None and len(w1) > 5 else None
    e21w = float(wr.ema21) if wr is not None and not np.isnan(wr.ema21) else np.nan
    e50w = float(wr.ema50) if wr is not None and not np.isnan(wr.ema50) else np.nan
    wclose = float(wr.close) if wr is not None else price
    w_up = (wclose > e50w and e21w > e50w) if not np.isnan(e50w) else (wclose > e21w if not np.isnan(e21w) else False)
    w_down = (not np.isnan(e50w)) and wclose < e50w and e21w < e50w and c["struct_w"]["code"] in ("DOWN", "NA")

    cands = [L["level"] for L in lv if L["level"] < price * 0.98]
    cands += [v for v in fib["ret"].values() if v < price * 0.98]
    cands += [v for v in (e21w, e50w) if not np.isnan(v) and v < price * 0.98]
    ded: list[float] = []
    for x in sorted(set(cands), reverse=True):
        if not ded or x <= ded[-1] * 0.96:
            ded.append(x)
    discount = price < vp["val"] or price <= fib["ret"][0.618]
    near_sup = bool(ded) and price <= ded[0] * 1.06
    if w_down and not discount:
        status = "🔴"
    elif (w_up and (near_sup or float(d1["rsi"].iloc[-1]) < 50)) or (not w_down and discount):
        status = "🟢"
    else:
        status = "🟡"

    t1 = price if status == "🟢" else (ded[0] if ded else price * 0.92)
    t2 = next((x for x in ded if x <= t1 * 0.94), t1 * 0.88)
    t3 = next((x for x in ded if x <= t2 * 0.93), t2 * 0.88)
    tiers = (t1, t2, t3)
    wts = TIER_W[status]
    avg = 1 / sum(w / p for w, p in zip(wts, tiers))
    stop_pct = float(np.clip(c["risk"]["vol_ann"] * 0.2, 0.08, 0.20))
    stop = t3 * (1 - stop_pct)

    res = sorted(L["level"] for L in lv if L["level"] > max(price, avg) * 1.10 and L["strength"] >= 2)
    tp1 = res[0] if res else (fib["hi52"] if fib["hi52"] > price * 1.10 else max(price, avg) * 1.35)
    ath = (c["fund"].get("ath") or 0) or (float(w1["high"].max()) if w1 is not None else 0)
    tp2 = ath if ath > tp1 * 1.10 else fib["ext1618"]
    tp2 = min(max(tp2, tp1 * 1.15), 2.2 * tp1)

    cf = c["cf"]
    good = [H for H in LONG_H if (s := cf["stats"].get(H)) and s["median"] > 0 and s["win"] >= 0.55]
    if good:
        hold = fmt_days(min(good), max(good)) if len(good) > 1 else fmt_days(good[0] * 0.66, good[0])
    else:
        hold = "สถิติยังไม่เป็นบวก — สะสมเฉพาะเมื่อพื้นฐานแข็ง"
    bl = cf["stats"].get(cf["best_long"]) if cf["best_long"] else None
    win_sys = (bl["win"] * 100) if bl else 50.0
    tier_fill = [1.0 if p >= price * 0.999 else touch_prob(c, p, 90) for p in tiers]
    return {"status": status, "tiers": tiers, "weights": wts, "avg": avg, "stop": stop, "stop_pct": stop_pct,
            "tp1": tp1, "tp2": tp2, "win_sys": float(np.clip(win_sys, 5, 95)), "hold": hold,
            "w_up": w_up, "w_down": w_down, "discount": discount, "e21w": e21w, "e50w": e50w,
            "tier_fill": tier_fill, "p_tp1_180": touch_prob(c, tp1, 180), "p_tp2_180": touch_prob(c, tp2, 180)}


def quant_score(c: dict) -> float:
    m, l, r = c["med"], c["long"], c["risk"]
    s = {"🟢": 25, "🟡": 12, "🔴": 0}[m["status"]] + {"🟢": 15, "🟡": 8, "🔴": 0}[l["status"]]
    s += 0.25 * m["win_sys"] + 0.15 * l["win_sys"]
    s += 8 * float(np.clip(c["bt"]["expectancy"], -1, 1))
    s += float(np.clip((r["rs90"] if not np.isnan(r["rs90"]) else 0) * 30, -8, 8))
    s += pattern_points(c.get("patterns", []))
    return float(np.clip(s, 0, 100))


def analyze_coin(sym: str, fr: dict, btc_d: pd.DataFrame, macro: dict, fund: dict,
                 now: pd.Timestamp | None = None) -> dict | None:
    """วิเคราะห์ 1 เหรียญครบทุก Agent (เชิงตัวเลข) · คืน None ถ้าข้อมูลไม่พอ"""
    if "1d" not in fr:
        return None
    raw1d = fr["1d"]
    price = float(raw1d["close"].iloc[-1])  # ราคาสด (แท่งปัจจุบัน)
    d1 = closed_only(add_indicators(raw1d), now)
    if len(d1) < MIN_BARS_1D:
        return None
    h4 = closed_only(add_indicators(fr["4h"]), now) if "4h" in fr else None
    w1 = closed_only(add_indicators(fr["1w"]), now) if "1w" in fr and len(fr["1w"]) > 3 else None
    vp = volume_profile(d1)
    c = {"symbol": sym, "ticker": sym[: -len(QUOTE)], "price": price, "d1": d1, "h4": h4, "w1": w1,
         "vp": vp, "fib": fib_levels(d1), "levels": sr_levels(d1, w1, vp, price),
         "struct_d": market_structure(d1, 5, 365),
         "struct_w": market_structure(w1, 3, 156) if w1 is not None and len(w1) > 20 else
         {"code": "NA", "label": "ข้อมูลไม่พอ", "bos": ""},
         "cf": conditional_forward(d1), "risk": risk_metrics(d1, btc_d), "pattern": candle_pattern(d1),
         "fund": fund, "last_closed": d1.index[-1]}
    c["med"] = medium_plan(c, macro)
    c["long"] = long_plan(c)
    c["fc"] = forecast_table(c)
    c["patterns"] = chart_patterns(d1)
    for p in c["patterns"]:
        p["p_target_60"] = touch_prob(c, p["target"], 60)  # โอกาสแตะเป้ารูปแบบใน 60 วัน (สถิติ)
    c["score"] = quant_score(c)
    c.pop("_fc", None)
    return c

"""Module 1-2 — Binance Market Data (ตรวจข้อมูล, ตัดแท่งที่ยังไม่ปิด) + Technical Indicators (registry)"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
from config import BINANCE_HOSTS, QUOTE, TF_LIMIT, TF_TTL, WORKERS, make_session


# ════════════════════ market_data ════════════════════
log = logging.getLogger(__name__)


class DataError(RuntimeError):
    """ข้อมูลจาก Binance ใช้ไม่ได้"""


def parse_klines(raw: list) -> pd.DataFrame:
    """แปลง klines ดิบ → DataFrame + ตรวจคุณภาพ (ซ้ำ/เรียงผิด/ราคา ≤ 0/high<low)"""
    if not raw:
        raise DataError("ไม่มีข้อมูลแท่งเทียน")
    a = np.asarray(raw, dtype=object)
    df = pd.DataFrame({
        "open": a[:, 1].astype(float), "high": a[:, 2].astype(float), "low": a[:, 3].astype(float),
        "close": a[:, 4].astype(float), "volume": a[:, 5].astype(float), "qvol": a[:, 7].astype(float),
        "taker_q": a[:, 10].astype(float),
        "close_time": pd.to_datetime(a[:, 6].astype(np.int64), unit="ms", utc=True),
    }, index=pd.to_datetime(a[:, 0].astype(np.int64), unit="ms", utc=True))
    df = df[~df.index.duplicated(keep="last")].sort_index()
    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (df["high"] < df["low"])
    if bad.any():
        log.warning("ตัดแท่งผิดปกติ %d แท่ง", int(bad.sum()))
        df = df[~bad]
    if df.empty:
        raise DataError("ข้อมูลผิดปกติทั้งหมด")
    return df


def closed_only(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """เหลือเฉพาะแท่งที่ปิดแล้ว — ป้องกันการใช้แท่งที่ยังไม่ปิดเป็นสัญญาณยืนยัน"""
    now = now or pd.Timestamp.now(tz="UTC")
    if "close_time" not in df:
        return df.iloc[:-1]
    return df[df["close_time"] < now]


class MarketData:
    """Binance public REST + host fallback + TTL cache (thread-safe)"""

    def __init__(self, hosts: tuple[str, ...] = BINANCE_HOSTS):
        self.hosts = hosts
        self.lock = threading.Lock()
        self.cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
        self.good = 0
        self.session = make_session()

    @property
    def host(self) -> str:
        return self.hosts[self.good]

    def _get(self, path: str, params: dict):
        order = [self.good] + [i for i in range(len(self.hosts)) if i != self.good]
        err: Exception | None = None
        for i in order:
            try:
                r = self.session.get(f"{self.hosts[i]}/{path}", params=params, timeout=15)
                if r.status_code in (403, 451):
                    raise DataError(f"HTTP {r.status_code} (บล็อกตามประเทศ)")
                r.raise_for_status()
                self.good = i
                return r.json()
            except Exception as e:  # noqa: BLE001 — ลอง host ถัดไป
                err = e
        raise DataError(f"Binance ใช้งานไม่ได้ทุก host: {err}")

    def ohlcv(self, symbol: str, tf: str, limit: int | None = None) -> pd.DataFrame:
        """แท่งเทียน (รวมแท่งล่าสุดที่ยังไม่ปิด — ใช้ closed_only() ก่อนคำนวณสัญญาณ)"""
        key, now = (symbol, tf), time.time()
        with self.lock:
            hit = self.cache.get(key)
        if hit and now - hit[0] < TF_TTL.get(tf, 600) and limit is None:
            return hit[1]
        df = parse_klines(self._get("klines", {"symbol": symbol, "interval": tf, "limit": limit or TF_LIMIT[tf]}))
        if limit is None:
            with self.lock:
                self.cache[key] = (now, df)
        return df

    def tickers(self) -> pd.DataFrame:
        """24h ticker ของคู่ USDT ทั้งหมด"""
        df = pd.DataFrame(self._get("ticker/24hr", {}))
        df = df[df["symbol"].str.endswith(QUOTE)].copy()
        for c in ("lastPrice", "priceChangePercent", "quoteVolume", "count"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["base"] = df["symbol"].str[: -len(QUOTE)]
        return df.set_index("symbol")


def fetch_frames(md: MarketData, symbols: list[str], tfs: tuple[str, ...],
                 progress: Callable[[float, str], None] | None = None) -> dict[str, dict[str, pd.DataFrame]]:
    """ดึงหลายเหรียญ × หลาย timeframe แบบขนาน · เหรียญที่ล้มเหลวจะถูกข้าม (บันทึก log)"""
    jobs = [(s, tf) for s in symbols for tf in tfs]
    out: dict[str, dict[str, pd.DataFrame]] = {s: {} for s in symbols}
    if not jobs:
        return out
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(md.ohlcv, s, tf): (s, tf) for s, tf in jobs}
        for f in as_completed(futs):
            s, tf = futs[f]
            try:
                df = f.result()
                if len(df) >= 30:
                    out[s][tf] = df
            except Exception as e:  # noqa: BLE001
                log.warning("fetch %s %s failed: %s", s, tf, e)
            done += 1
            if progress:
                progress(done / len(jobs), f"{done}/{len(jobs)}")
    return out


# ════════════════════ indicators ════════════════════
_REGISTRY: dict[str, Callable[[pd.DataFrame], None]] = {}


def indicator(name: str):
    """ลงทะเบียน indicator (ฟังก์ชันรับ DataFrame แล้วเพิ่มคอลัมน์ในตัว)"""
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def ema(s: pd.Series, n: int) -> pd.Series:
    """EMA ที่ไม่มีค่าในช่วง warm-up (EMA200 = NaN ถ้าแท่ง < 200)"""
    out = s.ewm(span=n, adjust=False).mean()
    out.iloc[: n - 1] = np.nan
    return out


@indicator("ema")
def _ema(d: pd.DataFrame) -> None:
    for n in (9, 20, 21, 50, 200):
        d[f"ema{n}"] = ema(d["close"], n)


@indicator("rsi")
def _rsi(d: pd.DataFrame) -> None:
    delta = d["close"].diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)


@indicator("macd")
def _macd(d: pd.DataFrame) -> None:
    c = d["close"]
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd"] = macd
    d["macd_sig"] = macd.ewm(span=9, adjust=False).mean()
    d["macd_h"] = macd - d["macd_sig"]


@indicator("atr_adx")
def _atr_adx(d: pd.DataFrame) -> None:
    h, l, c = d["high"], d["low"], d["close"]
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    upm, dnm = h.diff(), -l.diff()
    pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), d.index)
    mdm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), d.index)
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / d["atr"]
    mdi = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / d["atr"]
    d["adx"] = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).ewm(alpha=1 / 14, adjust=False).mean()
    d["pdi"], d["mdi"] = pdi, mdi


@indicator("bollinger")
def _bb(d: pd.DataFrame) -> None:
    m = d["close"].rolling(20).mean()
    sd = d["close"].rolling(20).std()
    d["bb_mid"], d["bb_up"], d["bb_lo"] = m, m + 2 * sd, m - 2 * sd
    d["bb_width"] = (d["bb_up"] - d["bb_lo"]) / m


@indicator("vwap")
def _vwap(d: pd.DataFrame) -> None:
    """Rolling VWAP 20 แท่ง (ตลาดคริปโตไม่มี session จึงใช้แบบ rolling)"""
    tp = (d["high"] + d["low"] + d["close"]) / 3
    d["vwap20"] = (tp * d["volume"]).rolling(20).sum() / d["volume"].rolling(20).sum().replace(0, np.nan)


@indicator("volume")
def _volume(d: pd.DataFrame) -> None:
    d["vol_ratio"] = d["qvol"] / d["qvol"].rolling(20).mean().shift(1)
    d["obv"] = (np.sign(d["close"].diff().fillna(0)) * d["qvol"]).cumsum()


def add_indicators(df: pd.DataFrame, only: list[str] | None = None) -> pd.DataFrame:
    """คืน DataFrame ใหม่พร้อม indicators ทั้งหมด (หรือเฉพาะที่ระบุใน only)"""
    d = df.copy()
    for name, fn in _REGISTRY.items():
        if only is None or name in only:
            fn(d)
    return d


def available() -> list[str]:
    return list(_REGISTRY)

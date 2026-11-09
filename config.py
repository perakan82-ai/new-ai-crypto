"""ค่าตั้งค่าทั้งหมด + HTTP session/cache + ฟังก์ชันจัดรูปแบบตัวเลข"""
from __future__ import annotations

__version__ = "6.0.0"

import functools
import numpy as np
import requests
import threading
import time
from datetime import timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ════════════════════ config ════════════════════
# ───────────── Market data ─────────────
# data-api.binance.vision มาก่อน: ใช้ได้จาก Server สหรัฐฯ (Streamlit Cloud / GitHub Actions) ที่ api.binance.com บล็อก (HTTP 451)
BINANCE_HOSTS = ("https://data-api.binance.vision/api/v3", "https://api.binance.com/api/v3")
QUOTE = "USDT"
TF_LIMIT = {"1h": 1000, "4h": 500, "1d": 1000, "1w": 1000}
TF_TTL = {"1h": 300, "4h": 600, "1d": 600, "1w": 1800}  # เก็บแท่งเทียนใช้ซ้ำ (วินาที)
WORKERS = 8
MIN_BARS_1D = 62

# ───────────── Costs (ใช้ทั้ง backtest และแผนจริง) ─────────────
FEE = 0.001       # ค่าธรรมเนียม Spot 0.1% ต่อฝั่ง
SLIPPAGE = 0.0005  # Slippage 0.05% ต่อฝั่ง

# ───────────── Quant ─────────────
FC_H = (7, 14, 30, 60, 90, 180)
SHRINK_K = 6
MED_H = (7, 14, 30, 60)
LONG_H = (90, 180, 365)
OOS_FRACTION = 0.3  # 30% ท้ายของข้อมูลใช้เป็น Out-of-sample

# ───────────── Signal engine ─────────────
SIGNAL_EXPIRY_H = 72          # ไม่ได้ของภายใน 72 ชม. → หมดอายุ
MIN_RR = 1.5
MIN_EDGE_PTS = 3.0            # โอกาส TP ก่อน SL ต้องสูงกว่าจุดคุ้มทุน ≥ 3 จุด
MIN_N_EFF = 8                 # ตัวอย่างอิสระขั้นต่ำของสถิติ
MIN_BT_TRADES = 5

# ───────────── Research-based (v5.1) ─────────────
# Trend ensemble: Donchian หลาย lookback รวมเป็นคะแนนเดียว (Zarattini, Pagani & Barbon 2025)
DONCHIAN_LB = (20, 40, 60, 90, 120)
# Momentum เทียบตลาด (cross-sectional): งานวิจัย SSRN 2949379 พบว่าในคริปโตแบบ cross-sectional ให้ Sharpe ดีกว่า
MOM_LB = (30, 90)          # วัน · ใช้ผลตอบแทนปรับความผันผวน (Grobys et al. 2025: momentum crash แรง → ต้องปรับตาม vol)
XS_HOLD = 7                # backtest การจัดอันดับ: ถือ 7 วันแล้วจัดใหม่
XS_TOP = 5                 # ถือ 5 อันดับแรก
# Mean reversion ระยะสั้น: Ekström (2025) พบ BTC รายชั่วโมงเด้งกลับหาค่าเฉลี่ย แต่รายวันเป็นเทรนด์
# → ใช้ "ย่อในเทรนด์ขาขึ้น": 1D ขาขึ้น + 4H ต่ำกว่าเส้นกลาง Bollinger ≤ -1.5 SD เป็นจังหวะเข้า
MR_Z_4H = -1.5

# ───────────── Focus coin (เจาะลึก) ─────────────
PEERS_MAX = 6              # เทียบกับเหรียญกลุ่มเดียวกันสูงสุด 6 ตัว (+ BTC/ETH เป็นเกณฑ์)
SECTORS = {
    "Layer 1": ["ETH", "SOL", "ADA", "AVAX", "NEAR", "SUI", "APT", "TON", "DOT", "ATOM", "ICP", "SEI", "BNB", "TRX",
                "ALGO", "HBAR", "KAS", "INJ", "TIA", "S"],
    "Payments": ["BTC", "XRP", "XLM", "LTC", "BCH", "ETC"],
    "DeFi": ["UNI", "AAVE", "ENA", "PENDLE", "LDO", "CRV", "JUP", "RAY", "HYPE", "ONDO", "JTO", "LINK"],
    "AI": ["FET", "TAO", "RENDER", "WLD", "VIRTUAL", "GRT"],
    "Meme": ["DOGE", "PEPE", "SHIB", "BONK", "WIF", "TRUMP", "PUMP"],
    "Layer 2": ["ARB", "OP", "POL", "IMX"],
    "Privacy": ["ZEC", "XMR"],
}

# ───────────── Watchlist + Scanner ─────────────
MAX_WATCH = 1              # v5.3: วิเคราะห์เจาะลึก 1 เหรียญ · ที่เหลือให้ AI Scanner คัดจากตลาดไว้เปรียบเทียบ
SCAN_TOP = 80              # สแกนจาก Volume สูงสุด 80 อันดับ
SCAN_DEFAULT_EXTRA = 7
MAX_SPREAD = 0.002         # bid-ask spread ≤ 0.2% (Fil & Kristoufek 2020: ต้นทุนการเทรดกินกำไรมาก)

# ───────────── Misc ─────────────
TZ_TH = timezone(timedelta(hours=7))
MAJORS = {"BTC", "ETH"}
TIER_W = {"🟢": (0.40, 0.35, 0.25), "🟡": (0.25, 0.40, 0.35), "🔴": (0.15, 0.40, 0.45)}
STATUS_TXT = {"🟢": "ซื้อได้", "🟡": "รอย่อ/Confirm", "🔴": "ยังไม่ควรซื้อ"}
REG_TH = {"UP": "ขาขึ้น", "TRANS": "เปลี่ยนผ่าน", "DOWN": "ขาลง"}
STABLES = {"USDT", "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "USD1", "PYUSD", "XUSD", "RLUSD",
           "BFUSD", "USDS", "EUR", "EURI", "AEUR", "GBP", "TRY", "BRL", "PAXG", "WBTC", "WBETH", "BNSOL"}
DEFAULT_WATCH = "SOL"

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite", "gemini-3-flash-preview"]
MODEL_FALLBACK = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]
RESEARCH_TTL = 6 * 3600

CG_BASE = "https://api.coingecko.com/api/v3"
LLAMA = "https://api.llama.fi"
RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
}
CG_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple", "BNB": "binancecoin",
    "ADA": "cardano", "DOGE": "dogecoin", "TRX": "tron", "AVAX": "avalanche-2", "DOT": "polkadot",
    "LINK": "chainlink", "NEAR": "near", "XLM": "stellar", "LTC": "litecoin", "BCH": "bitcoin-cash",
    "UNI": "uniswap", "AAVE": "aave", "ENA": "ethena", "HYPE": "hyperliquid", "ZEC": "zcash",
    "HBAR": "hedera-hashgraph", "SUI": "sui", "APT": "aptos", "ARB": "arbitrum", "OP": "optimism",
    "TON": "the-open-network", "ATOM": "cosmos", "ICP": "internet-computer", "FIL": "filecoin",
    "INJ": "injective-protocol", "TIA": "celestia", "SEI": "sei-network", "RENDER": "render-token",
    "FET": "fetch-ai", "TAO": "bittensor", "PEPE": "pepe", "SHIB": "shiba-inu", "WLD": "worldcoin-wld",
    "ONDO": "ondo-finance", "PUMP": "pump-fun", "JUP": "jupiter-exchange-solana", "PENDLE": "pendle",
    "LDO": "lido-dao", "CRV": "curve-dao-token", "ETC": "ethereum-classic", "ALGO": "algorand",
    "VET": "vechain", "POL": "polygon-ecosystem-token", "BONK": "bonk", "WIF": "dogwifcoin",
    "S": "sonic-3", "VIRTUAL": "virtual-protocol", "TRUMP": "official-trump", "XMR": "monero",
    "KAS": "kaspa", "JTO": "jito-governance-token", "RAY": "raydium", "GRT": "the-graph", "IMX": "immutable-x",
}



# ════════════════════ http ════════════════════
def make_session(user_agent: str = "crypto-ai-lab/4.0") -> requests.Session:
    """Session ที่ retry อัตโนมัติเมื่อเจอ 429/5xx"""
    s = requests.Session()
    retry = Retry(total=2, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET", "POST"), raise_on_status=False)
    ad = HTTPAdapter(max_retries=retry, pool_connections=24, pool_maxsize=24)
    s.mount("https://", ad)
    s.headers["User-Agent"] = user_agent
    return s


SESSION = make_session()


def get_json(url: str, params: dict | None = None, headers: dict | None = None, timeout: float = 20):
    """GET แล้วคืน JSON (raise เมื่อ HTTP error)"""
    r = SESSION.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def ttl_cache(seconds: int):
    """Cache ผลลัพธ์ตาม arguments เป็นเวลา `seconds` วินาที (ใช้ได้ทั้งใน Streamlit และ scan_job)"""
    def deco(fn):
        store: dict = {}
        lock = threading.Lock()

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            with lock:
                hit = store.get(key)
            if hit and now - hit[0] < seconds:
                return hit[1]
            val = fn(*args, **kwargs)
            with lock:
                store[key] = (now, val)
            return val

        wrapper.cache_clear = lambda: store.clear()  # type: ignore[attr-defined]
        return wrapper
    return deco


# ════════════════════ formatting ════════════════════
def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and not np.isfinite(x))


def fp(x) -> str:
    """ราคา USD ตามขนาด"""
    if _bad(x):
        return "-"
    if x >= 1000:
        return f"${x:,.0f}"
    if x >= 100:
        return f"${x:,.1f}"
    if x >= 1:
        return f"${x:,.2f}"
    if x >= 0.01:
        return f"${x:.4f}"
    return f"${x:.3g}"


def fpct(x, d: int = 1) -> str:
    return "-" if _bad(x) else f"{x * 100:+.{d}f}%"


def fprob(x) -> str:
    return "-" if _bad(x) else f"{x * 100:.0f}%"


def fbig(x) -> str:
    if not x:
        return "-"
    for v, s in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= v:
            return f"${x / v:.2f}{s}"
    return f"${x:.0f}"


def dist(c: dict, x: float) -> str:
    return fpct(x / c["price"] - 1)

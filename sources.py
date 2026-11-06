"""Fundamental (CoinGecko/DefiLlama/RSS/Fear&Greed) + News Agent"""
from __future__ import annotations

import concurrent.futures as cf
import html
import json
import logging
import math
import pandas as pd
import re
import requests
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from pydantic import BaseModel
from urllib.parse import quote_plus
from config import CG_BASE, CG_IDS, LLAMA, RSS_FEEDS, SESSION, get_json, ttl_cache


# ════════════════════ fundamentals ════════════════════
log = logging.getLogger(__name__)


def _cg_headers(key: str) -> dict:
    return {"x-cg-demo-api-key": key} if key else {}


def cg_markets(tickers: tuple[str, ...], key: str) -> dict:
    """symbol → row ของ CoinGecko (แมป id ก่อน แล้ว fallback ด้วย symbols)"""
    out: dict[str, dict] = {}
    try:
        ids = [CG_IDS[t] for t in tickers if t in CG_IDS]
        if ids:
            rows = get_json(f"{CG_BASE}/coins/markets", {"vs_currency": "usd", "ids": ",".join(ids),
                            "per_page": 250, "price_change_percentage": "30d"}, _cg_headers(key))
            for r in rows:
                out[r["symbol"].upper()] = r
        miss = [t for t in tickers if t not in out]
        if miss:
            rows = get_json(f"{CG_BASE}/coins/markets", {"vs_currency": "usd",
                            "symbols": ",".join(m.lower() for m in miss), "per_page": 250}, _cg_headers(key))
            for r in sorted(rows, key=lambda r: r.get("market_cap") or 0):
                out[r["symbol"].upper()] = r  # mcap สูงสุดเขียนทับท้ายสุด
    except Exception as e:  # noqa: BLE001
        log.warning("coingecko markets: %s", e)
    return out


def cg_global(key: str) -> dict:
    try:
        d = get_json(f"{CG_BASE}/global", headers=_cg_headers(key))["data"]
        return {"btc_dom": d["market_cap_percentage"].get("btc"), "eth_dom": d["market_cap_percentage"].get("eth"),
                "mcap_chg_24h": d.get("market_cap_change_percentage_24h_usd")}
    except Exception as e:  # noqa: BLE001
        log.warning("coingecko global: %s", e)
        return {}


def _cg_coin(cid: str, key: str) -> dict | None:
    try:
        d = get_json(f"{CG_BASE}/coins/{cid}", {"localization": "false", "tickers": "false",
                     "market_data": "false", "community_data": "false", "developer_data": "true",
                     "sparkline": "false"}, _cg_headers(key))
        dev = d.get("developer_data") or {}
        return {"categories": [c for c in (d.get("categories") or []) if c][:4],
                "commits_4w": dev.get("commit_count_4_weeks"), "name": d.get("name")}
    except Exception:  # noqa: BLE001
        return None


def cg_details(ids: tuple[str, ...], key: str) -> dict:
    with ThreadPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(lambda i: _cg_coin(i, key), ids))
    return {i: r for i, r in zip(ids, res) if r}


def llama_data() -> dict:
    out = {"protocols": [], "chains": [], "fees": []}

    def g(path, params=None):
        try:
            return get_json(f"{LLAMA}{path}", params, timeout=30)
        except Exception:  # noqa: BLE001
            return None

    with ThreadPoolExecutor(max_workers=3) as ex:
        p, c, f = ex.map(lambda a: g(*a), [("/protocols",), ("/v2/chains",),
                         ("/overview/fees", {"excludeTotalDataChart": "true",
                                             "excludeTotalDataChartBreakdown": "true"})])
    if p:
        out["protocols"] = [{"name": x.get("name"), "gecko_id": x.get("gecko_id"), "tvl": x.get("tvl")}
                            for x in p if x.get("gecko_id")]
    if c:
        out["chains"] = [{"name": x.get("name"), "gecko_id": x.get("gecko_id"), "tvl": x.get("tvl")} for x in c]
    if f:
        out["fees"] = [{"name": (x.get("displayName") or x.get("name") or ""), "gecko_id": x.get("gecko_id"),
                        "total30d": x.get("total30d")} for x in (f.get("protocols") or [])]
    return out


def rss_items() -> list[dict]:
    def one(src_url):
        src, url = src_url
        try:
            r = SESSION.get(url, timeout=12)
            root = ET.fromstring(r.content)
            return [{"src": src, "title": (it.findtext("title") or "").strip(),
                     "link": (it.findtext("link") or "").strip(), "date": (it.findtext("pubDate") or "")[:16]}
                    for it in root.iter("item")][:60]
        except Exception:  # noqa: BLE001
            return []

    with ThreadPoolExecutor(max_workers=3) as ex:
        return [x for lst in ex.map(one, RSS_FEEDS.items()) for x in lst]


def rss_for(ticker: str, name: str | None, items: list[dict], k: int = 6) -> list[dict]:
    pats = [re.compile(rf"\b{re.escape(ticker)}\b")]
    if name and len(name) > 4:
        pats.append(re.compile(rf"\b{re.escape(name)}\b", re.I))
    return [it for it in items if any(p.search(it["title"]) for p in pats)][:k]


@ttl_cache(600)
def fear_greed() -> dict | None:
    try:
        d = get_json("https://api.alternative.me/fng/", {"limit": 8})["data"]
        return {"value": int(d[0]["value"]), "label": d[0]["value_classification"],
                "d1": int(d[1]["value"]), "d7": int(d[-1]["value"])}
    except Exception:  # noqa: BLE001
        return None


@ttl_cache(1800)
def external_bundle(tickers: tuple[str, ...], key: str) -> dict:
    """ดึง CoinGecko / DefiLlama / RSS แบบขนาน (cache 30 นาที)"""
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_cg, f_gl, f_ll, f_rss = (ex.submit(cg_markets, tickers, key), ex.submit(cg_global, key),
                                   ex.submit(llama_data), ex.submit(rss_items))
        cg, glob, llama, rss = f_cg.result(), f_gl.result(), f_ll.result(), f_rss.result()
    det = cg_details(tuple(sorted({cg[t]["id"] for t in tickers if t in cg})), key)
    return {"cg": cg, "glob": glob, "llama": llama, "rss": rss, "det": det}


def fundamentals_for(ticker: str, cg: dict | None, det: dict | None, llama: dict) -> dict:
    if not cg:
        return {}
    gid, name = cg.get("id"), (cg.get("name") or "").lower()
    mcap, fdv = cg.get("market_cap") or 0, cg.get("fully_diluted_valuation") or 0
    tvl = next((x["tvl"] for x in llama["chains"] if x["gecko_id"] == gid and x["tvl"]), None)
    if tvl is None:
        tv = [x["tvl"] for x in llama["protocols"] if x["gecko_id"] == gid and x["tvl"]]
        tvl = sum(tv) if tv else None
    fees = [x["total30d"] for x in llama["fees"]
            if x["total30d"] and (x["gecko_id"] == gid or x["name"].lower() == name)]
    fees30 = max(fees) if fees else None
    circ, maxs = cg.get("circulating_supply"), cg.get("max_supply") or cg.get("total_supply")
    return {
        "name": cg.get("name"), "rank": cg.get("market_cap_rank"), "mcap": mcap, "fdv": fdv,
        "fdv_mcap": fdv / mcap if mcap and fdv else None, "circ_pct": circ / maxs if circ and maxs else None,
        "ath": cg.get("ath"), "ath_dist": (cg.get("ath_change_percentage") or 0) / 100,
        "tvl": tvl, "fees30d": fees30, "pf": mcap / (fees30 * 12) if fees30 and mcap else None,
        "categories": (det or {}).get("categories", []), "commits_4w": (det or {}).get("commits_4w"),
    }


# ════════════════════ news_agent ════════════════════
try:
    import feedparser
except ImportError:  # fallback: แยก RSS ด้วย ElementTree
    feedparser = None
try:
    import plotly.express as px
except ImportError:
    px = None

FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "The Block": "https://www.theblock.co/rss.xml",
    "CryptoSlate": "https://cryptoslate.com/feed/",
    "Bitcoin Magazine": "https://bitcoinmagazine.com/feed",
}
GNEWS = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
CACHE_TTL = 1800
HALF_LIFE_H = 48
UA = {"User-Agent": "Mozilla/5.0 (crypto-news-agent/1.0)"}

ALIASES = {
    "BTC": {"strong": ["Bitcoin"], "strip": ["Bitcoin Cash", "Bitcoin SV", "Bitcoin Magazine", "Bitcoin Depot", "Wrapped Bitcoin"]},
    "ETH": {"strong": ["Ethereum"], "weak": ["Ether"], "strip": ["Ethereum Classic"]},
    "SOL": {"strong": ["Solana"]},
    "XRP": {"strong": ["XRP", "XRP Ledger", "XRPL"], "weak": ["Ripple"]},
    "BNB": {"strong": ["BNB Chain", "BNB"]},
    "ADA": {"strong": ["Cardano"]},
    "DOGE": {"strong": ["Dogecoin"], "weak": ["DOGE"],
             "excl": [r"Department of Government Efficiency", r"DOGE (?:cuts|staff|team|office|agency|layoffs)", r"Musk'?s DOGE"]},
    "TRX": {"strong": ["TRON network", "TRON blockchain", "Tron DAO"], "weak": ["TRON", "Tron"], "excl": [r"Tron: (?:Ares|Legacy)"]},
    "AVAX": {"strong": ["AVAX", "Avalanche network", "Avalanche blockchain"], "weak": ["Avalanche"],
             "excl": [r"\bsnow\b", r"\bskier", r"Colorado Avalanche"]},
    "DOT": {"strong": ["Polkadot"]},
    "LINK": {"strong": ["Chainlink"]},
    "NEAR": {"strong": ["NEAR Protocol", "NEAR Intents", "NEAR Foundation", "NEAR AI", "Near Protocol"], "weak": ["NEAR"]},
    "XLM": {"strong": ["XLM", "Stellar Lumens", "Stellar Development Foundation", "Stellar network", "Stellar blockchain"],
            "weak": ["Stellar"], "excl": [r"stellar (?:performance|results|quarter|year|earnings|season|growth)"]},
    "LTC": {"strong": ["Litecoin"]},
    "BCH": {"strong": ["Bitcoin Cash"]},
    "UNI": {"strong": ["Uniswap"]},
    "AAVE": {"strong": ["Aave"]},
    "ENA": {"strong": ["Ethena", "USDe"]},
    "HYPE": {"strong": ["Hyperliquid", "HyperEVM"], "weak": ["HYPE"]},
    "ZEC": {"strong": ["Zcash", "ZEC"]},
    "HBAR": {"strong": ["Hedera", "HBAR"]},
    "SUI": {"strong": ["Sui Network", "Sui blockchain", "Mysten Labs"], "weak": ["Sui", "SUI"]},
    "APT": {"strong": ["Aptos"]},
    "ARB": {"strong": ["Arbitrum"]},
    "OP": {"strong": ["OP Mainnet", "Optimism Foundation", "OP Stack", "Superchain"], "weak": ["Optimism"],
           "excl": [r"(?:cautious|investor|market) optimism", r"optimism (?:about|over|that)"]},
    "TON": {"strong": ["Toncoin", "The Open Network"], "weak": ["TON"]},
    "ATOM": {"strong": ["Cosmos Hub", "Cosmos SDK"], "weak": ["Cosmos"]},
    "ICP": {"strong": ["Internet Computer", "DFINITY"]},
    "FIL": {"strong": ["Filecoin"]},
    "INJ": {"strong": ["Injective"]},
    "TIA": {"strong": ["Celestia"]},
    "SEI": {"strong": ["Sei Network", "Sei blockchain"], "weak": ["Sei"]},
    "RENDER": {"strong": ["Render Network"]},
    "FET": {"strong": ["Fetch.ai", "Artificial Superintelligence Alliance", "ASI Alliance"]},
    "TAO": {"strong": ["Bittensor"]},
    "PEPE": {"strong": ["PEPE"], "weak": ["Pepe"]},
    "SHIB": {"strong": ["Shiba Inu"]},
    "WLD": {"strong": ["Worldcoin", "World Network"]},
    "ONDO": {"strong": ["Ondo Finance"], "weak": ["Ondo"]},
    "PUMP": {"strong": ["Pump.fun", "PumpSwap"]},
    "JUP": {"strong": ["Jupiter Exchange", "Jupiter DEX", "Jupiter aggregator", "Jupiter Labs", "Jupuary"], "weak": ["Jupiter"],
            "excl": [r"planet", r"NASA", r"\bmoons?\b", r"telescope", r"astronom", r"Jupiter, (?:Florida|FL)", r"Jupiter Wellness"]},
    "PENDLE": {"strong": ["Pendle"]},
    "LDO": {"strong": ["Lido DAO", "Lido Finance"], "weak": ["Lido"]},
    "CRV": {"strong": ["Curve Finance"]},
    "ETC": {"strong": ["Ethereum Classic"]},
    "XMR": {"strong": ["Monero"]},
    "ALGO": {"strong": ["Algorand"]},
    "VET": {"strong": ["VeChain"]},
    "POL": {"strong": ["Polygon PoS", "Polygon Labs", "Polygon network"], "weak": ["Polygon"]},
    "BONK": {"strong": ["BONK"], "weak": ["Bonk"]},
    "WIF": {"strong": ["dogwifhat"]},
    "S": {"strong": ["Sonic Labs", "Sonic blockchain", "Sonic network"]},
    "VIRTUAL": {"strong": ["Virtuals Protocol"]},
    "TRUMP": {"strong": [r"\$TRUMP", "TRUMP memecoin", "TRUMP token", "Official Trump"]},
    "KAS": {"strong": ["Kaspa"]},
    "JTO": {"strong": ["Jito"]},
    "RAY": {"strong": ["Raydium"]},
    "GRT": {"strong": ["The Graph"]},
    "IMX": {"strong": ["Immutable X", "Immutable zkEVM"], "weak": ["Immutable"]},
}
CRYPTO_CTX = re.compile(r"crypto|token|blockchain|\bcoin|defi|web3|staking|\betf|layer[- ]?[12]|mainnet|testnet|"
                        r"exchange|binance|coinbase|on-?chain|altcoin|wallet|airdrop|protocol|ledger|stablecoin|\bdex\b|"
                        r"price|rally|whale|memecoin|validator|bitcoin|ethereum", re.I)

POS = {"surge": 1.0, "soar": 1.0, "rally": 0.9, "jump": 0.8, "climb": 0.6, "gain": 0.5, "rise": 0.5, "rises": 0.5,
       "bullish": 1.0, "record high": 1.0, "all-time high": 1.0, "breakout": 0.8, "rebound": 0.6, "recover": 0.5,
       "approval": 0.9, "approve": 0.9, "approved": 0.9, "launch": 0.5, "launches": 0.5, "partnership": 0.7,
       "partners": 0.6, "integrat": 0.6, "upgrade": 0.5, "adopt": 0.7, "inflow": 0.8, "buyback": 0.8, "listing": 0.6,
       "lists": 0.5, "milestone": 0.6, "accumulat": 0.6, "outperform": 0.7, "etf filing": 0.6, "golden cross": 0.7,
       "burn": 0.4, "mainnet": 0.4, "record": 0.5, "wins": 0.6, "expands": 0.5, "backs": 0.4, "invests": 0.6}
NEG = {"plunge": 1.0, "crash": 1.0, "tumble": 0.9, "slump": 0.8, "drop": 0.6, "falls": 0.6, "fall": 0.5, "sink": 0.7,
       "bearish": 1.0, "hack": 1.0, "exploit": 1.0, "breach": 0.9, "stolen": 0.9, "lawsuit": 0.8, "sues": 0.8,
       "charges": 0.7, "fraud": 1.0, "investigation": 0.7, "probe": 0.6, "outflow": 0.8, "delist": 0.9, "ban": 0.8,
       "liquidat": 0.7, "unlock": 0.5, "dump": 0.8, "sell-off": 0.8, "selloff": 0.8, "warning": 0.5, "warns": 0.5,
       "delay": 0.5, "outage": 0.8, "halt": 0.8, "death cross": 0.7, "downgrade": 0.6, "layoff": 0.6, "scam": 0.9,
       "rug": 0.9, "insolven": 1.0, "bankrupt": 1.0, "fine": 0.4, "penalty": 0.6, "loses": 0.6, "slides": 0.6}
_POS_RE = {k: re.compile(rf"\b{re.escape(k)}", re.I) for k in POS}
_NEG_RE = {k: re.compile(rf"\b{re.escape(k)}", re.I) for k in NEG}

_CACHE: dict = {}
_LOCK = threading.Lock()
_SESSION = requests.Session()
_SESSION.headers.update(UA)


def _cache_get(key):
    with _LOCK:
        v = _CACHE.get(key)
    return v[1] if v and time.time() - v[0] < CACHE_TTL else None


def _cache_set(key, val):
    with _LOCK:
        _CACHE[key] = (time.time(), val)
    return val


def clear_cache():
    with _LOCK:
        _CACHE.clear()


def _clean(text: str, n: int = 400) -> str:
    t = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", t).strip()[:n]


def _parse_date(s) -> datetime | None:
    if not s:
        return None
    try:
        d = parsedate_to_datetime(s)
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        except ValueError:
            return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _parse_feed(content: bytes) -> list[dict]:
    out = []
    if feedparser is not None:
        fp = feedparser.parse(content)
        for e in fp.entries:
            pub = None
            for k in ("published_parsed", "updated_parsed"):
                if e.get(k):
                    pub = datetime(*e[k][:6], tzinfo=timezone.utc)
                    break
            src = (e.get("source") or {}).get("title") if isinstance(e.get("source"), dict) else None
            out.append({"title": e.get("title", ""), "link": e.get("link", ""), "summary": e.get("summary", ""),
                        "published": pub, "src_name": src})
        return out
    import xml.etree.ElementTree as ET
    root = ET.fromstring(content)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for it in root.iter("item"):
        out.append({"title": it.findtext("title") or "", "link": it.findtext("link") or "",
                    "summary": it.findtext("description") or "", "published": _parse_date(it.findtext("pubDate")),
                    "src_name": it.findtext("source")})
    for it in root.findall(".//a:entry", ns):
        ln = it.find("a:link", ns)
        out.append({"title": it.findtext("a:title", "", ns), "link": ln.get("href") if ln is not None else "",
                    "summary": it.findtext("a:summary", "", ns),
                    "published": _parse_date(it.findtext("a:updated", None, ns)), "src_name": None})
    return out


def _fetch(url: str) -> list[dict]:
    r = _SESSION.get(url, timeout=12)
    r.raise_for_status()
    return _parse_feed(r.content)


def _gnews_query(ticker: str, name: str | None, days: int) -> str:
    spec = ALIASES.get(ticker, {})
    names = [n for n in spec.get("strong", []) if not n.startswith("\\")][:2]
    if not names and name:
        names = [name]
    terms = [f'"{n}"' for n in names] or [f'"{ticker} crypto"']
    if ticker not in ("S", "OP", "ONE"):
        terms.append(f'"{ticker} price"')
    return " OR ".join(terms) + f" when:{days}d"


def _word_re(term: str, case: bool) -> re.Pattern:
    if term.startswith("\\"):
        return re.compile(term)
    return re.compile(rf"(?<![\w$]){re.escape(term)}(?!\w)", 0 if case else re.I)


_REG_CACHE: dict = {}


def _registry(symbols: list[str], names: dict) -> dict:
    key = (tuple(sorted(symbols)), tuple(sorted((names or {}).items())))
    if key in _REG_CACHE:
        return _REG_CACHE[key]
    reg = {}
    for t in symbols:
        spec = ALIASES.get(t, {})
        strong = list(spec.get("strong", []))
        nm = (names or {}).get(t)
        if nm and len(nm) > 3 and nm.lower() not in {s.lower() for s in strong} and not spec.get("weak"):
            strong.append(nm)
        s_re = [_word_re(x, case=False) for x in strong]
        s_re += [re.compile(rf"\${re.escape(t)}\b"),
                 re.compile(rf"(?<!\w){re.escape(t)}\s*(?:token|coin|price|USDT?|holders|ETF)\b")]
        w_re = [_word_re(x, case=True) for x in spec.get("weak", [])]
        if t not in spec.get("weak", []) and len(t) >= 3:
            w_re.append(_word_re(t, case=True))
        reg[t] = {"strong": s_re, "weak": w_re,
                  "excl": [re.compile(x, re.I) for x in spec.get("excl", [])],
                  "strip": [re.compile(re.escape(x), re.I) for x in spec.get("strip", [])]}
    _REG_CACHE[key] = reg
    return reg


def tag_coins(title: str, summary: str, reg: dict) -> dict[str, int]:
    """คืน {ticker: relevance} (2 = อยู่ในหัวข่าว, 1 = อยู่ในเนื้อหา)"""
    out = {}
    for t, r in reg.items():
        best = 0
        for part, w in ((title, 2), (summary, 1)):
            txt = part
            for s in r["strip"]:
                txt = s.sub(" ", txt)
            if any(p.search(txt) for p in r["strong"]):
                best = max(best, w)
                continue
            full = f"{title} {summary}"
            if any(p.search(txt) for p in r["weak"]) and CRYPTO_CTX.search(full) and not any(e.search(full) for e in r["excl"]):
                best = max(best, w)
        if best:
            out[t] = best
    return out


def _norm_title(t: str) -> str:
    t = re.sub(r"\s+-\s+[^-]{2,40}$", "", t)
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def keyword_score(text: str) -> float:
    p = sum(w for k, w in POS.items() if _POS_RE[k].search(text))
    n = sum(w for k, w in NEG.items() if _NEG_RE[k].search(text))
    return 0.0 if p + n == 0 else float((p - n) / (p + n))


def collect_news(symbols, names=None, focus=None, days: int = 7, max_focus: int = 12, per_coin: int = 25):
    """คืน (items, src) โดย src = {แหล่ง: จำนวนข่าว} (แหล่งที่ล่ม = 0)"""
    symbols = [s.upper() for s in symbols]
    names = {k.upper(): v for k, v in (names or {}).items()}
    focus = [f.upper() for f in (focus or []) if f.upper() in symbols][:max_focus]
    key = ("collect", tuple(sorted(symbols)), tuple(focus), days)
    if (hit := _cache_get(key)) is not None:
        return hit

    jobs = {f"feed:{n}": u for n, u in FEEDS.items()}
    for t in focus:
        jobs[f"gnews:{t}"] = GNEWS.format(q=quote_plus(_gnews_query(t, names.get(t), days)))

    def get(k):
        if (c := _cache_get(("raw", k))) is not None:
            return k, c
        try:
            return k, _cache_set(("raw", k), _fetch(jobs[k]))
        except Exception:  # noqa: BLE001
            return k, None

    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        raw = dict(ex.map(get, jobs))

    reg = _registry(symbols, names)
    now = datetime.now(timezone.utc)
    items, seen_link, keys, src = [], set(), [], {}
    for k, entries in raw.items():
        origin, label = k.split(":", 1)
        src_name = label if origin == "feed" else f"Google News ({label})"
        src[src_name] = len(entries) if entries else 0
        for e in (entries or [])[: (80 if origin == "feed" else per_coin)]:
            pub = e["published"]
            age_h = (now - pub).total_seconds() / 3600 if pub else None
            if age_h is not None and age_h > days * 24:
                continue
            title = _clean(e["title"], 220)
            if not title:
                continue
            summary = _clean(e["summary"])
            link = e["link"]
            nt = _norm_title(title)
            tags = tag_coins(title, summary, reg)
            if origin == "gnews" and label in symbols:
                tags[label] = max(tags.get(label, 0), 2)
            dup_of = None
            if link in seen_link:
                dup_of = next((i for i in items if i["link"] == link), None)
            if dup_of is None:
                for it, kk in zip(items, keys):
                    if kk == nt or (abs(len(kk) - len(nt)) < 25 and SequenceMatcher(None, kk, nt).quick_ratio() > 0.85
                                    and SequenceMatcher(None, kk, nt).ratio() > 0.85):
                        dup_of = it
                        break
            if dup_of is not None:
                dup_of["dup"] += 1
                for t, w in tags.items():
                    dup_of["coins_rel"][t] = max(dup_of["coins_rel"].get(t, 0), w)
                continue
            source = e.get("src_name") or (label if origin == "feed" else (title.rsplit(" - ", 1)[-1] if " - " in title else "Google News"))
            if origin == "gnews" and " - " in title:
                title = title.rsplit(" - ", 1)[0]
            items.append({"title": title, "link": link, "source": source, "summary": summary,
                          "published": pub.isoformat() if pub else None,
                          "age_h": round(age_h, 1) if age_h is not None else None,
                          "coins_rel": dict(tags), "dup": 0, "origin": origin,
                          "kw": keyword_score(f"{title}. {title}. {summary}")})
            seen_link.add(link)
            keys.append(nt)
    items.sort(key=lambda i: (i["age_h"] if i["age_h"] is not None else 1e9))
    for n, it in enumerate(items, 1):
        it["id"] = n
        it["coins"] = sorted(it["coins_rel"], key=lambda t: -it["coins_rel"][t])
    return _cache_set(key, (items, src))


def coin_news(items, ticker: str, k: int = 12) -> list[dict]:
    t = ticker.upper()
    xs = [i for i in items if t in i["coins_rel"]]
    xs.sort(key=lambda i: (-i["coins_rel"][t], -min(i["dup"], 3), i["age_h"] if i["age_h"] is not None else 1e9))
    return xs[:k]


def headlines(items, ticker: str, k: int = 8) -> list[str]:
    return [f"[{i['id']}] {age_txt(i['age_h'])} {i['source']}: {i['title']}" for i in coin_news(items, ticker, k)]


def age_txt(h) -> str:
    if h is None:
        return "?"
    if h < 1:
        return f"{max(1, int(h * 60))} นาที"
    if h < 48:
        return f"{int(h)} ชม."
    return f"{int(h / 24)} วัน"


def _label(s: float) -> tuple[str, str]:
    if s >= 0.35:
        return "BULLISH", "🟢"
    if s >= 0.12:
        return "SLIGHTLY BULLISH", "🟢"
    if s <= -0.35:
        return "BEARISH", "🔴"
    if s <= -0.12:
        return "SLIGHTLY BEARISH", "🟠"
    return "NEUTRAL", "⚪"


def keyword_sentiment(items, symbols) -> dict:
    out = {}
    for t in symbols:
        xs = coin_news(items, t, 40)
        if not xs:
            continue
        ws, ss = [], []
        for i in xs:
            w = 0.5 ** ((i["age_h"] or 72) / HALF_LIFE_H) * i["coins_rel"][t] * (1 + 0.3 * min(i["dup"], 3))
            ws.append(w)
            ss.append(i["kw"])
        tw = sum(ws)
        s = float(sum(w * x for w, x in zip(ws, ss)) / (tw + 1.0)) if tw else 0.0
        conf = min(1.0, tw / 4) * (1 - math.exp(-len(xs) / 3))
        out[t] = {"score": round(s, 3), "confidence": round(conf, 2), "n": len(xs)}
    return out


class CoinSent(BaseModel):
    ticker: str
    sentiment: float
    confidence: float
    summary: str
    drivers: list[str]
    key_ids: list[int]


class NewsReport(BaseModel):
    overall: str
    themes: list[str]
    coins: list[CoinSent]


def analyze_news(items, symbols, llm=None, focus=None, max_coins: int = 14, per_coin: int = 10):
    """Sentiment รายเหรียญ · ใช้ AI (llm) ถ้ามี ไม่งั้นใช้ keyword · คืน (report, mode)"""
    symbols = [s.upper() for s in symbols]
    focus = [f.upper() for f in (focus or [])]
    kw = keyword_sentiment(items, symbols)
    order = [t for t in dict.fromkeys(focus + symbols) if t in kw][:max_coins]
    ctx, allowed = {}, {}
    for t in order:
        xs = coin_news(items, t, per_coin)
        allowed[t] = {i["id"] for i in xs}
        ctx[t] = [f"[{i['id']}] ({age_txt(i['age_h'])}, {i['source']}{', +' + str(i['dup']) + ' เว็บ' if i['dup'] else ''}) "
                  f"{i['title']} — {i['summary'][:160]}" for i in xs]
    ai, mode, overall, themes = {}, "keyword", "", []
    if llm and ctx:
        key = ("ai", tuple(order), tuple(sorted(i for s in allowed.values() for i in s)))
        res = _cache_get(key)
        if res is None:
            prompt = f"""คุณคือ News & Sentiment Agent ด้านคริปโต วิเคราะห์ข่าวต่อไปนี้ (ตัวเลขใน [] คือหมายเลขข่าว):
{json.dumps(ctx, ensure_ascii=False, indent=0)}

สำหรับแต่ละเหรียญ ให้:
- sentiment: -1 (ลบมาก) ถึง 1 (บวกมาก) = ผลกระทบต่อราคา 2-8 สัปดาห์ (ข่าวใหม่และข่าวที่ลงหลายเว็บสำคัญกว่า, ข่าวทั่วตลาดที่ไม่ได้เฉพาะเหรียญให้น้ำหนักน้อย)
- confidence: 0-1 (ข่าวน้อย/ไม่ชัด = ต่ำ)
- summary: สรุปภาษาไทย 1-2 ประโยค ว่าข่าวกำลังบอกอะไรกับราคา
- drivers: ประเด็นหลัก 1-3 ข้อ (ภาษาไทยสั้นๆ)
- key_ids: หมายเลขข่าว 1-4 ข่าวที่ใช้ตัดสิน (ต้องเป็นหมายเลขของเหรียญนั้นเท่านั้น)
และ overall = ภาพรวมข่าวตลาด 1-2 ประโยค, themes = ประเด็นร้อน 3-5 ข้อ
ห้ามแต่งข้อมูลที่ไม่มีในข่าว"""
            try:
                r = llm(prompt, NewsReport)
                if isinstance(r, dict):
                    r = NewsReport.model_validate(r)
                res = r if isinstance(r, NewsReport) else None
            except Exception:  # noqa: BLE001
                res = None
            if res is not None:
                _cache_set(key, res)
        if res is not None:
            mode, overall, themes = "ai", res.overall, list(res.themes)[:5]
            for c in res.coins:
                t = c.ticker.upper().replace("USDT", "").strip()
                if t in allowed:
                    ai[t] = {"score": max(-1.0, min(1.0, float(c.sentiment))),
                             "confidence": max(0.0, min(1.0, float(c.confidence))),
                             "summary": c.summary, "drivers": list(c.drivers)[:3],
                             "key_ids": [i for i in c.key_ids if i in allowed[t]][:4]}
    coins = {}
    for t, k in kw.items():
        a = ai.get(t)
        if a:
            s = 0.7 * a["score"] + 0.3 * k["score"]
            conf = max(min(1.0, 0.7 * a["confidence"] + 0.3 * k["confidence"]), 0.1)
        else:
            s, conf = k["score"], k["confidence"]
        lab, emo = _label(s)
        top = coin_news(items, t, 3)
        coins[t] = {"score": round(s, 3), "label": lab, "emoji": emo, "confidence": round(conf, 2), "n": k["n"],
                    "summary": (a or {}).get("summary") or (f"ข่าว {k['n']} ข่าว · ล่าสุด: {top[0]['title']}" if top else ""),
                    "drivers": (a or {}).get("drivers", []),
                    "key_ids": (a or {}).get("key_ids") or [i["id"] for i in top[:2]],
                    "kw_score": k["score"], "ai_score": (a or {}).get("score"), "mode": "ai" if a else "keyword"}
    if not overall and items:
        avg = sum(i["kw"] for i in items[:60]) / min(len(items), 60)
        overall = f"ข่าวล่าสุด {len(items)} ข่าว · โทนรวม {_label(avg)[1]} {_label(avg)[0]} (keyword)"
    return {"overall": overall, "themes": themes, "coins": coins}, mode


def news_score_map(report) -> dict:
    return {t: round(c["score"] * c["confidence"], 3) for t, c in (report or {}).get("coins", {}).items()}


def is_strong_negative(report, ticker: str, th: float = -0.45) -> bool:
    c = (report or {}).get("coins", {}).get(ticker.upper())
    return bool(c and c["score"] <= th and c["confidence"] >= 0.45)


def news_md(report, items, ticker: str, k: int = 6) -> str:
    c = (report or {}).get("coins", {}).get(ticker.upper())
    if not c:
        return "- ไม่พบข่าวเฉพาะเหรียญใน 7 วันล่าสุด"
    byid = {i["id"]: i for i in items}
    lines = [f"**{c['emoji']} {c['label']}** (คะแนน {c['score']:+.2f} · ความมั่นใจ {c['confidence'] * 100:.0f}% · {c['n']} ข่าว"
             f"{'' if c['mode'] == 'ai' else ' · keyword'}) — {c['summary']}"]
    if c["drivers"]:
        lines.append("ประเด็น: " + " · ".join(c["drivers"]))
    for i in c["key_ids"]:
        if i in byid:
            it = byid[i]
            lines.append(f"- 📌 [{it['title']}]({it['link']}) · {it['source']} · {age_txt(it['age_h'])}")
    for it in coin_news(items, ticker, k):
        if it["id"] not in c["key_ids"]:
            lines.append(f"- {'🟢' if it['kw'] > 0.2 else '🔴' if it['kw'] < -0.2 else '⚪'} [{it['title']}]({it['link']}) · "
                         f"{it['source']} · {age_txt(it['age_h'])}")
    return "\n".join(lines)


def render_news_tab(report, items, focus=None, src=None, mode="keyword", key: str = "news"):
    import streamlit as st
    focus = [f.upper() for f in (focus or [])]
    coins = (report or {}).get("coins", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ข่าว 7 วัน (ไม่ซ้ำ)", len(items))
    c2.metric("แหล่งที่ใช้ได้", f"{sum(1 for v in (src or {}).values() if v)}/{len(src or {})}")
    c3.metric("เหรียญที่มีข่าว", len(coins))
    c4.metric("โหมด Sentiment", "🧠 AI" if mode == "ai" else "🔤 Keyword")
    if report.get("overall"):
        st.info(f"**ภาพรวมข่าว:** {report['overall']}")
    if report.get("themes"):
        st.markdown("**ประเด็นร้อน:** " + " · ".join(f"`{t}`" for t in report["themes"]))
    if coins:
        df = pd.DataFrame([{"Ticker": t, "Sentiment": v["score"], "Label": v["label"], "ข่าว": v["n"],
                            "Confidence": v["confidence"], "แนะนำ": "⭐" if t in focus else ""} for t, v in coins.items()])
        order = sorted(coins, key=lambda t: (t not in focus, -abs(coins[t]["score"])))
        if px is not None:
            fig = px.bar(df.set_index("Ticker").loc[order].reset_index(), x="Ticker", y="Sentiment", color="Sentiment",
                         color_continuous_scale=["#dc2626", "#9ca3af", "#16a34a"], range_color=[-1, 1],
                         hover_data=["Label", "ข่าว", "Confidence"], template="plotly_dark", height=320)
            fig.update_layout(margin=dict(l=8, r=8, t=10, b=8), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        sel = st.selectbox("ดูข่าวของ", ["⭐ เหรียญที่แนะนำ/วิเคราะห์ก่อน"] + order, key=f"{key}_sel")
        show = order if sel.startswith("⭐") else [sel]
        for t in show:
            v = coins[t]
            with st.expander(f"{'⭐ ' if t in focus else ''}{t} {v['emoji']} {v['label']} ({v['score']:+.2f}) · {v['n']} ข่าว",
                             expanded=(len(show) == 1 or t in focus[:3])):
                st.markdown(news_md(report, items, t, 8))
    with st.expander(f"📰 ข่าวทั้งหมด ({len(items)})"):
        st.dataframe(pd.DataFrame([{"#": i["id"], "อายุ": age_txt(i["age_h"]), "แหล่ง": i["source"],
                                    "เหรียญ": ", ".join(i["coins"]), "หัวข่าว": i["title"], "ลิงก์": i["link"],
                                    "โทน": round(i["kw"], 2), "ซ้ำ": i["dup"]} for i in items]),
                     use_container_width=True, hide_index=True,
                     column_config={"ลิงก์": st.column_config.LinkColumn()})
        if src:
            st.caption("แหล่งข่าว: " + " · ".join(f"{k} {'✅' if v else '❌'} {v}" for k, v in src.items()))

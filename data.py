#!/usr/bin/env python3
"""
Binance → HuggingFace Dataset Pipeline
=======================================
严格对齐 README.md 中 LSTM 所需的全部特征维度:

  LSTM Input Shape: (batch, 60, 15)
  ┌─────────────────────────────────────────────────────────────────┐
  │  #   Feature               Category         Timeframe  Source  │
  │ ─── ─────────────────────  ──────────────── ───────── ──────── │
  │  1   rsi_14                Price-derived     1m        Candles  │
  │  2   ema_12                Price-derived     1m        Candles  │
  │  3   ema_26                Price-derived     1m        Candles  │
  │  4   ema_crossover_1m      Price-derived     1m        Candles  │
  │  5   atr_14                Price-derived     1m        Candles  │
  │  6   momentum_10           Price-derived     1m        Candles  │
  │  7   volatility_20         Price-derived     1m        Candles  │
  │  8   volume_ratio          Microstructure    1m        Candles  │
  │  9   ob_imbalance          Microstructure    1m        Candles  │
  │ 10   ema_crossover_15m     Multi-scale       15m       Resamp.  │
  │ 11   momentum_15m          Multi-scale       15m       Resamp.  │
  │ 12   ema_crossover_1h      Multi-scale       1h        Resamp.  │
  │ 13   momentum_1h           Multi-scale       1h        Resamp.  │
  │ 14   funding_rate          Web3 sentiment    8h        Futures  │
  │ 15   taker_buy_sell_ratio  Order flow        5m        Futures  │
  └─────────────────────────────────────────────────────────────────┘
  + alpha_score (rule-based, for ensemble mode — NOT direct LSTM input)

数据源 (Binance 公开 API, 无需 Key):
  ① Spot   K线  GET api.binance.com/api/v3/klines
  ② Futures K线  GET fapi.binance.com/fapi/v1/klines
  ③ 资金费率     GET fapi.binance.com/fapi/v1/fundingRate
  ④ Taker 多空比 GET fapi.binance.com/futures/data/takerlongshortRatio

依赖:  pip install requests pandas numpy tqdm
代理:  脚本顶部 PROXY 变量 (中国大陆必填)
运行:  python data.py
"""

import os
import sys
import time
import json
import math
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List, Dict, Any

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                          配 置 区                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── 66 种目标币 ──────────────────────────────────────────────────────
COINS: List[str] = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "TAO", "PEPE", "ADA", "ZEC",
    "PAXG", "LINK", "SUI", "TRUMP", "AVAX", "FET", "DOT", "LTC", "NEAR", "TRX",
    "UNI", "BONK", "WIF", "ENA", "PUMP", "WLD", "PENGU", "EIGEN", "HBAR", "ASTER",
    "AAVE", "APT", "FIL", "VIRTUAL", "CFX", "SHIB", "XPL", "ICP", "XLM", "S",
    "CAKE", "ARB", "WLFI", "CRV", "FLOKI", "ONDO", "ZEN", "SEI", "TON", "POL",
    "PENDLE", "TUT", "BIO", "LINEA", "PLUME", "100CHEEMS", "FORM", "AVNT", "OMNI",
    "LISTA", "OPEN", "SOMI", "HEMI", "EDEN", "MIRA", "BMT",
]

# ── Binance 交易对特殊映射 ────────────────────────────────────────────
SYMBOL_OVERRIDES: Dict[str, List[str]] = {
    "S":          ["SUSDT", "SONEUSDT"],
    "100CHEEMS":  ["1000CHEEMSUSDT", "100CHEEMSUSDT", "CHEEMSUSDT"],
    "PUMP":       ["PUMPUSDT", "PUMPFUNUSDT"],
    "WLFI":       ["WLFIUSDT"],
    "XPL":        ["XPLUSDT"],
    "SOMI":       ["SOMIUSDT"],
    "HEMI":       ["HEMIUSDT"],
    "EDEN":       ["EDENUSDT"],
    "MIRA":       ["MIRAUSDT"],
    "AVNT":       ["AVNTUSDT"],
    "LINEA":      ["LINEAUSDT"],
    "PLUME":      ["PLUMEUSDT"],
    "TUT":        ["TUTUTUSDT", "TUTUSDT"],
    "OPEN":       ["OPENUSDT"],
    "ASTER":      ["ASTERUSDT"],
    "BMT":        ["BMTUSDT"],
}

# ══════════════════════════════════════════════════════════════════════
#  ★★★  代理设置 (中国大陆必填)  ★★★
#
#  Clash:     "http://127.0.0.1:7890"
#  V2RayN:    "http://127.0.0.1:10809"
#  SOCKS5:    "socks5://127.0.0.1:7891"  (需 pip install requests[socks])
#  海外直连:   留空 ""
# ══════════════════════════════════════════════════════════════════════
PROXY = ""

# ── Binance API 端点 ─────────────────────────────────────────────────
SPOT_BASE_URLS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]
FUTURES_BASE_URLS = [
    "https://fapi.binance.com",
]

SPOT_KLINES_PATH    = "/api/v3/klines"
FUTURES_KLINES_PATH = "/fapi/v1/klines"
FUNDING_RATE_PATH   = "/fapi/v1/fundingRate"
TAKER_LS_PATH       = "/futures/data/takerlongshortRatio"

# ── 下载参数 ─────────────────────────────────────────────────────────
INTERVAL           = "1m"
DAYS               = 90          # README: "Recommended: fetch 90 days"
LIMIT_PER_REQUEST  = 1000
REQUEST_INTERVAL   = 0.12        # 请求间隔 (秒)
MAX_RETRIES        = 3
RETRY_WAIT         = 5
CONNECT_TIMEOUT    = 10
READ_TIMEOUT       = 30

# ── 特征参数 (对齐 README Feature Summary) ───────────────────────────
RSI_PERIOD          = 14         # README: RSI(14)
EMA_SHORT           = 12         # README: EMA(12)
EMA_LONG            = 26         # README: EMA(26)
ATR_PERIOD          = 14         # README: ATR(14)
MOMENTUM_PERIOD     = 10         # README: Momentum(10)
VOLATILITY_WINDOW   = 20         # README: Volatility(20)
VOLUME_RATIO_WINDOW = 20         # README: Volume ratio

# ── 多时间尺度 (README: "resampled 15m / 1h") ───────────────────────
RESAMPLE_15M = 15
RESAMPLE_1H  = 60

# ── Rule-Based Alpha 权重 (README: Design Decisions) ────────────────
ALPHA_W_RSI       = 0.3
ALPHA_W_MOMENTUM  = 0.3
ALPHA_W_EMA_CROSS = 0.3
ALPHA_W_VOL       = 0.1

# ── 数据切分 (README: "70 / 15 / 15") ───────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── 输出路径 ─────────────────────────────────────────────────────────
OUTPUT_DIR = Path.home() / "Desktop" / "HKSG"

# ── LSTM 输入特征列 (严格对齐 README Feature Summary) ────────────────
LSTM_FEATURE_COLUMNS: List[str] = [
    # ── 1m Price-derived (7) ──
    "rsi_14",
    "ema_12",
    "ema_26",
    "ema_crossover_1m",
    "atr_14",
    "momentum_10",
    "volatility_20",
    # ── Microstructure (2) ──
    "volume_ratio",
    "ob_imbalance",
    # ── Multi-scale trend (4) ──
    "ema_crossover_15m",
    "momentum_15m",
    "ema_crossover_1h",
    "momentum_1h",
    # ── Web3 sentiment (1) ──
    "funding_rate",
    # ── Order flow (1) ──
    "taker_buy_sell_ratio",
]
# Total: 15 features → LSTM input shape (batch, 60, 15)

# ── 辅助列 (非 LSTM 输入, 用于 ensemble / 标签计算) ──────────────────
AUX_COLUMNS: List[str] = [
    "alpha_score",           # Rule-based alpha (ensemble 模式用)
]

# ── K 线原始列定义 ───────────────────────────────────────────────────
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "_ignore",
]
FLOAT_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                       网 络 层                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

SESSION: Optional[requests.Session] = None
_WORKING_SPOT_BASE: Optional[str] = None
_WORKING_FUTURES_BASE: Optional[str] = None


def build_session() -> requests.Session:
    """构建带重试 + 代理的 Session"""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    if PROXY:
        session.proxies = {"http": PROXY, "https": PROXY}
        print(f"   🌐 代理: {PROXY}")
    else:
        env_p = (os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
                 or os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY"))
        if env_p:
            print(f"   🌐 环境变量代理: {env_p}")

    return session


def test_connectivity() -> Tuple[bool, bool]:
    """测试 Spot / Futures 连通性, 缓存可用 base URL"""
    global _WORKING_SPOT_BASE, _WORKING_FUTURES_BASE

    print("   🔍 测试 Binance API 连通性 ...")
    print()

    # Spot
    spot_ok = False
    for base in SPOT_BASE_URLS:
        try:
            r = SESSION.get(f"{base}/api/v3/ping",
                            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            if r.status_code == 200:
                print(f"   ✅ Spot 可用:    {base}")
                _WORKING_SPOT_BASE = base
                spot_ok = True
                break
            else:
                print(f"   ⚠  Spot {base} → HTTP {r.status_code}")
        except requests.exceptions.ProxyError:
            print(f"   ❌ Spot {base} 代理错误")
        except requests.exceptions.SSLError:
            print(f"   ❌ Spot {base} SSL 错误")
        except requests.exceptions.ConnectTimeout:
            print(f"   ❌ Spot {base} 连接超时")
        except requests.exceptions.ReadTimeout:
            print(f"   ❌ Spot {base} 读取超时")
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Spot {base} 无法连接")
        except Exception as e:
            print(f"   ❌ Spot {base} {type(e).__name__}: {e}")

    if not spot_ok:
        print("   ❌ 全部 Spot 端点不可用")

    # Futures
    futures_ok = False
    for base in FUTURES_BASE_URLS:
        try:
            r = SESSION.get(f"{base}/fapi/v1/ping",
                            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            if r.status_code == 200:
                print(f"   ✅ Futures 可用:  {base}")
                _WORKING_FUTURES_BASE = base
                futures_ok = True
                break
            else:
                print(f"   ⚠  Futures {base} → HTTP {r.status_code}")
        except requests.exceptions.ProxyError:
            print(f"   ❌ Futures {base} 代理错误")
        except requests.exceptions.SSLError:
            print(f"   ❌ Futures {base} SSL 错误")
        except requests.exceptions.ConnectTimeout:
            print(f"   ❌ Futures {base} 连接超时")
        except requests.exceptions.ReadTimeout:
            print(f"   ❌ Futures {base} 读取超时")
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Futures {base} 无法连接")
        except Exception as e:
            print(f"   ❌ Futures {base} {type(e).__name__}: {e}")

    if not futures_ok:
        print("   ❌ 全部 Futures 端点不可用")

    print()
    return spot_ok, futures_ok


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                   Binance API 请 求 层                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _api_get(url: str, params: dict, tag: str = "") -> Optional[Any]:
    """通用 GET, 带重试 / 限速 / 错误处理"""
    for attempt in range(MAX_RETRIES):
        try:
            resp = SESSION.get(url, params=params,
                               timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", RETRY_WAIT * 2))
                print(f"        ⏳ [{tag}] 429 限速, 等 {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code == 418:
                print(f"        🚫 [{tag}] 418 IP封禁, 等 120s")
                time.sleep(120)
                continue
            if resp.status_code != 200:
                return None

            data = resp.json()
            if isinstance(data, dict) and "code" in data:
                return None
            return data

        except requests.exceptions.ProxyError:
            print(f"        ❌ [{tag}] 代理错误 ({attempt+1}/{MAX_RETRIES})")
        except requests.exceptions.SSLError:
            print(f"        ❌ [{tag}] SSL错误 ({attempt+1}/{MAX_RETRIES})")
        except requests.exceptions.ConnectTimeout:
            print(f"        ❌ [{tag}] 连接超时 ({attempt+1}/{MAX_RETRIES})")
        except requests.exceptions.ReadTimeout:
            print(f"        ❌ [{tag}] 读取超时 ({attempt+1}/{MAX_RETRIES})")
        except requests.exceptions.ConnectionError:
            print(f"        ❌ [{tag}] 连接失败 ({attempt+1}/{MAX_RETRIES})")
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"        ❌ [{tag}] {type(e).__name__} ({attempt+1}/{MAX_RETRIES})")

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_WAIT)

    return None


# ╔══════════════════════════════════════════════════════════════════════╗
# ║          数 据 下 载 : K线 / 资金费率 / Taker多空比                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _download_klines_from_endpoint(
    base_url: str, path: str, symbol: str,
    start_ms: int, end_ms: int, source_name: str,
) -> Optional[List[list]]:
    """从单个端点分页下载 1m K线"""
    all_klines: List[list] = []
    cursor = start_ms
    url = f"{base_url}{path}"

    while cursor < end_ms:
        params = {
            "symbol": symbol, "interval": INTERVAL,
            "startTime": cursor, "endTime": end_ms,
            "limit": LIMIT_PER_REQUEST,
        }
        page = _api_get(url, params, tag=f"{symbol}/{source_name}")
        if page is None or len(page) == 0:
            break

        all_klines.extend(page)
        cursor = page[-1][0] + 60_000
        time.sleep(REQUEST_INTERVAL)

        if len(all_klines) % 30_000 < LIMIT_PER_REQUEST and all_klines:
            print(f"          {source_name}: {len(all_klines):>9,} candles ...")
        if len(page) < LIMIT_PER_REQUEST:
            break

    return all_klines if all_klines else None


def download_klines(
    symbol: str, start_ms: int, end_ms: int
) -> Tuple[Optional[List[list]], str]:
    """下载 K线: 先 Spot (多镜像), 后 Futures"""
    # Spot
    spot_order = ([_WORKING_SPOT_BASE] + [b for b in SPOT_BASE_URLS if b != _WORKING_SPOT_BASE]
                  if _WORKING_SPOT_BASE else SPOT_BASE_URLS)
    for base in spot_order:
        tag = f"Spot({base.split('//')[1].split('.')[0]})"
        result = _download_klines_from_endpoint(
            base, SPOT_KLINES_PATH, symbol, start_ms, end_ms, tag)
        if result is not None:
            return result, "Spot"

    # Futures
    futures_order = ([_WORKING_FUTURES_BASE] + [b for b in FUTURES_BASE_URLS if b != _WORKING_FUTURES_BASE]
                     if _WORKING_FUTURES_BASE else FUTURES_BASE_URLS)
    for base in futures_order:
        result = _download_klines_from_endpoint(
            base, FUTURES_KLINES_PATH, symbol, start_ms, end_ms, "Futures")
        if result is not None:
            return result, "Futures"

    return None, "N/A"


def download_funding_rate(
    symbol: str, start_ms: int, end_ms: int
) -> pd.DataFrame:
    """
    下载资金费率 — README: "Funding rate (perp sentiment)"
    来源: fapi/v1/fundingRate, 每 8 小时一个数据点
    """
    empty = pd.DataFrame(columns=["funding_time", "funding_rate"])
    if not _WORKING_FUTURES_BASE:
        return empty

    all_data: List[dict] = []
    cursor = start_ms
    url = f"{_WORKING_FUTURES_BASE}{FUNDING_RATE_PATH}"

    while cursor < end_ms:
        params = {"symbol": symbol, "startTime": cursor,
                  "endTime": end_ms, "limit": 1000}
        page = _api_get(url, params, tag=f"{symbol}/funding")
        if page is None or len(page) == 0:
            break
        all_data.extend(page)
        cursor = page[-1].get("fundingTime", 0) + 1
        time.sleep(REQUEST_INTERVAL)
        if len(page) < 1000:
            break

    if not all_data:
        return empty

    df = pd.DataFrame(all_data)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return (df[["funding_time", "funding_rate"]]
            .drop_duplicates().sort_values("funding_time").reset_index(drop=True))


def download_taker_ratio(
    symbol: str, start_ms: int, end_ms: int
) -> pd.DataFrame:
    """
    下载 Taker 多空比 — README: "Taker buy/sell ratio (aggressor flow)"
    来源: futures/data/takerlongshortRatio, 5 分钟粒度
    """
    empty = pd.DataFrame(columns=["taker_time", "taker_buy_sell_ratio"])
    if not _WORKING_FUTURES_BASE:
        return empty

    all_data: List[dict] = []
    cursor = start_ms
    url = f"{_WORKING_FUTURES_BASE}{TAKER_LS_PATH}"

    while cursor < end_ms:
        params = {"symbol": symbol, "period": "5m",
                  "startTime": cursor, "endTime": end_ms, "limit": 500}
        page = _api_get(url, params, tag=f"{symbol}/taker")
        if page is None or len(page) == 0:
            break
        all_data.extend(page)
        cursor = page[-1].get("timestamp", 0) + 1
        time.sleep(REQUEST_INTERVAL)
        if len(page) < 500:
            break

    if not all_data:
        return empty

    df = pd.DataFrame(all_data)
    df["taker_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["taker_buy_sell_ratio"] = df["buySellRatio"].astype(float)
    return (df[["taker_time", "taker_buy_sell_ratio"]]
            .drop_duplicates().sort_values("taker_time").reset_index(drop=True))


# ╔══════════════════════════════════════════════════════════════════════╗
# ║        特 征 计 算 — 严格对齐 README Feature Summary                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── 基础数学 ─────────────────────────────────────────────────────────

def _ema(data: np.ndarray, span: int) -> np.ndarray:
    out = np.empty_like(data, dtype=np.float64)
    alpha = 2.0 / (span + 1.0)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


# ── Feature #1: RSI(14) — README "Price-derived" ────────────────────

def calc_rsi(close: np.ndarray, period: int = RSI_PERIOD) -> np.ndarray:
    """Wilder RSI, 默认值 50 (中性)"""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


# ── Features #2,#3,#4: EMA(12), EMA(26), EMA crossover ─────────────

def calc_ema_crossover(
    close: np.ndarray, short: int = EMA_SHORT, long: int = EMA_LONG
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (ema_short, ema_long, crossover)
    crossover = (EMA_short - EMA_long) / EMA_long
    README: "EMA Crossover: (EMA12 - EMA26) / EMA26, scaled"
    """
    ema_s = _ema(close, short)
    ema_l = _ema(close, long)
    cross = np.where(ema_l != 0, (ema_s - ema_l) / ema_l, 0.0)
    return ema_s, ema_l, cross


# ── Feature #5: ATR(14) — README "Price-derived" ────────────────────

def calc_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    period: int = ATR_PERIOD,
) -> np.ndarray:
    """
    Average True Range — README: "ATR(14) measures recent true range"
    也被 RiskShield 用于动态止损: stop = entry - ATR * multiplier
    """
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    return _ema(tr, period)


# ── Feature #6: Momentum(10) — README "Price-derived" ───────────────

def calc_momentum(
    close: np.ndarray, period: int = MOMENTUM_PERIOD
) -> np.ndarray:
    """
    Rate of change: (close[i] - close[i-period]) / close[i-period]
    README: "Rate of change over 10 candles, normalized"
    """
    mom = np.zeros(len(close))
    for i in range(period, len(close)):
        if close[i - period] != 0:
            mom[i] = (close[i] - close[i - period]) / close[i - period]
    return mom


# ── Feature #7: Volatility(20) — README "Price-derived" ─────────────

def calc_volatility(
    close: np.ndarray, window: int = VOLATILITY_WINDOW
) -> np.ndarray:
    """收益率标准差 — README: "Volatility(20)"."""
    n = len(close)
    vol = np.zeros(n)
    if n < 2:
        return vol
    returns = np.zeros(n)
    for i in range(1, n):
        if close[i - 1] != 0:
            returns[i] = (close[i] - close[i - 1]) / close[i - 1]
    for i in range(window, n):
        vol[i] = np.std(returns[i - window + 1: i + 1])
    return vol


# ── Feature #8: Volume Ratio — README "Microstructure" ──────────────

def calc_volume_ratio(
    volume: np.ndarray, window: int = VOLUME_RATIO_WINDOW
) -> np.ndarray:
    """current_volume / SMA(volume, 20) — 量能异常检测"""
    vr = np.ones(len(volume))
    for i in range(window, len(volume)):
        avg = np.mean(volume[i - window: i])
        if avg > 0:
            vr[i] = volume[i] / avg
    return vr


# ── Feature #9: OB Imbalance — README "Microstructure" ──────────────

def calc_ob_imbalance(
    taker_buy_vol: np.ndarray, total_vol: np.ndarray
) -> np.ndarray:
    """
    订单簿失衡代理: (taker_buy / total - 0.5) * 2 ∈ [-1, 1]
    README: "Order book imbalance (microstructure)"
    离线用 taker_buy_base_volume 近似 (真实 L2 数据无法回溯)
    """
    ob = np.zeros(len(total_vol))
    mask = total_vol > 0
    ob[mask] = (taker_buy_vol[mask] / total_vol[mask] - 0.5) * 2.0
    return ob


# ── Alpha Score — README "Design Decisions: rule-based alpha" ────────

def calc_alpha_score(
    rsi: np.ndarray, momentum: np.ndarray,
    ema_cross: np.ndarray, volatility: np.ndarray,
) -> np.ndarray:
    """
    README 权重: RSI=0.3, Momentum=0.3, EMA_Cross=0.3, Vol=0.1
    用于 ensemble 模式, 非 LSTM 直接输入
    """
    rsi_s = (50.0 - rsi) / 50.0
    mom_s = np.tanh(momentum * 10.0)
    ema_s = np.tanh(ema_cross * 100.0)
    vol_s = -np.tanh(volatility * 100.0)
    return (ALPHA_W_RSI * rsi_s + ALPHA_W_MOMENTUM * mom_s
            + ALPHA_W_EMA_CROSS * ema_s + ALPHA_W_VOL * vol_s)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║        多 时 间 尺 度 — README "resampled 15m / 1h"                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """将 1m K 线重采样为 N 分钟 OHLCV"""
    df_r = df.set_index("open_time").copy()
    rule = f"{minutes}min"
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    resampled = df_r.resample(rule, label="left", closed="left").agg(agg).dropna()
    return resampled.reset_index()


def compute_multiscale_features(
    df: pd.DataFrame, minutes: int, suffix: str
) -> pd.DataFrame:
    """
    Features #10-#13: 重采样 → EMA crossover + Momentum →
    merge_asof backward (无未来泄露)
    README: "15m features capture intraday trend direction"
            "1h features capture session-level regime"
    """
    resampled = resample_ohlcv(df, minutes)
    if len(resampled) < 2:
        df[f"ema_crossover_{suffix}"] = 0.0
        df[f"momentum_{suffix}"] = 0.0
        return df

    close_r = resampled["close"].values.astype(np.float64)
    _, _, cross_r = calc_ema_crossover(close_r)
    mom_r = calc_momentum(close_r)

    resampled[f"ema_crossover_{suffix}"] = np.round(cross_r, 8)
    resampled[f"momentum_{suffix}"] = np.round(mom_r, 8)

    cols = ["open_time", f"ema_crossover_{suffix}", f"momentum_{suffix}"]
    merged = pd.merge_asof(
        df.sort_values("open_time"),
        resampled[cols].sort_values("open_time"),
        on="open_time", direction="backward",
    )
    merged[f"ema_crossover_{suffix}"] = merged[f"ema_crossover_{suffix}"].fillna(0.0)
    merged[f"momentum_{suffix}"] = merged[f"momentum_{suffix}"].fillna(0.0)
    return merged


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                数 据 处 理 主 管 线                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

def raw_to_dataframe(raw_klines: list, symbol: str) -> pd.DataFrame:
    """API 原始数据 → 干净 DataFrame"""
    df = pd.DataFrame(raw_klines, columns=KLINE_COLUMNS).drop(columns=["_ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce").astype(int)
    df.insert(0, "symbol", symbol)
    df = df.drop_duplicates(subset=["open_time"], keep="first")
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def merge_supplementary(
    df: pd.DataFrame,
    funding_df: pd.DataFrame,
    taker_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    合并 Web3 补充数据到 1m K 线
    README: "fail gracefully (features default to neutral values on timeout)"
      funding_rate   默认 0.0  (中性)
      taker_ratio    默认 1.0  (多空平衡)
    """
    df = df.sort_values("open_time").copy()

    # Feature #14: funding_rate
    if not funding_df.empty:
        df = pd.merge_asof(
            df, funding_df.sort_values("funding_time"),
            left_on="open_time", right_on="funding_time",
            direction="backward")
        if "funding_time" in df.columns:
            df.drop(columns=["funding_time"], inplace=True)
    if "funding_rate" not in df.columns:
        df["funding_rate"] = 0.0
    else:
        df["funding_rate"] = df["funding_rate"].fillna(0.0)

    # Feature #15: taker_buy_sell_ratio
    if not taker_df.empty:
        df = pd.merge_asof(
            df, taker_df.sort_values("taker_time"),
            left_on="open_time", right_on="taker_time",
            direction="backward")
        if "taker_time" in df.columns:
            df.drop(columns=["taker_time"], inplace=True)
    if "taker_buy_sell_ratio" not in df.columns:
        df["taker_buy_sell_ratio"] = 1.0
    else:
        df["taker_buy_sell_ratio"] = df["taker_buy_sell_ratio"].fillna(1.0)

    return df


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算全部 15 个 LSTM 特征 + 1 个辅助 alpha_score
    对齐 README Feature Summary 表 + Architecture 图
    """
    close  = df["close"].values.astype(np.float64)
    high   = df["high"].values.astype(np.float64)
    low    = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    tb_vol = df["taker_buy_base_volume"].values.astype(np.float64)

    # ── 1m Price-derived (#1 ~ #7) ──────────────────────────────
    rsi = calc_rsi(close)
    ema_s, ema_l, cross_1m = calc_ema_crossover(close)
    atr = calc_atr(high, low, close)
    momentum = calc_momentum(close)
    volatility = calc_volatility(close)
    vol_ratio = calc_volume_ratio(volume)

    df["rsi_14"]           = np.round(rsi, 4)
    df["ema_12"]           = np.round(ema_s, 6)
    df["ema_26"]           = np.round(ema_l, 6)
    df["ema_crossover_1m"] = np.round(cross_1m, 8)
    df["atr_14"]           = np.round(atr, 6)
    df["momentum_10"]      = np.round(momentum, 8)
    df["volatility_20"]    = np.round(volatility, 8)

    # ── Microstructure (#8, #9) ─────────────────────────────────
    df["volume_ratio"]     = np.round(vol_ratio, 4)
    df["ob_imbalance"]     = np.round(calc_ob_imbalance(tb_vol, volume), 4)

    # ── Multi-scale (#10, #11): 15m ─────────────────────────────
    df = compute_multiscale_features(df, RESAMPLE_15M, "15m")

    # ── Multi-scale (#12, #13): 1h ──────────────────────────────
    df = compute_multiscale_features(df, RESAMPLE_1H, "1h")

    # ── Web3 (#14, #15) 已在 merge_supplementary 中合并 ────────

    # ── Auxiliary: alpha_score (ensemble 用) ─────────────────────
    df["alpha_score"] = np.round(
        calc_alpha_score(
            df["rsi_14"].values, df["momentum_10"].values,
            df["ema_crossover_1m"].values, df["volatility_20"].values,
        ), 4)

    return df


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                   切 分 + 保 存 + 校 验                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

def split_chronological(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    README: "Time-series split (70 / 15 / 15)"
    "Chronological ordering is strictly enforced — no shuffling, no future leakage."
    """
    n = len(df)
    i_train = int(n * TRAIN_RATIO)
    i_val = int(n * (TRAIN_RATIO + VAL_RATIO))
    return {
        "train":      df.iloc[:i_train].copy(),
        "validation": df.iloc[i_train:i_val].copy(),
        "test":       df.iloc[i_val:].copy(),
    }


def save_splits(splits: Dict[str, pd.DataFrame], symbol: str) -> None:
    for name, sdf in splits.items():
        d = OUTPUT_DIR / "data" / name
        d.mkdir(parents=True, exist_ok=True)
        sdf.to_csv(d / f"{symbol}.csv", index=False)


def validate_features(df: pd.DataFrame, symbol: str) -> bool:
    """验证所有 LSTM 特征列都存在且无全 NaN"""
    missing = [c for c in LSTM_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"     {symbol:18s} ❌ 缺失特征列: {missing}")
        return False
    all_nan = [c for c in LSTM_FEATURE_COLUMNS if df[c].isna().all()]
    if all_nan:
        print(f"     {symbol:18s} ⚠  全 NaN 列: {all_nan}")
    return True


def is_already_complete(symbol: str) -> bool:
    """检查断点续传: 3个 split 都存在且含全部特征列"""
    for split in ("train", "validation", "test"):
        p = OUTPUT_DIR / "data" / split / f"{symbol}.csv"
        if not p.exists():
            return False
    try:
        sample = pd.read_csv(OUTPUT_DIR / "data" / "train" / f"{symbol}.csv", nrows=3)
        if not set(LSTM_FEATURE_COLUMNS).issubset(set(sample.columns)):
            return False
    except Exception:
        return False
    return True


def coin_to_candidates(coin: str) -> List[str]:
    if coin in SYMBOL_OVERRIDES:
        return SYMBOL_OVERRIDES[coin]
    return [f"{coin}USDT"]


def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def generate_gitattributes() -> None:
    (OUTPUT_DIR / ".gitattributes").write_text(
        "*.csv filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                           主 流 程                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main() -> None:
    global SESSION

    print()
    print("=" * 76)
    print("   📊  Binance → HuggingFace Dataset Pipeline")
    print("       Aligned with README.md LSTM Feature Specification")
    print("=" * 76)
    print(f"   币种:   {len(COINS)} 种")
    print(f"   间隔:   {INTERVAL}")
    print(f"   天数:   {DAYS} 天 (~{DAYS * 24 * 60:,} candles/symbol)")
    print(f"   切分:   train={TRAIN_RATIO:.0%} / val={VAL_RATIO:.0%} / test={TEST_RATIO:.0%}")
    print(f"   输出:   {OUTPUT_DIR}")
    print(f"   代理:   {PROXY if PROXY else '未设置 (中国大陆请设 PROXY)'}")
    print()

    # ── Session + 连通性 ─────────────────────────────────────────────
    SESSION = build_session()
    spot_ok, futures_ok = test_connectivity()

    if not spot_ok and not futures_ok:
        print("=" * 76)
        print("   ❌ 无法连接 Binance API!")
        print()
        print("   解决方法:")
        print("   1. 打开脚本, 找到 PROXY = \"\" 改为你的代理:")
        print("      PROXY = \"http://127.0.0.1:7890\"")
        print("   2. 确保代理软件 (Clash/V2Ray) 已启动")
        print("   3. SOCKS5 代理需: pip install requests[socks]")
        print("=" * 76)
        sys.exit(1)

    if not spot_ok:
        print("   ⚠  Spot 不可用, 仅用 Futures")
    if not futures_ok:
        print("   ⚠  Futures 不可用, funding_rate / taker_ratio 使用默认值")

    # ── 打印 LSTM 特征规格 ────────────────────────────────────────────
    print()
    print("   LSTM 输入特征 (对齐 README Feature Summary):")
    print("   ┌────┬───────────────────────┬──────────────────┬─────────┐")
    print("   │ #  │ Feature               │ Category         │ Source  │")
    print("   ├────┼───────────────────────┼──────────────────┼─────────┤")
    for i, feat in enumerate(LSTM_FEATURE_COLUMNS, 1):
        if i <= 7:
            cat, src = "Price-derived", "1m K线"
        elif i <= 9:
            cat, src = "Microstructure", "1m K线"
        elif i <= 13:
            cat, src = "Multi-scale", "重采样"
        elif i == 14:
            cat, src = "Web3 sentiment", "Futures"
        else:
            cat, src = "Order flow", "Futures"
        print(f"   │{i:>3} │ {feat:<21s} │ {cat:<16s} │ {src:<7s} │")
    print("   └────┴───────────────────────┴──────────────────┴─────────┘")
    print(f"   → LSTM input shape: (batch, 60, {len(LSTM_FEATURE_COLUMNS)})")
    print(f"   + alpha_score (ensemble 辅助, 非 LSTM 输入)")

    # ── 时间范围 ─────────────────────────────────────────────────────
    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=DAYS)
    start_ms = dt_to_ms(start_dt)
    end_ms   = dt_to_ms(end_dt)

    print(f"\n   时间: {start_dt:%Y-%m-%d %H:%M} → {end_dt:%Y-%m-%d %H:%M} UTC")
    print("=" * 76)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success: List[str] = []
    failed:  List[str] = []
    skipped: List[str] = []
    total_candles = 0
    t0 = time.time()

    for idx, coin in enumerate(COINS, 1):
        candidates = coin_to_candidates(coin)
        pre = f"[{idx:>2}/{len(COINS)}]"

        # ── 断点续传 ─────────────────────────────────────────────
        done = False
        for sc in candidates:
            if is_already_complete(sc):
                print(f"{pre} {sc:18s} ⏭  已存在, 跳过")
                skipped.append(sc)
                success.append(sc)
                done = True
                break
        if done:
            continue

        # ── 1) K 线 ──────────────────────────────────────────────
        raw_data, source, used_sym = None, "N/A", None
        for sc in candidates:
            print(f"{pre} {sc:18s} 📥 K线 ...")
            raw_data, source = download_klines(sc, start_ms, end_ms)
            if raw_data:
                used_sym = sc
                break
            print(f"     {sc:18s} ⚠  不可用, 下一候选")

        if not raw_data or not used_sym:
            print(f"     {coin:18s} ❌ 全部候选失败: {candidates}")
            failed.append(coin)
            continue

        df = raw_to_dataframe(raw_data, used_sym)
        d1 = df["open_time"].iloc[0].strftime("%Y-%m-%d")
        d2 = df["open_time"].iloc[-1].strftime("%Y-%m-%d")
        print(f"     {used_sym:18s} ✅ {source:7s} | {len(df):>9,} 条 | {d1} → {d2}")

        # ── 2) 资金费率 (Feature #14) ────────────────────────────
        print(f"     {used_sym:18s} 📥 资金费率 ...")
        funding_df = download_funding_rate(used_sym, start_ms, end_ms)
        n_fr = len(funding_df)
        print(f"     {used_sym:18s}    → {n_fr} 条"
              f"{'  (默认 0.0)' if n_fr == 0 else ''}")

        # ── 3) Taker 多空比 (Feature #15) ────────────────────────
        print(f"     {used_sym:18s} 📥 Taker 多空比 ...")
        taker_df = download_taker_ratio(used_sym, start_ms, end_ms)
        n_tk = len(taker_df)
        print(f"     {used_sym:18s}    → {n_tk} 条"
              f"{'  (默认 1.0)' if n_tk == 0 else ''}")

        # ── 4) 合并 Web3 补充数据 ────────────────────────────────
        df = merge_supplementary(df, funding_df, taker_df)

        # ── 5) 计算全部特征 ──────────────────────────────────────
        print(f"     {used_sym:18s} 🔧 计算 15 个 LSTM 特征 + alpha ...")
        df = compute_all_features(df)

        # ── 6) 校验特征完整性 ────────────────────────────────────
        if not validate_features(df, used_sym):
            failed.append(coin)
            continue

        # ── 7) 切分 + 保存 ──────────────────────────────────────
        splits = split_chronological(df)
        save_splits(splits, used_sym)

        nt = len(splits["train"])
        nv = len(splits["validation"])
        ne = len(splits["test"])
        print(f"     {used_sym:18s} 💾 train={nt:,}  val={nv:,}  test={ne:,}")

        # 确认特征列
        feat_present = sum(1 for c in LSTM_FEATURE_COLUMNS if c in df.columns)
        print(f"     {used_sym:18s}    LSTM 特征: {feat_present}/{len(LSTM_FEATURE_COLUMNS)} ✓  "
              f"共 {len(df.columns)} 列")
        print()

        success.append(used_sym)
        total_candles += len(df)

    # ── .gitattributes ───────────────────────────────────────────────
    generate_gitattributes()

    elapsed = time.time() - t0

    # ── meta.json ────────────────────────────────────────────────────
    meta = {
        "lstm_feature_columns": LSTM_FEATURE_COLUMNS,
        "lstm_input_dim": len(LSTM_FEATURE_COLUMNS),
        "lstm_seq_len": 60,
        "aux_columns": AUX_COLUMNS,
        "success_symbols": success,
        "failed_coins": failed,
        "total_candles": total_candles,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "days": DAYS,
        "split_ratio": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    meta_path = OUTPUT_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── 汇总 ─────────────────────────────────────────────────────────
    print()
    print("=" * 76)
    print("   📋  完 成 汇 总")
    print("=" * 76)
    print(f"   ✅ 成功:    {len(success):>3} / {len(COINS)}")
    if skipped:
        print(f"   ⏭  跳过:    {len(skipped):>3} (已存在)")
    if failed:
        print(f"   ❌ 失败:    {len(failed):>3}")
        for fc in failed:
            print(f"               {fc} → {coin_to_candidates(fc)}")
    print(f"   📊 总量:    {total_candles:>12,} 条 K 线")
    print(f"   ⏱  耗时:    {elapsed / 60:.1f} 分钟")
    print(f"   📁 输出:    {OUTPUT_DIR}")
    print()
    print("   LSTM 特征规格 (写入 meta.json):")
    print(f"   → input_dim = {len(LSTM_FEATURE_COLUMNS)}")
    print(f"   → seq_len   = 60")
    print(f"   → shape     = (batch, 60, {len(LSTM_FEATURE_COLUMNS)})")
    print(f"   → columns   = {LSTM_FEATURE_COLUMNS}")
    print()
    print("   CSV 结构:")
    print("   ╔═══════════════════════════════════════════════════════════╗")
    print("   ║ 原始 OHLCV (11 列):                                     ║")
    print("   ║   symbol, open_time, O/H/L/C, volume, close_time,       ║")
    print("   ║   quote_volume, num_trades, taker_buy_*                  ║")
    print("   ╠═══════════════════════════════════════════════════════════╣")
    print("   ║ LSTM 特征 (15 列):                                      ║")
    print("   ║   rsi_14, ema_12, ema_26, ema_crossover_1m, atr_14      ║")
    print("   ║   momentum_10, volatility_20, volume_ratio, ob_imbalance║")
    print("   ║   ema_crossover_15m, momentum_15m                        ║")
    print("   ║   ema_crossover_1h, momentum_1h                          ║")
    print("   ║   funding_rate, taker_buy_sell_ratio                     ║")
    print("   ╠═══════════════════════════════════════════════════════════╣")
    print("   ║ 辅助 (1 列):                                            ║")
    print("   ║   alpha_score (ensemble 用, 非 LSTM 输入)                ║")
    print("   ╚═══════════════════════════════════════════════════════════╝")
    print()
    print("=" * 76)


if __name__ == "__main__":
    main()
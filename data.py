#!/usr/bin/env python3
"""
Binance → HuggingFace Dataset Pipeline — 数据获取脚本
=====================================================

下载 66 种加密货币 90 天 1m K 线及补充数据（资金费率 / Taker 多空比），
预计算全部特征（含多时间尺度 15m / 1h），按 70/15/15 时间顺序切分，
保存为 HuggingFace Datasets 标准目录格式到 ~/Desktop/HKSG/。

Binance 公开 API（均无需 API Key）:
  ① Spot K 线        GET https://api.binance.com/api/v3/klines
  ② Futures K 线     GET https://fapi.binance.com/fapi/v1/klines
  ③ 资金费率         GET https://fapi.binance.com/fapi/v1/fundingRate
  ④ Taker 多空比     GET https://fapi.binance.com/futures/data/takerlongshortRatio

依赖安装:
    pip install requests pandas numpy tqdm

运行:
    python data.py

输出:
    ~/Desktop/HKSG/
    ├── .gitattributes
    └── data/
        ├── train/
        │   ├── BTCUSDT.csv
        │   └── ...
        ├── validation/
        │   └── ...
        └── test/
            └── ...
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
    print("[提示] 建议安装 tqdm 以显示进度条: pip install tqdm")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                          配 置 区                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

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
#  ★★★  代理设置  ★★★
#  如果你在中国大陆，需要设置代理才能访问 Binance API。
#  请根据你的 VPN / Clash / V2Ray 等工具的本地端口填写。
#  常见格式:
#    HTTP 代理:   "http://127.0.0.1:7890"
#    SOCKS5 代理: "socks5://127.0.0.1:7891"   (需 pip install requests[socks])
#  如果你不需要代理（海外直连），留空字符串 "" 即可。
# ══════════════════════════════════════════════════════════════════════
PROXY = "http://127.0.0.1:7890"   # ← ★ 在这里填写你的代理地址，例如 "http://127.0.0.1:7890"

# ── Binance API 端点（含镜像）──────────────────────────────────────────
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

# ── 下载参数 ──────────────────────────────────────────────────────────
INTERVAL           = "1m"
DAYS               = 90
LIMIT_PER_REQUEST  = 1000
REQUEST_INTERVAL   = 0.12
MAX_RETRIES        = 3
RETRY_WAIT         = 5
CONNECT_TIMEOUT    = 10   # 连接超时（秒）
READ_TIMEOUT       = 30   # 读取超时（秒）

# ── 特征参数 ─────────────────────────────────────────────────────────
RSI_PERIOD          = 14
EMA_SHORT           = 12
EMA_LONG            = 26
ATR_PERIOD          = 14
MOMENTUM_PERIOD     = 10
VOLATILITY_WINDOW   = 20
VOLUME_RATIO_WINDOW = 20

# ── 多时间尺度 ───────────────────────────────────────────────────────
RESAMPLE_15M_MINUTES = 15
RESAMPLE_1H_MINUTES  = 60

# ── Rule-Based Alpha 权重 ────────────────────────────────────────────
ALPHA_W_RSI       = 0.3
ALPHA_W_MOMENTUM  = 0.3
ALPHA_W_EMA_CROSS = 0.3
ALPHA_W_VOL       = 0.1

# ── 数据切分 ─────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── 输出路径 ─────────────────────────────────────────────────────────
OUTPUT_DIR = Path.home() / "Desktop" / "HKSG"

# ── K 线列定义 ───────────────────────────────────────────────────────
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
# ║                       工 具 函 数                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

def build_session() -> requests.Session:
    """构建带重试机制和代理的 requests.Session"""
    session = requests.Session()

    # 重试策略: 对 429/500/502/503/504 自动重试
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # 代理
    if PROXY:
        session.proxies = {
            "http":  PROXY,
            "https": PROXY,
        }
        print(f"   🌐 已设置代理: {PROXY}")
    else:
        # 也尝试从环境变量读取
        env_proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or \
                    os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
        if env_proxy:
            print(f"   🌐 检测到环境变量代理: {env_proxy}")

    return session


# 全局 Session
SESSION: Optional[requests.Session] = None

# 已验证可用的 base URL 缓存
_WORKING_SPOT_BASE: Optional[str] = None
_WORKING_FUTURES_BASE: Optional[str] = None


def coin_to_candidates(coin: str) -> List[str]:
    """返回某个 coin 在 Binance 上所有可能的 symbol 名称"""
    if coin in SYMBOL_OVERRIDES:
        return SYMBOL_OVERRIDES[coin]
    return [f"{coin}USDT"]


def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def is_already_complete(symbol: str) -> bool:
    """检查此 symbol 的 v3 完整数据是否已存在"""
    for split in ("train", "validation", "test"):
        csv_path = OUTPUT_DIR / "data" / split / f"{symbol}.csv"
        if not csv_path.exists():
            return False
    try:
        sample = pd.read_csv(
            OUTPUT_DIR / "data" / "train" / f"{symbol}.csv", nrows=3
        )
        required = {
            "rsi_14", "ema_crossover_1m", "ema_crossover_15m",
            "ema_crossover_1h", "funding_rate", "alpha_score",
        }
        if not required.issubset(set(sample.columns)):
            return False
    except Exception:
        return False
    return True


def test_connectivity() -> Tuple[bool, bool]:
    """
    启动时测试 Binance Spot / Futures API 连通性。
    返回 (spot_ok, futures_ok)
    """
    global _WORKING_SPOT_BASE, _WORKING_FUTURES_BASE

    print("   🔍 测试 Binance API 连通性 ...")
    print()

    # --- Spot ---
    spot_ok = False
    for base in SPOT_BASE_URLS:
        url = f"{base}/api/v3/ping"
        try:
            resp = SESSION.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            if resp.status_code == 200:
                print(f"   ✅ Spot 可用:    {base}")
                _WORKING_SPOT_BASE = base
                spot_ok = True
                break
            else:
                print(f"   ⚠  Spot {base} 返回 {resp.status_code}")
        except requests.exceptions.ProxyError:
            print(f"   ❌ Spot {base} 代理错误 — 请检查 PROXY 设置")
        except requests.exceptions.SSLError:
            print(f"   ❌ Spot {base} SSL 错误")
        except requests.exceptions.ConnectTimeout:
            print(f"   ❌ Spot {base} 连接超时")
        except requests.exceptions.ReadTimeout:
            print(f"   ❌ Spot {base} 读取超时")
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Spot {base} 无法连接")
        except Exception as e:
            print(f"   ❌ Spot {base} 错误: {type(e).__name__}: {e}")

    if not spot_ok:
        print("   ❌ 全部 Spot 端点均不可用")

    # --- Futures ---
    futures_ok = False
    for base in FUTURES_BASE_URLS:
        url = f"{base}/fapi/v1/ping"
        try:
            resp = SESSION.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            if resp.status_code == 200:
                print(f"   ✅ Futures 可用:  {base}")
                _WORKING_FUTURES_BASE = base
                futures_ok = True
                break
            else:
                print(f"   ⚠  Futures {base} 返回 {resp.status_code}")
        except requests.exceptions.ProxyError:
            print(f"   ❌ Futures {base} 代理错误 — 请检查 PROXY 设置")
        except requests.exceptions.SSLError:
            print(f"   ❌ Futures {base} SSL 错误")
        except requests.exceptions.ConnectTimeout:
            print(f"   ❌ Futures {base} 连接超时")
        except requests.exceptions.ReadTimeout:
            print(f"   ❌ Futures {base} 读取超时")
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Futures {base} 无法连接")
        except Exception as e:
            print(f"   ❌ Futures {base} 错误: {type(e).__name__}: {e}")

    if not futures_ok:
        print("   ❌ 全部 Futures 端点均不可用")

    print()
    return spot_ok, futures_ok


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    Binance API 请 求 层                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _api_get(url: str, params: dict, tag: str = "") -> Optional[Any]:
    """通用 GET 请求，含重试、速率限制处理"""
    for attempt in range(MAX_RETRIES):
        try:
            resp = SESSION.get(
                url, params=params,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", RETRY_WAIT * 2))
                print(f"        ⏳ [{tag}] 速率限制 429，等待 {wait}s")
                time.sleep(wait)
                continue

            if resp.status_code == 418:
                print(f"        🚫 [{tag}] IP 被暂时封禁 418，等待 120s")
                time.sleep(120)
                continue

            if resp.status_code != 200:
                return None

            data = resp.json()
            if isinstance(data, dict) and "code" in data:
                return None

            return data

        except requests.exceptions.ProxyError:
            print(f"        ❌ [{tag}] 代理连接失败 (尝试 {attempt+1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)
            continue

        except requests.exceptions.SSLError:
            print(f"        ❌ [{tag}] SSL 错误 (尝试 {attempt+1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)
            continue

        except requests.exceptions.ConnectTimeout:
            print(f"        ❌ [{tag}] 连接超时 (尝试 {attempt+1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)
            continue

        except requests.exceptions.ReadTimeout:
            print(f"        ❌ [{tag}] 读取超时 (尝试 {attempt+1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)
            continue

        except requests.exceptions.ConnectionError:
            print(f"        ❌ [{tag}] 连接失败 (尝试 {attempt+1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)
            continue

        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"        ❌ [{tag}] 请求异常: {type(e).__name__} (尝试 {attempt+1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)
            continue

    return None


# ── 1. K 线下载 ──────────────────────────────────────────────────────

def _download_klines_from_endpoint(
    base_url: str, path: str, symbol: str, start_ms: int, end_ms: int,
    source_name: str,
) -> Optional[List[list]]:
    """从单个端点分页下载全量 K 线"""
    all_klines: List[list] = []
    cursor = start_ms
    url = f"{base_url}{path}"

    while cursor < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  INTERVAL,
            "startTime": cursor,
            "endTime":   end_ms,
            "limit":     LIMIT_PER_REQUEST,
        }
        page = _api_get(url, params, tag=f"{symbol}/{source_name}")

        if page is None or len(page) == 0:
            break

        all_klines.extend(page)
        last_open_time = page[-1][0]
        cursor = last_open_time + 60_000

        time.sleep(REQUEST_INTERVAL)

        if len(all_klines) % 30_000 < LIMIT_PER_REQUEST and len(all_klines) > 0:
            print(f"          {source_name}: {len(all_klines):>9,} candles ...")

        if len(page) < LIMIT_PER_REQUEST:
            break

    return all_klines if len(all_klines) > 0 else None


def download_klines(
    symbol: str, start_ms: int, end_ms: int
) -> Tuple[Optional[List[list]], str]:
    """先尝试 Spot（多镜像），失败则 Futures"""

    # --- Spot ---
    if _WORKING_SPOT_BASE:
        # 优先使用已验证的 base
        spot_order = [_WORKING_SPOT_BASE] + [
            b for b in SPOT_BASE_URLS if b != _WORKING_SPOT_BASE
        ]
    else:
        spot_order = SPOT_BASE_URLS

    for base in spot_order:
        result = _download_klines_from_endpoint(
            base, SPOT_KLINES_PATH, symbol, start_ms, end_ms, f"Spot({base.split('//')[1].split('.')[0]})"
        )
        if result is not None:
            return result, "Spot"

    # --- Futures ---
    if _WORKING_FUTURES_BASE:
        futures_order = [_WORKING_FUTURES_BASE] + [
            b for b in FUTURES_BASE_URLS if b != _WORKING_FUTURES_BASE
        ]
    else:
        futures_order = FUTURES_BASE_URLS

    for base in futures_order:
        result = _download_klines_from_endpoint(
            base, FUTURES_KLINES_PATH, symbol, start_ms, end_ms, "Futures"
        )
        if result is not None:
            return result, "Futures"

    return None, "N/A"


# ── 2. 资金费率 ─────────────────────────────────────────────────────

def download_funding_rate(
    symbol: str, start_ms: int, end_ms: int
) -> pd.DataFrame:
    if not _WORKING_FUTURES_BASE:
        return pd.DataFrame(columns=["funding_time", "funding_rate"])

    all_data: List[dict] = []
    cursor = start_ms
    url = f"{_WORKING_FUTURES_BASE}{FUNDING_RATE_PATH}"

    while cursor < end_ms:
        params = {
            "symbol":    symbol,
            "startTime": cursor,
            "endTime":   end_ms,
            "limit":     1000,
        }
        page = _api_get(url, params, tag=f"{symbol}/funding")
        if page is None or len(page) == 0:
            break

        all_data.extend(page)
        cursor = page[-1].get("fundingTime", 0) + 1
        time.sleep(REQUEST_INTERVAL)

        if len(page) < 1000:
            break

    if not all_data:
        return pd.DataFrame(columns=["funding_time", "funding_rate"])

    df = pd.DataFrame(all_data)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return (
        df[["funding_time", "funding_rate"]]
        .drop_duplicates()
        .sort_values("funding_time")
        .reset_index(drop=True)
    )


# ── 3. Taker 多空比 ─────────────────────────────────────────────────

def download_taker_ratio(
    symbol: str, start_ms: int, end_ms: int
) -> pd.DataFrame:
    if not _WORKING_FUTURES_BASE:
        return pd.DataFrame(columns=["taker_time", "taker_buy_sell_ratio"])

    all_data: List[dict] = []
    cursor = start_ms
    url = f"{_WORKING_FUTURES_BASE}{TAKER_LS_PATH}"

    while cursor < end_ms:
        params = {
            "symbol":    symbol,
            "period":    "5m",
            "startTime": cursor,
            "endTime":   end_ms,
            "limit":     500,
        }
        page = _api_get(url, params, tag=f"{symbol}/taker")
        if page is None or len(page) == 0:
            break

        all_data.extend(page)
        cursor = page[-1].get("timestamp", 0) + 1
        time.sleep(REQUEST_INTERVAL)

        if len(page) < 500:
            break

    if not all_data:
        return pd.DataFrame(columns=["taker_time", "taker_buy_sell_ratio"])

    df = pd.DataFrame(all_data)
    df["taker_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["taker_buy_sell_ratio"] = df["buySellRatio"].astype(float)
    return (
        df[["taker_time", "taker_buy_sell_ratio"]]
        .drop_duplicates()
        .sort_values("taker_time")
        .reset_index(drop=True)
    )


# ╔══════════════════════════════════════════════════════════════════════╗
# ║               特 征 计 算 (纯 NumPy)                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _ema(data: np.ndarray, span: int) -> np.ndarray:
    """指数移动平均"""
    out = np.empty_like(data, dtype=np.float64)
    alpha = 2.0 / (span + 1.0)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


def calc_rsi(close: np.ndarray, period: int = RSI_PERIOD) -> np.ndarray:
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


def calc_ema_crossover(
    close: np.ndarray, short: int = EMA_SHORT, long: int = EMA_LONG
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ema_s = _ema(close, short)
    ema_l = _ema(close, long)
    cross = np.where(ema_l != 0, (ema_s - ema_l) / ema_l, 0.0)
    return ema_s, ema_l, cross


def calc_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    period: int = ATR_PERIOD,
) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return _ema(tr, period)


def calc_momentum(
    close: np.ndarray, period: int = MOMENTUM_PERIOD
) -> np.ndarray:
    mom = np.zeros(len(close))
    for i in range(period, len(close)):
        if close[i - period] != 0:
            mom[i] = (close[i] - close[i - period]) / close[i - period]
    return mom


def calc_volatility(
    close: np.ndarray, window: int = VOLATILITY_WINDOW
) -> np.ndarray:
    n = len(close)
    vol = np.zeros(n)
    if n < 2:
        return vol
    returns = np.zeros(n)
    for i in range(1, n):
        if close[i - 1] != 0:
            returns[i] = (close[i] - close[i - 1]) / close[i - 1]
    for i in range(window, n):
        vol[i] = np.std(returns[i - window + 1 : i + 1])
    return vol


def calc_volume_ratio(
    volume: np.ndarray, window: int = VOLUME_RATIO_WINDOW
) -> np.ndarray:
    vr = np.ones(len(volume))
    for i in range(window, len(volume)):
        avg = np.mean(volume[i - window : i])
        if avg > 0:
            vr[i] = volume[i] / avg
    return vr


def calc_ob_imbalance(
    taker_buy_vol: np.ndarray, total_vol: np.ndarray
) -> np.ndarray:
    ob = np.zeros(len(total_vol))
    mask = total_vol > 0
    ob[mask] = (taker_buy_vol[mask] / total_vol[mask] - 0.5) * 2.0
    return ob


def calc_alpha_score(
    rsi: np.ndarray,
    momentum: np.ndarray,
    ema_cross: np.ndarray,
    volatility: np.ndarray,
) -> np.ndarray:
    rsi_s = (50.0 - rsi) / 50.0
    mom_s = np.tanh(momentum * 10.0)
    ema_s = np.tanh(ema_cross * 100.0)
    vol_s = -np.tanh(volatility * 100.0)
    return (
        ALPHA_W_RSI * rsi_s
        + ALPHA_W_MOMENTUM * mom_s
        + ALPHA_W_EMA_CROSS * ema_s
        + ALPHA_W_VOL * vol_s
    )


# ╔══════════════════════════════════════════════════════════════════════╗
# ║          多 时 间 尺 度 重 采 样  (15m / 1h)                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """将 1m K 线重采样为 N 分钟 OHLCV"""
    df_r = df.set_index("open_time").copy()
    rule = f"{minutes}min"
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    resampled = df_r.resample(rule, label="left", closed="left").agg(agg_dict).dropna()
    resampled = resampled.reset_index()
    return resampled


def compute_multiscale_features(
    df: pd.DataFrame, minutes: int, suffix: str
) -> pd.DataFrame:
    """
    重采样 → 计算 EMA crossover + Momentum →
    merge_asof 回 1m 粒度 (backward fill, 无未来泄露)
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

    merge_cols = [
        "open_time",
        f"ema_crossover_{suffix}",
        f"momentum_{suffix}",
    ]
    merged = pd.merge_asof(
        df.sort_values("open_time"),
        resampled[merge_cols].sort_values("open_time"),
        on="open_time",
        direction="backward",
    )

    merged[f"ema_crossover_{suffix}"] = merged[f"ema_crossover_{suffix}"].fillna(0.0)
    merged[f"momentum_{suffix}"] = merged[f"momentum_{suffix}"].fillna(0.0)

    return merged


# ╔══════════════════════════════════════════════════════════════════════╗
# ║               数 据 处 理 + 特 征 管 线                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

def raw_to_dataframe(raw_klines: list, symbol: str) -> pd.DataFrame:
    """原始 API 响应 → 干净的 DataFrame"""
    df = pd.DataFrame(raw_klines, columns=KLINE_COLUMNS)
    df = df.drop(columns=["_ignore"])

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    for col in FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["num_trades"] = pd.to_numeric(
        df["num_trades"], errors="coerce"
    ).astype(int)

    df.insert(0, "symbol", symbol)
    df = df.drop_duplicates(subset=["open_time"], keep="first")
    df = df.sort_values("open_time")
    df = df.reset_index(drop=True)
    return df


def merge_supplementary(
    df: pd.DataFrame,
    funding_df: pd.DataFrame,
    taker_df: pd.DataFrame,
) -> pd.DataFrame:
    """合并资金费率 (8h) 和 Taker 比率 (5m) 到 1m K 线"""
    df = df.sort_values("open_time").copy()

    # 资金费率
    if not funding_df.empty:
        df = pd.merge_asof(
            df,
            funding_df.sort_values("funding_time"),
            left_on="open_time",
            right_on="funding_time",
            direction="backward",
        )
        if "funding_time" in df.columns:
            df = df.drop(columns=["funding_time"])

    if "funding_rate" not in df.columns:
        df["funding_rate"] = 0.0
    else:
        df["funding_rate"] = df["funding_rate"].fillna(0.0)

    # Taker 多空比
    if not taker_df.empty:
        df = pd.merge_asof(
            df,
            taker_df.sort_values("taker_time"),
            left_on="open_time",
            right_on="taker_time",
            direction="backward",
        )
        if "taker_time" in df.columns:
            df = df.drop(columns=["taker_time"])

    if "taker_buy_sell_ratio" not in df.columns:
        df["taker_buy_sell_ratio"] = 1.0
    else:
        df["taker_buy_sell_ratio"] = df["taker_buy_sell_ratio"].fillna(1.0)

    return df


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算全部特征:
      ① 1m: RSI, EMA12/26, EMA_crossover, ATR, Momentum, Volatility, Volume Ratio
      ② OB Imbalance (代理)
      ③ 15m: EMA_crossover, Momentum
      ④ 1h:  EMA_crossover, Momentum
      ⑤ Alpha Score (rule-based)
    """
    close  = df["close"].values.astype(np.float64)
    high   = df["high"].values.astype(np.float64)
    low    = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    tb_vol = df["taker_buy_base_volume"].values.astype(np.float64)

    # ── ① 1 分钟技术指标 ─────────────────────────────────────────
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
    df["volume_ratio"]     = np.round(vol_ratio, 4)

    # ── ② 订单簿失衡代理 ─────────────────────────────────────────
    df["ob_imbalance"] = np.round(calc_ob_imbalance(tb_vol, volume), 4)

    # ── ③ 15 分钟多尺度特征 ──────────────────────────────────────
    df = compute_multiscale_features(df, RESAMPLE_15M_MINUTES, "15m")

    # ── ④ 1 小时多尺度特征 ──────────────────────────────────────
    df = compute_multiscale_features(df, RESAMPLE_1H_MINUTES, "1h")

    # ── ⑤ Alpha Score ────────────────────────────────────────────
    df["alpha_score"] = np.round(
        calc_alpha_score(
            df["rsi_14"].values,
            df["momentum_10"].values,
            df["ema_crossover_1m"].values,
            df["volatility_20"].values,
        ),
        4,
    )

    return df


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                   切 分 + 保 存                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

def split_chronological(
    df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """严格按时间顺序切分，绝不 shuffle"""
    n = len(df)
    i_train = int(n * TRAIN_RATIO)
    i_val = int(n * (TRAIN_RATIO + VAL_RATIO))
    return {
        "train":      df.iloc[:i_train].copy(),
        "validation": df.iloc[i_train:i_val].copy(),
        "test":       df.iloc[i_val:].copy(),
    }


def save_splits(splits: Dict[str, pd.DataFrame], symbol: str) -> None:
    for split_name, split_df in splits.items():
        split_dir = OUTPUT_DIR / "data" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        filepath = split_dir / f"{symbol}.csv"
        split_df.to_csv(filepath, index=False)


def generate_gitattributes() -> None:
    content = "*.csv filter=lfs diff=lfs merge=lfs -text\n"
    (OUTPUT_DIR / ".gitattributes").write_text(content, encoding="utf-8")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                           主 流 程                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main() -> None:
    global SESSION

    print()
    print("=" * 76)
    print("   📊  Binance → HuggingFace Dataset Pipeline  (v3 Multi-Scale)")
    print("=" * 76)
    print(f"   币种数量:   {len(COINS)}")
    print(f"   K 线间隔:   {INTERVAL}")
    print(f"   历史天数:   {DAYS} 天")
    print(f"   数据切分:   train={TRAIN_RATIO:.0%} / val={VAL_RATIO:.0%} / test={TEST_RATIO:.0%}")
    print(f"   输出目录:   {OUTPUT_DIR}")
    if PROXY:
        print(f"   代理地址:   {PROXY}")
    else:
        print(f"   代理地址:   未设置 (如在中国大陆请在脚本顶部设置 PROXY)")
    print()

    # ── 构建 Session ─────────────────────────────────────────────────
    SESSION = build_session()

    # ── 连通性测试 ────────────────────────────────────────────────────
    spot_ok, futures_ok = test_connectivity()

    if not spot_ok and not futures_ok:
        print("=" * 76)
        print("   ❌ 无法连接任何 Binance API 端点！")
        print()
        print("   可能原因:")
        print("   1. 你在中国大陆，Binance 被墙 → 请设置代理")
        print("      打开脚本，找到 PROXY = \"\" 那一行，改为:")
        print("      PROXY = \"http://127.0.0.1:7890\"  (替换成你的代理端口)")
        print()
        print("   2. 网络本身不通 → 检查 Wi-Fi / 网线连接")
        print()
        print("   3. 代理设置了但代理软件没有运行 → 先启动 Clash / V2Ray 等")
        print()
        print("   4. 如果使用 SOCKS5 代理，需要:")
        print("      pip install requests[socks]")
        print("=" * 76)
        sys.exit(1)

    if not spot_ok:
        print("   ⚠  Spot 不可用，将仅使用 Futures 端点")
    if not futures_ok:
        print("   ⚠  Futures 不可用，资金费率和 Taker 比率将使用默认值")

    print()
    print("   特征列表 (14 特征列):")
    print("   ┌─ 1m 技术指标 ──────────────────────────────────────┐")
    print("   │  RSI(14)  EMA(12/26)  EMA_crossover  ATR(14)     │")
    print("   │  Momentum(10)  Volatility(20)  Volume_ratio      │")
    print("   ├─ 多时间尺度 ───────────────────────────────────────┤")
    print("   │  15m: EMA_crossover, Momentum                     │")
    print("   │  1h:  EMA_crossover, Momentum                     │")
    print("   ├─ Web3 补充 ────────────────────────────────────────┤")
    print("   │  Funding Rate / Taker Ratio / OB Imbalance        │")
    print("   ├─ 复合信号 ─────────────────────────────────────────┤")
    print("   │  Alpha Score (rule-based)                         │")
    print("   └────────────────────────────────────────────────────┘")

    # 时间范围
    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=DAYS)
    start_ms = dt_to_ms(start_dt)
    end_ms   = dt_to_ms(end_dt)

    print(f"\n   时间范围:  {start_dt.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"           → {end_dt.strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 76)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success_symbols: List[str] = []
    failed_coins:    List[str] = []
    skipped_symbols: List[str] = []
    total_candles = 0
    t_start = time.time()

    for idx, coin in enumerate(COINS, 1):
        candidates = coin_to_candidates(coin)
        prefix = f"[{idx:>2}/{len(COINS)}]"

        # ── 检查断点续传 ─────────────────────────────────────────
        already_done = False
        for sym_candidate in candidates:
            if is_already_complete(sym_candidate):
                print(
                    f"{prefix} {sym_candidate:18s} ⏭  已存在 (v3 完整)，跳过"
                )
                skipped_symbols.append(sym_candidate)
                success_symbols.append(sym_candidate)
                already_done = True
                break
        if already_done:
            continue

        # ── Step 1: 下载 K 线 ────────────────────────────────────
        raw_data = None
        source = "N/A"
        used_symbol = None

        for sym_candidate in candidates:
            print(f"{prefix} {sym_candidate:18s} 📥 尝试下载 K 线 ...")
            raw_data, source = download_klines(sym_candidate, start_ms, end_ms)
            if raw_data is not None:
                used_symbol = sym_candidate
                break
            else:
                print(
                    f"     {sym_candidate:18s} ⚠  K 线不可用，尝试下一候选 ..."
                )

        if raw_data is None or used_symbol is None:
            print(f"     {coin:18s} ❌ 全部候选均失败: {candidates}")
            failed_coins.append(coin)
            continue

        df = raw_to_dataframe(raw_data, used_symbol)
        date_from = df["open_time"].iloc[0].strftime("%Y-%m-%d")
        date_to   = df["open_time"].iloc[-1].strftime("%Y-%m-%d")
        print(
            f"     {used_symbol:18s} ✅ {source:7s} | {len(df):>9,} 条 "
            f"| {date_from} → {date_to}"
        )

        # ── Step 2: 下载资金费率 ─────────────────────────────────
        print(f"     {used_symbol:18s} 📥 资金费率 ...")
        funding_df = download_funding_rate(used_symbol, start_ms, end_ms)
        n_fr = len(funding_df)
        print(
            f"     {used_symbol:18s}    → {n_fr} 条"
            f"{'  (现货币种, 默认 0.0)' if n_fr == 0 else ''}"
        )

        # ── Step 3: 下载 Taker 多空比 ───────────────────────────
        print(f"     {used_symbol:18s} 📥 Taker 多空比 ...")
        taker_df = download_taker_ratio(used_symbol, start_ms, end_ms)
        n_tk = len(taker_df)
        print(
            f"     {used_symbol:18s}    → {n_tk} 条"
            f"{'  (默认 1.0)' if n_tk == 0 else ''}"
        )

        # ── Step 4: 合并补充数据 ─────────────────────────────────
        df = merge_supplementary(df, funding_df, taker_df)

        # ── Step 5: 计算全部特征 ─────────────────────────────────
        print(
            f"     {used_symbol:18s} 🔧 计算特征 (1m + 15m + 1h + alpha) ..."
        )
        df = compute_all_features(df)

        # ── Step 6: 切分 + 保存 ──────────────────────────────────
        splits = split_chronological(df)
        save_splits(splits, used_symbol)

        n_train = len(splits["train"])
        n_val   = len(splits["validation"])
        n_test  = len(splits["test"])

        print(
            f"     {used_symbol:18s} 💾 train={n_train:,}  "
            f"val={n_val:,}  test={n_test:,}"
        )
        print(
            f"     {used_symbol:18s}    共 {len(df.columns)} 列 | "
            f"特征: 8(1m) + 4(multi) + 3(web3) + 1(alpha) = 16 特征列"
        )
        print()

        success_symbols.append(used_symbol)
        total_candles += len(df)

    # ── 生成 .gitattributes ──────────────────────────────────────────
    generate_gitattributes()

    elapsed = time.time() - t_start

    # ── 保存元信息供 README 脚本使用 ─────────────────────────────────
    meta = {
        "success_symbols": success_symbols,
        "failed_coins":    failed_coins,
        "total_candles":   total_candles,
        "start_date":      start_dt.strftime("%Y-%m-%d"),
        "end_date":        end_dt.strftime("%Y-%m-%d"),
        "days":            DAYS,
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    meta_path = OUTPUT_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── 汇总报告 ─────────────────────────────────────────────────────
    print()
    print("=" * 76)
    print("   📋  完 成 汇 总")
    print("=" * 76)
    print(f"   ✅ 成功:      {len(success_symbols):>3} / {len(COINS)} 个币种")
    if skipped_symbols:
        print(
            f"   ⏭  跳过:      {len(skipped_symbols):>3} 个 (已存在完整数据)"
        )
    if failed_coins:
        print(f"   ❌ 失败:      {len(failed_coins):>3} 个")
        for fc in failed_coins:
            print(f"                  {fc} → 候选: {coin_to_candidates(fc)}")
    print(f"   📊 总数据量:  {total_candles:>12,} 条 K 线")
    print(f"   ⏱  耗时:      {elapsed / 60:.1f} 分钟")
    print(f"   📁 输出:      {OUTPUT_DIR}")
    print(f"   📄 元信息:    {meta_path}")
    print()
    print("   CSV 列结构 (共 ~27 列):")
    print("   ╔════════════════════════════════════════════════════════╗")
    print("   ║ 原始 OHLCV (12):                                     ║")
    print("   ║   symbol, open_time, O/H/L/C, volume, close_time,    ║")
    print("   ║   quote_volume, num_trades, taker_buy_*              ║")
    print("   ╠════════════════════════════════════════════════════════╣")
    print("   ║ 1m 技术指标 (8):                                     ║")
    print("   ║   rsi_14, ema_12, ema_26, ema_crossover_1m, atr_14,  ║")
    print("   ║   momentum_10, volatility_20, volume_ratio            ║")
    print("   ╠════════════════════════════════════════════════════════╣")
    print("   ║ 多时间尺度 (4):                                      ║")
    print("   ║   ema_crossover_15m, momentum_15m,                    ║")
    print("   ║   ema_crossover_1h, momentum_1h                      ║")
    print("   ╠════════════════════════════════════════════════════════╣")
    print("   ║ Web3 补充 (3):                                       ║")
    print("   ║   funding_rate, taker_buy_sell_ratio, ob_imbalance    ║")
    print("   ╠════════════════════════════════════════════════════════╣")
    print("   ║ 复合信号 (1):                                        ║")
    print("   ║   alpha_score                                         ║")
    print("   ╚════════════════════════════════════════════════════════╝")
    print()
    print("   下一步:")
    print(f"   1. 将 README.md 复制到 {OUTPUT_DIR}/")
    print("   2. 上传到 HuggingFace:")
    print("      pip install huggingface_hub")
    print("      huggingface-cli login")
    print(
        f"      huggingface-cli upload <username>/HKSG "
        f"{OUTPUT_DIR} . --repo-type dataset"
    )
    print()
    print("=" * 76)


if __name__ == "__main__":
    main()
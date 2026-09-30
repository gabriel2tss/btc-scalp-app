"""Fonte Dukascopy (forex, metais, índices): velas M1 bid/ask e ticks, desde ~2003.

Formato .bi5 = LZMA com registros big-endian. Horários já em UTC.
    ticks (1 arquivo/hora): ms desde o início da hora, ask, bid (inteiros), ask_vol, bid_vol (float32)
    velas (1 arquivo/dia):  s desde o início do dia, open, close, low, high (inteiros), volume (float32)
O mês na URL é 0-based.
"""

from __future__ import annotations

import lzma
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .http import NotFound, get

BASE = "https://datafeed.dukascopy.com/datafeed"

TICK_DTYPE = np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"), ("ask_vol", ">f4"), ("bid_vol", ">f4")])
CANDLE_DTYPE = np.dtype([("s", ">u4"), ("open", ">u4"), ("close", ">u4"), ("low", ">u4"),
                         ("high", ">u4"), ("volume", ">f4")])

# Divisor do preço inteiro. Padrão 1e5 (maioria do forex); exceções abaixo.
# Conferido na primeira coleta real contra a faixa de preço esperada (ver sanity_check).
PRICE_SCALE_OVERRIDES = {
    "XAUUSD": 1e3, "XAGUSD": 1e3,
    "USA500IDXUSD": 1e3, "USATECHIDXUSD": 1e3, "USA30IDXUSD": 1e3, "DEUIDXEUR": 1e3,
    "BTCUSD": 1e1, "ETHUSD": 1e1,
}


def price_scale(symbol: str) -> float:
    if symbol in PRICE_SCALE_OVERRIDES:
        return PRICE_SCALE_OVERRIDES[symbol]
    return 1e3 if symbol.endswith("JPY") else 1e5


def _fetch_bi5(url: str) -> bytes | None:
    try:
        raw = get(url).content
    except NotFound:
        return None
    return lzma.decompress(raw) if raw else b""


def decode_ticks(data: bytes, hour_start: pd.Timestamp, scale: float) -> pd.DataFrame:
    a = np.frombuffer(data, dtype=TICK_DTYPE)
    base_ms = hour_start.value // 1_000_000
    return pd.DataFrame({
        "time": pd.to_datetime(base_ms + a["ms"].astype("int64"), unit="ms", utc=True),
        "bid": a["bid"].astype("float64") / scale,
        "ask": a["ask"].astype("float64") / scale,
        "bid_volume": a["bid_vol"].astype("float32"),
        "ask_volume": a["ask_vol"].astype("float32"),
    })


def decode_candles(data: bytes, day_start: pd.Timestamp, scale: float) -> pd.DataFrame:
    a = np.frombuffer(data, dtype=CANDLE_DTYPE)
    base_s = day_start.value // 1_000_000_000
    return pd.DataFrame({
        "time": pd.to_datetime(base_s + a["s"].astype("int64"), unit="s", utc=True),
        "open": a["open"] / scale, "high": a["high"] / scale,
        "low": a["low"] / scale, "close": a["close"] / scale,
        "volume": a["volume"].astype("float64"),
    })


def _day_url(symbol: str, day: pd.Timestamp, side: str) -> str:
    return f"{BASE}/{symbol}/{day.year:04d}/{day.month - 1:02d}/{day.day:02d}/{side}_candles_min_1.bi5"


def _hour_url(symbol: str, hour: pd.Timestamp) -> str:
    return f"{BASE}/{symbol}/{hour.year:04d}/{hour.month - 1:02d}/{hour.day:02d}/{hour.hour:02d}h_ticks.bi5"


def _month_bounds(year: int, month: int, now: pd.Timestamp):
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    end = min(start + pd.offsets.MonthBegin(1), now.floor("D"))  # dia corrente ainda não fechou
    return start, end


def fetch_m1_month(symbol: str, year: int, month: int, now: pd.Timestamp, workers: int = 3) -> tuple[pd.DataFrame, dict]:
    """Velas M1 do mês: OHLC do bid + OHLC do ask + volumes. Minutos sem negociação são descartados."""
    scale = price_scale(symbol)
    start, end = _month_bounds(year, month, now)
    days = list(pd.date_range(start, end, freq="D", inclusive="left"))

    def one(day):
        out = {}
        for side in ("BID", "ASK"):
            data = _fetch_bi5(_day_url(symbol, day, side))
            out[side] = decode_candles(data, day, scale) if data else None
        return out

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(one, days))

    frames, missing_days = [], 0
    for day, r in zip(days, results):
        if r["BID"] is None or r["ASK"] is None:
            missing_days += 1
            continue
        b, a = r["BID"], r["ASK"]
        frames.append(pd.DataFrame({
            "time": b["time"],
            "open": b["open"], "high": b["high"], "low": b["low"], "close": b["close"],
            "ask_open": a["open"].values, "ask_high": a["high"].values,
            "ask_low": a["low"].values, "ask_close": a["close"].values,
            "bid_volume": b["volume"], "ask_volume": a["volume"].values,
        }))
    if not frames:
        return pd.DataFrame(), {"days_requested": len(days), "days_missing": missing_days, "flat_minutes_dropped": 0}
    df = pd.concat(frames, ignore_index=True)
    flat = (df["bid_volume"] == 0) & (df["ask_volume"] == 0)
    df = df[~flat].sort_values("time").reset_index(drop=True)
    return df, {"days_requested": len(days), "days_missing": missing_days,
                "flat_minutes_dropped": int(flat.sum()), "price_scale": scale}


def fetch_ticks_month(symbol: str, year: int, month: int, now: pd.Timestamp, workers: int = 16) -> tuple[pd.DataFrame, dict]:
    scale = price_scale(symbol)
    start, end = _month_bounds(year, month, now)
    hours = list(pd.date_range(start, end, freq="h", inclusive="left"))

    def one(h):
        data = _fetch_bi5(_hour_url(symbol, h))
        return None if data is None else decode_ticks(data, h, scale)

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(one, hours))
    missing = sum(r is None for r in results)
    frames = [r for r in results if r is not None and len(r)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df, {"hours_requested": len(hours), "hours_missing": missing, "price_scale": scale}


def sanity_check(df: pd.DataFrame, price_cols=("close",)) -> dict:
    """Checagens básicas de dados reais: preço positivo, ask >= bid, sem saltos absurdos."""
    col = price_cols[0] if price_cols[0] in df else "bid"
    p = df[col]
    out = {"min_price": float(p.min()), "max_price": float(p.max())}
    ask_col, bid_col = ("ask_close", "close") if "ask_close" in df else ("ask", "bid")
    spread = df[ask_col] - df[bid_col]
    out["negative_spread_rows"] = int((spread < 0).sum())
    out["median_spread"] = float(spread.median())
    r = np.log(p).diff().abs()
    out["max_abs_log_return"] = float(r.max())
    return out

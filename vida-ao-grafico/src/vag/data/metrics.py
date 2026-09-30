"""Métricas de derivativos da Binance (futuros USDT-M), a cada 5 min, de data.binance.vision.

Colunas: interesse aberto (contratos e valor), proporção comprados/vendidos dos maiores traders
(contas e posições), proporção de todas as contas e razão de volume agressor comprador/vendedor.
Existe desde ~set/2020 para BTCUSDT. Arquivos diários, conferidos pelo .CHECKSUM.

    python -m vag.data.metrics --start 2020-09-01
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..config import load_config
from . import binance

COLS = ["create_time", "symbol", "sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]


def path_for(cfg: dict, symbol: str):
    return cfg["data_dir"] / "metrics" / f"{symbol}_UM.parquet"


def _day(symbol: str, day: pd.Timestamp, tmp: str) -> pd.DataFrame | None:
    url = f"{binance.BASES['um']}/daily/metrics/{symbol}/{symbol}-metrics-{day:%Y-%m-%d}.zip"
    p = binance._download_verified(url, tmp)
    if p is None:
        return None
    t = binance._read_csv_from_zip(p, COLS).to_pandas()
    return t.drop(columns="symbol")


def download(symbol: str, start: str, end: str | None = None, workers: int = 8, log=print) -> pd.DataFrame:
    cfg = load_config()
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC").tz_convert(None).floor("D") - pd.Timedelta(days=1)
    days = pd.date_range(pd.Timestamp(start), end_ts, freq="D")
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(workers) as ex:
        parts = [p for p in ex.map(lambda d: _day(symbol, d, tmp), days) if p is not None]
    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df.pop("create_time"), utc=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    for c in df.columns:
        if c != "time":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    out = path_for(cfg, symbol)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="zstd")
    log(f"métricas {symbol}: {len(df):,} registros de {df.time.iloc[0]} a {df.time.iloc[-1]} "
        f"({len(parts)} de {len(days)} dias encontrados) -> {out}")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2020-09-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args(argv)
    download(args.symbol, args.start, args.end)
    return 0


if __name__ == "__main__":
    sys.exit(main())

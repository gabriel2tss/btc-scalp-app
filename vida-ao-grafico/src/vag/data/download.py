"""Download de dados reais de fontes públicas para Parquet (mês a mês, retomável).

Forex / metais / índices (Dukascopy, horários em UTC):
    python -m vag.data.download --source dukascopy --symbol EURUSD --kind m1    --start 2015-01
    python -m vag.data.download --source dukascopy --symbol EURUSD --kind ticks --start 2023-01

Cripto (Binance spot / futuros perpétuos USDT-M):
    python -m vag.data.download --source binance    --symbol BTCUSDT --kind m1 --start 2017-08
    python -m vag.data.download --source binance-um --symbol BTCUSDT --kind m1 --start 2019-09
    python -m vag.data.download --source binance --symbol BTCUSDT --kind trades --start 2024-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import pandas as pd

from ..config import load_config
from . import binance, dukascopy
from .collect import free_gb, iter_months
from .quality import find_gaps, summarize_gaps
from .storage import Manifest, month_path, write_month

KINDS = {"dukascopy": ("m1", "ticks"), "binance": ("m1", "trades"), "binance-um": ("m1", "trades")}


def storage_symbol(source: str, symbol: str) -> str:
    """Futuros ficam em pasta separada do spot: BTCUSDT (spot) x BTCUSDT_UM (perpétuo)."""
    return f"{symbol}_UM" if source == "binance-um" else symbol


def download_month(cfg: dict, manifest: Manifest, source: str, kind: str, symbol: str,
                   year: int, month: int, now: pd.Timestamp, force: bool = False) -> dict | None:
    market = "um" if source == "binance-um" else "spot"
    sym_out = storage_symbol(source, symbol)
    m_end = pd.Timestamp(year=year, month=month, day=1, tz="UTC") + pd.offsets.MonthBegin(1)
    complete = m_end <= now.floor("D")
    path = month_path(cfg["data_dir"], kind, sym_out, year, month)
    prev = manifest.get(kind, sym_out, year, month)
    if prev and prev.get("complete") and (path.exists() or prev.get("rows") == 0) and not force:
        return None

    comp, lvl = cfg["storage"]["compression"], cfg["storage"]["compression_level"]
    min_gap = pd.Timedelta(minutes=cfg["gaps"]["m1_min_gap_minutes" if kind == "m1" else "ticks_min_gap_minutes"])

    if source.startswith("binance") and kind == "trades":
        meta = binance.fetch_trades_month_to_parquet(symbol, year, month, now, path, comp, lvl, market)
        entry = {"source": source, "complete": complete, **meta}
        manifest.put(kind, sym_out, year, month, entry)
        return entry

    if kind == "m1" and source.startswith("binance"):
        df, meta = binance.fetch_m1_month(symbol, year, month, now, market)
    elif kind == "m1":
        df, meta = dukascopy.fetch_m1_month(symbol, year, month, now)
    else:
        df, meta = dukascopy.fetch_ticks_month(symbol, year, month, now)

    entry = {"source": source, "complete": complete, "rows": int(len(df)), "bytes": 0, **meta}
    if len(df):
        entry["bytes"] = int(write_month(df, path, comp, lvl))
        entry["first"], entry["last"] = str(df["time"].iloc[0]), str(df["time"].iloc[-1])
        entry["gaps"] = summarize_gaps(find_gaps(df["time"], min_gap))
        if source == "dukascopy":
            entry["sanity"] = dukascopy.sanity_check(df)
    manifest.put(kind, sym_out, year, month, entry)
    return entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=list(KINDS), required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--start", required=True, help="AAAA-MM")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m"), help="AAAA-MM (inclusive)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.kind not in KINDS[args.source]:
        ap.error(f"--kind para {args.source}: {KINDS[args.source]}")

    cfg = load_config()
    manifest = Manifest(cfg["data_dir"])
    now = pd.Timestamp.now(tz="UTC")
    for y, m in iter_months(args.start, args.end):
        free = free_gb(cfg["data_dir"])
        if free < cfg["disk"]["min_free_gb"]:
            print(f"Parando: só {free:.1f} GB livres (mínimo {cfg['disk']['min_free_gb']} GB).", file=sys.stderr)
            return 3
        e = download_month(cfg, manifest, args.source, args.kind, args.symbol, y, m, now, args.force)
        if e is None:
            print(f"  = {y}-{m:02d} já existe", flush=True)
            continue
        g = e.get("gaps", {})
        print(f"  + {y}-{m:02d}: {e['rows']:>12,} linhas  {e['bytes'] / 1024**2:8.2f} MB  "
              f"lacunas em dia útil: {g.get('weekday', '-')}", flush=True)

    for k, t in manifest.totals().items():
        print(f"{k}: {t['months']} meses, {t['rows']:,} linhas, {t['bytes'] / 1024**2:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

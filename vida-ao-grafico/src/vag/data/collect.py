"""Fase 1 — coleta do MT5 para Parquet (mês a mês, retomável).

Exemplo (no PC com MT5):
    python -m vag.data.collect --symbol EURUSD --kind m1 --start 2015-01 --end 2025-12
    python -m vag.data.collect --symbol EURUSD --kind ticks --start 2023-01 --end 2025-12
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone

import pandas as pd

from ..config import load_config
from .mt5_source import Source, normalize_rates, normalize_ticks, utc
from .quality import find_gaps, summarize_gaps
from .storage import Manifest, month_path, write_month

# O fuso do servidor desloca o mês em até ~3h; pedimos uma margem e recortamos em UTC.
FETCH_MARGIN = pd.Timedelta(hours=12)


def iter_months(start: str, end: str):
    p = pd.Period(start, "M")
    last = pd.Period(end, "M")
    while p <= last:
        yield p.year, p.month
        p += 1


def free_gb(path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1024**3


def collect_month(src: Source, cfg: dict, manifest: Manifest, kind: str, symbol: str,
                  year: int, month: int, now: datetime | None = None, force: bool = False) -> dict | None:
    now = now or datetime.now(timezone.utc)
    m_start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    m_end = m_start + pd.offsets.MonthBegin(1)
    complete = m_end <= pd.Timestamp(now)
    path = month_path(cfg["data_dir"], kind, symbol, year, month)

    prev = manifest.get(kind, symbol, year, month)
    if prev and prev.get("complete") and path.exists() and not force:
        return None  # já baixado

    st = cfg["server_time"]
    a = utc((m_start - FETCH_MARGIN).to_pydatetime())
    b = utc(min(m_end + FETCH_MARGIN, pd.Timestamp(now)).to_pydatetime())
    if kind == "m1":
        raw = src.rates_range(symbol, a, b)
        df = normalize_rates(raw, st["mode"], st["fixed_offset_hours"]) if raw is not None else pd.DataFrame()
        min_gap = pd.Timedelta(minutes=cfg["gaps"]["m1_min_gap_minutes"])
    elif kind == "ticks":
        raw = src.ticks_range(symbol, a, b)
        df = normalize_ticks(raw, st["mode"], st["fixed_offset_hours"]) if raw is not None else pd.DataFrame()
        min_gap = pd.Timedelta(minutes=cfg["gaps"]["ticks_min_gap_minutes"])
    else:
        raise ValueError(kind)

    if raw is None:
        print(f"  ! {symbol} {kind} {year}-{month:02d}: MT5 devolveu None ({src.last_error()})", file=sys.stderr)
    if not df.empty:
        df = df[(df["time"] >= m_start) & (df["time"] < m_end)].reset_index(drop=True)
    if df.empty:
        entry = {"rows": 0, "bytes": 0, "complete": complete, "source": "mt5"}
        manifest.put(kind, symbol, year, month, entry)
        return entry

    nbytes = write_month(df, path, cfg["storage"]["compression"], cfg["storage"]["compression_level"])
    entry = {
        "source": "mt5",
        "rows": int(len(df)),
        "bytes": int(nbytes),
        "first": str(df["time"].iloc[0]),
        "last": str(df["time"].iloc[-1]),
        "complete": complete,
        "server_time": dict(st),
        "gaps": summarize_gaps(find_gaps(df["time"], min_gap)),
    }
    manifest.put(kind, symbol, year, month, entry)
    return entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--kind", choices=["m1", "ticks"], required=True)
    ap.add_argument("--start", required=True, help="AAAA-MM")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m"), help="AAAA-MM (inclusive)")
    ap.add_argument("--force", action="store_true", help="rebaixa meses já completos")
    ap.add_argument("--mt5-path", default=None, help="caminho do terminal64.exe, se houver mais de um")
    args = ap.parse_args(argv)

    cfg = load_config()
    from .mt5_source import MT5Source

    src = MT5Source(args.mt5_path)
    manifest = Manifest(cfg["data_dir"])
    info = src.symbol_info(args.symbol)
    if info is None:
        print(f"Símbolo {args.symbol} não encontrado na corretora.", file=sys.stderr)
        return 2
    manifest.set_symbol_info(args.symbol, info)

    try:
        for y, m in iter_months(args.start, args.end):
            free = free_gb(cfg["data_dir"])
            if free < cfg["disk"]["min_free_gb"]:
                print(f"Parando: só {free:.1f} GB livres (mínimo {cfg['disk']['min_free_gb']} GB).", file=sys.stderr)
                return 3
            e = collect_month(src, cfg, manifest, args.kind, args.symbol, y, m, force=args.force)
            if e is None:
                print(f"  = {y}-{m:02d} já existe")
            else:
                g = e.get("gaps", {})
                print(f"  + {y}-{m:02d}: {e['rows']:>10,} linhas  {e['bytes']/1024**2:7.2f} MB  "
                      f"lacunas em dia útil: {g.get('weekday', 0)} ({g.get('weekday_missing_minutes', 0)} min)")
    finally:
        src.close()

    for k, t in manifest.totals().items():
        print(f"{k}: {t['months']} meses, {t['rows']:,} linhas, {t['bytes']/1024**2:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

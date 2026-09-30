"""Fonte Binance (cripto spot e futuros USDT-M): velas 1m e aggTrades, do repositório público data.binance.vision.

Meses fechados vêm do arquivo mensal; o mês corrente, dos arquivos diários.
Todo arquivo é conferido contra o .CHECKSUM (sha256) publicado pela Binance.
A partir de 2025 os timestamps do spot passaram de ms para µs — detectado automaticamente.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .http import NotFound, get

BASES = {
    "spot": "https://data.binance.vision/data/spot",
    "um": "https://data.binance.vision/data/futures/um",   # futuros perpétuos USDT-M (desde 2019-09 p/ BTC)
}

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
AGG_COLS = ["agg_id", "price", "qty", "first_id", "last_id", "time", "is_buyer_maker", "best_match"]
AGG_TYPES = {"agg_id": pa.int64(), "price": pa.float64(), "qty": pa.float64(), "first_id": pa.int64(),
             "last_id": pa.int64(), "time": pa.int64(), "is_buyer_maker": pa.bool_(), "best_match": pa.bool_()}


def _to_utc(ts: pd.Series | np.ndarray) -> pd.Series:
    ts = np.asarray(ts, dtype="int64")
    unit = "us" if ts.size and ts.max() > 10**14 else "ms"
    return pd.to_datetime(ts, unit=unit, utc=True)


def _download_verified(url: str, dest_dir: str) -> str | None:
    """Baixa o zip para disco (streaming) e confere o sha256. None se não existir."""
    try:
        expected = get(url + ".CHECKSUM").text.split()[0]
        r = get(url, stream=True, timeout=300)
    except NotFound:
        return None
    path = os.path.join(dest_dir, url.rsplit("/", 1)[1])
    h = hashlib.sha256()
    with open(path, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
            h.update(chunk)
    if h.hexdigest() != expected:
        os.remove(path)
        raise IOError(f"checksum não confere: {url}")
    return path


def _urls(kind: str, symbol: str, year: int, month: int, now: pd.Timestamp, market: str = "spot",
          daily: bool = False) -> list[str]:
    base = BASES[market]
    m_start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    m_end = m_start + pd.offsets.MonthBegin(1)
    sub = f"klines/{symbol}/1m" if kind == "m1" else f"aggTrades/{symbol}"
    name = f"{symbol}-1m" if kind == "m1" else f"{symbol}-aggTrades"
    if m_end <= now.floor("D") and not daily:
        return [f"{base}/monthly/{sub}/{name}-{year:04d}-{month:02d}.zip"]
    days = pd.date_range(m_start, min(m_end, now.floor("D")), freq="D", inclusive="left")
    return [f"{base}/daily/{sub}/{name}-{d:%Y-%m-%d}.zip" for d in days]


def _read_csv_from_zip(path: str, names: list[str]) -> pa.Table:
    with zipfile.ZipFile(path) as z:
        data = z.read(z.namelist()[0])
    # Alguns arquivos mais novos trazem cabeçalho; os antigos não.
    first = data.split(b"\n", 1)[0]
    skip = 1 if first[:1].isalpha() else 0
    return pacsv.read_csv(io.BytesIO(data), read_options=pacsv.ReadOptions(column_names=names, skip_rows=skip))


def _iter_csv_batches_from_zip(path: str, names: list[str], types: dict, block_size: int = 64 << 20):
    """Lê o CSV de dentro do zip em blocos: um mês de aggTrades descompactado passa de 5 GB.

    Os futuros USDT-M trazem 7 colunas (sem best_match); o spot, 8. Cabeçalho opcional.
    """
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            first = fh.readline()
        skip = 1 if first[:1].isalpha() else 0
        cols = names[: first.count(b",") + 1]
        opts = pacsv.ReadOptions(column_names=cols, skip_rows=skip, block_size=block_size)
        conv = pacsv.ConvertOptions(column_types={c: types[c] for c in cols})
        with z.open(name) as fh:
            yield from pacsv.open_csv(fh, read_options=opts, convert_options=conv)


def fetch_m1_month(symbol: str, year: int, month: int, now: pd.Timestamp, market: str = "spot") -> tuple[pd.DataFrame, dict]:
    urls = _urls("m1", symbol, year, month, now, market)
    frames, missing = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for u in urls:
            p = _download_verified(u, tmp)
            if p is None:
                missing += 1
                continue
            frames.append(_read_csv_from_zip(p, KLINE_COLS).to_pandas())
            os.remove(p)
    if not frames:
        return pd.DataFrame(), {"files": len(urls), "files_missing": missing}
    k = pd.concat(frames, ignore_index=True)
    df = pd.DataFrame({
        "time": _to_utc(k["open_time"]),
        "open": k["open"].astype("float64"), "high": k["high"].astype("float64"),
        "low": k["low"].astype("float64"), "close": k["close"].astype("float64"),
        "volume": k["volume"].astype("float64"),
        "quote_volume": k["quote_volume"].astype("float64"),
        "trades": k["trades"].astype("int64"),
        "taker_buy_volume": k["taker_buy_volume"].astype("float64"),
    })
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df, {"files": len(urls), "files_missing": missing}


def fetch_trades_month_to_parquet(symbol: str, year: int, month: int, now: pd.Timestamp,
                                  out_path, compression="zstd", level=9, market: str = "spot", daily: bool = False) -> dict:
    """aggTrades do mês direto para Parquet, arquivo por arquivo (não cabe tudo em memória).

    daily=True usa os arquivos diários mesmo para meses fechados: alguns mensais de aggTrades
    dos futuros vêm cortados em 20.000.000 de linhas, com dias inteiros faltando (ex.: 2020-04).
    """
    urls = _urls("trades", symbol, year, month, now, market, daily)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_out = tempfile.mkstemp(dir=out_path.parent, suffix=".tmp")
    os.close(fd)
    writer, rows, missing, first, last = None, 0, 0, None, None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for u in urls:
                p = _download_verified(u, tmp)
                if p is None:
                    missing += 1
                    continue
                for t in _iter_csv_batches_from_zip(p, AGG_COLS, AGG_TYPES):
                    if t.num_rows == 0:
                        continue
                    ts = _to_utc(t.column("time").to_numpy())
                    table = pa.table({
                        "time": pa.array(ts),
                        "price": t.column("price"),
                        "qty": t.column("qty"),
                        "is_buyer_maker": t.column("is_buyer_maker"),
                        "n_trades": pc.add(pc.subtract(t.column("last_id"), t.column("first_id")), 1).cast(pa.int32()),
                    })
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_out, table.schema, compression=compression, compression_level=level)
                    writer.write_table(table)
                    rows += table.num_rows
                    first = first if first is not None else ts[0]
                    last = ts[-1]
                os.remove(p)
        if writer is not None:
            writer.close()
            writer = None
            os.replace(tmp_out, out_path)
    finally:
        if writer is not None:   # no Windows, arquivo aberto não pode ser apagado
            writer.close()
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
    return {"files": len(urls), "files_missing": missing, "rows": rows,
            "first": str(first) if first is not None else None, "last": str(last) if last is not None else None,
            "bytes": out_path.stat().st_size if rows else 0}

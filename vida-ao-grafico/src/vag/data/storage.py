"""Armazenamento em Parquet+zstd particionado por ativo/ano, com manifesto.

Layout:
    data/<kind>/symbol=<SYM>/year=<YYYY>/<YYYY>-<MM>.parquet
    data/manifest.json
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def month_path(data_dir: Path, kind: str, symbol: str, year: int, month: int) -> Path:
    return data_dir / kind / f"symbol={symbol}" / f"year={year:04d}" / f"{year:04d}-{month:02d}.parquet"


def write_month(df: pd.DataFrame, path: Path, compression: str = "zstd", level: int = 9) -> int:
    """Grava atomicamente (tmp + rename) e devolve o tamanho em bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression=compression, compression_level=level)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path.stat().st_size


def read_symbol(data_dir: Path, kind: str, symbol: str, years: list[int] | None = None) -> pd.DataFrame:
    base = data_dir / kind / f"symbol={symbol}"
    files = sorted(base.glob("year=*/*.parquet"))
    if years is not None:
        wanted = {f"year={y:04d}" for y in years}
        files = [f for f in files if f.parent.name in wanted]
    if not files:
        return pd.DataFrame()
    return pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)


class Manifest:
    """Registro de tudo que foi baixado: fonte, período, linhas, lacunas, fuso."""

    def __init__(self, data_dir: Path):
        self.path = data_dir / "manifest.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"created_utc": _now(), "timezone": "UTC", "entries": {}}

    @staticmethod
    def key(kind: str, symbol: str, year: int, month: int) -> str:
        return f"{kind}/{symbol}/{year:04d}-{month:02d}"

    def get(self, kind: str, symbol: str, year: int, month: int) -> dict | None:
        return self.data["entries"].get(self.key(kind, symbol, year, month))

    def put(self, kind: str, symbol: str, year: int, month: int, entry: dict) -> None:
        entry = {**entry, "written_utc": _now()}
        self.data["entries"][self.key(kind, symbol, year, month)] = entry
        self.save()

    def set_symbol_info(self, symbol: str, info: dict) -> None:
        self.data.setdefault("symbols", {})[symbol] = info
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def totals(self) -> dict:
        out: dict[str, dict] = {}
        for k, e in self.data["entries"].items():
            kind, sym, _ = k.split("/")
            t = out.setdefault(f"{kind}/{sym}", {"months": 0, "rows": 0, "bytes": 0})
            t["months"] += 1
            t["rows"] += e.get("rows", 0)
            t["bytes"] += e.get("bytes", 0)
        return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

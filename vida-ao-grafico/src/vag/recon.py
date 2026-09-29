"""Fase 0 — reconhecimento do ambiente.

Roda no PC do Gabriel (Windows + MT5 aberto):
    python -m vag.recon
    python -m vag.recon --symbols EURUSD XAUUSD --first-year 2010

Checa: espaço em disco, Python, GPU/CUDA, MT5 (versão, conta, fuso do servidor)
e quanto histórico de M1 e de ticks a corretora oferece por ativo, com uma
estimativa de tamanho em disco medida (não chutada) em Parquet+zstd.
Gera reports/phase0_recon_<data>.json e .md.
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import load_config
from .data.mt5_source import normalize_rates, normalize_ticks, utc
from .data.server_time import estimate_offset_hours

M1_BARS_PER_YEAR = 260 * 24 * 60  # ordem de grandeza para forex (5 dias/semana)
TRADING_DAYS_PER_YEAR = 260


def check_system(cfg) -> dict:
    cfg["data_dir"].mkdir(parents=True, exist_ok=True)
    du = shutil.disk_usage(cfg["data_dir"])
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": __import__("os").cpu_count(),
        "disk_free_gb": round(du.free / 1024**3, 2),
        "disk_total_gb": round(du.total / 1024**3, 2),
        "data_dir": str(cfg["data_dir"]),
    }


def check_gpu() -> dict:
    out: dict = {"torch": None, "cuda_available": False, "devices": [], "nvidia_smi": None}
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
        if out["cuda_available"]:
            out["cuda_version"] = torch.version.cuda
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                out["devices"].append({"name": p.name, "vram_gb": round(p.total_memory / 1024**3, 1)})
    except ImportError:
        pass
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            out["nvidia_smi"] = r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return out


def parquet_bytes(df: pd.DataFrame, cfg) -> int:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf,
                   compression=cfg["storage"]["compression"],
                   compression_level=cfg["storage"]["compression_level"])
    return buf.tell()


def probe_week(year: int) -> tuple[datetime, datetime]:
    """Uma semana útil no meio de março do ano (evita feriados de fim de ano)."""
    d = datetime(year, 3, 10)
    d -= timedelta(days=d.weekday())  # segunda-feira
    return utc(d), utc(d + timedelta(days=5))


def probe_symbol(src, sym: str, cfg, first_year: int) -> dict:
    info = src.symbol_info(sym)
    if info is None:
        return {"symbol": sym, "available": False}
    st = cfg["server_time"]
    res: dict = {"symbol": sym, "available": True, "info": info}

    t = src.last_tick_time(sym)
    if t is not None:
        age = time.time() - t
        res["server_offset_estimate_h"] = estimate_offset_hours(t, time.time())
        res["last_tick_age_s_raw"] = round(age, 1)

    this_year = datetime.now(timezone.utc).year
    m1_years, tick_years = [], []
    m1_sample = tick_sample = None
    for y in range(this_year, first_year - 1, -1):
        a, b = probe_week(y)
        if b > datetime.now(timezone.utc):
            continue  # semana de teste deste ano ainda não aconteceu
        r = src.rates_range(sym, a, b)
        n = 0 if r is None else len(r)
        m1_years.append({"year": y, "bars_in_probe_week": n})
        if n and m1_sample is None:
            m1_sample = normalize_rates(r, st["mode"], st["fixed_offset_hours"])
        # ticks: um dia (quarta-feira) da semana de teste, para não pesar
        ta, tb = a + timedelta(days=2), a + timedelta(days=3)
        tk = src.ticks_range(sym, ta, tb)
        nt = 0 if tk is None else len(tk)
        tick_years.append({"year": y, "ticks_in_probe_day": nt})
        if nt and tick_sample is None:
            tick_sample = normalize_ticks(tk, st["mode"], st["fixed_offset_hours"])

    res["m1_by_year"] = m1_years
    res["ticks_by_year"] = tick_years
    res["m1_earliest_year"] = min((x["year"] for x in m1_years if x["bars_in_probe_week"]), default=None)
    res["ticks_earliest_year"] = min((x["year"] for x in tick_years if x["ticks_in_probe_day"]), default=None)

    if m1_sample is not None and len(m1_sample):
        bpr = parquet_bytes(m1_sample, cfg) / len(m1_sample)
        res["m1_bytes_per_bar"] = round(bpr, 2)
        res["m1_mb_per_year_est"] = round(bpr * M1_BARS_PER_YEAR / 1024**2, 1)
        res["m1_median_spread_points"] = float(m1_sample["spread"].median())
    if tick_sample is not None and len(tick_sample):
        bpt = parquet_bytes(tick_sample, cfg) / len(tick_sample)
        res["ticks_per_day_sample"] = int(len(tick_sample))
        res["ticks_bytes_per_tick"] = round(bpt, 2)
        res["ticks_mb_per_year_est"] = round(bpt * len(tick_sample) * TRADING_DAYS_PER_YEAR / 1024**2, 1)
    return res


def to_markdown(rep: dict) -> str:
    s, g = rep["system"], rep["gpu"]
    L = [f"# Fase 0 — Reconhecimento ({rep['generated_utc']})", "", "## Sistema", ""]
    L += [f"- Plataforma: {s['platform']}", f"- Python: {s['python']}", f"- CPUs: {s['cpu_count']}",
          f"- Disco livre: **{s['disk_free_gb']} GB** de {s['disk_total_gb']} GB (`{s['data_dir']}`)", ""]
    L += ["## GPU", ""]
    if g["cuda_available"]:
        L += [f"- CUDA {g.get('cuda_version')} via torch {g['torch']}"] + \
             [f"- {d['name']} — {d['vram_gb']} GB VRAM" for d in g["devices"]]
    else:
        L += [f"- CUDA via torch: **não** (torch: {g['torch'] or 'não instalado'})",
              f"- nvidia-smi: {g['nvidia_smi'] or 'não encontrado'}"]
    L += ["", "## MetaTrader 5", ""]
    mt = rep.get("mt5")
    if not mt or "error" in mt:
        L += [f"- **Falhou**: {mt.get('error') if mt else 'não executado'}"]
    else:
        acc = mt.get("account") or {}
        L += [f"- Versão: {mt.get('version')}", f"- Corretora: {acc.get('company')} / servidor {acc.get('server')}",
              f"- Modo de fuso configurado: `{rep['server_time_config']}`"]
    L += ["", "## Histórico disponível por ativo", "",
          "| Ativo | M1 desde | Ticks desde | MB/ano M1 | MB/ano ticks | ticks/dia | spread mediano (pts) | offset servidor (h) |",
          "|---|---|---|---|---|---|---|---|"]
    for p in rep.get("symbols", []):
        if not p.get("available"):
            L.append(f"| {p['symbol']} | — indisponível — | | | | | | |")
            continue
        L.append("| {symbol} | {m1} | {tk} | {m1mb} | {tkmb} | {tpd} | {spr} | {off} |".format(
            symbol=p["symbol"], m1=p.get("m1_earliest_year"), tk=p.get("ticks_earliest_year"),
            m1mb=p.get("m1_mb_per_year_est", "—"), tkmb=p.get("ticks_mb_per_year_est", "—"),
            tpd=p.get("ticks_per_day_sample", "—"), spr=p.get("m1_median_spread_points", "—"),
            off=p.get("server_offset_estimate_h", "—")))
    L += ["", "Notas:",
          "- \"desde\" = primeiro ano em que uma semana de teste em março devolveu dados. "
          "A primeira chamada pode disparar download do servidor; se os números parecerem baixos, rode de novo.",
          "- Offset do servidor só é confiável com o mercado aberto (tick recente). "
          "Corretoras no padrão NY-close mostram +2 (inverno) ou +3 (verão).",
          "- Tamanhos em Parquet+zstd, medidos numa amostra real.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="*", default=cfg["recon_candidates"])
    ap.add_argument("--first-year", type=int, default=2005)
    ap.add_argument("--mt5-path", default=None)
    args = ap.parse_args(argv)

    rep: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "system": check_system(cfg),
        "gpu": check_gpu(),
        "server_time_config": cfg["server_time"],
    }
    try:
        from .data.mt5_source import MT5Source

        src = MT5Source(args.mt5_path)
    except Exception as e:  # noqa: BLE001 — relatório precisa sair mesmo sem MT5
        rep["mt5"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        try:
            rep["mt5"] = src.terminal()
            rep["symbols"] = []
            for sym in args.symbols:
                print(f"sondando {sym}...", flush=True)
                rep["symbols"].append(probe_symbol(src, sym, cfg, args.first_year))
        finally:
            src.close()

    cfg["reports_dir"].mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    jp = cfg["reports_dir"] / f"phase0_recon_{stamp}.json"
    mp = cfg["reports_dir"] / f"phase0_recon_{stamp}.md"
    jp.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    md = to_markdown(rep)
    mp.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nRelatório salvo em {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

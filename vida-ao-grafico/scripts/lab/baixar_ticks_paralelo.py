"""Download de aggTrades BTCUSDT (futuros UM) com vários meses em paralelo.

    python par_download.py 2021-10 2022-12 4 [--daily] [--force] [--level 3]

Cada processo baixa um mês (streaming, checksum conferido); só o processo principal
escreve o manifesto, então não há corrida no manifest.json.
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from vag.config import load_config
from vag.data import binance
from vag.data.storage import Manifest, month_path

SYM, SYM_OUT = "BTCUSDT", "BTCUSDT_UM"


def work(args):
    y, m, level, daily = args
    cfg = load_config()
    now = pd.Timestamp.now(tz="UTC")
    path = month_path(cfg["data_dir"], "trades", SYM_OUT, y, m)
    t0 = time.time()
    try:
        meta = binance.fetch_trades_month_to_parquet(SYM, y, m, now, path, "zstd", level, "um", daily)
        return y, m, meta, time.time() - t0, None
    except Exception as e:  # o mês fica sem arquivo; a conferência de cobertura pega e o reparo refaz
        return y, m, None, time.time() - t0, repr(e)


def main():
    start, end, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    daily, force = "--daily" in sys.argv, "--force" in sys.argv
    level = int(sys.argv[sys.argv.index("--level") + 1]) if "--level" in sys.argv else 3
    cfg = load_config()
    man = Manifest(cfg["data_dir"])
    todo = []
    for p in pd.period_range(start, end, freq="M"):
        e = man.get("trades", SYM_OUT, p.year, p.month)
        if not force and e and e.get("complete") and month_path(cfg["data_dir"], "trades", SYM_OUT, p.year, p.month).exists():
            print(f"  = {p} já existe", flush=True)
            continue
        todo.append((p.year, p.month, level, daily))
    print(f"baixando {len(todo)} meses com {workers} processos (zstd {level}, diários={daily})", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(work, a) for a in todo]):
            y, m, meta, dt, err = f.result()
            if err:
                print(f"  ! {y}-{m:02d}: ERRO {err}", flush=True)
                continue
            man = Manifest(cfg["data_dir"])   # relê: outro processo (reparo) pode ter escrito
            man.put("trades", SYM_OUT, y, m, {"source": "binance-um", "complete": True, "daily_files": daily, **meta})
            print(f"  + {y}-{m:02d}: {meta['rows']:>12,} linhas {meta['bytes'] / 2**20:7.1f} MB em {dt / 60:.1f} min", flush=True)
    print(f"FIM {start}..{end} em {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

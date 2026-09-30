"""Experimento tick a tick numa janela do walk-forward.

Mesmas etapas, mesma configuração e mesma semente da rodada só com velas; a única
diferença é que cada minuto ganha as features de microestrutura dos ticks
(tick_features.py), somadas às 23 features da vela. Assim:
- gbm_cru (velas + ticks) x gbm_cru (só velas)  -> os ticks acrescentam informação aos indicadores?
- dialeto (velas + ticks) x dialeto (só velas)   -> o dialeto fica melhor com o dado detalhado?
- momentum_reversao só usa a vela: deve dar igual nas duas (checagem de que a comparação é justa)

    python -m vag.tick_experiment                                          # janela 0, config completa
    python -m vag.tick_experiment --windows 3 7                            # várias janelas (ticks até a última)
    python -m vag.tick_experiment --smoke --block 2020-03-15 2020-04-01 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
import yaml

from .config import CONFIG_DIR, PROJECT_ROOT, load_config
from .data.download import storage_symbol
from .data.storage import read_symbol
from .features import FEATURES, compute_features, forward_returns, valid_mask
from .tick_features import TICK_FEATURES, aggregate_ticks, compute_tick_features
from .walkforward import SMOKE, Logger, _merge, run_window, write_report


def load_timeline_ticks(cfg: dict, exp: dict, block, run_dir: Path, log) -> dict:
    cache = run_dir / f"timeline_ate_{block[1]}.parquet"   # serve para qualquer janela que termine até aqui
    cols = FEATURES + TICK_FEATURES
    horizons = exp["horizons_minutes"]
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        sym = storage_symbol(exp["data"]["source"], exp["data"]["symbol"])
        end = pd.Timestamp(block[1], tz="UTC")
        bars = read_symbol(cfg["data_dir"], "m1", sym)
        bars = bars.sort_values("time").drop_duplicates("time")
        bars = bars[bars["time"] < end + pd.Timedelta(days=1)].reset_index(drop=True)   # folga p/ o retorno futuro
        months = pd.period_range(bars["time"].iloc[0].tz_convert(None).to_period("M"),
                                 (end - pd.Timedelta(minutes=1)).tz_convert(None).to_period("M"), freq="M")
        base = cfg["data_dir"] / "trades" / f"symbol={sym}"
        files = [base / f"year={p.year:04d}" / f"{p.year:04d}-{p.month:02d}.parquet" for p in months]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"Faltam ticks de {sym}: {missing}. Rode o download --kind trades primeiro.")
        log(f"resumindo {len(files)} meses de ticks por minuto...")
        t0 = time.time()
        agg = aggregate_ticks(files, log)
        log(f"  {len(agg):,} minutos com ticks, {int(agg['n_agg'].sum()):,} ordens agressoras em {time.time() - t0:.0f}s")
        feats = compute_features(bars)
        tfe = compute_tick_features(bars, agg)
        log(f"  conferência ticks x vela: {tfe.attrs['check']}")
        fwd = forward_returns(bars, horizons)
        base_ok = valid_mask(feats)
        mask = base_ok & tfe.notna().all(axis=1)
        log(f"  minutos válidos: só vela {int(base_ok.sum()):,} | vela + ticks {int(mask.sum()):,}")
        df = pd.concat([feats, tfe, fwd.drop(columns="time")], axis=1)[mask].reset_index(drop=True)
        df.to_parquet(cache, compression="zstd")
    times = pd.to_datetime(df["time"], utc=True)
    breaks = np.r_[True, (times.diff().dt.total_seconds().values[1:] != 60)]
    return {"times": times, "X": df[cols].to_numpy(np.float32),
            "fwd": {h: df[f"fwd_{h}"].to_numpy(np.float64) for h in horizons}, "breaks": breaks}


def coverage(cfg: dict, exp: dict, start: str, end: str, log=print) -> list[str]:
    """Meses [start, end] cujos ticks não cobrem todos os minutos com volume na vela M1."""
    import pyarrow.parquet as pq
    sym = storage_symbol(exp["data"]["source"], exp["data"]["symbol"])
    bars = read_symbol(cfg["data_dir"], "m1", sym)
    bars_min = pd.DatetimeIndex(bars.loc[bars["volume"] > 0, "time"]).as_unit("ns").asi8 // 60_000_000_000
    bad = []
    for p in pd.period_range(start, end, freq="M"):
        f = cfg["data_dir"] / "trades" / f"symbol={sym}" / f"year={p.year:04d}" / f"{p.year:04d}-{p.month:02d}.parquet"
        lo = pd.Timestamp(p.start_time, tz="UTC").value // 60_000_000_000
        hi = pd.Timestamp(p.end_time, tz="UTC").value // 60_000_000_000
        want = bars_min[(bars_min >= lo) & (bars_min <= hi)]
        if not f.exists():
            log(f"  {p}: sem arquivo de ticks")
            bad.append(str(p))
            continue
        t = pq.read_table(f, columns=["time"])["time"]
        have = np.unique(t.cast(pa.timestamp("ms", tz=t.type.tz), safe=False).cast(pa.int64()).to_numpy() // 60_000)
        miss = np.setdiff1d(want, have, assume_unique=True)
        days = sorted({str(pd.Timestamp(m * 60_000, unit="ms").date()) for m in miss})
        log(f"  {p}: {len(miss):,} minutos com vela e sem ticks" + (f" (dias: {', '.join(days[:8])}{' ...' if len(days) > 8 else ''})" if len(miss) else ""))
        if len(miss) > 60:          # tolera poucos minutos soltos
            bad.append(str(p))
    log("MESES_COM_BURACO: " + " ".join(bad))
    return bad


def _row(v: dict | None) -> tuple:
    if not v or not v.get("escolha"):
        return ("—", 0, None, None, None)
    t = v["teste_taker"]
    return (v["escolha"], t["trades"], t["mean_net_bps"], t["hit"], t["t_stat"])


def _f(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def write_comparison(res: dict, base_path: Path, run_dir: Path, wi: int) -> Path:
    L = ["# Tick a tick x só velas — mesma janela, mesma configuração", ""]
    if not base_path.exists():
        L.append(f"(sem resultado da rodada só com velas em `{base_path}` para comparar)")
    else:
        b = json.loads(base_path.read_text(encoding="utf-8"))
        per = f"{res['periodos']['test'][0][:10]} → {res['periodos']['test'][1][:10]}"
        L += [f"Teste: {per}. Custo taker 12 bps ida e volta. Números do teste (nunca visto).", "",
              "| Método | Só velas: op. | média (bps) | acerto | t | Velas + ticks: op. | média (bps) | acerto | t |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m in ("dialeto", "gbm_dialeto", "gbm_cru", "momentum_reversao"):
            _, n0, m0, h0, t0 = _row(b["metodos"].get(m))
            _, n1, m1, h1, t1 = _row(res["metodos"].get(m))
            L.append(f"| {m} | {n0} | {_f(m0)} | {_f(h0, 3)} | {_f(t0, 2)} | {n1} | {_f(m1)} | {_f(h1, 3)} | {_f(t1, 2)} |")
        L += ["", "| | Só velas | Velas + ticks |", "|---|---|---|"]
        for k, lab in (("used_letters", "letras usadas"), ("perplexity", "perplexidade das letras")):
            L.append(f"| {lab} | {_f(float(b['alfabeto'][k]), 0)} | {_f(float(res['alfabeto'][k]), 0)} |")
        L.append(f"| LM: ganho sobre unigrama (nats) | {_f(b['lm']['val_gain_over_unigram_nats'], 3)} | "
                 f"{_f(res['lm']['val_gain_over_unigram_nats'], 3)} |")
        L.append(f"| surpresa × |retorno| | {_f(b['surpresa']['spearman_surpresa_vs_abs_ret'], 3)} | "
                 f"{_f(res['surpresa']['spearman_surpresa_vs_abs_ret'], 3)} |")
        L += ["", "Escolhas feitas na validação (podem diferir entre as duas rodadas):", ""]
        for m in ("dialeto", "gbm_dialeto", "gbm_cru", "momentum_reversao"):
            L.append(f"- {m}: só velas `{_row(b['metodos'].get(m))[0]}` | velas + ticks `{_row(res['metodos'].get(m))[0]}`")
        L += ["", "Leitura: momentum_reversao só usa a vela e deve dar igual nas duas colunas (checagem de justiça). "
              "gbm_cru com ticks x sem ticks responde se o dado detalhado acrescenta informação aos indicadores; "
              "dialeto com ticks x sem ticks, se o dialeto melhora com ele."]
    p = run_dir / f"comparacao_w{wi:02d}.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--windows", nargs="*", type=int, default=None, help="várias janelas (substitui --window)")
    ap.add_argument("--block", nargs=2, default=None, help="teste [início, fim) em UTC; padrão = bloco da janela")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--coverage", nargs=2, metavar=("AAAA-MM", "AAAA-MM"),
                    help="só confere se os ticks cobrem as velas nesses meses e sai")
    args = ap.parse_args(argv)

    cfg = load_config()
    exp = yaml.safe_load((CONFIG_DIR / "experiment.yaml").read_text(encoding="utf-8"))
    if args.coverage:
        coverage(cfg, exp, *args.coverage)
        return 0
    base_run = exp["run_name"]
    if args.smoke:
        exp = _merge(exp, SMOKE)
        exp = _merge(exp, {"lm": {"max_steps": 40, "eval_every": 20}})
    exp["run_name"] = f"{base_run}_ticks" + ("_smoke" if args.smoke else "")
    windows = args.windows or [args.window]
    if args.block and len(windows) > 1:
        ap.error("--block só com uma janela")
    blocks = {w: (args.block or exp["walkforward"]["test_blocks"][w]) for w in windows}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = PROJECT_ROOT / "runs" / exp["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment.yaml").write_text(yaml.safe_dump(exp, allow_unicode=True), encoding="utf-8")
    log = Logger(run_dir / "log.txt")
    gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU"
    log(f"experimento tick a tick {exp['run_name']} | janelas {windows} | dispositivo: {gpu}")
    np.random.seed(exp["seed"])
    torch.manual_seed(exp["seed"])
    last = max(blocks.values(), key=lambda b: pd.Timestamp(b[1]))
    tl = load_timeline_ticks(cfg, exp, last, run_dir, log)   # linhas além do teste de cada janela não são usadas
    for w in windows:
        t0 = time.time()
        res = run_window(w, blocks[w], tl, exp, run_dir, device, log)
        log(f"janela {w} concluída em {(time.time() - t0) / 60:.1f} min")
        p = write_comparison(res, PROJECT_ROOT / "runs" / base_run / f"w{w:02d}" / "result.json", run_dir, w)
        log(f"comparação: {p}")
        done = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(run_dir.glob("w*/result.json"))]
        write_report(done, exp, run_dir, args.smoke)
    return 0

if __name__ == "__main__":
    sys.exit(main())

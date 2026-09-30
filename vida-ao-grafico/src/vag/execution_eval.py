"""Execução realista com ordens limitadas (maker), simulada segundo a segundo com os ticks.

O "maker" do relatório supõe que toda ordem limitada é executada no preço desejado. Na prática,
a ordem de compra só é executada se o preço vier até ela — e vem com mais frequência justamente
quando o mercado está caindo (seleção adversa). Aqui:

- Sinais: os MESMOS da rodada só com velas (gbm_cru e gbm_dialeto de runs/<run>/wNN), regenerados
  de forma determinística e conferidos contra o result.json (tem que reproduzir o teste taker).
- Entrada: no fim do minuto do sinal, ordem limitada 1 tick melhor que o último preço (compra em
  close - tick, venda em close + tick). Executa só se algum negócio real passar ESTRITAMENTE do
  preço dentro de W_in segundos; senão cancela (operação perdida).
- Saída: após h minutos da execução, ordem limitada 1 tick além do último preço por W_out segundos;
  se não executar, sai a mercado (taxa taker + derrapagem).
- Taxas Binance futuros: maker 0,02% / taker 0,05% por lado; derrapagem 0,01% na ordem a mercado.
- Uma posição por vez; enquanto a ordem de entrada espera, novos sinais são ignorados.

    python -m vag.execution_eval --windows 0 1 2 3 4 5 6 7
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from . import baselines as B
from .alphabet import Alphabet
from .config import CONFIG_DIR, PROJECT_ROOT, load_config
from .data.download import storage_symbol
from .data.storage import read_symbol
from .evaluate import metrics, phrase_keys, simulate
from .walkforward import Logger, _roll_z, load_timeline, split_window, target_encode
from .words import Vocabulary

TICK = 0.1
FEES = {"maker": 0.0002, "taker": 0.0005, "slippage": 0.0001}
VARIANTS = {"limitada_1min": (60, 60), "limitada_5min": (300, 60)}   # (W_in, W_out) em segundos
METHODS = ("gbm_dialeto", "gbm_cru")


def regen_signals(w: int, tl: dict, exp: dict, base_dir, log) -> dict:
    """Refaz os sinais escolhidos na janela w (mesmos dados, mesma semente) e confere com o result.json."""
    wf, H = exp["walkforward"], exp["horizons_minutes"]
    sp = split_window(tl["times"], wf["test_blocks"][w], wf["embargo_minutes"], wf["val_fraction"])
    core, test, end = sp["core"], sp["test"], sp["end"]
    X, brk = tl["X"][:end], tl["breaks"][:end]
    fwd = {h: v[:end] for h, v in tl["fwd"].items()}
    wdir = base_dir / f"w{w:02d}"
    res = json.loads((wdir / "result.json").read_text(encoding="utf-8"))["metodos"]
    seed, bcfg = exp["seed"] + w, exp["baselines"]
    costs = exp["costs"]
    cost_taker = 2 * (costs["fee_per_side"] + costs["slippage_per_side"])

    alpha = Alphabet.load(wdir / "alphabet.pt")
    letters = alpha.encode(X, "cpu")
    vocab = Vocabulary.load(wdir / "vocab.json")
    keys = phrase_keys(vocab.phrases(letters, exp["dictionary"]["max_phrase_words"], brk), len(vocab))
    z = np.load(wdir / "lm_out.npz")
    extra = [_roll_z(z["surprise"])[:, None], _roll_z(z["entropy"])[:, None]]
    rest = np.arange(core[-1] + 1, end)
    for hh in H:
        te = np.zeros((end, len(keys)), dtype=np.float32)
        for f in np.array_split(core, 5):
            te[f] = target_encode(keys, fwd[hh], np.setdiff1d(core, f, assume_unique=True), f, hh)
        te[rest] = target_encode(keys, fwd[hh], core, rest, hh)
        extra.append(te)
    mats = {"gbm_cru": X, "gbm_dialeto": np.hstack([X] + extra).astype(np.float32)}

    out = {}
    for m in METHODS:
        ch = res[m]
        h = ch["horizonte"]
        sig = next(s for n, _, s in B.gbm_candidates(mats[m], fwd, core, [h], bcfg, seed, m, lambda *a: None)
                   if n == ch["escolha"])
        rep = metrics(simulate(sig, fwd[h], test, h, cost_taker))
        orig = ch["teste_taker"]
        ok = rep["trades"] == orig["trades"] and abs(rep["mean_net_bps"] - orig["mean_net_bps"]) < 0.5
        log(f"  janela {w} {m}: {ch['escolha']} h={h} | reprodução {'OK' if ok else 'DIFERENTE'} "
            f"({rep['trades']} op / {rep['mean_net_bps']:.2f} x {orig['trades']} op / {orig['mean_net_bps']:.2f})")
        out[m] = {"h": h, "sig": sig, "test": test, "ok": ok, "escolha": ch["escolha"],
                  "semaforo": ch["validacao"]["mean_net_bps"] > 0,
                  "taker": orig, "maker_simples": ch["teste_maker"]}
    return out


def second_arrays(cfg: dict, sym: str, t0: pd.Timestamp, t1: pd.Timestamp, log):
    """Mínimo, máximo, primeiro e último preço de cada segundo em [t0, t1) a partir dos ticks."""
    start_ms = t0.value // 1_000_000
    n = int((t1 - t0).total_seconds())
    smin, smax = np.full(n, np.inf), np.full(n, -np.inf)
    sfirst, slast = np.full(n, np.nan), np.full(n, np.nan)
    months = pd.period_range(t0.tz_convert(None).to_period("M"),
                             (t1 - pd.Timedelta(milliseconds=1)).tz_convert(None).to_period("M"), freq="M")
    for p in months:
        f = cfg["data_dir"] / "trades" / f"symbol={sym}" / f"year={p.year:04d}" / f"{p.year:04d}-{p.month:02d}.parquet"
        if not f.exists():
            log(f"    (sem ticks de {p}; operações que precisem dele são descartadas)")
            continue
        for b in pq.ParquetFile(f).iter_batches(batch_size=4_000_000, columns=["time", "price"]):
            tcol = b.column("time")
            ms = tcol.cast(pa.timestamp("ms", tz=tcol.type.tz), safe=False).cast(pa.int64()).to_numpy()
            pr = b.column("price").to_numpy()
            sec = (ms - start_ms) // 1000
            keep = (sec >= 0) & (sec < n)
            if not keep.any():
                continue
            sec, pr = sec[keep], pr[keep]
            if not (sec[1:] >= sec[:-1]).all():
                o = np.argsort(sec, kind="stable")
                sec, pr = sec[o], pr[o]
            st = np.flatnonzero(np.r_[True, sec[1:] != sec[:-1]])
            u = sec[st]
            smin[u] = np.minimum(smin[u], np.minimum.reduceat(pr, st))
            smax[u] = np.maximum(smax[u], np.maximum.reduceat(pr, st))
            new = np.isnan(sfirst[u])
            sfirst[u[new]] = pr[st][new]
            slast[u] = pr[np.r_[st[1:] - 1, len(pr) - 1]]
    have = ~np.isnan(slast)
    idx = np.where(have, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    last_ff = slast[idx]
    last_ff[: np.argmax(have)] = np.nan
    return start_ms, n, smin, smax, sfirst, last_ff


def load_funding(cfg: dict, symbol: str, t0: pd.Timestamp, t1: pd.Timestamp, log) -> pd.DataFrame:
    """Histórico de funding (a cada 8 h) do perpétuo, de data.binance.vision, com cache em data/funding."""
    import tempfile
    from .data import binance
    cache = cfg["data_dir"] / "funding" / f"{symbol}_UM.parquet"
    if cache.exists():
        fr = pd.read_parquet(cache)
    else:
        parts = []
        with tempfile.TemporaryDirectory() as tmp:
            for p in pd.period_range("2020-01", pd.Timestamp.now(tz="UTC").tz_convert(None).to_period("M") - 1, freq="M"):
                url = f"{binance.BASES['um']}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{p.year:04d}-{p.month:02d}.zip"
                path = binance._download_verified(url, tmp)
                if path is None:
                    continue
                t = binance._read_csv_from_zip(path, ["calc_time", "funding_interval_hours", "last_funding_rate"]).to_pandas()
                parts.append(t[["calc_time", "last_funding_rate"]])
        fr = pd.concat(parts, ignore_index=True).rename(columns={"calc_time": "ms", "last_funding_rate": "rate"})
        fr = fr.astype({"ms": "int64", "rate": "float64"}).sort_values("ms").reset_index(drop=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        fr.to_parquet(cache)
        log(f"  funding: {len(fr):,} cobranças baixadas ({pd.to_datetime(fr.ms.iloc[0], unit='ms'):%Y-%m} a "
            f"{pd.to_datetime(fr.ms.iloc[-1], unit='ms'):%Y-%m})")
    return fr[(fr.ms >= t0.value // 1_000_000) & (fr.ms < t1.value // 1_000_000)]


def sim_limit(sig, test, times_ms, close, h, S, w_in, w_out, fund_sec=None, fund_rate=None):
    """Retorna as operações executadas. Ordem que executa já no 1º segundo conta como TAKER (ela cruzou o
    livro ao ser enviada); funding é cobrado/recebido em cada horário de funding que a posição atravessa."""
    start_ms, n, smin, smax, sfirst, slast = S
    cand = test[sig[test] != 0]
    busy = -1
    rows, tried = [], 0
    for t in cand:
        T = int((times_ms[t] + 60_000 - start_ms) // 1000)   # fim do minuto t, em segundos
        if T < busy:
            continue
        if T + w_in + h * 60 + w_out + 2 >= n:
            continue
        d = int(sig[t])
        P = close[t] - TICK if d > 0 else close[t] + TICK
        tried += 1
        hit = np.flatnonzero(smin[T:T + w_in] < P) if d > 0 else np.flatnonzero(smax[T:T + w_in] > P)
        if len(hit) == 0:
            busy = T + w_in                                # ordem cancelada
            continue
        tf = T + int(hit[0])
        fee_in = FEES["taker"] if hit[0] == 0 else FEES["maker"]
        te = tf + h * 60
        last = slast[te - 1]
        Pe = last + TICK if d > 0 else last - TICK
        hit2 = np.flatnonzero(smax[te:te + w_out] > Pe) if d > 0 else np.flatnonzero(smin[te:te + w_out] < Pe)
        if len(hit2):
            px, tx = Pe, te + int(hit2[0])
            fee_out = FEES["taker"] if hit2[0] == 0 else FEES["maker"]
            maker_exit = hit2[0] != 0
        else:
            tx = te + w_out
            px = sfirst[tx] if np.isfinite(sfirst[tx]) else slast[tx]
            px = px * (1 - FEES["slippage"]) if d > 0 else px * (1 + FEES["slippage"])
            fee_out, maker_exit = FEES["taker"], False
        funding = 0.0
        if fund_sec is not None:
            k = (fund_sec > tf) & (fund_sec <= tx)
            funding = d * float(fund_rate[k].sum())       # comprado paga taxa positiva; vendido recebe
        gross = d * np.log(px / P)
        rows.append((t, d, gross, gross - fee_in - fee_out - funding, maker_exit, hit[0] != 0, funding))
        busy = tx + 1
    return pd.DataFrame(rows, columns=["t", "dir", "gross", "net", "saida_maker", "entrada_maker", "funding"]), tried


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", nargs="+", type=int, default=list(range(8)))
    args = ap.parse_args(argv)

    cfg = load_config()
    exp = yaml.safe_load((CONFIG_DIR / "experiment.yaml").read_text(encoding="utf-8"))
    base_dir = PROJECT_ROOT / "runs" / exp["run_name"]
    out_dir = PROJECT_ROOT / "runs" / f"{exp['run_name']}_execucao"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "log.txt")
    log(f"execução realista com ordens limitadas | janelas {args.windows} | variantes {VARIANTS}")
    tl = load_timeline(cfg, exp, base_dir, log)
    sym = storage_symbol(exp["data"]["source"], exp["data"]["symbol"])
    bars = read_symbol(cfg["data_dir"], "m1", sym)
    close = pd.Series(bars["close"].to_numpy(), index=pd.DatetimeIndex(bars["time"]).as_unit("ns"))
    times_ns = pd.DatetimeIndex(tl["times"]).as_unit("ns")
    close_arr = close.reindex(times_ns).to_numpy()
    times_ms = times_ns.asi8 // 1_000_000

    results = []
    for w in args.windows:
        t0 = time.time()
        blk = exp["walkforward"]["test_blocks"][w]
        sigs = regen_signals(w, tl, exp, base_dir, log)
        t_a, t_b = pd.Timestamp(blk[0], tz="UTC"), pd.Timestamp(blk[1], tz="UTC") + pd.Timedelta(hours=3)
        S = second_arrays(cfg, sym, t_a, t_b, log)
        fr = load_funding(cfg, exp["data"]["symbol"], t_a, t_b, log)
        fund_sec = ((fr.ms.to_numpy() - S[0]) // 1000).astype(np.int64)
        fund_rate = fr.rate.to_numpy()
        for m, s in sigs.items():
            for vname, (w_in, w_out) in VARIANTS.items():
                tr, tried = sim_limit(s["sig"], s["test"], times_ms, close_arr, s["h"], S, w_in, w_out, fund_sec, fund_rate)
                mt = metrics(tr.rename(columns={"net": "net"}))
                r = {"janela": w, "metodo": m, "variante": vname, "escolha": s["escolha"], "h": s["h"],
                     "reproducao_ok": s["ok"], "semaforo": bool(s["semaforo"]),
                     "sinais_tentados": tried, "executadas": int(len(tr)),
                     "taxa_execucao": len(tr) / tried if tried else None,
                     "saida_maker": float(tr["saida_maker"].mean()) if len(tr) else None,
                     "entrada_maker": float(tr["entrada_maker"].mean()) if len(tr) else None,
                     "funding_bps": float(tr["funding"].mean() * 1e4) if len(tr) else None,
                     "bruto_bps": float(tr["gross"].mean() * 1e4) if len(tr) else None,
                     "liquido_bps": mt["mean_net_bps"], "total_pct": mt["total_net_pct"], "t": mt["t_stat"],
                     "taker_bps": s["taker"]["mean_net_bps"], "taker_op": s["taker"]["trades"],
                     "maker_simples_bps": s["maker_simples"]["mean_net_bps"],
                     "taker_total_pct": s["taker"]["total_net_pct"], "maker_simples_total_pct": s["maker_simples"]["total_net_pct"]}
                results.append(r)
                log(f"    {m:12s} {vname:14s}: entrada maker {r['entrada_maker'] or 0:.0%}, saída maker {r['saida_maker'] or 0:.0%}, "
                    f"funding {r['funding_bps'] if r['funding_bps'] is None else round(r['funding_bps'], 2)} bps/op")
                log(f"    {m:12s} {vname:14s}: {tried} sinais, {len(tr)} executadas ({r['taxa_execucao'] or 0:.0%}), "
                    f"bruto {r['bruto_bps'] if r['bruto_bps'] is None else round(r['bruto_bps'], 1)} bps, "
                    f"líquido {mt['mean_net_bps'] if mt['mean_net_bps'] is None else round(mt['mean_net_bps'], 1)} bps "
                    f"| taker {s['taker']['mean_net_bps']:.1f} | maker simples {s['maker_simples']['mean_net_bps']:.1f}")
        del S
        log(f"janela {w} concluída em {(time.time() - t0) / 60:.1f} min")
        (out_dir / "result.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    df = pd.DataFrame(results)
    L = ["# Execução realista com ordens limitadas (ticks, segundo a segundo)", "",
         "Mesmos sinais da rodada só com velas. Entrada limitada 1 tick melhor que o último preço; executa só se um "
         "negócio real passar do preço. Saída limitada por 1 min, senão a mercado. Maker 0,02%/lado, taker 0,05% + 0,01%. "
         "Ordem executada já no 1º segundo paga taxa taker (cruzou o livro); funding real da Binance descontado.", "",
         "## Soma das janelas", "",
         "| Método | Variante | Regra | Janelas | Sinais | Executadas | Líquido (bps/op) | Soma (%) | Taker (soma %) | Maker simples (soma %) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        for vname in VARIANTS:
            for regra, sel in (("sempre", slice(None)), ("semáforo", True)):
                d = df[(df.metodo == m) & (df.variante == vname)]
                if regra == "semáforo":
                    d = d[d.semaforo]
                ex = d.executadas.sum()
                tot = d.total_pct.sum()
                L.append(f"| {m} | {vname} | {regra} | {len(d)} | {d.sinais_tentados.sum()} | {ex} | "
                         f"{(tot * 100 / ex) if ex else float('nan'):.1f} | {tot:+.1f} | {d.taker_total_pct.sum():+.1f} | "
                         f"{d.maker_simples_total_pct.sum():+.1f} |")
    L += ["", "## Por janela", "", "| Janela | Método | Variante | Semáforo | Execução | Entrada maker | Saída maker | Funding (bps) | Bruto (bps) | Líquido (bps) | Taker (bps) | Maker simples (bps) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, nd=1: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"
    for r in results:
        L.append(f"| {r['janela']} | {r['metodo']} | {r['variante']} | {'🟢' if r['semaforo'] else '🔴'} | "
                 f"{r['executadas']}/{r['sinais_tentados']} ({f((r['taxa_execucao'] or 0) * 100, 0)}%) | "
                 f"{f((r['entrada_maker'] or 0) * 100, 0)}% | {f((r['saida_maker'] or 0) * 100, 0)}% | {f(r['funding_bps'], 2)} | "
                 f"{f(r['bruto_bps'])} | {f(r['liquido_bps'])} | {f(r['taker_bps'])} | {f(r['maker_simples_bps'])} |")
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    log(f"relatório: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

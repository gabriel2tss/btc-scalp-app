"""Execução realista (ticks, segundo a segundo) da estratégia de barreiras + veredito contra os critérios travados.

Para cada sinal de teste salvo por barrier_eval (uma posição por vez):
- entrada: ordem limitada 1 tick melhor que o último preço, por até 60 s; executa só se um negócio real passar
  do preço; se executar no 1º segundo, cruzou o livro e paga taxa taker; se não executar, cancela;
- take: ordem limitada no alvo (executa quando um negócio passa do alvo) -> taxa maker;
- stop: a mercado quando algum negócio toca o stop; preço de saída = o pior entre o stop e o último negócio
  daquele segundo, menos derrapagem -> taxa taker; take e stop no mesmo segundo -> stop;
- tempo: a mercado ao fim do horizonte -> taxa taker + derrapagem;
- funding real em cada cobrança atravessada. Estresse: taxas, derrapagem e funding x1,5.

    python -m vag.barrier_exec
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

from .barriers import CONFIGS
from .config import CONFIG_DIR, PROJECT_ROOT, load_config
from .criteria import verify as criteria_ok
from .data.download import storage_symbol
from .data.storage import read_symbol
from .evaluate import metrics
from .execution_eval import FEES, TICK, load_funding, second_arrays
from .walkforward import Logger, load_timeline

W_IN = 60


def sim(idx, side, Xtake, r, H, times_ms, close, S, fund_sec, fund_rate):
    start_ms, n, smin, smax, sfirst, slast = S
    rows, busy, tried = [], -1, 0
    for t, d, X in zip(idx, side, Xtake):
        T = int((times_ms[t] + 60_000 - start_ms) // 1000)
        if T < busy or T + W_IN + H * 60 + 5 >= n:
            continue
        d = int(d)
        P = close[t] - TICK if d > 0 else close[t] + TICK
        tried += 1
        hit = np.flatnonzero(smin[T:T + W_IN] < P) if d > 0 else np.flatnonzero(smax[T:T + W_IN] > P)
        if len(hit) == 0:
            busy = T + W_IN
            continue
        tf = T + int(hit[0])
        taker_in = hit[0] == 0
        Y = X / r
        tp, sl = (P * np.exp(X), P * np.exp(-Y)) if d > 0 else (P * np.exp(-X), P * np.exp(Y))
        a, b = tf + 1, tf + H * 60
        if d > 0:
            it, is_ = np.flatnonzero(smax[a:b] > tp), np.flatnonzero(smin[a:b] <= sl)
        else:
            it, is_ = np.flatnonzero(smin[a:b] < tp), np.flatnonzero(smax[a:b] >= sl)
        i_t = int(it[0]) if len(it) else 10**9
        i_s = int(is_[0]) if len(is_) else 10**9
        if i_s <= i_t and i_s < 10**9:                      # stop (inclusive empate no mesmo segundo)
            tx = a + i_s
            px = min(sl, slast[tx]) if d > 0 else max(sl, slast[tx])
            kind, market = "stop", True
        elif i_t < 10**9:
            tx, px, kind, market = a + i_t, tp, "take", False
        else:
            tx = b
            px = sfirst[tx] if np.isfinite(sfirst[tx]) else slast[tx]
            kind, market = "tempo", True
        k = (fund_sec > tf) & (fund_sec <= tx)
        rows.append({"t": int(t), "dir": d, "gross": d * np.log(px / P), "tipo": kind,
                     "fee_in": FEES["taker"] if taker_in else FEES["maker"],
                     "fee_out": FEES["taker"] if market else FEES["maker"],
                     "slip": FEES["slippage"] if market else 0.0, "funding": d * float(fund_rate[k].sum())})
        busy = tx + 1
    df = pd.DataFrame(rows, columns=["t", "dir", "gross", "tipo", "fee_in", "fee_out", "slip", "funding"])
    return df, tried


def with_costs(df, mult=1.0):
    d = df.copy()
    d["net"] = d["gross"] - mult * (d["fee_in"] + d["fee_out"] + d["slip"] + d["funding"])
    return d


def pooled(dfs):
    d = pd.concat(dfs) if dfs else pd.DataFrame(columns=["net"])
    return metrics(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", nargs="+", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_config()
    exp = yaml.safe_load((CONFIG_DIR / "experiment.yaml").read_text(encoding="utf-8"))
    crit = yaml.safe_load((CONFIG_DIR / "success_criteria.yaml").read_text(encoding="utf-8"))
    out_dir = PROJECT_ROOT / "runs" / f"{exp['run_name']}_barreiras"
    log = Logger(out_dir / "log_execucao.txt")
    log(f"execução realista das barreiras | critérios travados e íntegros: {criteria_ok()}")
    res = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    if args.windows is not None:
        res = [r for r in res if r["janela"] in args.windows]

    tl = load_timeline(cfg, exp, PROJECT_ROOT / "runs" / exp["run_name"], log)
    times = pd.DatetimeIndex(tl["times"]).as_unit("ns")
    times_ms = times.asi8 // 1_000_000
    sym = storage_symbol(exp["data"]["source"], exp["data"]["symbol"])
    bars = read_symbol(cfg["data_dir"], "m1", sym)
    close = bars.set_index(pd.DatetimeIndex(bars["time"]).as_unit("ns"))["close"].reindex(times).to_numpy()

    trades = {}      # (variante, janela) -> DataFrame
    rows = []
    for w in sorted({r["janela"] for r in res}):
        blk = exp["walkforward"]["test_blocks"][w]
        t_a, t_b = pd.Timestamp(blk[0], tz="UTC"), pd.Timestamp(blk[1], tz="UTC") + pd.Timedelta(hours=6)
        S = second_arrays(cfg, sym, t_a, t_b, log)
        fr = load_funding(cfg, exp["data"]["symbol"], t_a, t_b, log)
        fund_sec = ((fr.ms.to_numpy() - S[0]) // 1000).astype(np.int64)
        for r in [x for x in res if x["janela"] == w and x.get("escolha", True) is not None and x.get("config")]:
            v = r["variante"]
            z = np.load(out_dir / f"sinais_w{w:02d}_{v}.npz")
            a, rr, H = CONFIGS[int(z["ci"])]
            df, tried = sim(z["idx"], z["side"], z["X"], rr, H, times_ms, close, S, fund_sec, fr.rate.to_numpy())
            trades[(v, w)] = df
            m1, m15 = metrics(with_costs(df)), metrics(with_costs(df, 1.5))
            row = {"janela": w, "variante": v, "semaforo": r["semaforo"], "config": r["config"], "sinais": tried,
                   "executadas": int(len(df)), "tipos": df["tipo"].value_counts().to_dict() if len(df) else {},
                   "entrada_maker": float((df.fee_in == FEES["maker"]).mean()) if len(df) else None,
                   "real": m1, "estresse": m15, "velas": r["teste_velas"]}
            rows.append(row)
            g = lambda m: "—" if m["mean_net_bps"] is None else f"{m['mean_net_bps']:+.1f} bps (PF {m['pf'] and round(m['pf'], 2)})"
            log(f"  janela {w} {v:15s} {'🟢' if r['semaforo'] else '🔴'}: {len(df)}/{tried} executadas | real {g(m1)} | "
                f"estresse {g(m15)} | velas {g(r['teste_velas'])} | {row['tipos']}")
        del S

    # veredito: só janelas com semáforo verde operam (estratégia pré-registrada)
    L = ["# Barreiras — execução realista e veredito contra os critérios travados", "",
         f"Critérios íntegros (hash da trava confere): **{criteria_ok()}**. Janelas avaliadas: "
         f"{sorted({r['janela'] for r in rows})}.", ""]
    verdict = {}
    for v in sorted({r["variante"] for r in rows}):
        rs = [r for r in rows if r["variante"] == v]
        green = [r for r in rs if r["semaforo"]]
        real = pooled([with_costs(trades[(v, r["janela"])]) for r in green])
        stress = pooled([with_costs(trades[(v, r["janela"])], 1.5) for r in green])
        passing = [r["janela"] for r in green if (r["real"]["mean_net_bps"] or -1) > 0 and (r["real"]["pf"] or 0) >= 1.0]
        edge_bps = crit["min_net_edge_per_trade"] * 100
        checks = {
            f"média líquida >= {edge_bps:.0f} bps": (real["mean_net_bps"] or -1e9) >= edge_bps,
            f"fator de lucro >= {crit['min_profit_factor_oos']}": (real["pf"] or 0) >= crit["min_profit_factor_oos"],
            f"t >= {crit['min_pooled_t_stat']}": (real["t_stat"] or -1e9) >= crit["min_pooled_t_stat"],
            f"janelas positivas >= {crit['min_walkforward_windows_passing']} de {crit['total_walkforward_windows']}":
                len(passing) >= crit["min_walkforward_windows_passing"],
            f"estresse x{crit['cost_stress']['multiplier']}: média >= {crit['cost_stress']['min_net_edge_per_trade'] * 100:.0f} bps":
                (stress["mean_net_bps"] or -1e9) >= crit["cost_stress"]["min_net_edge_per_trade"] * 100,
        }
        verdict[v] = {"passou": all(checks.values()), "checks": checks, "real": real, "estresse": stress,
                      "janelas_positivas": passing, "janelas_operadas": [r["janela"] for r in green]}
        L += [f"## {v}: {'✅ PASSOU' if all(checks.values()) else '❌ NÃO PASSOU'}", "",
              f"Operou {len(green)} de {len(rs)} janelas (semáforo verde: {[r['janela'] for r in green]}). "
              f"{real['trades']} operações; média {real['mean_net_bps'] and round(real['mean_net_bps'], 1)} bps, "
              f"PF {real['pf'] and round(real['pf'], 2)}, t {real['t_stat'] and round(real['t_stat'], 2)}; "
              f"estresse {stress['mean_net_bps'] and round(stress['mean_net_bps'], 1)} bps.", ""]
        L += [f"- {'✅' if ok else '❌'} {k}" for k, ok in checks.items()]
        L += ["", "| Janela | Semáforo | Barreiras | Executadas | Real (bps) | PF | Estresse (bps) | Velas (bps) |", "|---|---|---|---|---|---|---|---|"]
        for r in rs:
            f = lambda x, nd=1: "—" if x is None else f"{x:.{nd}f}"
            L.append(f"| {r['janela']} | {'🟢' if r['semaforo'] else '🔴'} | a={r['config']['a']} {r['config']['take_stop']}:1 "
                     f"H={r['config']['H']} | {r['executadas']}/{r['sinais']} | {f(r['real']['mean_net_bps'])} | "
                     f"{f(r['real']['pf'], 2)} | {f(r['estresse']['mean_net_bps'])} | {f(r['velas']['mean_net_bps'])} |")
        L.append("")
    (out_dir / "veredito.md").write_text("\n".join(L), encoding="utf-8")
    (out_dir / "veredito.json").write_text(json.dumps({"veredito": verdict, "janelas": rows}, indent=2, ensure_ascii=False,
                                                     default=float), encoding="utf-8")
    log(f"veredito: {out_dir / 'veredito.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Walk-forward do modelo de barreiras (triplo limite): só números x números + dialeto.

Por janela (mesmos cortes treino | embargo | validação | embargo | teste da rodada btc_m1_v1):
  1. para cada configuração de barreiras (12) e lado (compra/venda), um GBM aprende, no treino, o
     resultado da operação em unidades de risco (resultado / take);
  2. valor esperado de cada lado = previsão x take; opera o lado de maior valor esperado quando ele,
     menos o custo, passa do limiar (quantis 99,5 / 99 / 98 / 95% do treino, e > 0);
  3. configuração e limiar escolhidos só na validação (>= 30 operações); semáforo: opera o teste só se,
     na validação, média líquida >= +5 bps e PF >= 1,15;
  4. teste avaliado uma vez (simulação com velas, custos de referência). Os sinais do teste ficam
     salvos para a execução realista com ticks (python -m vag.barrier_exec).

Alfabeto, palavras e LM (dialeto) são os já treinados em runs/btc_m1_v1/wNN.

    python -m vag.barrier_eval --windows 0 1 2 3 4 5 6 7 8 9 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from .alphabet import Alphabet
from .barriers import (CONFIGS, COST_ENTRY, barrier_outcomes, candle_sim, contiguous_until, derivatives_features,
                       funding_cum, vol_1m)
from .config import CONFIG_DIR, PROJECT_ROOT, load_config
from .data.download import storage_symbol
from .data.metrics import path_for as metrics_path
from .data.storage import read_symbol
from .evaluate import metrics, phrase_keys
from .execution_eval import load_funding
from .walkforward import Logger, _roll_z, load_timeline, split_window, target_encode
from .words import Vocabulary

QUANTS = (0.995, 0.99, 0.98, 0.95)
STRIDE = 10                      # rótulos de 1-4 h se sobrepõem muito; 1 a cada 10 minutos basta para treinar
SEMAFORO = {"min_bps": 5.0, "min_pf": 1.15}
EXIT_COST_EST = 0.0004           # custo médio de saída esperado, para o limiar de decisão


def labels(tl, bars_o, bars_h, bars_l, bars_c, tns, out_dir, log, min_take=None):
    cache = out_dir / "rotulos.npz"
    if cache.exists():
        z = np.load(cache)
        return {ci: (z[f"X{ci}"], {1: (z[f"o{ci}_1"], z[f"k{ci}_1"], z[f"t{ci}_1"]),
                                  -1: (z[f"o{ci}_-1"], z[f"k{ci}_-1"], z[f"t{ci}_-1"])}) for ci in range(len(CONFIGS))}
    t0 = time.time()
    sig1 = vol_1m(bars_c)
    cu = contiguous_until(tns)
    lab, save = {}, {}
    for ci, (a, r, H) in enumerate(CONFIGS):
        X, Y, res = barrier_outcomes(bars_o, bars_h, bars_l, bars_c, sig1, a, r, H, cu,
                                     **({} if min_take is None else {'min_take': min_take}))
        lab[ci] = (X.astype(np.float32), {s: (res[s][0].astype(np.float32), res[s][1], res[s][2]) for s in (1, -1)})
        save[f"X{ci}"] = lab[ci][0]
        for s in (1, -1):
            save[f"o{ci}_{s}"], save[f"k{ci}_{s}"], save[f"t{ci}_{s}"] = lab[ci][1][s]
        take = np.mean(res[1][2][np.isfinite(res[1][0])] == 1)
        log(f"  rótulos a={a} take:stop={r}:1 H={H}: take mediano {np.nanmedian(X) * 1e4:.0f} bps | "
            f"compra bate o take em {take:.1%} dos minutos")
    np.savez(cache, **save)
    log(f"  rótulos calculados em {(time.time() - t0) / 60:.1f} min")
    return lab


def dialect_cols(w, tl, end, core, keys_src_dir, exp):
    wdir = keys_src_dir / f"w{w:02d}"
    X, brk = tl["X"][:end], tl["breaks"][:end]
    alpha = Alphabet.load(wdir / "alphabet.pt")
    letters = alpha.encode(X, "cpu")
    vocab = Vocabulary.load(wdir / "vocab.json")
    keys = phrase_keys(vocab.phrases(letters, exp["dictionary"]["max_phrase_words"], brk), len(vocab))
    z = np.load(wdir / "lm_out.npz")
    assert len(z["surprise"]) == end
    cols = [_roll_z(z["surprise"])[:, None], _roll_z(z["entropy"])[:, None]]
    rest = np.arange(core[-1] + 1, end)
    targets = [(h, tl["fwd"][h][:end]) for h in exp["horizons_minutes"]]            # significado: direção
    targets += [(h, np.abs(tl["fwd"][h][:end])) for h in (15, 60)]                  # significado: tamanho do movimento
    for h, y in targets:
        te = np.zeros((end, len(keys)), dtype=np.float32)
        for f in np.array_split(core, 5):
            te[f] = target_encode(keys, y, np.setdiff1d(core, f, assume_unique=True), f, h)
        te[rest] = target_encode(keys, y, core, rest, h)
        cols.append(te)
    return np.hstack(cols).astype(np.float32)


def fit_ev(Xv, y_R, Xc, core, pidx, end, bcfg, seed):
    """Treina no treino (1 a cada STRIDE min) e prevê só onde é preciso (amostra do treino, validação, teste)."""
    sub = core[::STRIDE]
    ok = np.isfinite(y_R[sub])
    m = HistGradientBoostingRegressor(max_iter=bcfg["gbm_max_iter"], learning_rate=bcfg["gbm_learning_rate"],
                                      max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
    m.fit(Xv[sub][ok], y_R[sub][ok])
    ev = np.full(end, np.nan)
    ev[pidx] = m.predict(Xv[pidx]) * Xc[pidx]
    return ev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", nargs="+", type=int, default=list(range(11)))
    ap.add_argument("--variants", nargs="+", default=["so_numeros", "numeros_dialeto"])
    ap.add_argument("--config", default=str(CONFIG_DIR / "experiment.yaml"))
    args = ap.parse_args(argv)

    cfg = load_config()
    exp = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_dir = PROJECT_ROOT / "runs" / exp["run_name"]
    out_dir = PROJECT_ROOT / "runs" / f"{exp['run_name']}_barreiras"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "log.txt")
    log(f"modelo de barreiras | janelas {args.windows} | variantes {args.variants} | {len(CONFIGS)} configurações")

    tl = load_timeline(cfg, exp, base_dir, log)
    times = pd.DatetimeIndex(tl["times"]).as_unit("ns")
    tns = times.asi8
    sym = storage_symbol(exp["data"]["source"], exp["data"]["symbol"])
    bars = read_symbol(cfg["data_dir"], "m1", sym)
    bars = bars.set_index(pd.DatetimeIndex(bars["time"]).as_unit("ns")).reindex(times)
    o, h, l, c = (bars[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    bc = exp.get("barreiras", {})
    min_take = bc["min_take_bps"] / 1e4 if "min_take_bps" in bc else None
    lab = labels(tl, o, h, l, c, tns, out_dir, log, min_take)
    semaforo = bc.get("semaforo", SEMAFORO)
    stress = float(bc.get("estresse_custo", 1.5))
    cost_rt = None
    if bc.get("custo") == "spread":           # ouro: spread real do minuto de entrada + derrapagem
        sp = bars["spread_bps"].to_numpy(np.float64)
        cost_rt = (np.r_[sp[1:], np.nan] + bc.get("derrapagem_bps", 0.0)) / 1e4
        cost_rt = np.where(np.isfinite(cost_rt), cost_rt, np.nanmedian(cost_rt))
        log(f"  custo pelo spread: mediana {np.median(cost_rt) * 1e4:.2f} bps por operação")
    cost_est = float(np.median(cost_rt)) if cost_rt is not None else COST_ENTRY + EXIT_COST_EST

    if bc.get("funding", True):
        funding = load_funding(cfg, exp["data"]["symbol"], times[0], times[-1] + pd.Timedelta(days=1), log)
        fcum = funding_cum(tns, funding)
    else:
        funding, fcum = None, np.zeros(len(tns))
    if bc.get("derivativos", True):
        D = derivatives_features(times, pd.read_parquet(metrics_path(cfg, exp["data"]["symbol"])), funding).to_numpy()
    else:
        D = np.zeros((len(tns), 0), dtype=np.float32)
    log(f"  derivativos: {D.shape[1]} colunas")
    T = None
    if any(v.endswith("ticks") for v in args.variants):
        from .tick_features import TICK_FEATURES, feature_table
        tf = feature_table(cfg, sym, log)
        ti = pd.DatetimeIndex(tf["time"])
        ti = (ti.tz_localize("UTC") if ti.tz is None else ti.tz_convert("UTC")).as_unit("ns")
        T = tf.set_index(ti)[TICK_FEATURES].reindex(times).to_numpy(np.float32)
        if np.isfinite(T).all(1).mean() < 0.9:
            raise SystemExit("features de ticks não alinharam com a timeline (fuso/horário)")
        log(f"  ticks: {np.isfinite(T).all(1).mean():.0%} dos minutos com as 10 features de ticks")

    wf, bcfg = exp["walkforward"], exp["baselines"]
    results = json.loads((out_dir / "result.json").read_text(encoding="utf-8")) if (out_dir / "result.json").exists() else []
    done = {(r["janela"], r["variante"]) for r in results}
    for w in args.windows:
        t0 = time.time()
        sp = split_window(tl["times"], wf["test_blocks"][w], wf["embargo_minutes"], wf["val_fraction"])
        core, val, test, end = sp["core"], sp["val"], sp["test"], sp["end"]
        X_num = np.hstack([tl["X"][:end], D[:end]]).astype(np.float32)
        todo = [v for v in args.variants if (w, v) not in done]
        dia = dialect_cols(w, tl, end, core, base_dir, exp) if any("dialeto" in v for v in todo) else None
        mats = {"so_numeros": lambda: X_num,
                "numeros_dialeto": lambda: np.hstack([X_num, dia]),
                "numeros_ticks": lambda: np.hstack([X_num, T[:end]]),
                "numeros_dialeto_ticks": lambda: np.hstack([X_num, T[:end], dia])}
        seed = exp["seed"] + w
        core_s = core[::STRIDE]
        pidx = np.r_[core_s, val, test]
        for vname in todo:
            Xv = mats[vname]().astype(np.float32)
            best = None
            for ci in range(len(CONFIGS)):
                Xc, outc = lab[ci]
                ev = {s: fit_ev(Xv, outc[s][0][:end] / Xc[:end], Xc, core, pidx, end, bcfg, seed) for s in (1, -1)}
                side = np.where(ev[1] >= ev[-1], 1, -1).astype(np.int8)
                net_ev = np.maximum(ev[1], ev[-1]) - cost_est
                for q in QUANTS:
                    thr = max(float(np.nanquantile(net_ev[core_s], q)), 0.0)
                    sig = np.where(net_ev > thr, side, 0).astype(np.int8)
                    mv = metrics(candle_sim(sig, val, outc, fcum, cost_rt=cost_rt))
                    if mv["trades"] >= 30 and (best is None or mv["total_net_pct"] > best[2]["total_net_pct"]):
                        best = (ci, q, mv, sig, thr)
            if best is None:
                r = {"janela": w, "variante": vname, "escolha": None, "semaforo": False}
                results.append(r)
                log(f"  janela {w} {vname}: nenhuma configuração com >= 30 operações na validação")
                continue
            ci, q, mv, sig, thr = best
            a, rr, H = CONFIGS[ci]
            green = (mv["mean_net_bps"] or -1e9) >= semaforo["min_bps"] and (mv["pf"] or 0) >= semaforo["min_pf"]
            tr = candle_sim(sig, test, lab[ci][1], fcum, cost_rt=cost_rt)
            mt_stress = metrics(candle_sim(sig, test, lab[ci][1], fcum, cost_mult=stress, cost_rt=cost_rt))
            mt = metrics(tr)
            np.savez_compressed(out_dir / f"sinais_w{w:02d}_{vname}.npz", idx=test[sig[test] != 0],
                                side=sig[test][sig[test] != 0], X=lab[ci][0][test][sig[test] != 0], ci=ci)
            r = {"janela": w, "variante": vname, "periodo": wf["test_blocks"][w], "config": {"a": a, "take_stop": rr, "H": H},
                 "limiar_q": q, "semaforo": bool(green), "validacao": mv, "teste_velas": mt, "teste_estresse": mt_stress,
                 "teste_velas_tipos": tr["tipo"].value_counts().to_dict() if len(tr) else {},
                 "teste_velas_compra": metrics(tr[tr.dir > 0]) if len(tr) else {},
                 "teste_velas_venda": metrics(tr[tr.dir < 0]) if len(tr) else {}}
            results.append(r)
            f = lambda x: "—" if x is None else f"{x:+.1f}"
            log(f"  janela {w} {vname:15s}: a={a} {rr}:1 H={H} q{q} | validação {f(mv['mean_net_bps'])} bps "
                f"(PF {mv['pf'] and round(mv['pf'], 2)}, {mv['trades']} op) {'🟢' if green else '🔴'} | "
                f"teste (velas) {f(mt['mean_net_bps'])} bps, PF {mt['pf'] and round(mt['pf'], 2)}, {mt['trades']} op")
            (out_dir / "result.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        log(f"janela {w} concluída em {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())

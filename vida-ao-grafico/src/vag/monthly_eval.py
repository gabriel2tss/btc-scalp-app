"""Treino MENSAL dentro do bloco de teste de uma janela (velas + ticks).

Na receita original, o modelo aprende até ~3 meses antes do teste e opera 6 meses sem atualização;
os meses mais recentes (a validação) nunca entram no treino. Aqui, para cada mês M do bloco de teste:
  1. validação = os 3 meses antes de M (até M - embargo); treino = tudo antes disso (menos o embargo);
  2. candidatos (horizonte x limiar) treinados no treino e escolhidos na validação, como em run_window;
  3. o escolhido é RETREINADO com treino + validação (tudo até M - embargo), dicionário incluído;
  4. opera o mês M às cegas; o "semáforo" registra se a validação do escolhido era positiva.
Alfabeto, palavras e LM (surpresa/entropia) são os da janela, fixos: só dicionário e GBM são mensais.
Duas variantes decididas antes de olhar o teste: sem peso e com peso maior para o recente (meia-vida 180 dias).

    python -m vag.monthly_eval --window 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from . import baselines as B
from .alphabet import Alphabet
from .config import CONFIG_DIR, PROJECT_ROOT, load_config
from .evaluate import metrics, phrase_keys, simulate
from .tick_experiment import load_timeline_ticks
from .walkforward import Logger, _roll_z, target_encode
from .words import Vocabulary

HALF_LIFE_DAYS = 180
VAL_MONTHS = 3
VARIANTS = {"sem_peso": None, "peso_recente": HALF_LIFE_DAYS}


def te_cols(keys: dict, fwd: dict, horizons, fit_idx: np.ndarray, n: int) -> list[np.ndarray]:
    """Significado das frases: fora-da-dobra dentro de fit_idx; aprendido em fit_idx para o que vem depois."""
    cols = []
    rest = np.arange(fit_idx[-1] + 1, n)
    for h in horizons:
        te = np.zeros((n, len(keys)), dtype=np.float32)
        for f in np.array_split(fit_idx, 5):
            te[f] = target_encode(keys, fwd[h], np.setdiff1d(fit_idx, f, assume_unique=True), f, h)
        te[rest] = target_encode(keys, fwd[h], fit_idx, rest, h)
        cols.append(te)
    return cols


def fit_predict(X, fwd, fit_idx, h, cfg, seed, times_ns, half_life):
    """Mesmo GBM de baselines.gbm_candidates, com peso opcional por idade da amostra."""
    sub = fit_idx[::3]
    y = fwd[h][sub]
    ok = np.isfinite(y) & np.isfinite(X[sub]).all(1)
    clip = np.nanquantile(np.abs(y[ok]), 0.99)
    w = None
    if half_life:
        age_days = (times_ns[fit_idx[-1]] - times_ns[sub][ok]) / 86_400e9
        w = 0.5 ** (age_days / half_life)
    m = HistGradientBoostingRegressor(max_iter=cfg["gbm_max_iter"], learning_rate=cfg["gbm_learning_rate"],
                                      max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
                                      random_state=seed)
    m.fit(X[sub][ok], np.clip(y[ok], -clip, clip), sample_weight=w)
    pred = np.full(len(X), np.nan)
    good = np.isfinite(X).all(1)
    pred[good] = m.predict(X[good])
    return pred


def signal(pred, fit_idx, q):
    thr = np.nanquantile(np.abs(pred[fit_idx]), q)
    return np.where(np.abs(pred) > thr, np.sign(pred), 0).astype(np.int8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=3)
    args = ap.parse_args(argv)
    wi = args.window

    cfg = load_config()
    exp = yaml.safe_load((CONFIG_DIR / "experiment.yaml").read_text(encoding="utf-8"))
    tick_dir = PROJECT_ROOT / "runs" / f"{exp['run_name']}_ticks"
    wdir = tick_dir / f"w{wi:02d}"
    out_dir = PROJECT_ROOT / "runs" / f"{exp['run_name']}_ticks_mensal_w{wi:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "log.txt")
    block = exp["walkforward"]["test_blocks"][wi]
    H, bcfg, seed = exp["horizons_minutes"], exp["baselines"], exp["seed"] + wi
    emb = pd.Timedelta(minutes=exp["walkforward"]["embargo_minutes"])
    costs = exp["costs"]
    cost = 2 * (costs["fee_per_side"] + costs["slippage_per_side"])
    log(f"treino mensal na janela {wi} {block} | variantes {list(VARIANTS)} | validação {VAL_MONTHS} meses")

    tl = load_timeline_ticks(cfg, exp, block, tick_dir, log)
    times = tl["times"]
    end = int(np.searchsorted(times.values, pd.Timestamp(block[1], tz="UTC").to_datetime64()))
    X, brk = tl["X"][:end], tl["breaks"][:end]
    fwd = {h: v[:end] for h, v in tl["fwd"].items()}
    tv = times.values[:end]
    tns = tv.astype("datetime64[ns]").astype(np.int64)

    alpha = Alphabet.load(wdir / "alphabet.pt")
    letters = alpha.encode(X, "cpu")
    vocab = Vocabulary.load(wdir / "vocab.json")
    keys = phrase_keys(vocab.phrases(letters, exp["dictionary"]["max_phrase_words"], brk), len(vocab))
    z = np.load(wdir / "lm_out.npz")
    assert len(z["surprise"]) == end, (len(z["surprise"]), end)
    lm_cols = [_roll_z(z["surprise"])[:, None], _roll_z(z["entropy"])[:, None]]

    def idx_at(ts):
        return int(np.searchsorted(tv, pd.Timestamp(ts).to_datetime64()))

    months = pd.date_range(pd.Timestamp(block[0], tz="UTC"), pd.Timestamp(block[1], tz="UTC"), freq="MS")
    trades = {(m, v): [] for m in ("gbm_dialeto", "gbm_cru") for v in VARIANTS}
    rows = []
    for ms, me in zip(months[:-1], months[1:]):
        t0 = time.time()
        test = np.arange(idx_at(ms), idx_at(me))
        v0 = ms - pd.DateOffset(months=VAL_MONTHS)
        val = np.arange(idx_at(v0), idx_at(ms - emb))
        core = np.arange(0, idx_at(v0 - emb))
        full = np.arange(0, idx_at(ms - emb))           # treino + validação: tudo o que se sabe antes de M
        XD_sel = np.hstack([X] + lm_cols + te_cols(keys, fwd, H, core, end)).astype(np.float32)
        XD_ref = np.hstack([X] + lm_cols + te_cols(keys, fwd, H, full, end)).astype(np.float32)
        for method, (Xs, Xr) in {"gbm_dialeto": (XD_sel, XD_ref), "gbm_cru": (X, X)}.items():
            for vname, hl in VARIANTS.items():
                best = None
                for h in H:
                    pred = fit_predict(Xs, fwd, core, h, bcfg, seed, tns, hl)
                    for q in B.QUANTILES:
                        mv = metrics(simulate(signal(pred, core, q), fwd[h], val, h, cost))
                        if mv["trades"] >= 30 and (best is None or mv["total_net_pct"] > best[2]["total_net_pct"]):
                            best = (h, q, mv)
                if best is None:
                    rows.append({"mes": f"{ms:%Y-%m}", "metodo": method, "variante": vname, "escolha": None})
                    continue
                h, q, mv = best
                sig = signal(fit_predict(Xr, fwd, full, h, bcfg, seed, tns, hl), full, q)
                tr = simulate(sig, fwd[h], test, h, cost)
                tr["semaforo"] = mv["mean_net_bps"] is not None and mv["mean_net_bps"] > 0
                trades[(method, vname)].append(tr)
                mt = metrics(tr)
                rows.append({"mes": f"{ms:%Y-%m}", "metodo": method, "variante": vname, "escolha": f"h={h} q{q}",
                             "val_bps": mv["mean_net_bps"], "val_op": mv["trades"], "semaforo_verde": bool(tr["semaforo"].all()) if len(tr) else mv["mean_net_bps"] > 0,
                             "teste_op": mt["trades"], "teste_bps": mt["mean_net_bps"], "teste_total_pct": mt["total_net_pct"]})
                log(f"  {ms:%Y-%m} {method:12s} {vname:13s} h={h:3d} q{q}: validação {mv['mean_net_bps']:+.1f} bps "
                    f"({mv['trades']} op) -> mês {mt['trades']} op, {mt['mean_net_bps'] if mt['mean_net_bps'] is None else round(mt['mean_net_bps'], 1)} bps")
        log(f"mês {ms:%Y-%m} concluído em {(time.time() - t0) / 60:.1f} min")

    base = json.loads((wdir / "result.json").read_text(encoding="utf-8"))["metodos"]
    L = [f"# Treino mensal x receita original — janela {wi} ({block[0]} → {block[1]}), velas + ticks", "",
         "Receita original: modelo treinado até ~3 meses antes do teste, sem atualização por 6 meses. "
         "Mensal: dicionário e GBM refeitos todo mês com tudo até o mês anterior (incluindo a validação). "
         "Custo taker 12 bps ida e volta.", "",
         "| Método | Receita original (6 meses parado) | Mensal sem peso | Mensal c/ peso recente | Mensal + semáforo (sem peso) | Mensal + semáforo (peso recente) |",
         "|---|---|---|---|---|---|"]
    summary = {}
    for method in ("gbm_dialeto", "gbm_cru"):
        cells = [f"{base[method]['teste_taker']['mean_net_bps']:.1f} bps ({base[method]['teste_taker']['trades']} op)"]
        for vname in VARIANTS:
            allt = pd.concat(trades[(method, vname)]) if trades[(method, vname)] else pd.DataFrame(columns=["t", "dir", "net", "semaforo"])
            mt = metrics(allt)
            summary[f"{method}/{vname}"] = mt
            cells.append(f"{mt['mean_net_bps']:.1f} bps ({mt['trades']} op, t {mt['t_stat']:.2f})" if mt["trades"] else "sem operações")
        for vname in VARIANTS:
            allt = pd.concat(trades[(method, vname)]) if trades[(method, vname)] else pd.DataFrame(columns=["t", "dir", "net", "semaforo"])
            g = allt[allt["semaforo"].astype(bool)] if len(allt) else allt
            mt = metrics(g)
            summary[f"{method}/{vname}/semaforo"] = mt
            cells.append(f"{mt['mean_net_bps']:.1f} bps ({mt['trades']} op)" if mt["trades"] else "0 (ficou de fora)")
        L.append(f"| {method} | " + " | ".join(cells) + " |")
    L += ["", "## Mês a mês", "", "| Mês | Método | Variante | Escolha | Validação (bps, op) | Semáforo | Mês às cegas (op, bps) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["escolha"]:
            L.append(f"| {r['mes']} | {r['metodo']} | {r['variante']} | — | <30 op | — | — |")
            continue
        tb = "—" if r["teste_bps"] is None else f"{r['teste_bps']:.1f}"
        L.append(f"| {r['mes']} | {r['metodo']} | {r['variante']} | {r['escolha']} | {r['val_bps']:+.1f} ({r['val_op']}) | "
                 f"{'🟢' if r['semaforo_verde'] else '🔴'} | {r['teste_op']} op, {tb} |")
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps({"resumo": summary, "meses": rows}, indent=2, ensure_ascii=False,
                                                    default=float), encoding="utf-8")
    log(f"relatório: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

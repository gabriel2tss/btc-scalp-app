"""Modelo CONGELADO: o GBM + dialeto de uma janela (velas + ticks), sem retreinar, aplicado em períodos posteriores.

Reconstrói exatamente o que a janela de origem aprendeu:
- alfabeto, palavras e LM: carregados de runs/<run>_ticks/wNN (não são retreinados);
- dicionários (significado das frases) e GBM: refeitos de forma determinística com os mesmos
  dados de treino da janela de origem e a mesma semente;
- regra de operação: o horizonte e o limiar que a validação da janela de origem escolheu.
Depois aplica essa mesma regra, sem nenhuma atualização, nos blocos de teste pedidos.

Primeiro reproduz o teste da própria janela de origem: tem que dar o mesmo resultado; se não der,
a reconstrução não é fiel e o resto não vale.

    python -m vag.frozen_eval --windows 3 7            # origem: janela 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from . import baselines as B
from .alphabet import Alphabet
from .config import CONFIG_DIR, PROJECT_ROOT, load_config
from .evaluate import breakdown, metrics, phrase_keys, session_of, simulate
from .features import FEATURES
from .lm import MarketGPT, surprise_entropy
from .tick_experiment import load_timeline_ticks
from .walkforward import Logger, _roll_z, split_window, target_encode
from .words import Vocabulary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", nargs="+", type=int, required=True)
    ap.add_argument("--source-window", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    cfg = load_config()
    exp = yaml.safe_load((CONFIG_DIR / "experiment.yaml").read_text(encoding="utf-8"))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sw = args.source_window
    tick_dir = PROJECT_ROOT / "runs" / f"{exp['run_name']}_ticks"
    src_dir = tick_dir / f"w{sw:02d}"
    out_dir = PROJECT_ROOT / "runs" / f"{exp['run_name']}_ticks_congelado_w{sw:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "log.txt")

    src = json.loads((src_dir / "result.json").read_text(encoding="utf-8"))
    choice = src["metodos"]["gbm_dialeto"]
    h, name = choice["horizonte"], choice["escolha"]
    q = float(name.rsplit("_q", 1)[1])
    blocks_all = exp["walkforward"]["test_blocks"]
    windows = [sw] + [w for w in args.windows if w != sw]
    last = max((blocks_all[w] for w in windows), key=lambda b: b[1])
    log(f"modelo congelado da janela {sw}: {name} (h={h} min, limiar q{q}) | aplicar em {windows}")

    # a timeline é a mesma da receita (mesmo cache em runs/<run>_ticks); features causais: linhas antigas não mudam
    tl = load_timeline_ticks(cfg, exp, last, tick_dir, log)
    X, brk, times, fwd, H = tl["X"], tl["breaks"], tl["times"], tl["fwd"], exp["horizons_minutes"]
    wf = exp["walkforward"]
    core = split_window(times, blocks_all[sw], wf["embargo_minutes"], wf["val_fraction"])["core"]

    alpha = Alphabet.load(src_dir / "alphabet.pt")
    letters = alpha.encode(X, device)
    vocab = Vocabulary.load(src_dir / "vocab.json")
    keys = phrase_keys(vocab.phrases(letters, exp["dictionary"]["max_phrase_words"], brk), len(vocab))

    lm_cfg = exp["lm"]
    model = MarketGPT(alpha.k, lm_cfg["context"], lm_cfg["n_layer"], lm_cfg["n_head"], lm_cfg["d_model"],
                      lm_cfg["dropout"]).to(device)
    model.load_state_dict(torch.load(src_dir / "lm.pt", map_location=device, weights_only=False)["best_model"])
    log("calculando surpresa/entropia com o LM congelado...")
    surprise, entropy = surprise_entropy(model, letters, brk, device, bool(lm_cfg["amp"]))
    del model
    old = np.load(src_dir / "lm_out.npz")["surprise"]
    n_chk = len(old) - lm_cfg["context"]              # o fim da série antiga usa outra janela deslizante
    same = np.allclose(old[:n_chk], surprise[:n_chk], atol=1e-3, equal_nan=True)
    log(f"  surpresa igual à da janela {sw} original: {same}")

    # mesmas colunas extras, na mesma ordem, que run_window monta para o gbm_dialeto
    extra = [_roll_z(surprise)[:, None], _roll_z(entropy)[:, None]]
    rest = np.arange(core[-1] + 1, len(X))
    for hh in H:
        te = np.zeros((len(X), len(keys)), dtype=np.float32)
        for f in np.array_split(core, 5):
            te[f] = target_encode(keys, fwd[hh], np.setdiff1d(core, f, assume_unique=True), f, hh)
        te[rest] = target_encode(keys, fwd[hh], core, rest, hh)
        extra.append(te)
    XD = np.hstack([X] + extra).astype(np.float32)
    cands = B.gbm_candidates(XD, fwd, core, [h], exp["baselines"], exp["seed"] + sw, "gbm_dialeto", log)
    sig = next(s for n, _, s in cands if n == name)

    costs = exp["costs"]
    cost_taker = 2 * (costs["fee_per_side"] + costs["slippage_per_side"])
    cost_maker = 2 * (0.0002 + costs["slippage_per_side"])
    sess = session_of(times)
    vr = X[:, FEATURES.index("vol_regime")]
    vq = np.nanquantile(vr[core], [1 / 3, 2 / 3])
    vol_lab = np.select([vr < vq[0], vr < vq[1]], ["vol_baixa", "vol_media"], "vol_alta")
    dir_lab = np.where(sig > 0, "compra", np.where(sig < 0, "venda", "nada"))

    results = {}
    for w in windows:
        test = split_window(times, blocks_all[w], wf["embargo_minutes"], wf["val_fraction"])["test"]
        tr = simulate(sig, fwd[h], test, h, cost_taker)
        results[w] = {"periodo": blocks_all[w], "teste_taker": metrics(tr),
                      "teste_maker": metrics(simulate(sig, fwd[h], test, h, cost_maker)),
                      "acaso_mesmo_n": B.random_baseline(len(tr), test, fwd[h], h, cost_taker, exp["seed"] + sw),
                      "por_sessao": breakdown(tr, sess), "por_volatilidade": breakdown(tr, vol_lab),
                      "por_direcao": breakdown(tr, dir_lab)}
        t = results[w]["teste_taker"]
        log(f"  janela {w} {blocks_all[w]}: {t['trades']} op, média {t['mean_net_bps']} bps, acerto {t['hit']}, "
            f"PF {t['pf']}, t {t['t_stat']}")
        for k, v in results[w]["por_direcao"].items():
            log(f"    {k}: {v['trades']} op, média {v['mean_net_bps']} bps, acerto {v['hit']}")
    orig = choice["teste_taker"]
    rep = results[sw]["teste_taker"]
    ok = rep["trades"] == orig["trades"] and abs(rep["mean_net_bps"] - orig["mean_net_bps"]) < 0.5
    log(f"reprodução da janela {sw}: {'OK' if ok else 'AVISO: diferente do original'} "
        f"({rep['trades']} op / {rep['mean_net_bps']:.2f} bps x original {orig['trades']} op / {orig['mean_net_bps']:.2f} bps)")

    def f(x, nd=1):
        return "—" if x is None else f"{x:.{nd}f}"

    L = [f"# Modelo congelado da janela {sw} (velas + ticks) aplicado sem retreinar", "",
         f"Regra: `{name}` — operação de {h} min, mesmo limiar, mesmos dicionários e LM de {blocks_all[sw][0][:7]}. "
         f"Custo taker 12 bps ida e volta. Reprodução da janela {sw}: **{'OK' if ok else 'DIFERENTE — não confiar'}**.", "",
         "| Janela | Teste | Op. | Média líq. (bps) | Acerto | PF | t | Acaso p95 (bps) | Maker (bps) | Receita retreinada (bps) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for w, r in results.items():
        t, a = r["teste_taker"], r["acaso_mesmo_n"]
        rp = tick_dir / f"w{w:02d}" / "result.json"
        rec = json.loads(rp.read_text(encoding="utf-8"))["metodos"]["gbm_dialeto"]["teste_taker"]["mean_net_bps"] if rp.exists() else None
        L.append(f"| {w}{' (origem)' if w == sw else ''} | {r['periodo'][0]} → {r['periodo'][1]} | {t['trades']} | "
                 f"{f(t['mean_net_bps'])} | {f(t['hit'], 3)} | {f(t['pf'], 2)} | {f(t['t_stat'], 2)} | "
                 f"{f(a.get('p95_bps'))} | {f(r['teste_maker']['mean_net_bps'])} | {f(rec)} |")
    L += ["", "Por direção (compra x venda):", ""]
    for w, r in results.items():
        L.append(f"- janela {w}: " + " | ".join(f"{k} {v['trades']} op, {f(v['mean_net_bps'])} bps"
                                                 for k, v in r["por_direcao"].items()))
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps({str(k): v for k, v in results.items()}, indent=2,
                                                    ensure_ascii=False, default=float), encoding="utf-8")
    log(f"relatório: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

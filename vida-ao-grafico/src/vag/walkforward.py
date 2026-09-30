"""Orquestrador walk-forward (Fases 2 a 6), feito para rodar a madrugada sozinho.

    python -m vag.walkforward --smoke           # teste rápido de ponta a ponta (2 janelas, modelos mínimos)
    python -m vag.walkforward                   # rodada completa
    python -m vag.walkforward --windows 0 1     # só algumas janelas

Retomável: cada etapa de cada janela grava seu resultado; se cair, rodar de novo
continua de onde parou. Tudo em runs/<run_name>/ (log.txt, w00/, w01/, ..., report.md).

Em cada janela:
  treino_core | embargo | validação | embargo | TESTE
  - alfabeto, palavras, LM, dicionário e GBMs aprendem só com treino_core
  - todos os parâmetros (horizonte, ocorrências mínimas, FDR, limiares) são escolhidos na validação
  - o teste é avaliado uma única vez com o que foi escolhido
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from . import baselines as B
from .alphabet import Alphabet, code_stats, train_alphabet
from .config import CONFIG_DIR, PROJECT_ROOT, load_config
from .data.download import storage_symbol
from .data.storage import read_symbol
from .evaluate import (breakdown, build_dictionary, dictionary_signal, metrics, phrase_keys, session_of, simulate)
from .features import FEATURES, compute_features, forward_returns, valid_mask
from .lm import MarketGPT, surprise_entropy, train_lm
from .words import Vocabulary, learn_bpe

SMOKE = {
    "alphabet": {"epochs": 1},
    "words": {"merges": 100},
    "lm": {"max_steps": 300, "eval_every": 100, "n_layer": 2, "d_model": 128, "n_head": 4},
    "baselines": {"gbm_max_iter": 40},
}


class Logger:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *msg):
        line = f"[{datetime.now():%H:%M:%S}] " + " ".join(str(m) for m in msg)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _roll_z(x: np.ndarray, win: int = 1440) -> np.ndarray:
    s = pd.Series(x)
    return ((s - s.rolling(win, min_periods=60).mean()) / s.rolling(win, min_periods=60).std()).values


# ---------------- dados + features (uma vez por rodada) ----------------

def load_timeline(cfg: dict, exp: dict, run_dir: Path, log) -> dict:
    cache = run_dir / "timeline.parquet"
    horizons = exp["horizons_minutes"]
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        sym = storage_symbol(exp["data"]["source"], exp["data"]["symbol"])
        log(f"lendo velas de {sym}...")
        bars = read_symbol(cfg["data_dir"], "m1", sym)
        if bars.empty:
            raise SystemExit(f"Sem dados em {cfg['data_dir']} para {sym}. Rode o download primeiro.")
        bars = bars.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        log(f"  {len(bars):,} velas de {bars['time'].iloc[0]} a {bars['time'].iloc[-1]}")
        t0 = time.time()
        feats = compute_features(bars)
        fwd = forward_returns(bars, horizons)
        df = pd.concat([feats, fwd.drop(columns="time")], axis=1)
        df = df[valid_mask(feats)].reset_index(drop=True)
        df.to_parquet(cache, compression="zstd")
        log(f"  features calculadas em {time.time() - t0:.0f}s: {len(df):,} minutos válidos")
    times = pd.to_datetime(df["time"], utc=True)
    breaks = np.r_[True, (times.diff().dt.total_seconds().values[1:] != 60)]
    return {
        "times": times,
        "X": df[FEATURES].to_numpy(np.float32),
        "fwd": {h: df[f"fwd_{h}"].to_numpy(np.float64) for h in horizons},
        "breaks": breaks,
    }


def split_window(times: pd.Series, block, embargo_min: int, val_frac: float) -> dict:
    a, b = pd.Timestamp(block[0], tz="UTC"), pd.Timestamp(block[1], tz="UTC")
    emb = pd.Timedelta(minutes=embargo_min)
    tv = times.values
    i_test0, i_test1 = np.searchsorted(tv, a.to_datetime64()), np.searchsorted(tv, b.to_datetime64())
    i_train1 = np.searchsorted(tv, (a - emb).to_datetime64())
    i_val0 = int(i_train1 * (1 - val_frac))
    i_core1 = np.searchsorted(tv, (times.iloc[i_val0] - emb).to_datetime64())
    return {
        "core": np.arange(0, i_core1), "val": np.arange(i_val0, i_train1), "test": np.arange(i_test0, i_test1),
        "end": int(i_test1),
        "desc": {"core": [str(times.iloc[0]), str(times.iloc[i_core1 - 1])],
                 "val": [str(times.iloc[i_val0]), str(times.iloc[i_train1 - 1])],
                 "test": [str(a), str(b)], "test_minutes": int(i_test1 - i_test0)},
    }


def target_encode(keys_by_n: dict, fwd: np.ndarray, fit_idx: np.ndarray, apply_idx: np.ndarray,
                  horizon: int, shrink: float = 200.0) -> np.ndarray:
    """Significado médio (encolhido para 0) da frase em cada t, aprendido só em fit_idx."""
    out = np.zeros((len(apply_idx), len(keys_by_n)), dtype=np.float32)
    for j, n in enumerate(sorted(keys_by_n)):
        d = build_dictionary(keys_by_n[n], fwd, fit_idx, horizon)
        val = (d["mean"] * d["count"] / (d["count"] + shrink)).astype(np.float32)
        pos = pd.Index(d.index).get_indexer(keys_by_n[n][apply_idx])
        out[:, j] = np.where(pos >= 0, val.values[np.clip(pos, 0, None)], 0.0)
    return out * 1e4


# ---------------- uma janela ----------------

def run_window(wi: int, block, tl: dict, exp: dict, run_dir: Path, device: str, log) -> dict:
    wdir = run_dir / f"w{wi:02d}"
    wdir.mkdir(exist_ok=True)
    done = wdir / "result.json"
    if done.exists():
        log(f"janela {wi}: já concluída")
        return json.loads(done.read_text(encoding="utf-8"))

    wf, seed = exp["walkforward"], exp["seed"] + wi
    sp = split_window(tl["times"], block, wf["embargo_minutes"], wf["val_fraction"])
    core, val, test, end = sp["core"], sp["val"], sp["test"], sp["end"]
    log(f"=== janela {wi}: treino {sp['desc']['core']} | validação {sp['desc']['val']} | teste {sp['desc']['test']}")
    if len(test) == 0:
        log("  sem dados de teste nesta janela; pulando")
        return {}
    X, brk, fwd, H = tl["X"][:end], tl["breaks"][:end], {h: v[:end] for h, v in tl["fwd"].items()}, exp["horizons_minutes"]
    timings = {}

    # 1. alfabeto
    t0 = time.time()
    ap = wdir / "alphabet.pt"
    if ap.exists():
        alpha = Alphabet.load(ap)
        a_stats = json.loads((wdir / "alphabet.json").read_text())
    else:
        log("  [1/5] alfabeto (VQ-VAE)")
        alpha, a_stats = train_alphabet(X[core], exp["alphabet"], device, seed, log)
        alpha.save(ap)
        (wdir / "alphabet.json").write_text(json.dumps(a_stats, indent=2))
    letters = alpha.encode(X, device)
    a_stats["test"] = code_stats(letters[test], alpha.k)
    log(f"    letras usadas no treino: {a_stats['used_letters']}/{alpha.k}, perplexidade {a_stats['perplexity']:.0f}; "
        f"no teste: {a_stats['test']['used_letters']}")
    timings["alphabet_s"] = time.time() - t0

    # 2. palavras
    t0 = time.time()
    vp = wdir / "vocab.json"
    if vp.exists():
        vocab = Vocabulary.load(vp)
    else:
        log("  [2/5] palavras (BPE)")
        w = exp["words"]
        vocab = learn_bpe(letters[core], alpha.k, w["merges"], w["max_word_len"], w["min_pair_count"], brk[core], log)
        vocab.save(vp)
    n_max = exp["dictionary"]["max_phrase_words"]
    phrases = vocab.phrases(letters, n_max, brk)
    keys = phrase_keys(phrases, len(vocab))
    wl = np.array([len(w) for w in vocab.words])
    timings["words_s"] = time.time() - t0
    log(f"    vocabulário: {len(vocab)} palavras, comprimento médio (uso) {wl[phrases[test, 0]].mean():.2f} min")

    # 3. modelo de linguagem
    t0 = time.time()
    lm_out = wdir / "lm_out.npz"
    lm_cfg = exp["lm"]
    if lm_out.exists():
        z = np.load(lm_out)
        surprise, entropy = z["surprise"], z["entropy"]
        lm_stats = json.loads((wdir / "lm.json").read_text())
    else:
        log("  [3/5] modelo de linguagem (mini-GPT)")
        model, lm_stats = train_lm(letters[core], brk[core], letters[val], brk[val], alpha.k, lm_cfg, device, seed,
                                   wdir / "lm.pt", log)
        surprise, entropy = surprise_entropy(model, letters, brk, device, bool(lm_cfg["amp"]))
        np.savez_compressed(lm_out, surprise=surprise, entropy=entropy)
        (wdir / "lm.json").write_text(json.dumps(lm_stats, indent=2))
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    timings["lm_s"] = time.time() - t0
    log(f"    LM: validação {lm_stats['best_val_loss']:.4f} vs unigrama {lm_stats['unigram_loss']:.4f} nats")

    # 4. dicionário + seleção na validação + teste
    t0 = time.time()
    log("  [4/5] dicionário de significados e seleção na validação")
    costs = exp["costs"]
    cost_taker = 2 * (costs["fee_per_side"] + costs["slippage_per_side"])
    cost_maker = 2 * (0.0002 + costs["slippage_per_side"])
    dcfg = exp["dictionary"]
    dicts = {h: {n: build_dictionary(keys[n], fwd[h], core, h) for n in keys} for h in H}

    def dialect_candidates():
        for h in H:
            for mc in sorted({50, 100, 300, 1000, dcfg["min_train_count"]}):
                for q in (dcfg["fdr_q"], 0.2):
                    for edge in (0.0, cost_taker / 2, cost_taker):
                        for nm in range(1, n_max + 1):
                            sig, nsel = dictionary_signal(keys, dicts[h], mc, q, edge, nm, end)
                            if nsel:
                                yield (f"dialeto h={h} min_oc={mc} fdr={q} edge={edge * 1e4:.0f}bps n<={nm}", h, sig)

    results = {}
    best_d = B.pick_on_val(dialect_candidates(), val, fwd, H, cost_taker)

    # baselines
    col = {f: i for i, f in enumerate(FEATURES)}
    best_m = B.pick_on_val(B.momentum_candidates(X, col, core, H), val, fwd, H, cost_taker)
    best_g = B.pick_on_val(B.gbm_candidates(X, fwd, core, H, exp["baselines"], seed, "gbm_cru", log), val, fwd, H, cost_taker)

    # GBM + dialeto: features cruas + surpresa/entropia (relativas) + significado das frases
    s_z, e_z = _roll_z(surprise), _roll_z(entropy)
    extra_cols = [s_z[:, None], e_z[:, None]]
    for h in H:
        te = np.zeros((end, len(keys)), dtype=np.float32)
        folds = np.array_split(core, 5)   # fora-da-dobra dentro do treino (evita decorar o próprio rótulo)
        for f in folds:
            te[f] = target_encode(keys, fwd[h], np.setdiff1d(core, f, assume_unique=True), f, h)
        rest = np.arange(core[-1] + 1, end)
        te[rest] = target_encode(keys, fwd[h], core, rest, h)
        extra_cols.append(te)
    XD = np.hstack([X] + extra_cols).astype(np.float32)
    best_gd = B.pick_on_val(B.gbm_candidates(XD, fwd, core, H, exp["baselines"], seed, "gbm_dialeto", log),
                            val, fwd, H, cost_taker)
    timings["dictionary_baselines_s"] = time.time() - t0

    # 5. teste (uma vez)
    log("  [5/5] avaliação no teste (período nunca visto)")
    sess = session_of(tl["times"][:end])
    vr = X[:, col["vol_regime"]]
    vq = np.nanquantile(vr[core], [1 / 3, 2 / 3])
    vol_lab = np.select([vr < vq[0], vr < vq[1]], ["vol_baixa", "vol_media"], "vol_alta")
    eq = np.nanquantile(entropy[core], [1 / 3, 2 / 3])
    ent_lab = np.select([entropy < eq[0], entropy < eq[1]], ["fala_clara", "fala_media"], "fala_confusa")

    for name, best in [("dialeto", best_d), ("momentum_reversao", best_m), ("gbm_cru", best_g), ("gbm_dialeto", best_gd)]:
        if best is None:
            results[name] = {"escolha": None, "motivo": "nenhuma configuração com >= 30 operações na validação"}
            continue
        pname, h, sig, vm = best
        tr_taker = simulate(sig, fwd[h], test, h, cost_taker)
        results[name] = {
            "escolha": pname, "horizonte": h, "validacao": vm,
            "teste_taker": metrics(tr_taker),
            "teste_maker": metrics(simulate(sig, fwd[h], test, h, cost_maker)),
            "acaso_mesmo_n": B.random_baseline(len(tr_taker), test, fwd[h], h, cost_taker, seed),
            "por_sessao": breakdown(tr_taker, sess),
            "por_volatilidade": breakdown(tr_taker, vol_lab),
            "por_clareza_da_fala": breakdown(tr_taker, ent_lab),
        }
        t = results[name]["teste_taker"]
        log(f"    {name:18s} [{pname}] teste: {t['trades']} op, média {t['mean_net_bps']} bps, "
            f"acerto {t['hit']}, PF {t['pf']}")

    # a surpresa antecipa movimento? (útil como filtro, mesmo sem direção)
    h_mid = 15 if 15 in H else H[0]
    ok = np.isfinite(surprise[test]) & np.isfinite(fwd[h_mid][test])
    clarity = {
        "spearman_surpresa_vs_abs_ret": float(pd.Series(surprise[test][ok]).corr(pd.Series(np.abs(fwd[h_mid][test][ok])), method="spearman")),
        "spearman_entropia_vs_abs_ret": float(pd.Series(entropy[test][ok]).corr(pd.Series(np.abs(fwd[h_mid][test][ok])), method="spearman")),
        "horizonte": h_mid,
    }

    # frases aprovadas (matéria-prima da "voz" na Fase 7)
    if best_d is not None:
        h = best_d[1]
        rows = []
        for n in keys:
            d = dicts[h][n]
            d = d[d["count"] >= 100].copy()
            d["n_words"] = n
            rows.append(d)
        top = pd.concat(rows).sort_values("p").head(500)
        top.to_csv(wdir / f"frases_top_h{h}.csv")

    res = {"janela": wi, "periodos": sp["desc"], "alfabeto": a_stats,
           "lm": {k: v for k, v in lm_stats.items() if k != "history"},
           "vocabulario": len(vocab), "metodos": results, "surpresa": clarity, "tempos_s": timings,
           "custos_ida_volta": {"taker": cost_taker, "maker": cost_maker}}
    done.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return res


# ---------------- relatório ----------------

def _fmt(v, nd=1):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def write_report(results: list[dict], exp: dict, run_dir: Path, smoke: bool) -> Path:
    L = [f"# Walk-forward — {exp['run_name']}{' (SMOKE: modelos mínimos, não interpretar)' if smoke else ''}",
         "", f"Gerado em {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Ativo: {exp['data']['symbol']} "
         f"({exp['data']['source']}). Custos por operação (ida e volta): taker "
         f"{2 * (exp['costs']['fee_per_side'] + exp['costs']['slippage_per_side']) * 1e4:.0f} bps.", "",
         "Todos os parâmetros foram escolhidos na validação; os números abaixo são do teste, nunca visto.", "",
         "| Janela (teste) | Método | Escolha | Op. | Média líq. (bps) | Acerto | PF | t | Acaso (bps, p95) | Maker (bps) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    wins = {m: 0 for m in ("dialeto", "momentum_reversao", "gbm_cru", "gbm_dialeto")}
    beats = 0
    counted = 0
    for r in results:
        if not r:
            continue
        counted += 1
        per = f"{r['periodos']['test'][0][:10]} → {r['periodos']['test'][1][:10]}"
        for m, v in r["metodos"].items():
            if not v.get("escolha"):
                L.append(f"| {per} | {m} | — | 0 | — | — | — | — | — | — |")
                continue
            t, mk, rnd = v["teste_taker"], v["teste_maker"], v["acaso_mesmo_n"]
            if t["mean_net_bps"] is not None and t["mean_net_bps"] > 0:
                wins[m] += 1
            L.append(f"| {per} | {m} | {v['escolha']} | {t['trades']} | {_fmt(t['mean_net_bps'])} | {_fmt(t['hit'], 3)} | "
                     f"{_fmt(t['pf'], 2)} | {_fmt(t['t_stat'], 2)} | {_fmt(rnd.get('p95_bps'))} | {_fmt(mk['mean_net_bps'])} |")
        gd, gc = r["metodos"].get("gbm_dialeto", {}), r["metodos"].get("gbm_cru", {})
        if gd.get("escolha") and gc.get("escolha"):
            a, b = gd["teste_taker"]["mean_net_bps"], gc["teste_taker"]["mean_net_bps"]
            if a is not None and b is not None and a > b:
                beats += 1
    L += ["", "## Resumo", "",
          f"- Janelas avaliadas: {counted}",
          *[f"- {m}: resultado líquido positivo em {k} de {counted} janelas" for m, k in wins.items()],
          f"- GBM **com** dialeto ganhou do GBM só com números crus em {beats} de {counted} janelas "
          "(a pergunta central: o dialeto acrescenta informação?)", "",
          "## Modelo de linguagem e alfabeto", "",
          "| Janela | Letras usadas (treino/teste) | Perplexidade letras | LM val (nats) | Unigrama | Ganho | Surpresa×|ret| | Entropia×|ret| |",
          "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if not r:
            continue
        a, lm, s = r["alfabeto"], r["lm"], r["surpresa"]
        L.append(f"| {r['janela']} | {a['used_letters']}/{a['test']['used_letters']} | {a['perplexity']:.0f} | "
                 f"{lm['best_val_loss']:.3f} | {lm['unigram_loss']:.3f} | {lm['val_gain_over_unigram_nats']:.3f} | "
                 f"{s['spearman_surpresa_vs_abs_ret']:.3f} | {s['spearman_entropia_vs_abs_ret']:.3f} |")
    L += ["", "Leitura: *Ganho* = quanto o LM prevê a próxima letra melhor que só a frequência das letras "
          "(0 = não aprendeu gramática nenhuma). Correlação surpresa×|retorno| > 0 = a surpresa antecipa movimento "
          "(útil como filtro de clareza, mesmo sem prever direção).", "",
          "Detalhes por sessão, volatilidade e clareza da fala em `wNN/result.json`; frases mais significativas em "
          "`wNN/frases_top_h*.csv`.", "",
          "Limitações desta versão: funding (a cada 8h) ainda não descontado; surpresa/entropia do LM no trecho de "
          "treino são dentro da amostra (mitigado usando valores relativos às últimas 24h)."]
    p = run_dir / "report.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_DIR / "experiment.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--windows", nargs="*", type=int)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    cfg = load_config()
    exp = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.smoke:
        exp = _merge(exp, SMOKE)
        exp["run_name"] += "_smoke"
    windows = args.windows if args.windows is not None else (
        [0, 1] if args.smoke else list(range(len(exp["walkforward"]["test_blocks"]))))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = PROJECT_ROOT / "runs" / exp["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment.yaml").write_text(yaml.safe_dump(exp, allow_unicode=True), encoding="utf-8")
    log = Logger(run_dir / "log.txt")
    gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU"
    log(f"rodada {exp['run_name']} | dispositivo: {gpu} | janelas {windows}")

    np.random.seed(exp["seed"])
    torch.manual_seed(exp["seed"])
    tl = load_timeline(cfg, exp, run_dir, log)
    results = []
    for wi in windows:
        t0 = time.time()
        results.append(run_window(wi, exp["walkforward"]["test_blocks"][wi], tl, exp, run_dir, device, log))
        log(f"janela {wi} concluída em {(time.time() - t0) / 60:.1f} min")
        write_report(results, exp, run_dir, args.smoke)
    p = write_report(results, exp, run_dir, args.smoke)
    log(f"relatório: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

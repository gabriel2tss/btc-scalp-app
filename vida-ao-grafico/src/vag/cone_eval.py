"""Cone de futuros: o LM do dialeto "imagina" muitas continuações do gráfico e medimos, às cegas, se acerta.

Para uma janela (LM, alfabeto e cortes da rodada btc_m1_v1, sem retreinar):
  1. tradutor letra -> movimento: no TREINO, para cada letra, os retornos de 1 min que aconteceram quando ela
     apareceu, divididos pela volatilidade recente; ao imaginar, sorteia um deles e multiplica pela volatilidade
     do momento;
  2. em N momentos sorteados do TESTE, o LM lê os últimos (512 - H) minutos e imagina P futuros de H minutos
     (amostragem letra a letra, com memória das leituras anteriores: KV cache);
  3. mede: largura do cone x tamanho real (e x previsão ingênua pela volatilidade da última hora);
     inclinação do cone x direção real; "futuros que batem o take antes do stop" x resultado real.

    python -m vag.cone_eval --window 7 --moments 500 --paths 100 --horizon 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from .alphabet import Alphabet
from .barriers import CONFIGS, vol_1m
from .config import CONFIG_DIR, PROJECT_ROOT, load_config
from .data.download import storage_symbol
from .data.storage import read_symbol
from .lm import MarketGPT
from .walkforward import Logger, load_timeline, split_window


def _heads(t, h):
    B, T, d = t.shape
    return t.view(B, T, h, d // h).transpose(1, 2)


@torch.no_grad()
def prefill(model, ctx):
    """Lê o contexto inteiro uma vez e guarda chaves/valores de cada camada (KV cache)."""
    T = ctx.shape[1]
    x = model.tok(ctx) + model.pos(torch.arange(T, device=ctx.device))
    cache = []
    for b in model.blocks:
        a = b.attn
        q, k, v = F.linear(b.ln1(x), a.in_proj_weight, a.in_proj_bias).chunk(3, -1)
        q, k, v = (_heads(t, a.num_heads) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(x.shape)
        x = x + a.out_proj(o)
        x = x + b.mlp(b.ln2(x))
        cache.append((k, v))
    return model.head(model.ln(x[:, -1])), cache


@torch.no_grad()
def step(model, tok, pos, cache):
    """Uma letra nova por caminho, atendendo a tudo o que já foi lido (sem reler o contexto)."""
    x = model.tok(tok)[:, None] + model.pos(torch.tensor([pos], device=tok.device))
    new = []
    for b, (K, V) in zip(model.blocks, cache):
        a = b.attn
        q, k, v = F.linear(b.ln1(x), a.in_proj_weight, a.in_proj_bias).chunk(3, -1)
        q, k, v = (_heads(t, a.num_heads) for t in (q, k, v))
        K, V = torch.cat([K, k], 2), torch.cat([V, v], 2)
        o = F.scaled_dot_product_attention(q, K, V).transpose(1, 2).reshape(x.shape)
        x = x + a.out_proj(o)
        x = x + b.mlp(b.ln2(x))
        new.append((K, V))
    return model.head(model.ln(x[:, -1])), new


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--moments", type=int, default=500)
    ap.add_argument("--paths", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=60)
    args = ap.parse_args(argv)
    w, P, H = args.window, args.paths, args.horizon

    cfg = load_config()
    exp = yaml.safe_load((CONFIG_DIR / "experiment.yaml").read_text(encoding="utf-8"))
    base = PROJECT_ROOT / "runs" / exp["run_name"]
    wdir = base / f"w{w:02d}"
    out_dir = PROJECT_ROOT / "runs" / f"{exp['run_name']}_cone"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "log.txt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"cone de futuros | janela {w} | {args.moments} momentos x {P} futuros x {H} min | {dev}")

    tl = load_timeline(cfg, exp, base, log)
    wf = exp["walkforward"]
    sp = split_window(tl["times"], wf["test_blocks"][w], wf["embargo_minutes"], wf["val_fraction"])
    core, test, end = sp["core"], sp["test"], sp["end"]
    times = pd.DatetimeIndex(tl["times"]).as_unit("ns")
    bars = read_symbol(cfg["data_dir"], "m1", storage_symbol(exp["data"]["source"], exp["data"]["symbol"]))
    bars = bars.set_index(pd.DatetimeIndex(bars["time"]).as_unit("ns")).reindex(times)
    h_, l_, c_ = (bars[k].to_numpy(np.float64) for k in ("high", "low", "close"))
    lc = np.log(c_)
    lr = np.r_[np.nan, np.diff(lc)]
    s1 = vol_1m(c_)
    rnorm = lr / np.r_[np.nan, s1[:-1]]

    alpha = Alphabet.load(wdir / "alphabet.pt")
    letters = alpha.encode(tl["X"][:end], dev).astype(np.int64)
    brk = tl["breaks"][:end]

    # tradutor letra -> retorno normalizado, aprendido só no treino
    rng = np.random.default_rng(exp["seed"] + w)
    ok = np.isfinite(rnorm[core])
    allpool = rnorm[core][ok]
    pools = []
    for k in range(alpha.k):
        p = rnorm[core][ok & (letters[core] == k)]
        pools.append(p if len(p) >= 50 else allpool)
    pool_len = torch.tensor([len(p) for p in pools], device=dev)
    maxlen = int(pool_len.max())
    pool_mat = torch.full((alpha.k, maxlen), float("nan"), device=dev)
    for k, p in enumerate(pools):
        pool_mat[k, :len(p)] = torch.from_numpy(p.astype(np.float32)).to(dev)

    lm = exp["lm"]
    model = MarketGPT(alpha.k, lm["context"], lm["n_layer"], lm["n_head"], lm["d_model"], lm["dropout"]).to(dev).eval()
    model.load_state_dict(torch.load(wdir / "lm.pt", map_location=dev, weights_only=False)["best_model"])
    ctx_len = lm["context"] - H

    # checagem: prefill + passos == leitura completa
    seq = torch.from_numpy(letters[core[1000]:core[1000] + 40]).to(dev)[None]
    full = model(seq)[0, -1]
    lg, cache = prefill(model, seq[:, :30])
    for i in range(30, 40):
        lg, cache = step(model, seq[:, i], i, cache)
    log(f"  checagem da memória (KV cache): diferença máxima {float((lg[0] - full).abs().max()):.2e}")

    cand = test[(test >= ctx_len) & (test + H < end)]
    cand = cand[[not brk[t - ctx_len + 1:t + H + 1].any() for t in cand]]
    moments = np.sort(rng.choice(cand, size=min(args.moments, len(cand)), replace=False))
    ci = CONFIGS.index((1.0, 1, 60)) if H == 60 else None
    X_take = np.maximum(1.0 * s1 * np.sqrt(H), 3 * 0.0008)

    rows, t0 = [], time.time()
    gen = torch.Generator(device=dev).manual_seed(exp["seed"] + w)
    for n, t in enumerate(moments):
        ctx = torch.from_numpy(letters[t - ctx_len + 1:t + 1]).to(dev)[None]
        lg, cache = prefill(model, ctx)
        cache = [(K.expand(P, -1, -1, -1).contiguous(), V.expand(P, -1, -1, -1).contiguous()) for K, V in cache]
        lg = lg.expand(P, -1)
        rets = torch.empty((P, H), device=dev)
        for i in range(H):
            tok = torch.multinomial(F.softmax(lg, -1), 1, generator=gen)[:, 0]
            j = (torch.rand(P, device=dev, generator=gen) * pool_len[tok]).long()
            rets[:, i] = pool_mat[tok, j]
            if i < H - 1:
                lg, cache = step(model, tok, ctx_len + i, cache)
        path = torch.cumsum(rets * float(s1[t]), 1).cpu().numpy()        # log-preço relativo ao fechamento de t
        fin = path[:, -1]
        Xt = X_take[t]
        up = (path >= Xt).argmax(1).astype(float); up[~(path >= Xt).any(1)] = np.inf
        dn = (path <= -Xt).argmax(1).astype(float); dn[~(path <= -Xt).any(1)] = np.inf
        long_out = np.where(up < dn, Xt, np.where(dn < up, -Xt, fin))
        real_fin = lc[t + H] - lc[t]
        real_rng = np.log(h_[t + 1:t + H + 1].max() / l_[t + 1:t + H + 1].min())
        rows.append({"t": int(t), "cone_largura": float(np.std(fin)), "cone_amplitude": float(np.median(path.max(1) - path.min(1))),
                     "cone_inclinacao": float(np.mean(fin)), "cone_p_sobe": float((fin > 0).mean()),
                     "cone_ev_compra": float(long_out.mean() / Xt), "ingenuo_largura": float(s1[t] * np.sqrt(H)),
                     "real_final": float(real_fin), "real_abs": float(abs(real_fin)), "real_amplitude": float(real_rng)})
        if (n + 1) % 100 == 0:
            log(f"  {n + 1}/{len(moments)} momentos ({(time.time() - t0) / 60:.1f} min)")
    df = pd.DataFrame(rows)
    if ci is not None:
        z = np.load(PROJECT_ROOT / "runs" / f"{exp['run_name']}_barreiras" / "rotulos.npz")
        df["real_barreira_compra"] = z[f"o{ci}_1"][df.t] / z[f"X{ci}"][df.t]
    df.to_csv(out_dir / f"cone_w{w:02d}.csv", index=False)

    res = {
        "largura_x_tamanho_real": spearman(df.cone_largura.values, df.real_abs.values),
        "ingenuo_x_tamanho_real": spearman(df.ingenuo_largura.values, df.real_abs.values),
        "amplitude_cone_x_amplitude_real": spearman(df.cone_amplitude.values, df.real_amplitude.values),
        "ingenuo_x_amplitude_real": spearman(df.ingenuo_largura.values, df.real_amplitude.values),
        "inclinacao_x_direcao_real": spearman(df.cone_inclinacao.values, df.real_final.values),
        "acerto_direcao_20pct_mais_inclinados": None,
    }
    q = df.cone_inclinacao.abs().quantile(0.8)
    top = df[df.cone_inclinacao.abs() >= q]
    res["acerto_direcao_20pct_mais_inclinados"] = float((np.sign(top.cone_inclinacao) == np.sign(top.real_final)).mean())
    if "real_barreira_compra" in df:
        res["ev_compra_cone_x_resultado_real"] = spearman(df.cone_ev_compra.values, df.real_barreira_compra.values)
    for k, v in res.items():
        log(f"  {k}: {v:+.3f}" if v is not None else f"  {k}: —")
    (out_dir / f"cone_w{w:02d}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"concluído em {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())

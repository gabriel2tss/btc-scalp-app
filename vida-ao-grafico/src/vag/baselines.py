"""Baselines obrigatórios (seção 7 do briefing), nos MESMOS dados, horizontes e custos.

1. aleatório            — mesma quantidade de operações, direção sorteada
2. persistência/reversão — segue ou inverte a tendência de 15 min acima de um limiar
3. estatístico direto    — gradient boosting nas features cruas
4. estatístico + dialeto — o mesmo, com surpresa/entropia do LM e o significado das frases

A comparação 3 × 4 responde a pergunta central: o dialeto acrescenta informação
além dos números crus?
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .evaluate import metrics, simulate

QUANTILES = [0.90, 0.95, 0.98, 0.99, 0.995]


def pick_on_val(candidates, val_idx, fwd_by_h, horizons, cost_rt, min_trades=30):
    """candidates: lista de (nome_param, h, sinal_completo). Escolhe o maior total líquido na validação."""
    best = None
    for name, h, sig in candidates:
        m = metrics(simulate(sig, fwd_by_h[h], val_idx, h, cost_rt))
        if m["trades"] < min_trades:
            continue
        if best is None or m["total_net_pct"] > best[3]["total_net_pct"]:
            best = (name, h, sig, m)
    return best


def momentum_candidates(feats: np.ndarray, col: dict, train_idx, horizons):
    tr = feats[:, col["trend_15"]]
    out = []
    for qv in QUANTILES:
        thr = np.nanquantile(np.abs(tr[train_idx]), qv)
        base = np.where(np.abs(tr) > thr, np.sign(tr), 0).astype(np.int8)
        for h in horizons:
            out.append((f"momentum_q{qv}", h, base))
            out.append((f"reversao_q{qv}", h, (-base).astype(np.int8)))
    return out


def gbm_candidates(X: np.ndarray, fwd_by_h: dict, fit_idx: np.ndarray, horizons, cfg: dict, seed: int, tag: str, log=print):
    """Treina um GBM por horizonte (regressão do retorno futuro) e gera sinais por limiar de confiança."""
    out = []
    sub = fit_idx[::3]   # amostras vizinhas são quase iguais; subamostrar acelera sem perder informação
    for h in horizons:
        y = fwd_by_h[h][sub]
        ok = np.isfinite(y) & np.isfinite(X[sub]).all(1)
        clip = np.nanquantile(np.abs(y[ok]), 0.99)
        model = HistGradientBoostingRegressor(max_iter=cfg["gbm_max_iter"], learning_rate=cfg["gbm_learning_rate"],
                                              max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
                                              random_state=seed)
        model.fit(X[sub][ok], np.clip(y[ok], -clip, clip))
        pred = np.full(len(X), np.nan)
        good = np.isfinite(X).all(1)
        pred[good] = model.predict(X[good])
        for qv in QUANTILES:
            thr = np.nanquantile(np.abs(pred[fit_idx]), qv)
            sig = np.where(np.abs(pred) > thr, np.sign(pred), 0).astype(np.int8)
            out.append((f"{tag}_q{qv}", h, sig))
        log(f"    {tag} h={h} treinado")
    return out


def random_baseline(n_trades: int, test_idx, fwd, h, cost_rt, seed, reps=200) -> dict:
    """Distribuição do resultado de operar ao acaso o mesmo número de vezes."""
    rng = np.random.default_rng(seed)
    ok = test_idx[np.isfinite(fwd[test_idx])]
    if n_trades == 0 or len(ok) == 0:
        return {}
    means = []
    for _ in range(reps):
        t = rng.choice(ok, min(n_trades, len(ok)), replace=False)
        d = rng.choice([-1, 1], len(t))
        means.append(np.mean(d * fwd[t] - cost_rt))
    means = np.array(means) * 1e4
    return {"mean_net_bps_avg": float(means.mean()), "p95_bps": float(np.quantile(means, 0.95))}

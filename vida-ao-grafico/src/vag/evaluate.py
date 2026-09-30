"""Fases 5 e 6 — dicionário de significados, simulação de operações e métricas.

Regras de honestidade:
- Dicionário e parâmetros são aprendidos no TREINO e escolhidos na VALIDAÇÃO
  (fatia final do treino). O TESTE só é usado uma vez, no fim.
- Operação executável: decide no fechamento de t, entra na abertura de t+1,
  sai no fechamento de t+h, uma posição por vez, custos de ida e volta descontados.
- Significância com n efetivo (ocorrências que não se sobrepõem no horizonte) e
  correção de Benjamini–Hochberg para as milhares de frases testadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------- frases ----------

def phrase_keys(phrases: np.ndarray, vocab_size: int) -> dict[int, np.ndarray]:
    """{n: chave int64 da frase com as n últimas palavras terminando em t} (-1 = indisponível)."""
    V = np.int64(vocab_size + 1)
    out = {}
    key = np.zeros(len(phrases), dtype=np.int64)
    ok = np.ones(len(phrases), dtype=bool)
    for n in range(phrases.shape[1]):
        w = phrases[:, n].astype(np.int64)
        ok &= w >= 0
        key = key * V + (w + 1)
        out[n + 1] = np.where(ok, key, -1)
    return out


def build_dictionary(keys: np.ndarray, fwd: np.ndarray, idx: np.ndarray, horizon: int) -> pd.DataFrame:
    """Estatísticas do que acontece depois de cada frase, nas posições `idx`."""
    k, r = keys[idx], fwd[idx]
    ok = (k >= 0) & np.isfinite(r)
    df = pd.DataFrame({"key": k[ok], "r": r[ok], "t": idx[ok]}).sort_values(["key", "t"], kind="stable")
    new_occ = (df["key"].values != np.roll(df["key"].values, 1)) | (np.diff(df["t"].values, prepend=-10**9) >= horizon)
    df["new"] = new_occ
    df["up"] = df["r"] > 0
    g = df.groupby("key")
    d = pd.DataFrame({"count": g["r"].size(), "mean": g["r"].mean(), "std": g["r"].std(ddof=1),
                      "hit_up": g["up"].mean(), "n_eff": g["new"].sum()})
    se = d["std"] / np.sqrt(d["n_eff"].clip(lower=1))
    d["t_stat"] = d["mean"] / se.replace(0, np.nan)
    d["p"] = 2 * norm.sf(d["t_stat"].abs())
    return d


def bh_select(p: pd.Series, q: float) -> pd.Series:
    """Benjamini–Hochberg: máscara das hipóteses aprovadas com FDR q."""
    p = p.fillna(1.0)
    m = len(p)
    if m == 0:
        return p.astype(bool)
    order = np.argsort(p.values)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p.values[order] <= thresh
    kmax = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    sel = np.zeros(m, dtype=bool)
    sel[order[:kmax]] = True
    return pd.Series(sel, index=p.index)


# ---------- operações ----------

def simulate(signal: np.ndarray, fwd: np.ndarray, idx: np.ndarray, horizon: int, cost_rt: float) -> pd.DataFrame:
    """signal[t] em {-1,0,+1}. Opera só em posições `idx` (ordenadas), uma posição por vez."""
    cand = idx[(signal[idx] != 0) & np.isfinite(fwd[idx])]
    trades, busy_until = [], -1
    for t in cand:
        if t <= busy_until:
            continue
        d = signal[t]
        trades.append((t, d, d * fwd[t] - cost_rt))
        busy_until = t + horizon
    return pd.DataFrame(trades, columns=["t", "dir", "net"])


def metrics(tr: pd.DataFrame) -> dict:
    n = len(tr)
    if n == 0:
        return {"trades": 0, "mean_net_bps": None, "hit": None, "pf": None, "total_net_pct": 0.0, "t_stat": None}
    x = tr["net"].values
    gains, losses = x[x > 0].sum(), -x[x < 0].sum()
    sd = x.std(ddof=1) if n > 1 else np.nan
    return {
        "trades": int(n),
        "mean_net_bps": float(x.mean() * 1e4),
        "hit": float((x > 0).mean()),
        "pf": float(gains / losses) if losses > 0 else None,
        "total_net_pct": float(x.sum() * 100),
        "t_stat": float(x.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else None,
    }


def dictionary_signal(keys_by_n: dict[int, np.ndarray], dicts: dict[int, pd.DataFrame], min_count: int,
                      q: float, min_edge: float, n_max: int, length: int) -> tuple[np.ndarray, int]:
    """Sinal a partir das frases aprovadas; a frase mais longa (mais específica) tem prioridade."""
    sig = np.zeros(length, dtype=np.int8)
    decided = np.zeros(length, dtype=bool)
    n_sel = 0
    for n in range(n_max, 0, -1):
        d = dicts[n]
        d = d[d["count"] >= min_count]
        if d.empty:
            continue
        sel = bh_select(d["p"], q) & (d["mean"].abs() > min_edge)
        d = d[sel]
        n_sel += len(d)
        if d.empty:
            continue
        k = keys_by_n[n]
        pos = pd.Index(d.index).get_indexer(k)
        hit = (pos >= 0) & ~decided
        sig[hit] = np.sign(d["mean"].values[pos[hit]]).astype(np.int8)
        decided |= hit
    return sig, n_sel


# ---------- contexto ----------

def session_of(times: pd.Series) -> np.ndarray:
    h = pd.to_datetime(times, utc=True).dt.hour.values
    return np.select([h < 7, h < 13, h < 21], ["asia", "londres", "nova_york"], "fechamento")


def breakdown(tr: pd.DataFrame, labels: np.ndarray) -> dict:
    if tr.empty:
        return {}
    lab = labels[tr["t"].values]
    return {str(k): metrics(tr[lab == k]) for k in pd.unique(lab)}

"""Barreiras (triplo limite): rótulos, features de derivativos e simulação das operações.

Rótulo de cada minuto t, para compra e para venda, numa configuração (a, r, H):
  take X = max(a x volatilidade esperada em H, mínimo), stop Y = X / r, tempo máximo H minutos.
  Entrada na abertura de t+1. Do minuto t+1 ao t+H, com máximas e mínimas de 1 min:
    toca o take -> +X; toca o stop -> -Y; os dois no mesmo minuto -> stop (conservador);
    nenhum -> sai no fechamento de t+H.
  Resultado em log-retorno; o modelo aprende em "unidades de risco" (resultado / X).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VOL_WIN = 240                 # volatilidade de 1 min estimada nas últimas 4 h
CONFIGS = [(a, r, H) for H in (60, 240) for a in (1.0, 1.5, 2.0) for r in (1, 2)]
TAKE, STOP, TIME = 1, 2, 3

# custos de referência (seleção na validação; a nota final vem da execução com ticks)
COST_ENTRY = 0.0004           # ~35% das entradas como maker (0,02%) e o resto cruzando o livro (0,05%)
COST_EXIT = {TAKE: 0.0002, STOP: 0.0006, TIME: 0.0006}   # take = limitada; stop e tempo = mercado + derrapagem
MIN_TAKE = 3 * 0.0008         # take >= 3x o custo típico de ida e volta (~8 bps)


def vol_1m(close: np.ndarray) -> np.ndarray:
    r = pd.Series(np.log(close)).diff()
    return r.rolling(VOL_WIN, min_periods=VOL_WIN).std().to_numpy()


def barrier_outcomes(o, h, l, c, sig1, a, r, H, contiguous_until, min_take=MIN_TAKE):
    """Resultado (log), minuto de saída relativo (k) e tipo, para compra e venda, em cada t."""
    n = len(c)
    X = np.maximum(a * sig1 * np.sqrt(H), min_take)
    Y = X / r
    le = np.full(n, np.nan)
    le[:-1] = np.log(o[1:])
    lh, ll, lc = np.log(h), np.log(l), np.log(c)
    res = {}
    for side in (1, -1):
        out = np.full(n, np.nan)
        kk = np.zeros(n, dtype=np.int16)
        typ = np.zeros(n, dtype=np.int8)
        und = np.isfinite(le) & np.isfinite(X) & (np.arange(n) + H < contiguous_until)
        tp = le + side * X          # nível do take
        sl = le - side * Y          # nível do stop
        for k in range(1, H + 1):
            m = n - k
            hk, lk = lh[k:], ll[k:]
            if side > 0:
                take, stop = hk >= tp[:m], lk <= sl[:m]
            else:
                take, stop = lk <= tp[:m], hk >= sl[:m]
            u = und[:m]
            s_new = u & stop
            t_new = u & take & ~stop
            out[:m][s_new] = -Y[:m][s_new]
            out[:m][t_new] = X[:m][t_new]
            kk[:m][s_new | t_new] = k
            typ[:m][s_new] = STOP
            typ[:m][t_new] = TAKE
            und[:m] &= ~(s_new | t_new)
        m = n - H
        rest = und[:m]
        out[:m][rest] = side * (lc[H:][rest] - le[:m][rest])
        kk[:m][rest] = H
        typ[:m][rest] = TIME
        res[side] = (out, kk, typ)
    return X, Y, res


def contiguous_until(times_ns: np.ndarray) -> np.ndarray:
    """Para cada t, o primeiro índice onde a sequência de minutos quebra (rótulos não atravessam buracos)."""
    n = len(times_ns)
    brk = np.r_[np.diff(times_ns) != 60_000_000_000, True]      # brk[i]: entre i e i+1 há buraco
    nxt = np.full(n, n)
    pos = np.flatnonzero(brk)
    idx = np.searchsorted(pos, np.arange(n))
    ok = idx < len(pos)
    nxt[ok] = pos[idx[ok]] + 1
    return nxt


def derivatives_features(times: pd.DatetimeIndex, metrics: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Interesse aberto, posicionamento e funding, alinhados por minuto só com o que já era conhecido
    (registro de 5 min usado a partir de 5 min depois; funding: última cobrança já ocorrida)."""
    m = metrics.copy()
    m["time"] = pd.to_datetime(m["time"], utc=True) + pd.Timedelta(minutes=5)
    oi = m["sum_open_interest"].where(m["sum_open_interest"] > 0)
    lo = np.log(oi)
    day = 288
    f = pd.DataFrame({"time": m["time"]})
    f["oi_chg_1h"] = lo.diff(12)
    f["oi_chg_4h"] = lo.diff(48)
    f["oi_chg_z"] = (f["oi_chg_1h"] - f["oi_chg_1h"].rolling(day, min_periods=day // 2).mean()) / \
        f["oi_chg_1h"].rolling(day, min_periods=day // 2).std()
    for col, name in (("sum_toptrader_long_short_ratio", "ls_top_z"), ("count_long_short_ratio", "ls_all_z"),
                      ("sum_taker_long_short_vol_ratio", "taker_ratio_z")):
        x = np.log(m[col].where(m[col] > 0))
        f[name] = (x - x.rolling(day, min_periods=day // 2).mean()) / x.rolling(day, min_periods=day // 2).std()
    fr = funding.copy()
    fr["time"] = pd.to_datetime(fr["ms"], unit="ms", utc=True)
    left = pd.DataFrame({"time": pd.DatetimeIndex(times).as_unit("ns")})
    f["time"] = pd.DatetimeIndex(f["time"]).as_unit("ns")
    fr["time"] = pd.DatetimeIndex(fr["time"]).as_unit("ns")
    out = pd.merge_asof(left, f.sort_values("time"), on="time", direction="backward", tolerance=pd.Timedelta(minutes=30))
    out = pd.merge_asof(out, fr[["time", "rate"]].sort_values("time").rename(columns={"rate": "funding_last"}),
                        on="time", direction="backward", tolerance=pd.Timedelta(hours=9))
    out["funding_last"] = out["funding_last"] * 1e4
    cols = ["oi_chg_1h", "oi_chg_4h", "oi_chg_z", "ls_top_z", "ls_all_z", "taker_ratio_z", "funding_last"]
    return out[cols].replace([np.inf, -np.inf], np.nan).clip(-8, 8).astype(np.float32)


def funding_cum(times_ns: np.ndarray, funding: pd.DataFrame) -> np.ndarray:
    """F[i] = soma das taxas de funding cobradas até o fim do minuto i (para custo de posição)."""
    fm = (funding["ms"].to_numpy().astype(np.int64) * 1_000_000)
    idx = np.searchsorted(times_ns, fm, side="right") - 1       # minuto em que a cobrança acontece
    acc = np.zeros(len(times_ns))
    ok = (idx >= 0) & (idx < len(times_ns))
    np.add.at(acc, idx[ok], funding["rate"].to_numpy()[ok])
    return np.cumsum(acc)


def candle_sim(side, idx, outcomes, fcum, cost_mult=1.0, cost_rt=None):
    """Uma posição por vez. side[t] em {-1, 0, +1}; outcomes = {1: (out, k, typ), -1: (...)}.
    cost_rt: custo de ida e volta por minuto (ex.: spread do ouro); sem ele, custos de referência do BTC."""
    cand = idx[side[idx] != 0]
    rows, busy = [], -1
    for t in cand:
        if t <= busy:
            continue
        d = int(side[t])
        out, kk, typ = outcomes[d][0][t], int(outcomes[d][1][t]), int(outcomes[d][2][t])
        if not np.isfinite(out) or kk == 0:
            continue
        fund = d * (fcum[t + kk] - fcum[t])
        base = cost_rt[t] if cost_rt is not None else COST_ENTRY + COST_EXIT[typ]
        net = out - cost_mult * (base + fund)
        rows.append((t, d, out, net, typ, kk))
        busy = t + kk
    return pd.DataFrame(rows, columns=["t", "dir", "gross", "net", "tipo", "k"])

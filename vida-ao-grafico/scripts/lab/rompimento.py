"""Checagem exploratória: o tamanho do movimento é previsível? E dá para lucrar deixando o preço escolher o lado?

Previsão de tamanho SEM treino (só passado): volatilidade das últimas 60 velas de 1 min x raiz(60).
Rompimento (OCO): no fim do minuto t, compra-stop em close*e^{+k s} e venda-stop em close*e^{-k s}, válidas 60 min;
a primeira acionada vira a posição (a outra cancela); se as duas no mesmo minuto, descarta (ambíguo).
Após acionar: take = +2k s, stop = volta ao preço do meio (-k s), tempo máximo 60 min; mesmo minuto -> stop.
Custos: entrada stop = taker 0,05% + derrapagem 0,02%; take = maker 0,02%; stop/tempo = taker 0,05% + 0,01%.
Filtro de tamanho: s no top 10% dos últimos 30 dias (só passado). Uma operação por vez.
"""
import numpy as np
import pandas as pd
from vag.config import load_config
from vag.data.storage import read_symbol

b = read_symbol(load_config()["data_dir"], "m1", "BTCUSDT_UM").sort_values("time").reset_index(drop=True)
t = pd.to_datetime(b["time"], utc=True)
o, h, l, c = (b[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
lr = np.r_[np.nan, np.diff(np.log(c))]
s60 = pd.Series(lr).rolling(60).std().to_numpy() * np.sqrt(60)          # tamanho previsto (log)
fut = np.full(len(c), np.nan)
fut[:-60] = np.abs(np.log(c[60:] / c[:-60]))                             # tamanho realizado em 60 min
rng = np.full(len(c), np.nan)                                            # amplitude realizada em 60 min
hh = pd.Series(h[::-1]).rolling(60).max().to_numpy()[::-1]; ll = pd.Series(l[::-1]).rolling(60).min().to_numpy()[::-1]
rng[:-1] = np.log(hh[1:] / ll[1:])
m = (t >= "2021-01-01") & np.isfinite(s60) & np.isfinite(fut)
print("== 1) O TAMANHO é previsível? (2021-01 a 2026-09, sem treino)")
print(f"  correlação (Spearman) tamanho previsto x movimento real em 60 min: {pd.Series(s60[m]).corr(pd.Series(fut[m]), method='spearman'):.2f}")
print(f"  correlação tamanho previsto x amplitude (máx-mín) em 60 min:      {pd.Series(s60[m]).corr(pd.Series(rng[m]), method='spearman'):.2f}")
dec = pd.qcut(pd.Series(s60[m]), 10, labels=False)
tab = pd.DataFrame({"dec": dec.values, "fut": fut[m] * 1e4, "rng": rng[m] * 1e4})
g = tab.groupby("dec").agg(mov_mediano=("fut", "median"), amplitude_mediana=("rng", "median"),
                           p_mov_60bps=("fut", lambda x: (x > 60).mean()))
print(g.round(2).to_string())

thr = pd.Series(s60).rolling(43200, min_periods=10000).quantile(0.9).shift(1).to_numpy()
FEE = {"in": 0.0005 + 0.0002, "take": 0.0002, "mkt": 0.0006}
def oco(k, filt, t0, t1):
    idx = np.flatnonzero(filt & (t >= t0) & (t < t1) & np.isfinite(s60))
    rows, busy = [], -1
    for i in idx:
        if i <= busy or i + 125 >= len(c):
            continue
        s = s60[i]; up, dn = c[i] * np.exp(k * s), c[i] * np.exp(-k * s)
        hi, lo = h[i + 1:i + 61], l[i + 1:i + 61]
        ju, jd = np.flatnonzero(hi >= up), np.flatnonzero(lo <= dn)
        fu, fd = (ju[0] if len(ju) else 999), (jd[0] if len(jd) else 999)
        if fu == fd == 999:
            busy = i + 60; continue
        if fu == fd:
            busy = i + 1 + fu; continue
        d, E, j = (1, up, fu) if fu < fd else (-1, dn, fd)
        e0 = i + 1 + j
        tp, sl = E * np.exp(d * 2 * k * s), E * np.exp(-d * k * s)
        H2, L2 = h[e0 + 1:e0 + 61], l[e0 + 1:e0 + 61]
        if d > 0:
            it, is_ = np.flatnonzero(H2 >= tp), np.flatnonzero(L2 <= sl)
        else:
            it, is_ = np.flatnonzero(L2 <= tp), np.flatnonzero(H2 >= sl)
        a_t, a_s = (it[0] if len(it) else 999), (is_[0] if len(is_) else 999)
        # o próprio minuto da entrada também pode tocar o stop (conservador)
        if (d > 0 and l[e0] <= sl) or (d < 0 and h[e0] >= sl):
            a_s = -1
        if a_s <= a_t and a_s < 999:
            px, fee, ex = sl, FEE["mkt"], e0 + max(a_s, 0) + 1
        elif a_t < 999:
            px, fee, ex = tp, FEE["take"], e0 + a_t + 1
        else:
            px, fee, ex = c[e0 + 60], FEE["mkt"], e0 + 60
        g_ = d * np.log(px / E)
        rows.append((i, g_, g_ - FEE["in"] - fee))
        busy = ex
    return pd.DataFrame(rows, columns=["i", "gross", "net"])

print("\n== 2) ROMPIMENTO (o preço escolhe o lado), 60 min, custos de ordem a mercado")
for k in (0.5, 1.0):
    for nome, filt in (("todos os minutos", np.ones(len(c), bool)), ("só tamanho previsto no top 10%", s60 > thr)):
        tr = oco(k, filt, "2021-01-01", "2026-09-01")
        tr["ano"] = t[tr["i"]].dt.year.values
        yr = tr.groupby("ano")["net"].mean() * 1e4
        n = len(tr); mu = tr.net.mean() * 1e4; tt = tr.net.mean() / (tr.net.std(ddof=1) / np.sqrt(n))
        print(f"  k={k} {nome:32s}: {n:6d} op | bruto {tr.gross.mean() * 1e4:+6.1f} | líquido {mu:+6.1f} bps/op (t {tt:+.1f}) | "
              f"por ano: " + " ".join(f"{a}:{v:+.1f}" for a, v in yr.items()))

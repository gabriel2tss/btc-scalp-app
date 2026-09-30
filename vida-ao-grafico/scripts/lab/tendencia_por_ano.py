"""Seguir tendência no BTC, por ano (2020-2025 + último ano), a partir dos ticks. ATENÇÃO: o % é soma de
retornos diários (aproximação); para dólares reais com alavancagem e liquidação use simulador_binance.py.

Regras IDÊNTICAS às do teste do último ano (tendencia_btc.py), sem nenhuma mudança:
  dia UTC; velas diárias dos aggTrades; sinal no fechamento de D, posição em D+1 aberta no 1º negócio do dia;
  custo 0,06% por lado a cada mudança; funding real; momentum N em {7,14,28,56} (compra/venda e só compra),
  momentum 28d semanal, fluxo de ticks 7d, comprar e segurar.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from vag.config import load_config

cfg = load_config()
dd = cfg["data_dir"]
cache = dd / "tick_features" / "btc_diario_dos_ticks.parquet"
if cache.exists():
    D = pd.read_parquet(cache)
else:
    parts = []
    for p in pd.period_range("2020-01", "2026-09", freq="M"):
        f = dd / "trades" / "symbol=BTCUSDT_UM" / f"year={p.year:04d}" / f"{p.year:04d}-{p.month:02d}.parquet"
        for b in pq.ParquetFile(f).iter_batches(batch_size=4_000_000, columns=["time", "price", "qty", "is_buyer_maker"]):
            tc = b.column("time")
            ms = tc.cast(pa.timestamp("ms", tz=tc.type.tz), safe=False).cast(pa.int64()).to_numpy()
            px, q = b.column("price").to_numpy(), b.column("qty").to_numpy()
            buy = ~b.column("is_buyer_maker").to_numpy(zero_copy_only=False)
            if not (ms[1:] >= ms[:-1]).all():
                o = np.argsort(ms, kind="stable"); ms, px, q, buy = ms[o], px[o], q[o], buy[o]
            day = ms // 86_400_000
            st = np.flatnonzero(np.r_[True, day[1:] != day[:-1]]); en = np.r_[st[1:], len(day)]
            parts.append(pd.DataFrame({"day": day[st], "open": px[st], "close": px[en - 1], "first_ms": ms[st], "last_ms": ms[en - 1],
                                       "buy": np.add.reduceat(np.where(buy, q, 0.0), st), "sell": np.add.reduceat(np.where(buy, 0.0, q), st),
                                       "n": np.diff(np.r_[st, len(day)])}))
        print("  ticks:", p, flush=True)
    raw = pd.concat(parts).sort_values(["day", "first_ms"])
    g = raw.groupby("day")
    D = pd.DataFrame({"open": g.open.first(), "close": raw.sort_values(["day", "last_ms"]).groupby("day").close.last(),
                      "buy": g.buy.sum(), "sell": g.sell.sum(), "n": g.n.sum()})
    D.index = pd.to_datetime(D.index * 86_400_000, unit="ms", utc=True)
    D.to_parquet(cache)
print(f"{len(D)} dias dos ticks ({int(D.n.sum()):,} negócios), {D.index[0]:%Y-%m-%d} a {D.index[-1]:%Y-%m-%d}")

fr = pd.read_parquet(dd / "funding" / "BTCUSDT_UM.parquet")
fund = fr.assign(day=pd.to_datetime(fr.ms, unit="ms", utc=True).dt.floor("D")).groupby("day").rate.sum().reindex(D.index).fillna(0.0)
day_ret = np.log(D.close / D.open)
COST = 0.0006

sig = {}
for N in (7, 14, 28, 56):
    m = np.sign(np.log(D.close / D.close.shift(N)))
    sig[f"momentum {N}d"] = m.shift(1)
    sig[f"momentum {N}d só compra"] = m.clip(lower=0).shift(1)
m28 = np.sign(np.log(D.close / D.close.shift(28)))
sig["momentum 28d semanal"] = m28.where(D.index.dayofweek == 0).ffill().shift(1)
flow = np.sign((D.buy - D.sell).rolling(7).sum())
sig["fluxo ticks 7d"] = flow.shift(1)
sig["comprar e segurar"] = pd.Series(1.0, index=D.index)

periods = {"2020": ("2020-03-01", "2021-01-01"), "2021": ("2021-01-01", "2022-01-01"), "2022": ("2022-01-01", "2023-01-01"),
           "2023": ("2023-01-01", "2024-01-01"), "2024": ("2024-01-01", "2025-01-01"), "2025": ("2025-01-01", "2026-01-01"),
           "último ano": ("2025-09-29", "2026-09-29")}
btc = {k: (D.close[(D.index >= a) & (D.index < b)].iloc[-1] / D.close[(D.index >= a) & (D.index < b)].iloc[0] - 1) * 100
       for k, (a, b) in periods.items()}

def net_of(s):
    pos = s.reindex(D.index).fillna(0.0)
    chg = pos.diff().abs().fillna(pos.abs())
    return pos * day_ret - chg * COST - pos * fund

tab, sh, dds = {}, {}, {}
for name, s in sig.items():
    net = net_of(s)
    row = {}
    for k, (a, b) in periods.items():
        x = net[(net.index >= pd.Timestamp(a, tz="UTC")) & (net.index < pd.Timestamp(b, tz="UTC"))]
        row[k] = round(x.sum() * 100, 1)
    tab[name] = row
    x = net[(net.index >= "2021-01-01") & (net.index < "2026-09-29")]
    sh[name] = round(x.mean() / x.std() * np.sqrt(365), 2)
    eq = x.cumsum(); dds[name] = round((eq - eq.cummax()).min() * 100, 1)
T = pd.DataFrame(tab).T
yrs = ["2020", "2021", "2022", "2023", "2024", "2025"]
T["anos positivos"] = [f"{int((T.loc[n, yrs].dropna() > 0).sum())}/{int(T.loc[n, yrs].notna().sum())}" for n in T.index]
T["Sharpe 2021-26"] = pd.Series(sh); T["pior queda 2021-26 (%)"] = pd.Series(dds)
pd.set_option("display.width", 250)
print("\nResultado por período (%, sem alavancagem, já com taxa e funding):")
print(T.to_string())
print("\nBTC no período (%):", {k: round(v, 1) for k, v in btc.items()})

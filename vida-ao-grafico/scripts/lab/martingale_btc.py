"""Martingale no BTC (futuro Binance), cada um dos últimos 12 meses ISOLADO (começa com US$ 100, termina no fim do mês).

Parâmetros FIXADOS antes de rodar:
 (A0) sem martingale: 1 operação por dia (abre no 1º negócio do dia, fecha no último), direção = momentum 14d,
      posição de US$ 100 (1x do capital inicial).
 (A)  martingale diário: igual, mas perdeu o dia -> dobra o tamanho no dia seguinte; ganhou -> volta a US$ 100.
 (B)  grid martingale (preço médio), minuto a minuto: abre US$ 100 na direção do momentum 14d do dia; a cada 1%
      contra o preço da última ordem, nova ordem com o DOBRO da anterior (até 10 níveis); fecha tudo quando o
      preço volta 0,5% a favor do preço médio; depois recomeça. Fim do mês: fecha o que estiver aberto.
 Custos: 0,06% por ordem (taker 0,05% + derrapagem 0,01%) sobre o valor; funding real às 00, 08 e 16 UTC.
 Liquidação (margem cruzada, US$ 100 na conta): saldo no pior preço <= 0,4% do valor total das posições -> zera.
"""
import numpy as np
import pandas as pd

from vag.config import load_config
from vag.data.storage import read_symbol

cfg = load_config(); dd = cfg["data_dir"]
FEE, MMR = 0.0006, 0.004
D = pd.read_parquet(dd / "tick_features" / "btc_diario_dos_ticks.parquet")
D.index = pd.DatetimeIndex(D.index).as_unit("ns")
sig = np.sign(np.log(D.close / D.close.shift(14))).shift(1)
b = read_symbol(dd, "m1", "BTCUSDT_UM", years=[2025, 2026])
b["time"] = pd.to_datetime(b["time"], utc=True)
b = b[b.time >= "2025-09-01"].reset_index(drop=True)
day = b.time.dt.floor("D")
hl = b.groupby(day).agg(high=("high", "max"), low=("low", "min"))
hl.index = pd.DatetimeIndex(hl.index).as_unit("ns")
D = D.join(hl)
fr = pd.read_parquet(dd / "funding" / "BTCUSDT_UM.parquet")
fr["t"] = pd.to_datetime(fr.ms, unit="ms", utc=True).dt.floor("min")
fmap = dict(zip(fr.t, fr.rate))
fund_day = fr.groupby(pd.DatetimeIndex(fr.t.dt.floor("D")).as_unit("ns")).rate.sum()
months = pd.period_range("2025-10", "2026-09", freq="M")


def daily(month, martingale):
    X = D[(D.index >= month.start_time.tz_localize("UTC")) & (D.index < (month.end_time + pd.Timedelta(1, "ns")).tz_localize("UTC"))]
    E, size, maxsize, busted = 100.0, 100.0, 100.0, None
    for d, r in X.iterrows():
        s = sig.get(d, np.nan)
        if not np.isfinite(s) or s == 0 or E <= 0:
            continue
        q = size / r.open
        E -= size * FEE
        worst = r.low if s > 0 else r.high
        if E + s * q * (worst - r.open) <= MMR * q * worst:
            busted = d; E = 0.0; break
        pnl = s * q * (r.close - r.open) - s * size * fund_day.get(d, 0.0)
        E += pnl - q * r.close * FEE
        maxsize = max(maxsize, size)
        size = size * 2 if (martingale and pnl < 0) else 100.0
    return E, busted, maxsize


def grid(month, step=0.01, tp=0.005, levels=10):
    a = month.start_time.tz_localize("UTC"); e = (month.end_time + pd.Timedelta(1, "ns")).tz_localize("UTC")
    M = b[(b.time >= a) & (b.time < e)]
    E = 100.0
    side, orders, busted, cycles, maxlvl = 0, [], None, 0, 0     # orders: (preço, quantidade)
    O, H, L, C, T = M.open.values, M.high.values, M.low.values, M.close.values, M.time.values
    for i in range(len(M)):
        t = pd.Timestamp(T[i]).tz_localize("UTC") if pd.Timestamp(T[i]).tz is None else pd.Timestamp(T[i])
        if side == 0:
            s = sig.get(t.floor("D"), np.nan)
            if not np.isfinite(s) or s == 0:
                continue
            side = int(s); q = 100.0 / O[i]; orders = [(O[i], q)]; E -= 100.0 * FEE
        rate = fmap.get(t)
        qty = sum(q for _, q in orders)
        if rate is not None:
            E -= side * qty * O[i] * rate
        # novas ordens do grid (preço andou contra)
        while len(orders) < levels:
            last_p, last_q = orders[-1]
            lvl_p = last_p * (1 - step) if side > 0 else last_p * (1 + step)
            hit = (L[i] <= lvl_p) if side > 0 else (H[i] >= lvl_p)
            if not hit:
                break
            nq = 2 * last_q; orders.append((lvl_p, nq)); E -= nq * lvl_p * FEE
        maxlvl = max(maxlvl, len(orders))
        qty = sum(q for _, q in orders); avg = sum(p * q for p, q in orders) / qty
        worst = L[i] if side > 0 else H[i]
        if E + side * qty * (worst - avg) <= MMR * qty * worst:
            busted = t; E = 0.0; break
        tgt = avg * (1 + tp) if side > 0 else avg * (1 - tp)
        if (H[i] >= tgt) if side > 0 else (L[i] <= tgt):
            E += side * qty * (tgt - avg) - qty * tgt * FEE
            side, orders, cycles = 0, [], cycles + 1
    if side != 0 and busted is None:
        qty = sum(q for _, q in orders); avg = sum(p * q for p, q in orders) / qty
        E += side * qty * (C[-1] - avg) - qty * C[-1] * FEE
    return E, busted, cycles, maxlvl


rows = []
for m in months:
    e0, b0, _ = daily(m, False)
    e1, b1, mx = daily(m, True)
    e2, b2, cyc, lvl = grid(m)
    rows.append({"mês": str(m),
                 "sem martingale (US$)": round(e0 - 100, 2),
                 "martingale diário (US$)": ("QUEBROU " + b1.strftime("%d/%m")) if b1 is not None else round(e1 - 100, 2),
                 "maior posição (US$)": round(mx),
                 "grid martingale (US$)": ("QUEBROU " + pd.Timestamp(b2).strftime("%d/%m")) if b2 is not None else round(e2 - 100, 2),
                 "ciclos do grid": cyc, "nível máx. do grid": lvl})
R = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(R.to_string(index=False))
num = lambda c: pd.to_numeric(R[c], errors="coerce")
for c in ("sem martingale (US$)", "martingale diário (US$)", "grid martingale (US$)"):
    x = num(c); q = R[c].astype(str).str.startswith("QUEBROU").sum()
    print(f"{c:26s}: meses com lucro {int((x > 0).sum())}/12 | quebrou {q}/12 | soma dos meses (quebra = -100) "
          f"{(x.fillna(-100)).sum():+.2f}")

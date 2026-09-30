"""Versão prática: US$ 100 na Binance (futuro BTCUSDT), alavancagem Lx, mesmas regras de tendência, mês a mês.

- Sinal decidido no fechamento de D (00:00 UTC), execução no 1º negócio de D+1 (abertura das velas dos ticks).
- A cada nova entrada: quantidade = L x saldo / preço; enquanto o sinal não muda, a quantidade fica fixa.
- Custos: 0,05% (taker) + 0,01% (derrapagem) sobre o valor da posição em cada abertura e fechamento;
  funding real (soma das cobranças do dia) sobre o valor da posição.
- Liquidação: se no pior preço do dia (mínima p/ comprado, máxima p/ vendido) o saldo ficar <= margem de
  manutenção (0,4% do valor da posição, faixa 1 da Binance), a conta é liquidada e vai a zero.
"""
import numpy as np
import pandas as pd

from vag.config import load_config
from vag.data.storage import read_symbol

cfg = load_config()
dd = cfg["data_dir"]
D = pd.read_parquet(dd / "tick_features" / "btc_diario_dos_ticks.parquet")
D.index = pd.DatetimeIndex(D.index).as_unit("ns")
# máxima e mínima do dia: das velas de 1 min (mesmos negócios dos ticks, max/min idênticos)
b = read_symbol(dd, "m1", "BTCUSDT_UM")
t = pd.to_datetime(b["time"], utc=True)
hl = b.groupby(t.dt.floor("D")).agg(high=("high", "max"), low=("low", "min"))
hl.index = pd.DatetimeIndex(hl.index).as_unit("ns")
D = D.join(hl)
fr = pd.read_parquet(dd / "funding" / "BTCUSDT_UM.parquet")
fd = pd.to_datetime(fr.ms, unit="ms", utc=True).dt.floor("D")
fund = fr.groupby(pd.DatetimeIndex(fd).as_unit("ns")).rate.sum().reindex(D.index).fillna(0.0)

# ---- sinais (fixados no teste de 30/09/2026) ----
sig = {}
for N in (14, 28):
    sig[f"momentum {N}d"] = np.sign(np.log(D.close / D.close.shift(N))).shift(1)
sig["fluxo ticks 7d"] = np.sign((D.buy - D.sell).rolling(7).sum()).shift(1)
sig["comprar e segurar"] = pd.Series(1.0, index=D.index)
FEE, MMR = 0.0006, 0.004


def simulate(s, lev, a, e, capital=100.0):
    s = s.reindex(D.index).fillna(0.0)
    X = D[(D.index >= pd.Timestamp(a, tz="UTC")) & (D.index < pd.Timestamp(e, tz="UTC"))]
    E, side, q, ref = capital, 0, 0.0, None
    fees = funding_paid = 0.0
    liq, log = None, []
    for day, r in X.iterrows():
        tgt = int(s.loc[day])
        if side != 0:                                    # marca a posição até a abertura do dia
            E += side * q * (r.open - ref)
        ref = r.open
        if tgt != side:
            if side != 0:
                c = q * r.open * FEE; E -= c; fees += c
            side, q = tgt, 0.0
            if tgt != 0 and E > 0:
                q = lev * E / r.open
                c = q * r.open * FEE; E -= c; fees += c
        if side != 0:
            f = side * q * r.open * fund.loc[day]; E -= f; funding_paid += f
            worst = r.low if side > 0 else r.high
            if E + side * q * (worst - r.open) <= MMR * q * worst:
                liq, E = day, 0.0
                log.append((day, E)); break
            E += side * q * (r.close - r.open)
            ref = r.close
        log.append((day, E))
    eq = pd.Series(dict(log))
    return eq, fees, funding_paid, liq


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="US$ 100 (ou --capital) na Binance com alavancagem, mês a mês")
    ap.add_argument("--estrategia", default="momentum 14d", choices=list(sig))
    ap.add_argument("--alavancagem", type=float, nargs="+", default=[1, 2, 3, 5, 10])
    ap.add_argument("--inicio", default="2025-10-01")
    ap.add_argument("--fim", default="2026-09-29")
    ap.add_argument("--capital", type=float, default=100.0)
    args = ap.parse_args()
    cols = {}
    for lev in args.alavancagem:
        eq, fees, fpaid, liq = simulate(sig[args.estrategia], lev, args.inicio, args.fim, args.capital)
        m = eq.groupby(eq.index.strftime("%Y-%m")).last()
        prev = pd.concat([pd.Series({"x": args.capital}), m]).shift(1).iloc[1:]
        cols[f"{lev:g}x mês"] = (m - prev).round(2)
        cols[f"{lev:g}x saldo"] = m.round(2)
        pk = eq.cummax()
        print(f"{lev:g}x: saldo final US$ {eq.iloc[-1]:.2f} | pior queda {((eq - pk) / pk).min() * 100:.1f}% | "
              f"taxas US$ {fees:.2f} | funding US$ {fpaid:.2f} | liquidada: {liq.strftime('%Y-%m-%d') if liq is not None else 'não'}")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(cols).fillna("liquidada").to_string())

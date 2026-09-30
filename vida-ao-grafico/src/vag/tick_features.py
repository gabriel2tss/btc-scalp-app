"""Experimento: microestrutura tick a tick (aggTrades) resumida por minuto.

A vela de 1 min já traz volume, nº de negócios e volume do comprador agressor.
Os ticks acrescentam o que a vela esconde:
- direção das ordens grandes (>= US$ 100 mil e >= US$ 1 milhão por ordem agressora)
- agressão por CONTAGEM de ordens (a vela só dá por volume)
- varreduras: quantos negócios cada ordem agressora gerou (consumiu vários níveis do livro)
- fluxo nos últimos 10 segundos do minuto
- onde o preço de fato negociou no minuto (VWAP) e o vai-e-vem intraminuto (variância realizada)

Mesmas regras de features.py: só janelas finitas para trás; a linha t usa ticks <= fim do minuto t.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .features import ATR_WIN, LONG_WIN, _roll_z

BIG_USD = 100_000
HUGE_USD = 1_000_000
LATE_SECONDS = 50          # segundos 50-59 do minuto

TICK_FEATURES = [
    "tk_cnt_imb", "tk_big_imb", "tk_big_share_z", "tk_big_imb_15", "tk_huge_imb_60",
    "tk_sweep_z", "tk_late_imb", "tk_vwap_dev", "tk_path_z", "tk_max_share",
]

_SUM_COLS = ["n_agg", "n_fill", "vol", "buy_vol", "n_buy", "pv", "big_buy", "big_sell",
             "huge_buy", "huge_sell", "late_signed", "rv"]


def _reduce(minute: np.ndarray, cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Soma por minuto (max para max_qty). `minute` precisa estar ordenado; usa reduceat (rápido)."""
    starts = np.flatnonzero(np.r_[True, minute[1:] != minute[:-1]])
    out = {k: (np.maximum if k == "max_qty" else np.add).reduceat(v, starts) for k, v in cols.items()}
    out["minute"] = minute[starts]
    return out


def _aggregate_batch(b: pa.RecordBatch) -> dict[str, np.ndarray]:
    tcol = b.column("time")
    ms = tcol.cast(pa.timestamp("ms", tz=tcol.type.tz), safe=False).cast(pa.int64()).to_numpy()
    price = b.column("price").to_numpy()
    qty = b.column("qty").to_numpy()
    buy = ~b.column("is_buyer_maker").to_numpy(zero_copy_only=False)   # comprador foi o agressor
    n_fill = b.column("n_trades").to_numpy().astype(np.float64)
    if len(ms) > 1 and not (ms[1:] >= ms[:-1]).all():
        o = np.argsort(ms, kind="stable")
        ms, price, qty, buy, n_fill = ms[o], price[o], qty[o], buy[o], n_fill[o]
    minute = ms // 60_000
    notional = price * qty
    big, huge = notional >= BIG_USD, notional >= HUGE_USD
    late = (ms // 1000) % 60 >= LATE_SECONDS
    lp = np.log(price)
    d = np.diff(lp, prepend=lp[:1])
    d[np.r_[True, minute[1:] != minute[:-1]]] = 0.0      # não atravessa a fronteira do minuto
    return _reduce(minute, {
        "n_agg": np.ones(len(ms)), "n_fill": n_fill, "vol": qty,
        "buy_vol": np.where(buy, qty, 0.0), "n_buy": buy.astype(np.float64), "pv": notional,
        "big_buy": np.where(big & buy, qty, 0.0), "big_sell": np.where(big & ~buy, qty, 0.0),
        "huge_buy": np.where(huge & buy, qty, 0.0), "huge_sell": np.where(huge & ~buy, qty, 0.0),
        "late_signed": np.where(late, np.where(buy, qty, -qty), 0.0), "rv": d * d, "max_qty": qty,
    })


def aggregate_ticks(files: list[Path], log=print) -> pd.DataFrame:
    """Lê os Parquet de aggTrades em blocos e devolve uma linha por minuto (índice = início do minuto, UTC)."""
    parts = []
    for f in files:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=4_000_000, columns=["time", "price", "qty", "is_buyer_maker", "n_trades"]):
            if b.num_rows:
                parts.append(_aggregate_batch(b))
        log(f"  ticks resumidos: {f.name}")
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    minute = cat.pop("minute")
    if not (minute[1:] >= minute[:-1]).all():
        o = np.argsort(minute, kind="stable")
        minute, cat = minute[o], {k: v[o] for k, v in cat.items()}
    red = _reduce(minute, cat)                 # junta minutos que caíram em dois blocos
    idx = pd.to_datetime(red.pop("minute") * 60_000, unit="ms", utc=True)
    return pd.DataFrame(red, index=pd.DatetimeIndex(idx, name="time"))[_SUM_COLS + ["max_qty"]]


def compute_tick_features(bars: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """bars: velas M1 (time, open, high, low, close). agg: saída de aggregate_ticks. Alinha pelas velas."""
    t = pd.DatetimeIndex(pd.to_datetime(bars["time"], utc=True)).as_unit("ns")
    a = agg.set_axis(pd.DatetimeIndex(agg.index).as_unit("ns")).reindex(t)
    a.index = bars.index
    h, l, c = (bars[k].astype("float64") for k in ("high", "low", "close"))
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_WIN, min_periods=ATR_WIN).mean().replace(0, np.nan)

    def neutral(x: pd.Series, value: float = 0.0) -> pd.Series:
        # um minuto sem ticks (ou sem amplitude) não pode apagar 1 dia inteiro de _roll_z(min_periods=1440)
        return x.replace([np.inf, -np.inf], np.nan).fillna(value)

    vol = a["vol"].replace(0, np.nan)
    f = pd.DataFrame(index=bars.index)
    f["tk_cnt_imb"] = 2 * a["n_buy"] / a["n_agg"].replace(0, np.nan) - 1
    big_net = (a["big_buy"] - a["big_sell"]).fillna(0.0)
    vol0 = a["vol"].fillna(0.0)
    f["tk_big_imb"] = big_net / vol
    f["tk_big_share_z"] = _roll_z(neutral((a["big_buy"] + a["big_sell"]) / vol), LONG_WIN)
    f["tk_big_imb_15"] = big_net.rolling(15).sum() / vol0.rolling(15).sum().replace(0, np.nan)
    f["tk_huge_imb_60"] = (a["huge_buy"] - a["huge_sell"]).fillna(0.0).rolling(60).sum() / vol0.rolling(60).sum().replace(0, np.nan)
    f["tk_sweep_z"] = _roll_z(neutral(np.log(a["n_fill"] / a["n_agg"].replace(0, np.nan))), LONG_WIN)
    f["tk_late_imb"] = a["late_signed"] / vol
    f["tk_vwap_dev"] = (c - a["pv"] / vol) / atr
    # caminho percorrido x amplitude: minuto "picotado" (vai-e-vem) x minuto limpo
    rng = np.log(h / l).replace(0, np.nan)
    f["tk_path_z"] = _roll_z(neutral(np.log(np.sqrt(a["rv"]) / rng)), LONG_WIN)
    f["tk_max_share"] = a["max_qty"] / vol
    f.loc[a["vol"].isna(), :] = np.nan        # minuto sem ticks: linha inválida (valid_mask a descarta)
    f = f[TICK_FEATURES].replace([np.inf, -np.inf], np.nan).clip(-8, 8).astype("float32")
    # checagem: os ticks precisam bater com a vela (volume e volume comprador)
    ok = a["vol"].notna() & (bars["volume"] > 0)
    rel = ((a["vol"] - bars["volume"]).abs() / bars["volume"])[ok]
    rel_buy = ((a["buy_vol"] - bars["taker_buy_volume"]).abs() / bars["volume"])[ok]
    f.attrs["check"] = {"minutes_with_ticks": int(ok.sum()),
                        "vol_mismatch_p99": float(rel.quantile(0.99)) if len(rel) else None,
                        "buy_mismatch_p99": float(rel_buy.quantile(0.99)) if len(rel_buy) else None}
    return f

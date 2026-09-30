"""Fase 2 — o "estado rico" de cada minuto.

Regras:
- Só janelas FINITAS voltadas para trás (rolling), nada de EWM com memória infinita.
  Assim, calcular sobre um buffer das últimas LOOKBACK velas dá exatamente o mesmo
  resultado que sobre o histórico inteiro: o código do backtest é o código ao vivo.
- Tudo relativo à volatilidade recente, para comparar épocas diferentes.
- A linha t usa apenas velas <= t (a vela t já fechou no momento da decisão).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ATR_WIN = 60          # volatilidade de curto prazo (1h)
LONG_WIN = 1440       # 1 dia
# Maior cadeia de dependência: rolling(1440) de algo que já depende de rolling(60)... + folga.
LOOKBACK = LONG_WIN + ATR_WIN + 300

FEATURES = [
    # forma da vela (relativa ao ATR)
    "ret", "body", "upper_wick", "lower_wick", "range",
    # participação e agressão
    "vol_z", "trades_z", "trade_size_z", "taker_imb", "taker_imb_15",
    # regime
    "vol_regime", "trend_15", "trend_60", "trend_240",
    # posição no contexto
    "pos_60", "pos_240", "pos_1440", "pos_day",
    # tempo
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # qualidade do dado
    "gap",
]


def _roll_z(x: pd.Series, win: int) -> pd.Series:
    m = x.rolling(win, min_periods=win).mean()
    s = x.rolling(win, min_periods=win).std()
    # série constante na janela (ex.: ouro, onde volume e nº de negócios são a mesma contagem de ticks) -> z = 0
    return ((x - m) / s.replace(0, np.nan)).where(s != 0, 0.0)


def _pos(close: pd.Series, high: pd.Series, low: pd.Series, win: int) -> pd.Series:
    hh = high.rolling(win, min_periods=win).max()
    ll = low.rolling(win, min_periods=win).min()
    return ((close - ll) / (hh - ll).replace(0, np.nan)) * 2 - 1   # -1 = fundo, +1 = topo


def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    """bars: time (UTC), open, high, low, close, volume, trades, taker_buy_volume. Ordenado por time."""
    o, h, l, c = (bars[k].astype("float64") for k in ("open", "high", "low", "close"))
    t = pd.to_datetime(bars["time"], utc=True)
    prev_c = c.shift(1)

    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_WIN, min_periods=ATR_WIN).mean().replace(0, np.nan)
    atr_long = tr.rolling(LONG_WIN, min_periods=LONG_WIN).mean().replace(0, np.nan)

    f = pd.DataFrame(index=bars.index)
    f["ret"] = (c - prev_c) / atr
    f["body"] = (c - o) / atr
    f["upper_wick"] = (h - np.maximum(o, c)) / atr
    f["lower_wick"] = (np.minimum(o, c) - l) / atr
    f["range"] = (h - l) / atr

    vol = bars["volume"].astype("float64")
    trades = bars["trades"].astype("float64")
    lv, lt = np.log1p(vol), np.log1p(trades)
    f["vol_z"] = _roll_z(lv, LONG_WIN)
    f["trades_z"] = _roll_z(lt, LONG_WIN)
    f["trade_size_z"] = _roll_z(lv - lt, LONG_WIN)
    buy = bars["taker_buy_volume"].astype("float64")
    f["taker_imb"] = (2 * buy / vol.replace(0, np.nan) - 1).fillna(0.0)
    f["taker_imb_15"] = (2 * buy.rolling(15).sum() / vol.rolling(15).sum().replace(0, np.nan) - 1)

    f["vol_regime"] = np.log(atr / atr_long)
    for n in (15, 60, 240):
        f[f"trend_{n}"] = (c - c.shift(n)) / (atr * np.sqrt(n))

    f["pos_60"] = _pos(c, h, l, 60)
    f["pos_240"] = _pos(c, h, l, 240)
    f["pos_1440"] = _pos(c, h, l, 1440)
    day = t.dt.floor("D")
    dh = h.groupby(day.values).cummax()
    dl = l.groupby(day.values).cummin()
    f["pos_day"] = (((c - dl) / (dh - dl).replace(0, np.nan)) * 2 - 1).fillna(0.0)

    hour = t.dt.hour + t.dt.minute / 60.0
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    dow = t.dt.dayofweek + hour / 24.0
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)

    dt_min = t.diff().dt.total_seconds() / 60.0
    f["gap"] = np.log1p((dt_min - 1).clip(lower=0).fillna(0.0))

    f = f[FEATURES].replace([np.inf, -np.inf], np.nan).clip(-8, 8).astype("float32")
    f.insert(0, "time", t.values)
    return f


def valid_mask(feats: pd.DataFrame) -> pd.Series:
    """Linhas com todas as features definidas (fora do aquecimento e de divisões por zero)."""
    return feats[FEATURES].notna().all(axis=1)


def forward_returns(bars: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Retorno executável: decide no fechamento de t, ENTRA na abertura de t+1, sai no fechamento de t+h.

    Só serve para rótulos/avaliação — nunca como feature. Exige minutos contíguos
    (se houver buraco no meio do horizonte, o rótulo vira NaN).
    """
    t = pd.to_datetime(bars["time"], utc=True).reset_index(drop=True)
    o = bars["open"].astype("float64").reset_index(drop=True)
    c = bars["close"].astype("float64").reset_index(drop=True)
    entry = o.shift(-1)
    out = pd.DataFrame({"time": t})
    for hz in horizons:
        exit_ = c.shift(-hz)
        contiguous = (t.shift(-hz) - t) == pd.Timedelta(minutes=hz)
        out[f"fwd_{hz}"] = np.where(contiguous, np.log(exit_ / entry), np.nan)
    return out

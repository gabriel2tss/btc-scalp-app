"""Acesso ao MetaTrader 5 e normalização dos dados para UTC.

Só roda no Windows com o terminal MT5 aberto e logado. Tudo que não depende do
terminal (normalização, conversão de fuso) fica em funções puras testáveis.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .server_time import server_to_utc


class Source(Protocol):
    def rates_range(self, symbol: str, start: datetime, end: datetime) -> np.ndarray | None: ...
    def ticks_range(self, symbol: str, start: datetime, end: datetime) -> np.ndarray | None: ...
    def symbol_info(self, symbol: str) -> dict | None: ...
    def last_tick_time(self, symbol: str) -> float | None: ...
    def last_error(self) -> Any: ...


class MT5Source:
    """Implementação real, usando o pacote `MetaTrader5`."""

    def __init__(self, path: str | None = None, login: int | None = None,
                 password: str | None = None, server: str | None = None):
        import MetaTrader5 as mt5  # import tardio: só existe no Windows

        self.mt5 = mt5
        kwargs = {k: v for k, v in dict(login=login, password=password, server=server).items() if v}
        ok = mt5.initialize(path, **kwargs) if path else mt5.initialize(**kwargs)
        if not ok:
            raise RuntimeError(f"mt5.initialize() falhou: {mt5.last_error()}")

    def close(self) -> None:
        self.mt5.shutdown()

    def terminal(self) -> dict:
        ti, ai = self.mt5.terminal_info(), self.mt5.account_info()
        return {
            "version": list(self.mt5.version() or []),
            "terminal": ti._asdict() if ti else None,
            "account": {k: getattr(ai, k) for k in ("server", "company", "currency", "trade_mode")} if ai else None,
        }

    def _select(self, symbol: str) -> None:
        self.mt5.symbol_select(symbol, True)

    def rates_range(self, symbol, start, end):
        self._select(symbol)
        return self.mt5.copy_rates_range(symbol, self.mt5.TIMEFRAME_M1, start, end)

    def ticks_range(self, symbol, start, end):
        self._select(symbol)
        return self.mt5.copy_ticks_range(symbol, start, end, self.mt5.COPY_TICKS_ALL)

    def symbol_info(self, symbol):
        self._select(symbol)
        si = self.mt5.symbol_info(symbol)
        if si is None:
            return None
        keep = ("name", "description", "path", "digits", "point", "trade_contract_size",
                "currency_base", "currency_profit", "spread", "spread_float",
                "trade_tick_size", "trade_tick_value", "volume_min", "volume_step")
        return {k: getattr(si, k) for k in keep if hasattr(si, k)}

    def last_tick_time(self, symbol):
        self._select(symbol)
        t = self.mt5.symbol_info_tick(symbol)
        return float(t.time) if t else None

    def last_error(self):
        return self.mt5.last_error()


def utc(dt: datetime) -> datetime:
    """O MT5 recomenda passar datetimes com tzinfo UTC para evitar deslocamento local."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def normalize_rates(raw: np.ndarray, tz_mode: str, fixed_offset_hours: float = 0.0) -> pd.DataFrame:
    """Barras M1 do MT5 -> DataFrame com `time` em UTC."""
    df = pd.DataFrame(raw)
    if df.empty:
        return df
    server = pd.to_datetime(df["time"], unit="s")
    out = pd.DataFrame({
        "time": server_to_utc(server, tz_mode, fixed_offset_hours),
        "open": df["open"].astype("float64"),
        "high": df["high"].astype("float64"),
        "low": df["low"].astype("float64"),
        "close": df["close"].astype("float64"),
        "tick_volume": df["tick_volume"].astype("int64"),
        "spread": df["spread"].astype("int32"),          # em points; ver symbol_info.point no manifesto
        "real_volume": df["real_volume"].astype("int64"),
    })
    return out.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


def normalize_ticks(raw: np.ndarray, tz_mode: str, fixed_offset_hours: float = 0.0) -> pd.DataFrame:
    """Ticks do MT5 -> DataFrame com `time` em UTC (precisão de ms)."""
    df = pd.DataFrame(raw)
    if df.empty:
        return df
    server = pd.to_datetime(df["time_msc"], unit="ms")
    out = pd.DataFrame({
        "time": server_to_utc(server, tz_mode, fixed_offset_hours),
        "bid": df["bid"].astype("float64"),
        "ask": df["ask"].astype("float64"),
        "last": df["last"].astype("float64"),
        "volume": df["volume_real"].astype("float64"),
        "flags": df["flags"].astype("uint32"),
    })
    # Ticks podem repetir o mesmo ms; ordenação estável preserva a ordem de chegada.
    return out.sort_values("time", kind="stable").reset_index(drop=True)

"""Fonte MT5 falsa: gera barras/ticks sintéticos no horário do servidor (NY+7h)."""

from __future__ import annotations

import numpy as np
import pandas as pd

RATES_DTYPE = [("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
               ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")]
TICKS_DTYPE = [("time", "<i8"), ("bid", "<f8"), ("ask", "<f8"), ("last", "<f8"), ("volume", "<u8"),
               ("time_msc", "<i8"), ("flags", "<u4"), ("volume_real", "<f8")]


def utc_to_server_epoch(ts_utc: pd.DatetimeIndex) -> np.ndarray:
    ny = ts_utc.tz_convert("America/New_York").tz_localize(None)
    server = ny + pd.Timedelta(hours=7)
    return (server - pd.Timestamp(0)) // pd.Timedelta(seconds=1)


def forex_minutes(start_utc, end_utc) -> pd.DatetimeIndex:
    idx = pd.date_range(start_utc, end_utc, freq="1min", inclusive="left")
    # fecha sex 21:00 UTC -> dom 21:00 UTC (aproximação)
    dow, hr = idx.dayofweek, idx.hour
    closed = (dow == 5) | ((dow == 4) & (hr >= 21)) | ((dow == 6) & (hr < 21))
    return idx[~closed]


class FakeSource:
    def __init__(self, drop_utc: tuple[str, str] | None = None):
        self.drop = drop_utc

    def _minutes(self, start, end):
        idx = forex_minutes(pd.Timestamp(start), pd.Timestamp(end))
        if self.drop:
            a, b = pd.Timestamp(self.drop[0], tz="UTC"), pd.Timestamp(self.drop[1], tz="UTC")
            idx = idx[(idx < a) | (idx >= b)]
        return idx

    def rates_range(self, symbol, start, end):
        # O MT5 interpreta start/end no relógio do servidor; para o teste basta
        # cobrir a janela com folga — o coletor recorta em UTC depois.
        idx = self._minutes(pd.Timestamp(start) - pd.Timedelta(hours=4), pd.Timestamp(end) + pd.Timedelta(hours=4))
        arr = np.zeros(len(idx), dtype=RATES_DTYPE)
        arr["time"] = utc_to_server_epoch(idx)
        px = 1.1 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, len(idx)))
        arr["open"], arr["close"] = px, px
        arr["high"], arr["low"] = px + 1e-4, px - 1e-4
        arr["tick_volume"], arr["spread"] = 30, 8
        return arr

    def ticks_range(self, symbol, start, end):
        idx = self._minutes(pd.Timestamp(start), pd.Timestamp(end))[::10]
        arr = np.zeros(len(idx), dtype=TICKS_DTYPE)
        e = utc_to_server_epoch(idx)
        arr["time"], arr["time_msc"] = e, e * 1000
        arr["bid"], arr["ask"] = 1.1, 1.1001
        return arr

    def symbol_info(self, symbol):
        return {"name": symbol, "digits": 5, "point": 1e-5}

    def last_tick_time(self, symbol):
        return None

    def last_error(self):
        return (1, "ok")

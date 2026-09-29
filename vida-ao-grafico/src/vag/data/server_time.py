"""Conversão do horário do servidor MT5 para UTC.

O pacote MetaTrader5 devolve `time`/`time_msc` como segundos/ms desde a época,
mas contados no relógio do SERVIDOR da corretora, não em UTC. Sem corrigir
isso, sessões (Ásia/Londres/NY) e horários ficam deslocados em 2-3 horas.
"""

from __future__ import annotations

import pandas as pd

NY_CLOSE_SHIFT = pd.Timedelta(hours=7)  # servidor = horário de Nova York + 7h


def server_to_utc(server_naive: pd.Series, mode: str, fixed_offset_hours: float = 0.0) -> pd.Series:
    """Converte uma série de datetimes ingênuos (horário do servidor) para UTC tz-aware."""
    s = pd.to_datetime(server_naive)
    if mode == "utc":
        return s.dt.tz_localize("UTC")
    if mode == "fixed":
        return (s - pd.Timedelta(hours=fixed_offset_hours)).dt.tz_localize("UTC")
    if mode == "ny_close":
        ny_naive = s - NY_CLOSE_SHIFT
        # A hora ambígua da virada de horário de verão cai no domingo de manhã
        # em NY (mercado de forex fechado); tratamos como horário padrão.
        ny = ny_naive.dt.tz_localize(
            "America/New_York", ambiguous=False, nonexistent="shift_forward"
        )
        return ny.dt.tz_convert("UTC")
    raise ValueError(f"modo de horário do servidor desconhecido: {mode!r}")


def estimate_offset_hours(server_epoch_s: float, utc_epoch_s: float) -> float:
    """Estima o offset do servidor a partir de um tick recente (arredonda para 0,5 h)."""
    return round((server_epoch_s - utc_epoch_s) / 1800.0) / 2.0

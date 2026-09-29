"""Checagens de qualidade: lacunas no tempo."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd


@dataclass
class Gap:
    start: str
    end: str
    minutes: float
    weekend: bool


def _spans_weekend(a: pd.Timestamp, b: pd.Timestamp) -> bool:
    """Verdadeiro se o intervalo (a, b) contém algum instante de sábado (UTC).

    O mercado de forex fecha ~sex 21-22h UTC e reabre ~dom 21-22h UTC, então
    toda lacuna de fim de semana atravessa o sábado.
    """
    if (b - a) >= pd.Timedelta(days=7):
        return True
    day = a.normalize()
    while day <= b:
        if day.dayofweek == 5 and day + pd.Timedelta(days=1) > a and day < b:
            return True
        day += pd.Timedelta(days=1)
    return False


def find_gaps(times: pd.Series, min_gap: pd.Timedelta) -> list[Gap]:
    """Lista lacunas maiores que `min_gap` numa série ordenada de timestamps UTC."""
    t = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    if len(t) < 2:
        return []
    d = t.diff()
    idx = d.index[d > min_gap]
    gaps = []
    for i in idx:
        a, b = t.iloc[i - 1], t.iloc[i]
        gaps.append(Gap(str(a), str(b), (b - a).total_seconds() / 60.0, _spans_weekend(a, b)))
    return gaps


def summarize_gaps(gaps: list[Gap], top: int = 10) -> dict:
    weekday = [g for g in gaps if not g.weekend]
    return {
        "total": len(gaps),
        "weekend": len(gaps) - len(weekday),
        "weekday": len(weekday),
        "weekday_missing_minutes": round(sum(g.minutes for g in weekday), 1),
        "largest_weekday": [asdict(g) for g in sorted(weekday, key=lambda g: -g.minutes)[:top]],
    }

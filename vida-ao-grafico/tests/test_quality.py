import pandas as pd

from vag.data.quality import find_gaps, summarize_gaps


def test_weekend_vs_weekday_gaps():
    t = pd.Series(pd.to_datetime([
        "2024-01-05 20:58", "2024-01-05 20:59",       # sexta
        "2024-01-07 22:00", "2024-01-07 22:01",       # domingo (lacuna de fim de semana)
        "2024-01-08 10:00", "2024-01-08 10:30",       # segunda (lacuna de 30 min em dia útil)
    ], utc=True))
    s = summarize_gaps(find_gaps(t, pd.Timedelta(minutes=2)))
    assert s["weekend"] == 1
    assert s["weekday"] == 2
    assert s["largest_weekday"][0]["minutes"] > 30

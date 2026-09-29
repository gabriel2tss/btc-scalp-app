import pandas as pd

from vag.data.server_time import estimate_offset_hours, server_to_utc


def test_ny_close_winter_is_utc_plus_2():
    s = pd.Series(pd.to_datetime(["2024-01-15 12:00"]))
    assert server_to_utc(s, "ny_close").iloc[0] == pd.Timestamp("2024-01-15 10:00", tz="UTC")


def test_ny_close_summer_is_utc_plus_3():
    s = pd.Series(pd.to_datetime(["2024-07-15 12:00"]))
    assert server_to_utc(s, "ny_close").iloc[0] == pd.Timestamp("2024-07-15 09:00", tz="UTC")


def test_ny_close_follows_us_dst_not_eu():
    # Entre a virada dos EUA (10/mar/2024) e da Europa (31/mar/2024) o servidor já está em +3.
    s = pd.Series(pd.to_datetime(["2024-03-20 12:00"]))
    assert server_to_utc(s, "ny_close").iloc[0] == pd.Timestamp("2024-03-20 09:00", tz="UTC")


def test_fixed_and_utc_modes():
    s = pd.Series(pd.to_datetime(["2024-01-15 12:00"]))
    assert server_to_utc(s, "fixed", 2).iloc[0] == pd.Timestamp("2024-01-15 10:00", tz="UTC")
    assert server_to_utc(s, "utc").iloc[0] == pd.Timestamp("2024-01-15 12:00", tz="UTC")


def test_estimate_offset():
    assert estimate_offset_hours(1_000_000 + 3 * 3600 + 40, 1_000_000) == 3.0
    assert estimate_offset_hours(1_000_000 + 5.5 * 3600, 1_000_000) == 5.5

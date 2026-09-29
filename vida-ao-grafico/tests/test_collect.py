from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq
import pytest

from fake_mt5 import FakeSource
from vag.config import load_config
from vag.data.collect import collect_month, iter_months
from vag.data.storage import Manifest, month_path, read_symbol


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["data_dir"] = tmp_path / "data"
    return c


NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def test_m1_month_is_utc_clipped_and_manifested(cfg):
    man = Manifest(cfg["data_dir"])
    e = collect_month(FakeSource(), cfg, man, "m1", "EURUSD", 2024, 3, now=NOW)
    df = pq.read_table(month_path(cfg["data_dir"], "m1", "EURUSD", 2024, 3)).to_pandas()
    assert str(df["time"].dt.tz) == "UTC"
    assert df["time"].min() >= pd.Timestamp("2024-03-01", tz="UTC")
    assert df["time"].max() < pd.Timestamp("2024-04-01", tz="UTC")
    assert df["time"].is_monotonic_increasing and df["time"].is_unique
    # Semana cheia de forex fica ~ 5 dias * 1440; o mês tem 21 dias úteis.
    assert 20 * 1440 < len(df) < 23 * 1440
    assert e["complete"] and e["rows"] == len(df)
    assert e["gaps"]["weekday"] == 0  # sessão sintética não tem buracos em dia útil
    assert man.get("m1", "EURUSD", 2024, 3)["rows"] == len(df)


def test_detects_injected_feed_gap(cfg):
    src = FakeSource(drop_utc=("2024-03-13 10:00", "2024-03-13 11:00"))
    e = collect_month(src, cfg, Manifest(cfg["data_dir"]), "m1", "EURUSD", 2024, 3, now=NOW)
    assert e["gaps"]["weekday"] == 1
    assert e["gaps"]["largest_weekday"][0]["start"].startswith("2024-03-13 09:59")


def test_resume_skips_complete_months_but_refetches_current(cfg):
    man = Manifest(cfg["data_dir"])
    assert collect_month(FakeSource(), cfg, man, "m1", "EURUSD", 2024, 3, now=NOW) is not None
    assert collect_month(FakeSource(), cfg, man, "m1", "EURUSD", 2024, 3, now=NOW) is None
    mid = datetime(2024, 5, 15, tzinfo=timezone.utc)
    e = collect_month(FakeSource(), cfg, man, "m1", "EURUSD", 2024, 5, now=mid)
    assert not e["complete"]
    assert collect_month(FakeSource(), cfg, man, "m1", "EURUSD", 2024, 5, now=mid) is not None


def test_ticks_and_read_back(cfg):
    man = Manifest(cfg["data_dir"])
    for y, m in iter_months("2024-02", "2024-03"):
        collect_month(FakeSource(), cfg, man, "ticks", "EURUSD", y, m, now=NOW)
    df = read_symbol(cfg["data_dir"], "ticks", "EURUSD")
    assert df["time"].is_monotonic_increasing
    assert set(df.columns) == {"time", "bid", "ask", "last", "volume", "flags"}
    assert man.totals()["ticks/EURUSD"]["months"] == 2


def test_iter_months_crosses_year():
    assert list(iter_months("2023-11", "2024-02")) == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]

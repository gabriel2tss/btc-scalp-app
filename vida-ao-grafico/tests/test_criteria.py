import yaml

import vag.criteria as C


def test_lock_requires_filled_and_detects_tampering(tmp_path, monkeypatch):
    crit, lock = tmp_path / "c.yaml", tmp_path / "c.lock"
    monkeypatch.setattr(C, "CRITERIA", crit)
    monkeypatch.setattr(C, "LOCK", lock)
    crit.write_text((C.CONFIG_DIR / "success_criteria.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    assert C.lock() == 1 and not lock.exists()

    c = yaml.safe_load(crit.read_text())
    c.update(assets=["EURUSD"], horizons_minutes=[15], min_net_edge_per_trade=0.3, min_net_edge_unit="pips",
             min_profit_factor_oos=1.1, min_walkforward_windows_passing=6, total_walkforward_windows=8,
             min_phrase_occurrences=200)
    crit.write_text(yaml.safe_dump(c))
    assert C.lock() == 0 and C.verify()
    assert C.lock() == 1  # não trava duas vezes

    crit.write_text(crit.read_text().replace("200", "50"))
    assert not C.verify()

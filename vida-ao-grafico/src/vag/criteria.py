"""Pré-registro dos critérios de sucesso (seção 8 do briefing).

    python -m vag.criteria check   # mostra o que falta preencher
    python -m vag.criteria lock    # trava (grava hash); depois disso, não se altera
    python -m vag.criteria verify  # usado pela Fase 6: falha se o arquivo mudou
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone

import yaml

from .config import CONFIG_DIR

CRITERIA = CONFIG_DIR / "success_criteria.yaml"
LOCK = CONFIG_DIR / "success_criteria.lock"

REQUIRED = ["assets", "horizons_minutes", "min_net_edge_per_trade", "min_net_edge_unit",
            "min_walkforward_windows_passing", "total_walkforward_windows", "min_phrase_occurrences"]


def _digest() -> str:
    return hashlib.sha256(CRITERIA.read_bytes()).hexdigest()


def missing() -> list[str]:
    c = yaml.safe_load(CRITERIA.read_text(encoding="utf-8"))
    miss = [k for k in REQUIRED if c.get(k) in (None, [], "")]
    if c.get("min_hit_rate_oos") is None and c.get("min_profit_factor_oos") is None:
        miss.append("min_hit_rate_oos ou min_profit_factor_oos")
    return miss


def lock() -> int:
    miss = missing()
    if miss:
        print("Não dá para travar; faltam: " + ", ".join(miss))
        return 1
    if LOCK.exists():
        print(f"Já travado em {json.loads(LOCK.read_text())['locked_utc']}. Não se altera critério depois.")
        return 1
    LOCK.write_text(json.dumps({"sha256": _digest(),
                                "locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2))
    print("Critérios travados. Faça commit de config/success_criteria.* agora.")
    return 0


def verify() -> bool:
    if not LOCK.exists():
        return False
    return json.loads(LOCK.read_text())["sha256"] == _digest()


def main(argv: list[str] | None = None) -> int:
    cmd = (argv or sys.argv[1:] or ["check"])[0]
    if cmd == "check":
        miss = missing()
        print("Tudo preenchido." if not miss else "Faltam: " + ", ".join(miss))
        print("Travado e íntegro." if verify() else ("TRAVA VIOLADA" if LOCK.exists() else "Ainda não travado."))
        return 0
    if cmd == "lock":
        return lock()
    if cmd == "verify":
        ok = verify()
        print("ok" if ok else "FALHOU: critérios não travados ou alterados após a trava.")
        return 0 if ok else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())

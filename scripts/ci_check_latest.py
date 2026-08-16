"""Check de CI / pre-commit: valida artifacts/latest/ contra el contrato.

Falla (exit 1) si el artefacto publicado tiene VIOLACIONES DURAS del contrato
(mezcla de procedencia, set incompleto, look-ahead en la serie, git=nogit, R6
multistart < 20). Es la red de seguridad que impide que la regresion de
publicacion (V0/--quick pisando latest/, sets mezclados) pase inadvertida: aunque
la promocion ya lo bloquea en promote_run, este check lo re-verifica sobre lo que
hay en disco AHORA.

No pasa --allow-provisional a proposito: un artefacto --quick (multistart < 20) es
un fallo de CI, no un pase provisional. Si latest/ esta vacio (sin artefacto aun),
el check pasa en verde (no hay nada que este mal todavia).

    python scripts/ci_check_latest.py         # exit 0 si latest/ es valido
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irfn.outputs.contract import _format, validate_artifact  # noqa: E402

LATEST = ROOT / "artifacts" / "latest"


def main() -> int:
    irfn = LATEST / "irfn.json"
    if not irfn.exists():
        print("CI latest: artifacts/latest/ vacio (sin artefacto aun) -> OK.")
        return 0

    res = validate_artifact(LATEST)  # allow_provisional=False: --quick es fallo
    if res.ok:
        estado = "PROMOVIBLE" if res.promotable else "valido (no promovible por guardarrail)"
        print(f"CI latest: OK -- version={res.version} run_id={res.run_id} [{estado}].")
        return 0

    print("CI latest: FALLO -- artifacts/latest/ tiene violaciones duras del contrato:")
    print(_format(res))
    print("\nCorrige re-promoviendo un artefacto valido con scripts/promote.py.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

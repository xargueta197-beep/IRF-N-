"""Promocion de un artefacto a artifacts/latest/ -- UNICO punto de entrada.

Las corridas (run_pipeline / run_v2 / run_v3) escriben su set completo en
artifacts/[<slug>/]runs/<run_id>/ y NO tocan latest/. Este script es la unica
forma de que un artefacto llegue a latest/, y solo lo hace si pasa el contrato
(outputs/contract.py) y el guardarrail anti-downgrade (outputs/publish.py).

Uso:
    python scripts/promote.py runs/<run_id>
    python scripts/promote.py artifacts/btc/runs/<run_id> --slug btc
    python scripts/promote.py runs/<run_id> --force-downgrade   # V<3, deja registro

Codigo de salida 0 si promovio; 1 si el contrato/guardarrail lo rechazo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irfn.outputs.contract import ContractViolation  # noqa: E402
from irfn.outputs.publish import promote_run  # noqa: E402


def _resolve(run_dir_arg: str, slug: str | None) -> tuple[Path, Path]:
    run_dir = Path(run_dir_arg)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    base = ROOT / "artifacts" / slug if slug else ROOT / "artifacts"
    return run_dir, base / "latest"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Promueve un run a artifacts/latest/ (unico punto de promocion).")
    ap.add_argument("run_dir", help="Directorio del run (p.ej. runs/<run_id> o ruta absoluta).")
    ap.add_argument("--slug", default=None, help="Sub-activo (p.ej. btc). Sin esto, latest = artifacts/latest.")
    ap.add_argument("--force-downgrade", action="store_true",
                    help="Permite promover version < V3 o provisional (queda en publish_log.jsonl).")
    ap.add_argument("--allow-provisional", action="store_true")
    ap.add_argument("--tolerance-days", type=int, default=3)
    ap.add_argument("--no-clean-tree-check", action="store_true",
                    help="No verificar que el arbol git este limpio (R5).")
    args = ap.parse_args(argv)

    run_dir, latest_dir = _resolve(args.run_dir, args.slug)
    if not run_dir.is_dir():
        print(f"ERROR: no existe el directorio del run: {run_dir}")
        return 1

    try:
        res = promote_run(
            run_dir, latest_dir=latest_dir,
            repo_root=None if args.no_clean_tree_check else ROOT,
            force_downgrade=args.force_downgrade,
            allow_provisional=args.allow_provisional,
            tolerance_days=args.tolerance_days,
            triggered_by=f"scripts/promote.py {' '.join(argv or sys.argv[1:])}",
        )
    except ContractViolation as e:
        print(str(e))
        return 1

    print(f"PROMOVIDO: version={res.version} run_id={res.run_id} -> {latest_dir}")
    if res.provisional_reasons:
        print("  (promovido como provisional/forzado -- registrado en publish_log.jsonl)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

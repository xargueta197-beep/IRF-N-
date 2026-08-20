"""Estado de reapertura de las capas bloqueadas por DATOS (avisos #2/#3/#8).

Las capas M4/V2 (sorpresa/consenso) y M2+H (Hawkes como covariable en el
walk-forward pre-registrado) NO estan apagadas por un bug ni por una decision
de diseno: les faltan datos (consenso historico y anios de corpus GDELT). Eso
es un blocker del mundo, no de ingenieria (R8: se documenta, no se esconde).

El plan `reports/plan_mejoras_avisos_2026-08-18.md` (Franja E) pide que la
REAPERTURA sea observable y automatica -- que exista una condicion explicita y
verificable, no "cuando alguien se acuerde". Este script es esa condicion:

  * lee SOLO la cobertura ya publicada en el artefacto (no calcula nada del
    modelo, no re-corre nada, no toca artifacts/ -- espiritu R9), y
  * la compara contra umbrales derivados de config/ (fuente unica de verdad),
    no numeros magicos.

Umbrales (de config/, no hardcodeados):
  * Consenso (M4/V2): reabrir cuando el total de releases con consenso
    acumulados alcanza `v2.delta_mle.min_events_total`. Es el mismo umbral que
    usa el fit conjunto de delta antes de siquiera intentarse.
  * GDELT/lambda_N_z (M2+H): reabrir cuando el corpus cubre el span completo
    del walk-forward pre-registrado = train_years + n_blocks * test_months.
    Con menos, la malla de 6 bloques NO cabe y ENCOGERLA para "alcanzar" la
    cobertura esta PROHIBIDO (R8, aviso #3 del plan).

Uso:
    python scripts/check_reopen_status.py
    python scripts/check_reopen_status.py --artifact artifacts/btc/latest/irfn.json

Codigo de salida:
    0 siempre que pueda leer el artefacto (es un lector de estado, no un gate).
    2 si no encuentra el artefacto.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irfn.config import BaseConfig  # noqa: E402


def _bar(done: int, need: int, width: int = 28) -> str:
    frac = 0.0 if need <= 0 else min(1.0, done / need)
    filled = int(round(frac * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {frac * 100:5.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--artifact",
        default=str(ROOT / "artifacts" / "latest" / "irfn.json"),
        help="Ruta a irfn.json (por defecto: artifacts/latest/irfn.json).",
    )
    args = ap.parse_args()

    art_path = Path(args.artifact)
    if not art_path.exists():
        print(f"ERROR: no existe el artefacto {art_path}", file=sys.stderr)
        return 2

    cfg = BaseConfig.load()
    consensus_need = cfg.v2.delta_mle.min_events_total
    wf = cfg.walkforward
    # Span del walk-forward pre-registrado, en dias de calendario. Derivado de
    # config, no un 2555 magico: train + n_blocks bloques de test consecutivos.
    span_years = wf.train_years + wf.n_blocks * wf.test_months / 12.0
    gdelt_need_days = round(span_years * 365.25)

    art = json.loads(art_path.read_text(encoding="utf-8"))
    model = art.get("model", {})

    # --- Consenso (M4/V2) ---
    news_cov = (model.get("news_layer_params") or {}).get("coverage") or {}
    consensus_have = sum(int(v.get("n_with_consensus", 0)) for v in news_cov.values())
    consensus_ready = consensus_have >= consensus_need

    # --- GDELT/lambda_N_z (M2+H) ---
    hawkes_cov = (model.get("hawkes_layer_params") or {}).get("coverage") or {}
    gdelt_have_days = int(hawkes_cov.get("n_days", 0))
    gdelt_ready = gdelt_have_days >= gdelt_need_days

    print("=" * 66)
    print(f" ESTADO DE REAPERTURA  ({art.get('version', '?')} run_id={art.get('run_id', '?')} asof={art.get('asof', '?')})")
    print("=" * 66)

    print("\n[M4/V2] Capa de sorpresa (consenso historico)")
    print(f"  umbral (config v2.delta_mle.min_events_total): >= {consensus_need} releases con consenso")
    print(f"  acumulado hoy: {consensus_have}")
    print("  " + _bar(consensus_have, consensus_need))
    if consensus_ready:
        print("  -> REABRIR: hay consenso suficiente. Re-correr scripts/run_v2.py (M4).")
    else:
        print(f"  -> ESPERAR: faltan {consensus_need - consensus_have}. Mantener scripts/capture_consensus.py vivo")
        print("     (o pagar Trading Economics point-in-time; decision del director).")

    print("\n[M2+H] lambda_N_z como covariable en el walk-forward pre-registrado")
    print(f"  umbral (config walkforward: {wf.train_years}a train + {wf.n_blocks}x{wf.test_months}m test)")
    print(f"          = {span_years:.1f} anios ~= {gdelt_need_days} dias de cobertura GDELT")
    print(f"  cobertura hoy: {gdelt_have_days} dias")
    print("  " + _bar(gdelt_have_days, gdelt_need_days))
    if gdelt_ready:
        print("  -> REABRIR: el corpus cubre la malla de 6 bloques. Re-correr la ablacion M2+H.")
    else:
        falta = gdelt_need_days - gdelt_have_days
        print(f"  -> ESPERAR: faltan ~{falta} dias de backfill GDELT hacia atras. Mantener")
        print("     scripts/capture_headlines.py vivo. PROHIBIDO encoger la malla (R8, aviso #3).")

    print("\n" + "-" * 66)
    print(" Ninguna de estas reaperturas se arregla con codigo: son blockers de")
    print(" datos. Este chequeo las hace observables; correrlo periodicamente")
    print(" (o en CI) convierte la reapertura en automatica, no en 'ojala'.")
    print("-" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

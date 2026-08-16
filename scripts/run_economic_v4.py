"""Corre el walk-forward economico PRE-REGISTRADO (Test 5 de V4, parte economica).

    python scripts/run_economic_v4.py

Regla y criterios congelados en docs/preregistro_regla_trading.md (aprobado por
el director 2026-07-18) y parametrizados en config/base.yaml seccion v4.economic
(R7: cero numeros magicos en el codigo). Consume artifacts/latest/history.parquet
(el walk-forward OOS vigente) y escribe artifacts/latest/validation.json + copia
inmutable junto al run correspondiente.

La linea BTC NO se corre (pre-registro seccion 7): su artefacto es K=1 — no hay
regimen de alta volatilidad que evitar, la regla es vacua por construccion. Se
deja constancia aqui y en el reporte, no se finge un resultado.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irfn.config import BaseConfig  # noqa: E402
from irfn.validation.economic import economic_walkforward  # noqa: E402

ARTIFACTS = ROOT / "artifacts" / "latest"


def main() -> None:
    base = BaseConfig.load()
    econ = base.v4.economic

    history = pd.read_parquet(ARTIFACTS / "history.parquet").set_index("fecha")
    irfn = json.loads((ARTIFACTS / "irfn.json").read_text(encoding="utf-8"))
    run_id = irfn["run_id"]

    result = economic_walkforward(
        history,
        p_threshold=econ.p_threshold,
        w_reduced=econ.w_reduced,
        cost_bps=econ.cost_bps,
        cost_bps_sensitivity=list(econ.cost_bps_sensitivity),
        n_boot=econ.n_boot,
        seed=base.model.seed,
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id_consumed": run_id,
        "asset": "SPY",
        "provisional_quick": True,  # el walk-forward consumido no cumple R6 aun
        "btc_note": (
            "linea BTC no aplicable: su artefacto es K=1 (sin regimen de alta "
            "volatilidad que evitar); la regla es vacua por construccion."
        ),
        **result,
    }

    # Fase 3: no escribir a latest/. validation.json se deja junto a su run en
    # runs/<run_id>/; llega a latest/ solo si ese run se promueve (scripts/promote.py).
    run_dir = ROOT / "artifacts" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "validation.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    m = result["main"]
    print(f"run_id consumido: {run_id}  (n={result['n_obs']} dias OOS)")
    print(f"regla: salir a efectivo si P(alta vol) > {econ.p_threshold} "
          f"(w_reducido={econ.w_reduced}, costo {econ.cost_bps} pb)")
    print(f"Sharpe estrategia:  {m['sharpe_strategy']:.3f}")
    print(f"Sharpe buy&hold:    {m['sharpe_buyhold']:.3f}")
    lo, hi = m["sharpe_diff_ci95"]
    print(f"Sharpe diferencia:  {m['sharpe_diff']:.3f}  IC95 [{lo:.3f}, {hi:.3f}] "
          f"(bloque={m['sharpe_diff_block_length']})")
    print(f"exposicion: {m['exposure_fraction']:.1%}  cambios: {m['n_position_changes']}  "
          f"turnover: {m['total_turnover']:.0f}")
    print(f"max drawdown: estrategia {m['max_drawdown_strategy']:.1%} "
          f"vs buy&hold {m['max_drawdown_buyhold']:.1%}")
    for c in result["cost_sensitivity"]:
        clo, chi = c["sharpe_diff_ci95"]
        print(f"  sensibilidad {c['cost_bps']:.0f} pb: diff {c['sharpe_diff']:.3f} "
              f"IC95 [{clo:.3f}, {chi:.3f}]")
    print(f"\nVEREDICTO pre-registrado: {'EXITO' if result['success'] else 'NO SUPERA'} "
          f"(criterio: IC de la diferencia excluye 0 por arriba a {econ.cost_bps} pb)")
    print(f"escrito: {out}")


if __name__ == "__main__":
    main()

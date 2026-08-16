"""Diagnostico de ESTABILIDAD del optimo K=2 dist=t para BTC (@diagnostic_only).

Motivacion (decision del director, 2026-08-15): el kselect R6 de BTC dejo un
conflicto BIC-vs-Hansen:
  - BIC ganador = K=1 t (14559.11), robusto (18/20 arranques).
  - Bootstrap LR (Hansen) K=2 vs K=1 (dist t) = 53.18, p=0.020 -> K=2 significativo.
PERO el optimo de K=2 t solo lo alcanzo 1/20 arranques: no se puede confiar en su
log-verosimilitud ni en su BIC hasta confirmar que ese optimo es real y no un
artefacto numerico del multistart.

Esta corrida NO decide el titular. NO toca _load_titular_K_dist. NO re-corre
run_v3. NO publica ni escribe en artifacts/. NO captura GDELT. Solo produce la
tabla comparativa (BIC + estabilidad + Hansen) de tres alternativas, en
reports/diag_btc_k2t_stability.md, para que el director decida con evidencia.

    python scripts/diag_btc_k2t_stability.py            # 100 arranques (default)
    python scripts/diag_btc_k2t_stability.py --starts 100 --nboot 49

Que compara (los tres candidatos que pidio el director):
  - K=1 t     : ganador BIC actual (18/20 robusto)
  - K=2 t     : el que gano Hansen, pendiente de estabilizar (1/20 en R6)
  - K=2 normal: alternativa ya estable (10/20 en R6), referencia si K=2 t no cuaja

Reproducibilidad (R6): mismo seed del proyecto; `fit` genera los arranques de
forma determinista desde el seed, asi que los primeros 20 arranques de esta
corrida de 100 son EXACTAMENTE los del kselect R6 (superset), y el n_converged
es directamente comparable con el de la tabla de 20.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from irfn.config import BaseConfig  # noqa: E402
from irfn.validation.tests_stat import bic_table, bootstrap_lr_test  # noqa: E402

import run_v1  # noqa: E402  -- reutiliza load_sample + _configure_asset (misma muestra BTC)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("diag_btc_k2t")

REPORTS = ROOT / "reports"
# Checkpoints DIAGNOSTICOS aparte (no comparten con el kselect publicado): esta
# corrida no debe reanudar ni contaminar los checkpoints de artifacts/btc/.
CKPT = ROOT / "artifacts" / "btc" / "checkpoints" / "diag_k2t_stability"


def _fit_cell(r, K: int, dist: str, n_starts: int, seed: int) -> dict:
    """Un (K, dist) con n_starts arranques via bic_table (celda unica).

    Devuelve BIC/loglik/n_converged. n_converged/n_starts = fraccion de arranques
    que alcanzaron el optimo global (dentro de la tolerancia de estimate.py): la
    medida de estabilidad que pidio el director."""
    t0 = time.time()
    row = bic_table(
        r, k_candidates=[K], dists=[dist], n_starts=n_starts, seed=seed,
        checkpoint_path=CKPT / f"bic_K{K}_{dist}_{n_starts}.pkl",
    )[0]
    row["seconds"] = round(time.time() - t0, 1)
    row["conv_frac"] = row["n_converged"] / row["n_starts"]
    log.info(
        "  K=%d %-6s: BIC=%.2f loglik=%.2f | estabilidad %d/%d = %.0f%% | %.0fs",
        K, dist, row["bic"], row["loglik"], row["n_converged"], row["n_starts"],
        100 * row["conv_frac"], row["seconds"],
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnostico de estabilidad K=2 t para BTC (no publica).")
    ap.add_argument("--starts", type=int, default=100, help="Arranques del multistart (director: ~100).")
    ap.add_argument("--nboot", type=int, default=49, help="Replicas del bootstrap LR (Hansen).")
    args = ap.parse_args()

    base = BaseConfig.load()
    seed = base.model.seed
    # Configura el activo BTC en run_v1 (fija ASSET_* globals) y carga EXACTAMENTE
    # la misma muestra que el kselect (2987 obs, 2018-05-13..2026-07-16).
    run_v1._configure_asset(base, "BTC")
    returns, X = run_v1.load_sample(base)
    r = returns.to_numpy(dtype=float)
    log.info("muestra BTC: %d obs, %s a %s (seed=%d, starts=%d)",
             len(r), returns.index[0].date(), returns.index[-1].date(), seed, args.starts)

    CKPT.mkdir(parents=True, exist_ok=True)

    # --- 1) Los tres candidatos con los MISMOS n_starts (comparacion en igualdad) ---
    log.info("=== BIC + estabilidad con %d arranques (los tres candidatos) ===", args.starts)
    cells = {
        "K1_t": _fit_cell(r, 1, "t", args.starts, seed),
        "K2_t": _fit_cell(r, 2, "t", args.starts, seed),
        "K2_normal": _fit_cell(r, 2, "normal", args.starts, seed),
    }

    # --- 2) Hansen (bootstrap LR) K=1 t vs K=2 t con el optimo ESTABILIZADO ---
    # n_starts_data = args.starts asegura que el K=2 observado del test use el
    # optimo de 100 arranques (no el de 1/20 del R6): asi el lr_obs se mide contra
    # el optimo correcto. Recalcula si el p=0.020 se sostiene.
    log.info("=== Hansen (bootstrap LR) K=2 vs K=1 dist=t con optimo estabilizado (%d replicas) ===", args.nboot)
    t0 = time.time()
    hansen = bootstrap_lr_test(
        r, K_null=1, K_alt=2, dist="t", n_boot=args.nboot,
        n_starts_data=args.starts, n_starts_boot=base.v1.ktest.boot_n_starts, seed=seed,
        checkpoint_path=CKPT / f"hansen_K2vs1_t_data{args.starts}.pkl", checkpoint_every=5,
    )
    hansen["seconds"] = round(time.time() - t0, 1)
    log.info("  LR_obs=%.2f p=%.3f [%d/%d replicas] (%.0fs)",
             hansen["lr_obs"], hansen["p_value"], hansen["n_boot_ok"], hansen["n_boot"], hansen["seconds"])

    # --- 3) Reporte comparativo (R8: se escribe pase lo que pase; diagnostico) ---
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset": "BTC", "seed": seed, "n_starts": args.starts,
        "sample": {"n_obs": len(r), "start": str(returns.index[0].date()),
                   "end": str(returns.index[-1].date())},
        "r6_reference": {  # de artifacts/btc/latest/v1_kselect.json (kselect R6, 20 arranques)
            "K1_t": {"bic": 14559.11, "conv": "18/20"},
            "K2_t": {"bic": 14569.94, "conv": "1/20"},
            "K2_normal": {"bic": 14665.57, "conv": "10/20"},
            "hansen_K2vs1_t": {"lr_obs": 53.18, "p_value": 0.020},
        },
        "cells": cells,
        "hansen_K2vs1_t_stabilized": hansen,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "diag_btc_k2t_stability.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")

    _write_report(out)
    log.info("LISTO. reports/diag_btc_k2t_stability.md (y .json). NO se toco artifacts/ ni el loader.")


def _fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _write_report(out: dict) -> None:
    c = out["cells"]
    r6 = out["r6_reference"]
    h = out["hansen_K2vs1_t_stabilized"]
    ns = out["n_starts"]
    L: list[str] = []
    L.append(f"# Diagnostico de estabilidad K=2 t — BTC (@diagnostic_only)\n")
    L.append(f"generado: {out['generated_at']}  |  seed: {out['seed']}  |  "
             f"muestra: {out['sample']['n_obs']} obs ({out['sample']['start']}..{out['sample']['end']})\n")
    L.append("**Esta corrida NO decide el titular, NO publica, NO toca `_load_titular_K_dist` "
             "ni `run_v3`.** Responde una sola pregunta: el optimo de K=2 t que gano el test de "
             "Hansen (1/20 en el kselect R6) es real o un artefacto del multistart?\n")

    L.append(f"## Los tres candidatos con {ns} arranques (comparacion en igualdad)\n")
    L.append(f"| modelo | BIC ({ns} arr.) | log-lik | estabilidad ({ns} arr.) | ref. R6 (20 arr.) |")
    L.append("| :-- | --: | --: | :-- | :-- |")
    L.append(f"| K=1 t | {_fmt(c['K1_t']['bic'])} | {_fmt(c['K1_t']['loglik'])} | "
             f"{c['K1_t']['n_converged']}/{ns} ({100*c['K1_t']['conv_frac']:.0f}%) | "
             f"BIC {r6['K1_t']['bic']}, {r6['K1_t']['conv']} |")
    L.append(f"| K=2 t | {_fmt(c['K2_t']['bic'])} | {_fmt(c['K2_t']['loglik'])} | "
             f"{c['K2_t']['n_converged']}/{ns} ({100*c['K2_t']['conv_frac']:.0f}%) | "
             f"BIC {r6['K2_t']['bic']}, {r6['K2_t']['conv']} |")
    L.append(f"| K=2 normal | {_fmt(c['K2_normal']['bic'])} | {_fmt(c['K2_normal']['loglik'])} | "
             f"{c['K2_normal']['n_converged']}/{ns} ({100*c['K2_normal']['conv_frac']:.0f}%) | "
             f"BIC {r6['K2_normal']['bic']}, {r6['K2_normal']['conv']} |")
    L.append("")
    L.append("> BIC menor = mejor. La columna de estabilidad es la fraccion de arranques que "
             "alcanzo el optimo global (tolerancia de estimate.py): baja = superficie multimodal, "
             "el optimo no es de fiar.\n")

    L.append("## Hansen (bootstrap LR) K=2 vs K=1 (dist t) con el optimo estabilizado\n")
    L.append(f"- Ajuste observado de K=2 con {ns} arranques (no 1/20): "
             f"log-lik_alt={_fmt(h['loglik_alt'])}, log-lik_null={_fmt(h['loglik_null'])}.")
    L.append(f"- **LR_obs = {_fmt(h['lr_obs'])}, p = {_fmt(h['p_value'],3)}** "
             f"({h['n_boot_ok']}/{h['n_boot']} replicas ok).")
    L.append(f"- Referencia R6 (optimo 1/20): LR_obs={r6['hansen_K2vs1_t']['lr_obs']}, "
             f"p={r6['hansen_K2vs1_t']['p_value']}.\n")

    # Lectura mecanica (no decide; solo describe que muestran los numeros).
    frac = c["K2_t"]["conv_frac"]
    dbic = c["K2_t"]["bic"] - c["K1_t"]["bic"]
    L.append("## Lectura (descriptiva; la decision es del director)\n")
    estab = ("SE ESTABILIZO" if frac >= 0.30 else
             "SIGUE INESTABLE" if frac < 0.10 else "PARCIALMENTE estable")
    L.append(f"- Estabilidad de K=2 t: {estab} ({c['K2_t']['n_converged']}/{ns}).")
    L.append(f"- BIC(K=2 t) - BIC(K=1 t) = {_fmt(dbic)} "
             f"({'K=1 t sigue mejor por BIC' if dbic > 0 else 'K=2 t pasa a ser mejor por BIC'}).")
    L.append(f"- Hansen con el optimo estabilizado: p={_fmt(h['p_value'],3)} "
             f"({'sigue significativo al 5%' if h['p_value'] < 0.05 else 'YA NO significativo al 5%'}).\n")

    (REPORTS / "diag_btc_k2t_stability.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()

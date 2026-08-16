"""Test JUSTO de M3 (macro): ablacion M0-M3 CON penalizacion L1 (lambda por CV
dentro del train de cada bloque), la especificacion canonica de CLAUDE.md.

Motivo (validation_v4.md, Test 7): la unica corrida de M3 hasta ahora
(`run_v2 --jobs -1`) uso `l1_grid=[0.0]` -- SIN L1 -- y ahi hasta M2 (que con L1
mejora a M1) aparecia degradado por sobreajuste de los beta. Este script corre la
misma escalera M0-M3 sobre la muestra alineada a macro, pero con la malla L1 real,
para saber si la macro aporta de verdad. NO publica ni toca artifacts/latest:
solo escribe reports/ablation_m3_l1.md y su checkpoint propio.

    python scripts/run_m3_l1.py --jobs -1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from irfn.config import BaseConfig  # noqa: E402
from irfn.validation.ablation import full_ladder_specs, run_ablation  # noqa: E402
import run_v2  # reutiliza load_sample, try_macro_covariates, _load_titular_K_dist  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("run_m3_l1")
REPORTS = ROOT / "reports"


def _fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablacion M0-M3 CON L1 (test justo de macro).")
    ap.add_argument("--jobs", type=int, default=1,
                    help="-1 = todos los nucleos menos uno; 1 = serie.")
    args = ap.parse_args()
    n_jobs = max(1, (os.cpu_count() or 1) - 1) if args.jobs == -1 else max(1, args.jobs)

    base = BaseConfig.load()
    run_v2._configure_asset(base, None)
    v1, seed = base.v1, base.model.seed

    log.info("cargando muestra (ancla + tecnicas)...")
    returns, X_tech = run_v2.load_sample(base)
    Kt, distt = run_v2._load_titular_K_dist(base)
    log.info("K=%d dist=%s", Kt, distt)

    log.info("construyendo covariables macro (M3)...")
    X_macro, blocker = run_v2.try_macro_covariates(base, returns.index)
    if blocker:
        log.error("M3 BLOQUEADO: %s -- no se puede correr el test justo.", blocker)
        sys.exit(1)

    # Escalera M0-M3 (M4/M5 fuera: sin datos). M2 = tecnico, M3 = tecnico + macro.
    specs = full_ladder_specs(
        Kt, distt,
        covariates_m2=list(base.v1.covariates_ablation_m2),
        covariates_m3=list(base.v2.covariates_ablation_m3),
    )[:4]  # M0, M1, M2, M3

    # Muestra alineada al tramo comun sin NaN (mismo trato que run_v2: recorta
    # warm-up del z-score macro; todos los peldanos en las MISMAS fechas -> DM valido).
    X_all = pd.concat([X_tech, X_macro], axis=1)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()].dropna()
    returns_abl = returns.loc[X_all.index]
    log.info("muestra alineada: %d obs (%s..%s), covs=%s",
             len(X_all), X_all.index[0].date(), X_all.index[-1].date(), list(X_all.columns))

    n_starts = base.v0.wf_n_starts  # R6
    log.info("=== ablacion M0-M3 CON L1 (grid=%s), n_starts=%d, n_jobs=%d ===",
             v1.tvtp.l1_grid, n_starts, n_jobs)
    abl = run_ablation(
        returns_abl, X_all, specs, seed=seed, n_starts=n_starts,
        train_years=base.walkforward.train_years, test_months=base.walkforward.test_months,
        n_blocks_min=base.walkforward.n_blocks,
        l1_grid=list(v1.tvtp.l1_grid),
        cv_val_frac=v1.tvtp.cv_val_frac, cv_n_starts=v1.tvtp.cv_n_starts,
        l1_smooth_eps=v1.tvtp.l1_smooth_eps,
        checkpoint_dir=ROOT / "artifacts" / "checkpoints" / "m3_l1",
        n_jobs=n_jobs,
    )

    # --- Reporte (R8) ---
    L = ["# Ablacion M0-M3 CON L1 -- test justo de la macro (M3)\n",
         f"generado: {datetime.now(timezone.utc).isoformat()}  |  K={Kt}, dist={distt}  |  "
         f"n_starts={n_starts}, L1 grid={list(v1.tvtp.l1_grid)}\n",
         f"muestra: {len(X_all)} obs, {X_all.index[0].date()}..{X_all.index[-1].date()}, "
         f"{abl.table[0]['n_blocks']} bloques\n",
         "\n## Log-loss OOS por peldano\n",
         "| modelo | covs | n_oos | log-loss OOS/obs | lambdas L1 por bloque |",
         "| :-- | :-- | --: | --: | :-- |"]
    for row in abl.table:
        lam = row.get("l1_lambdas_por_bloque")
        L.append(f"| {row['model']} | {row['covariates'] or '-'} | {row['n_oos']} | "
                 f"{_fmt(row['oos_logloss_per_obs'])} | {lam if lam else '-'} |")
    L += ["\n## Diebold-Mariano entre peldanos (DM<0 => A mejor)\n",
          "| A vs B | DM stat | p-value | dif. media |",
          "| :-- | --: | --: | --: |"]
    for d in abl.dm_pairs:
        L.append(f"| {d['model_a']} vs {d['model_b']} | {_fmt(d['dm_stat'],3)} | "
                 f"{_fmt(d['p_value'],3)} | {_fmt(d['mean_diff'],5)} |")
    dm_m3 = next((d for d in abl.dm_pairs if d["model_a"] == "M3" and d["model_b"] == "M2"), None)
    if dm_m3:
        better = dm_m3["dm_stat"] < 0
        dist = dm_m3["p_value"] < 0.10
        L.append("\n## Veredicto M3 (macro) CON L1\n")
        if better and dist:
            L.append(f"**M3 APORTA**: con L1, la macro mejora la densidad predictiva OOS sobre M2 "
                     f"de forma distinguible (DM={_fmt(dm_m3['dm_stat'],3)}, p={_fmt(dm_m3['p_value'],3)}).")
        elif better:
            L.append(f"**M3 nominalmente mejor pero NO distinguible** (DM={_fmt(dm_m3['dm_stat'],3)}, "
                     f"p={_fmt(dm_m3['p_value'],3)}): con L1 la macro ya no degrada, pero no acredita aporte.")
        else:
            L.append(f"**M3 no aporta** (DM={_fmt(dm_m3['dm_stat'],3)}, p={_fmt(dm_m3['p_value'],3)}): "
                     f"incluso con L1 la macro no mejora a M2.")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "ablation_m3_l1.md").write_text("\n".join(L), encoding="utf-8")
    log.info("escrito reports/ablation_m3_l1.md")

    # volcado JSON crudo para post-proceso
    (REPORTS / "ablation_m3_l1.json").write_text(
        json.dumps({"table": abl.table, "dm_pairs": abl.dm_pairs}, indent=2, default=str),
        encoding="utf-8")
    log.info("LISTO. Veredicto M3 vs M2: %s",
             f"DM={_fmt(dm_m3['dm_stat'],3)} p={_fmt(dm_m3['p_value'],3)}" if dm_m3 else "n/a")


if __name__ == "__main__":
    main()

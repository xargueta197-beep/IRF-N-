"""Ablacion M0..M5: una variable a la vez, jamas todas juntas (guia P8.1).

Escalera de modelos (cada peldano aisla UN aporte):

    M0  GARCH de un solo regimen               -- el piso; si el HMM no le gana, no hay proyecto
    M1  HMM K regimenes, P constante           -- el aporte de tener regimenes
    M2  + TVTP tecnico (sma_gap, bb_width_z)   -- el aporte de transiciones variables
    M3  + macro (slope_2s10y, hy_oas_z)        -- el aporte del macro           (V1+ con ALFRED)
    M4  + surprise_index                       -- noticias calendarizadas       (V2)
    M5  + lambda_N (Hawkes)                    -- flujo de noticias             (V3)

Este modulo define el MARCO completo; en V1 corren M0-M2 (y el modelo titular de
V1, TVTP con sma_gap + hy_oas_z, si la capa ALFRED esta disponible). M3/M4/M5 se
declaran en `full_ladder_specs` (V2: macro + sorpresa; V3: + lambda_N de Hawkes)
para que el diseno quede escrito ANTES de correrlos (R8). La pregunta que aisla
M5 vs M4: el flujo de noticias aporta POR ENCIMA de la sorpresa calendarizada?

--------------------------------------------------------------------------------
Metrica primaria: log-loss predictiva FUERA DE MUESTRA en walk-forward
--------------------------------------------------------------------------------
Se usa la log-densidad predictiva un-paso-adelante del RETORNO (el denominador
de la actualizacion de Hamilton, por observacion, en el tramo de test de cada
bloque; la perdida es su negativo). Por que esta y no la log-loss de regimen
sobre el proxy argmax: (1) es una regla de puntuacion PROPIA sobre algo
OBSERVABLE (el retorno), sin proxies; (2) es comparable entre modelos con
DISTINTO K (el proxy de regimen no lo es: cambia el espacio de clases); (3) es
exactamente la perdida que compara el Diebold-Mariano. Las metricas de
calibracion de regimen (Brier, ECE) se reportan como secundarias solo entre
modelos del mismo K.

Todos los modelos corren sobre la MISMA muestra alineada (mismo indice de
retornos, mismas ventanas de bloques): las perdidas quedan emparejadas por fecha
y el DM es valido. Alinear distinto por modelo invalidaria la comparacion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from irfn.validation.tests_stat import diebold_mariano
from irfn.validation.walkforward import WalkForwardResult, walk_forward

logger = logging.getLogger("irfn.ablation")


@dataclass(frozen=True)
class ModelSpec:
    """Especificacion declarativa de un peldano de la ablacion."""

    name: str
    K: int
    dist: str
    covariates: tuple[str, ...] | None    # None = matriz de transicion constante
    description: str


@dataclass
class AblationResult:
    specs: list[ModelSpec]
    wf_results: dict[str, WalkForwardResult]
    table: list[dict]                      # una fila por modelo (log-loss OOS, etc.)
    dm_pairs: list[dict] = field(default_factory=list)   # DM entre peldanos consecutivos
    loss_series: dict[str, pd.Series] = field(default_factory=dict)


def full_ladder_specs(
    K: int,
    dist: str,
    *,
    covariates_m2: list[str],
    covariates_m3: list[str],
    news_covariate: str = "surprise_index",
    hawkes_covariate: str = "lambda_N_z",
) -> list[ModelSpec]:
    """Escalera declarativa M0..M5 (diseno ANTES de correr, R8): cada peldano
    aisla UN aporte, acumulando covariables sobre el anterior.

        M0  GARCH un regimen (K=1), sin covariables            -- el piso
        M1  HMM K regimenes, P constante                       -- aporte de regimenes
        M2  + TVTP tecnico (covariates_m2)                      -- aporte de transiciones variables
        M3  + macro (covariates_m3, SUPERSET de covariates_m2)  -- aporte del macro (V1+, ALFRED)
        M4  + surprise_index (covariates_m3 + news_covariate)   -- noticias calendarizadas (V2)
        M5  + lambda_N_z (M4 + hawkes_covariate)                -- flujo de noticias (V3)

    `covariates_m3` debe incluir ya las de `covariates_m2` (M3 = "tecnico +
    macro", no "solo macro"): la escalera aisla el APORTE MARGINAL de cada
    peldano via el DM entre consecutivos, y eso solo es valido si cada peldano
    contiene estrictamente al anterior. M5 vs M4 responde la pregunta de V3:
    el flujo de titulares aporta POR ENCIMA de la sorpresa calendarizada?
    lambda_N_z y surprise_index entran como covariables SEPARADAS (guia 6.7:
    prohibido colapsarlas en un "news score").

    M4 y M5 se declaran SIEMPRE, existan o no sus columnas en los datos del
    llamador: run_ablation las valida y falla con un mensaje claro si faltan.
    Es responsabilidad del orquestador (scripts/run_v2.py / run_v3.py) decidir
    si hay datos para correrlas o si documenta el bloqueante SIN invocar
    run_ablation con ellas (R8: un bloqueante documentado > una excepcion sin
    contexto).
    """
    covariates_m4 = list(covariates_m3) + [news_covariate]
    covariates_m5 = covariates_m4 + [hawkes_covariate]
    return [
        ModelSpec("M0", 1, dist, None, "GARCH un solo regimen (piso)"),
        ModelSpec("M1", K, dist, None, f"HMM K={K} P constante"),
        ModelSpec("M2", K, dist, tuple(covariates_m2), f"+TVTP tecnico {covariates_m2}"),
        ModelSpec("M3", K, dist, tuple(covariates_m3), f"+macro {covariates_m3}"),
        ModelSpec("M4", K, dist, tuple(covariates_m4), f"+surprise_index {covariates_m4}"),
        ModelSpec("M5", K, dist, tuple(covariates_m5), f"+lambda_N_z (Hawkes) {covariates_m5}"),
    ]


def run_ablation(
    returns: pd.Series,
    X_all: pd.DataFrame | None,
    specs: list[ModelSpec],
    *,
    seed: int,
    n_starts: int,
    train_years: int,
    test_months: int,
    n_blocks_min: int,
    l1_grid: list[float] | None = None,
    cv_val_frac: float | None = None,
    cv_n_starts: int | None = None,
    l1_smooth_eps: float | None = None,
    checkpoint_dir: Path | None = None,
    n_jobs: int = 1,
) -> AblationResult:
    """Corre la escalera de ablacion sobre la MISMA muestra y malla de bloques.

    `returns` y `X_all` deben venir YA alineados (mismo indice, sin NaN): la
    alineacion se hace UNA vez sobre el conjunto de covariables mas grande de la
    escalera, para que M0 y M2 vean exactamente las mismas fechas.

    Si `checkpoint_dir` se da, el walk-forward de cada peldano (M0, M1, ...)
    reanuda por bloque (validation/walkforward.py) via un archivo propio
    `{checkpoint_dir}/wf_{spec.name}.pkl`; un peldano ya terminado en una
    corrida anterior se carga entero sin recalcular nada.
    """
    wf_results: dict[str, WalkForwardResult] = {}
    table: list[dict] = []
    loss_series: dict[str, pd.Series] = {}

    for spec in specs:
        if spec.covariates:
            if X_all is None:
                raise ValueError(f"{spec.name} requiere covariables y X_all es None.")
            missing = [c for c in spec.covariates if c not in X_all.columns]
            if missing:
                raise ValueError(f"{spec.name}: covariables ausentes de X_all: {missing}")
            X = X_all[list(spec.covariates)]
        else:
            X = None

        logger.info("Ablacion %s: K=%d dist=%s covs=%s", spec.name, spec.K, spec.dist, spec.covariates)
        ckpt_path = checkpoint_dir / f"wf_{spec.name}.pkl" if checkpoint_dir is not None else None
        wf = walk_forward(
            returns,
            K=spec.K,
            seed=seed,
            n_starts=n_starts,
            train_years=train_years,
            test_months=test_months,
            n_blocks_min=n_blocks_min,
            X=X,
            dist=spec.dist,
            l1_grid=l1_grid if X is not None else None,
            cv_val_frac=cv_val_frac,
            cv_n_starts=cv_n_starts,
            l1_smooth_eps=l1_smooth_eps,
            checkpoint_path=ckpt_path,
            n_jobs=n_jobs,
        )
        wf_results[spec.name] = wf

        ll = wf.oos_frame["loglik_obs"]
        loss_series[spec.name] = -ll                       # perdida = -log densidad predictiva
        table.append(
            {
                "model": spec.name,
                "description": spec.description,
                "K": spec.K,
                "dist": spec.dist,
                "covariates": list(spec.covariates) if spec.covariates else [],
                "n_blocks": wf.n_blocks,
                "n_oos": int(len(ll)),
                "oos_logloss_per_obs": float(-ll.mean()),  # PRIMARIA: menor es mejor
                "oos_loglik_per_obs": float(ll.mean()),
                "l1_lambdas_por_bloque": [b.l1_lambda for b in wf.blocks] if X is not None else None,
            }
        )

    # DM entre peldanos consecutivos: cada comparacion aisla UN aporte.
    dm_pairs: list[dict] = []
    for prev, curr in zip(specs[:-1], specs[1:]):
        a, b = loss_series[curr.name], loss_series[prev.name]
        common = a.index.intersection(b.index)
        res = diebold_mariano(a.loc[common].to_numpy(), b.loc[common].to_numpy())
        res.update({"model_a": curr.name, "model_b": prev.name})
        dm_pairs.append(res)

    return AblationResult(
        specs=list(specs),
        wf_results=wf_results,
        table=table,
        dm_pairs=dm_pairs,
        loss_series=loss_series,
    )

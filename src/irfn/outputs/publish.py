"""Guardian anti-smoother (R1) y generador del payload de artifacts/.

Ninguna ruta que llegue a `publish` puede contener el smoother de Kim, filtrado o
no: el chequeo de FORBIDDEN_KEYS es agresivo a proposito (el look-ahead no se
evita con buenas intenciones, se evita con un raise). `build_payload` arma el
diccionario del contrato a partir de la corrida del pipeline y la validacion,
calculando entropia, confianza, duracion esperada y estadisticas condicionales.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from irfn.outputs.schema import IRFNOutput
from irfn.validation.bootstrap import bootstrap_regime_stats

FORBIDDEN_KEYS = {"xi_smoothed", "smoothed", "kim_smoother", "xi_tT"}


class LookAheadViolation(Exception):
    """Se lanza cuando un payload de publicacion contiene una clave prohibida por R1."""


def _check_forbidden_keys(flat: str) -> None:
    for key in FORBIDDEN_KEYS:
        if key in flat:
            raise LookAheadViolation(
                f"'{key}' encontrado en el payload de publicacion. "
                f"Regla R1: jamas se publica el smoother."
            )


def publish(payload: dict, out_path: Path) -> None:
    """Verifica ausencia de claves prohibidas, valida payload contra el
    contrato, y escribe artifacts/latest/irfn.json a disco.

    El chequeo de FORBIDDEN_KEYS corre PRIMERO, sobre el payload crudo, antes
    de cualquier validacion de esquema (defensa en profundidad: R1 no debe
    depender de que el esquema siga siendo estricto en el futuro). payload
    debe validar contra IRFNOutput; esta funcion no repara ni completa
    payloads invalidos, solo los rechaza.
    """
    _check_forbidden_keys(json.dumps(payload, default=str))

    validated = IRFNOutput.model_validate(payload)
    contract = validated.to_contract_dict()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(contract, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Construccion del payload del contrato a partir de la corrida del pipeline.
# ---------------------------------------------------------------------------

def confidence_label(entropy_norm: float, mid: float, high: float) -> str:
    """Etiqueta de confianza segun la entropia normalizada H/ln(K) in [0,1].

    Umbrales de config (cero constantes magicas aqui). Por encima de `high` el
    modelo reparte la probabilidad de forma tan pareja que declararse por un
    regimen seria inventar una senal que no existe: "el modelo no distingue".
    """
    if entropy_norm >= high:
        return "el modelo no distingue"
    if entropy_norm >= mid:
        return "media"
    return "alta"


def expected_duration_days(P: np.ndarray) -> list[float]:
    """Duracion esperada de cada regimen: E[D_k] = 1/(1 - p_kk).

    Para una cadena de Markov, el tiempo de permanencia en k es geometrico con
    exito (1 - p_kk); su media es 1/(1 - p_kk) dias.
    """
    diag = np.clip(np.diag(np.asarray(P, dtype=float)), 0.0, 1.0 - 1e-12)
    return [float(1.0 / (1.0 - d)) for d in diag]


def conditional_stats(
    r_pct: np.ndarray,
    argmax_idx: np.ndarray,
    labels: list[str],
    asset: str,
    *,
    bootstrap_n_boot: int,
    bootstrap_block_len: float,
    bootstrap_ci_level: float,
    bootstrap_min_obs: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, dict]]:
    """Estadisticas de retorno por regimen (argmax de ξ_{t|t}), anualizadas, con
    su intervalo de confianza por bootstrap estacionario (Politis-Romano, V2;
    validation/bootstrap.py).

    r_pct son log-retornos en escala porcentual (como los usa el pipeline); se
    convierten a decimal para las metricas. Por cada regimen se toman los dias
    asignados a el y se calcula media anualizada (x252), vol anualizada (xsqrt252),
    Sharpe y maxima caida (sobre el equity de esos dias). Es descriptivo: NO es un
    backtest de estrategia (no se opera el regimen), solo caracteriza cada estado.

    Cada metrica es {"value", "ci_low", "ci_high"} (schema.MetricWithCI). El IC
    es None cuando el regimen tiene menos de `bootstrap_min_obs` dias (config
    v2.bootstrap.min_obs): con pocos datos se reporta el punto, nunca un
    intervalo inventado. bootstrap_seed se desplaza por regimen (+k) para que
    las replicas de cada regimen sean independientes pero reproducibles.
    """
    r_dec = np.asarray(r_pct, dtype=float) / 100.0
    idx = np.asarray(argmax_idx)
    out: dict[str, dict] = {}
    for k, label in enumerate(labels):
        sel = idx == k
        rk = r_dec[sel]
        stats = bootstrap_regime_stats(
            rk, n_boot=bootstrap_n_boot, block_len=bootstrap_block_len,
            ci_level=bootstrap_ci_level, seed=bootstrap_seed + k, min_obs=bootstrap_min_obs,
        )
        entry: dict[str, dict] = {}
        for name, (point, lo, hi) in stats.items():
            entry[name] = {
                "value": point if np.isfinite(point) else 0.0,
                "ci_low": lo,
                "ci_high": hi,
            }
        out[label] = entry
    return {asset: out}


def _v0_news_block() -> dict:
    """Bloque de noticias vacio para V0/V1: la capa de sorpresa/Hawkes es V2/V3. Se
    rellena con neutros y la atribucion es 100% precio, 0% noticias. El warning
    correspondiente lo agrega build_payload."""
    return {
        "surprise_index": 0.0,
        "lambda_N": 0.0,
        "lambda_N_z": 0.0,
        "branching_ratio": 0.0,
        "branching_ratio_ci_low": None,
        "branching_ratio_ci_high": None,
        "expected_cascade": 0.0,
        "expected_cascade_bounded": True,
        "attribution": {"price": 1.0, "news": 0.0},
    }


def default_hawkes_layer_params(blocker: str | None = None) -> dict:
    """hawkes_layer_params neutro: capa inactiva, nada estimado. Default de
    V0-V2 (la capa Hawkes no existia) y estado de V3 cuando el corpus de
    titulares o FinBERT no alcanzan -- `blocker` documenta por que (R8)."""
    return {
        "active": False,
        "mu_N": None, "alpha": None, "beta": None,
        "se_mu_N": None, "se_alpha": None, "se_beta": None,
        "mean_mark": None, "branching_ratio": None,
        "branching_ratio_se": None, "branching_ratio_ci_low": None, "branching_ratio_ci_high": None,
        "expected_cascade": None, "expected_cascade_bounded": None,
        "stationary": None,
        "ks_stat": None, "ks_pvalue": None, "ks_passed": None,
        "n_events": 0, "n_starts": 0, "starts_at_best": 0,
        "coverage": {"first_day": None, "last_day": None, "n_days": 0,
                     "n_missing_days": 0, "n_censored_days": 0},
        "reflexive_threshold": None,
        "blocker": blocker,
    }


def default_news_layer_params(blocker: str | None = None) -> dict:
    """news_layer_params neutro: capa inactiva, nada estimado. Es el default de
    V0/V1 (sin capa de noticias) y el estado de V2 mientras no haya datos
    suficientes -- `blocker` documenta por que (R8), None si simplemente no aplica
    (V0/V1, donde la capa todavia no existe en absoluto)."""
    return {
        "active": False,
        "delta": None,
        "delta_se": None,
        "surprise_start_date": None,
        "indicators": [],
        "coverage": {},
        "blocker": blocker,
    }


def build_payload(
    *,
    asof: date,
    version: str,
    run_id: str,
    git_commit: str,
    config_hash: str,
    K: int,
    labels: list[str],
    xi_history: np.ndarray,          # (T, K) ξ_{t|t} para momentum y estado de hoy
    P: np.ndarray,
    entropy_mid: float,
    entropy_high: float,
    seed: int,
    n_multistart: int,
    converged: bool,
    r_pct: np.ndarray,               # log-retornos % alineados con xi_history
    argmax_idx: np.ndarray,          # argmax por dia (para conditional_stats)
    asset: str,
    validation_ref: str,
    warnings: list[str],
    spec: str | None = None,
    tvtp: bool = False,
    covariates: list[str] | None = None,
    transition_matrix_today: np.ndarray | None = None,
    bootstrap_n_boot: int,
    bootstrap_block_len: float,
    bootstrap_ci_level: float,
    bootstrap_min_obs: int,
    news_layer: list[str] | None = None,
    news_block: dict | None = None,
    news_layer_params: dict | None = None,
    hawkes_layer_params: dict | None = None,
) -> dict:
    """Arma el diccionario del contrato (outputs/schema.py). No escribe a disco;
    eso lo hace `publish`, que ademas corre el guardian anti-smoother (R1).

    Parametros V1 (opcionales; los defaults reproducen el payload de V0):
      spec : cadena descriptiva del modelo (P constante Normal por defecto).
      tvtp : True si la matriz de transicion es variable en el tiempo (V1).
      covariates : nombres de las covariables del logit de transicion.
      transition_matrix_today : matriz de transicion CONDICIONAL evaluada en
        x_asof (la ultima covariable rezagada disponible). Con TVTP es distinta
        de la matriz de interceptos: es la que la pantalla 1 usa para el texto
        "la probabilidad de pasar a risk-off es X%". Si es None se publica P
        (caso V0, matriz constante).

    Bootstrap (v2.bootstrap; sin defaults a proposito -- vienen de config, cero
    constantes magicas): parametros del IC por bootstrap estacionario de
    conditional_stats (pantalla 4).

    Parametros V2 (opcionales; los defaults reproducen el payload de V0/V1 SIN
    capa de noticias):
      news_layer : nombres de covariables de noticias activas en el logit
        (["surprise_index"] cuando la capa esta prendida; [] si no).
      news_block : dict {"surprise_index","lambda_N","lambda_N_z",
        "branching_ratio","expected_cascade","attribution"} para "hoy". None =
        bloque neutro de _v0_news_block (100% atribucion a precio).
      news_layer_params : dict de outputs.publish.default_news_layer_params (w_i,
        delta, surprise_start_date, cobertura, bloqueante). None = neutro/inactivo.
    """
    xi_history = np.asarray(xi_history, dtype=float)
    xi_today = xi_history[-1]
    H = float(-np.sum(np.where(xi_today > 0, xi_today * np.log(xi_today), 0.0)))
    H_max = float(np.log(K)) if K > 1 else 0.0
    H_norm = H / H_max if H_max > 0 else 0.0

    if xi_history.shape[0] > 5:
        momentum = (xi_today - xi_history[-6]).tolist()
    else:
        momentum = [0.0] * K

    news_warnings = list(warnings)
    layer_params = news_layer_params if news_layer_params is not None else default_news_layer_params()
    hawkes_params = (hawkes_layer_params if hawkes_layer_params is not None
                     else default_hawkes_layer_params())
    if news_block is None:
        news_warnings.append(
            "Capa de noticias no implementada (surprise index y Hawkes son V2/V3); "
            "atribucion 100% precio por construccion."
        )
    elif not layer_params.get("active", False):
        blocker = layer_params.get("blocker")
        news_warnings.append(
            "Capa de noticias (V2) inactiva: " + (blocker or "surprise_layer=false en config/news.yaml.")
        )
    if news_block is not None and not hawkes_params.get("active", False):
        h_blocker = hawkes_params.get("blocker")
        news_warnings.append(
            "Capa de Hawkes (V3) inactiva: " + (h_blocker or "capa aun no corrida en este artefacto.")
        )
    if hawkes_params.get("active") and hawkes_params.get("stationary") is False:
        # No deberia llegar aqui (el orquestador trata n>=1 como bloqueante y
        # desactiva la capa), pero si llega, el warning viaja con el artefacto.
        news_warnings.append(
            "Branching ratio n >= 1: proceso de Hawkes explosivo. Es un bug o un "
            "problema de datos; el numero NO es interpretable como indicador."
        )

    P_today = P if transition_matrix_today is None else transition_matrix_today

    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc),
        "git_commit": git_commit,
        "config_hash": config_hash,
        "asof": asof,
        "version": version,
        "model": {
            "K": K,
            "spec": spec or "MS-GJR-GARCH (Haas et al. 2004), P constante, innovaciones Normales",
            "tvtp": tvtp,
            "covariates": covariates or [],
            "news_layer": news_layer or [],
            "estimation": {
                "method": "MLE L-BFGS-B multistart",
                "multistart": n_multistart,
                "seed": seed,
                "converged": converged,
            },
            "news_layer_params": layer_params,
            "hawkes_layer_params": hawkes_params,
        },
        "regime": {
            "labels": labels,
            "xi_filtered": xi_today.tolist(),
            "entropy": H,
            "entropy_max": H_max,
            "confidence": confidence_label(H_norm, entropy_mid, entropy_high),
            "expected_duration_days": expected_duration_days(P),
            "argmax": labels[int(xi_today.argmax())],
            "xi_momentum_5d": momentum,
        },
        "transition_matrix_today": np.asarray(P_today, dtype=float).tolist(),
        "news": news_block if news_block is not None else _v0_news_block(),
        "conditional_stats": conditional_stats(
            r_pct, argmax_idx, labels, asset,
            bootstrap_n_boot=bootstrap_n_boot, bootstrap_block_len=bootstrap_block_len,
            bootstrap_ci_level=bootstrap_ci_level, bootstrap_min_obs=bootstrap_min_obs,
            bootstrap_seed=seed,
        ),
        "warnings": news_warnings,
        "validation_ref": validation_ref,
        "disclaimer": (
            "IRF-N es un indicador de investigacion, no una senal de trading ni una "
            "promesa de rendimiento. Usa solo informacion disponible en la fecha de "
            "publicacion (point-in-time)."
        ),
    }
    return payload

"""Guardian anti-smoother (R1) y generador del payload de artifacts/.

Ninguna ruta que llegue a `publish` puede contener el smoother de Kim, filtrado o
no: el chequeo de FORBIDDEN_KEYS es agresivo a proposito (el look-ahead no se
evita con buenas intenciones, se evita con un raise). `build_payload` arma el
diccionario del contrato a partir de la corrida del pipeline y la validacion,
calculando entropia, confianza, duracion esperada y estadisticas condicionales.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from irfn.outputs.contract import (
    ContractViolation,
    _version_rank,
    validate_artifact,
)
from irfn.outputs.schema import IRFNOutput
from irfn.validation.bootstrap import bootstrap_regime_stats

FORBIDDEN_KEYS = {"xi_smoothed", "smoothed", "kim_smoother", "xi_tT"}

# Rango minimo de version promovible a latest/ sin --force-downgrade. La regresion
# que motivo todo esto fue publicar V0 encima de V3; el guardarrail lo prohibe.
MIN_PROMOTABLE_RANK = 3  # V3


class LookAheadViolation(Exception):
    """Se lanza cuando un payload de publicacion contiene una clave prohibida por R1."""


def _check_forbidden_keys(flat: str) -> None:
    for key in FORBIDDEN_KEYS:
        if key in flat:
            raise LookAheadViolation(
                f"'{key}' encontrado en el payload de publicacion. "
                f"Regla R1: jamas se publica el smoother."
            )


# ---------------------------------------------------------------------------
# Promocion atomica a artifacts/latest/ (UNICO punto de escritura de latest/).
#
# Flujo (Fase 3 de la remediacion): las corridas escriben SIEMPRE en
# artifacts/runs/<run_id>/. Para que un artefacto llegue a latest/ hay que
# promoverlo con `promote_run`, que (1) escribe el manifiesto, (2) corre el
# validador del contrato, (3) aplica el guardarrail anti-downgrade y (4) hace el
# swap atomico. Nada de copiar archivo por archivo: latest/ nunca queda en un
# estado intermedio, y si el proceso muere a mitad del swap se recupera solo.
# ---------------------------------------------------------------------------

def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_manifest(run_dir: Path) -> Path:
    """Escribe run_dir/manifest.json (run_id + version + sha256 de cada archivo).
    Es lo que le permite al contrato afirmar 'procedencia unica'."""
    run_dir = Path(run_dir)
    irfn = json.loads((run_dir / "irfn.json").read_text(encoding="utf-8"))
    files: dict[str, dict] = {}
    for p in sorted(run_dir.iterdir()):
        if p.is_file() and p.name != "manifest.json":
            files[p.name] = {"sha256": _sha256(p), "size": p.stat().st_size}
    manifest = {
        "run_id": irfn.get("run_id"),
        "version": irfn.get("version"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _recover_latest(latest_dir: Path) -> None:
    """Si latest/ no existe pero quedo un .latest.trash.* de un swap interrumpido,
    restaura el mas reciente. Idempotente."""
    latest_dir = Path(latest_dir)
    if latest_dir.exists():
        return
    parent = latest_dir.parent
    trashes = sorted(parent.glob(".latest.trash.*"))
    if trashes:
        os.replace(str(trashes[-1]), str(latest_dir))
        for t in trashes[:-1]:
            shutil.rmtree(t, ignore_errors=True)


def _atomic_swap(run_dir: Path, latest_dir: Path) -> None:
    """Reemplaza latest/ por una copia completa de run_dir mediante renames
    atomicos (cada os.replace tiene destino inexistente => atomico en NTFS/POSIX).

    Invariante: latest/ SOLO puede observarse como (a) el set viejo completo,
    (b) inexistente por un instante entre los dos renames -recuperable desde el
    trash-, o (c) el set nuevo completo. NUNCA una mezcla archivo-por-archivo.
    """
    run_dir = Path(run_dir)
    latest_dir = Path(latest_dir)
    parent = latest_dir.parent
    parent.mkdir(parents=True, exist_ok=True)

    staging = parent / f".latest.staging.{os.getpid()}.{int(time.time()*1000)}"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(run_dir, staging)
    # marcador de carpeta versionada (lo pide .gitignore)
    (staging / ".gitkeep").write_text("", encoding="utf-8")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    trash = parent / f".latest.trash.{ts}"
    if latest_dir.exists():
        os.replace(str(latest_dir), str(trash))   # rename 1: dst inexistente
    try:
        os.replace(str(staging), str(latest_dir))  # rename 2: dst inexistente
    except BaseException:
        # restaurar el estado anterior si el segundo rename fallo
        if not latest_dir.exists() and trash.exists():
            os.replace(str(trash), str(latest_dir))
        raise
    if trash.exists():
        shutil.rmtree(trash, ignore_errors=True)


def _append_publish_log(artifacts_root: Path, res, *, triggered_by: str, forced: bool) -> None:
    line = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": res.run_id,
        "version": res.version,
        "forced": forced,
        "provisional": bool(res.provisional_reasons),
        "triggered_by": triggered_by,
    }
    log_path = Path(artifacts_root) / "publish_log.jsonl"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")


def promote_run(
    run_dir: Path,
    *,
    latest_dir: Path,
    repo_root: Path | None = None,
    force_downgrade: bool = False,
    allow_provisional: bool = False,
    triggered_by: str = "desconocido",
    tolerance_days: int = 5,
):
    """UNICO punto de escritura de artifacts/latest/. Promueve run_dir a latest/
    solo si pasa el contrato y el guardarrail. Devuelve el ContractResult.

    - Violaciones DURAS del contrato (procedencia mezclada, set incompleto, hueco
      de frescura, git=nogit, ...): NUNCA se promueven, ni con force_downgrade.
    - version < V3 o corrida provisional (R6): bloqueado salvo force_downgrade
      explicito, que ademas queda registrado en artifacts/publish_log.jsonl.
    """
    run_dir = Path(run_dir)
    latest_dir = Path(latest_dir)

    write_manifest(run_dir)
    _recover_latest(latest_dir)  # por si un swap anterior murio a la mitad

    res = validate_artifact(
        run_dir, tolerance_days=tolerance_days,
        allow_provisional=allow_provisional, repo_root=repo_root,
    )
    if res.violations:
        detail = "\n".join("  - " + v for v in res.violations)
        raise ContractViolation(
            f"Promocion RECHAZADA ({run_dir}): violaciones duras del contrato:\n{detail}")

    blockers: list[str] = []
    if _version_rank(res.version) < MIN_PROMOTABLE_RANK:
        blockers.append(f"version {res.version} < V{MIN_PROMOTABLE_RANK}")
    if res.provisional_reasons:
        blockers.append("corrida provisional (R6): " + "; ".join(res.provisional_reasons))
    if blockers and not force_downgrade:
        raise ContractViolation(
            "Promocion BLOQUEADA por el guardarrail: " + "; ".join(blockers) +
            ". Requiere force_downgrade=True explicito (queda en publish_log.jsonl).")

    _atomic_swap(run_dir, latest_dir)
    _append_publish_log(latest_dir.parent, res, triggered_by=triggered_by, forced=bool(blockers))
    return res


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
    expected_durations: list[float] | None = None,
    degenerate_duration_days: float = 0.0,
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

    Regimen DEGENERADO (F4): si `expected_durations[k] < degenerate_duration_days`
    (E[D]=1/(1-p_kk) ~ 1 dia, un 'absorbe-outliers' que no persiste), se publica
    el punto de TODAS sus metricas pero se SUPRIME el IC (None): sus dias son
    excursiones sueltas, no una serie con dependencia de corto plazo, asi que el
    bootstrap por bloques finge una precision que no existe. Mismo criterio que
    la app (components.DEGENERATE_DURATION_DAYS). degenerate_duration_days=0.0
    (default) desactiva la supresion (compat V0/V1 sin el umbral configurado).
    """
    r_dec = np.asarray(r_pct, dtype=float) / 100.0
    idx = np.asarray(argmax_idx)
    edd = expected_durations if expected_durations is not None else []
    out: dict[str, dict] = {}
    for k, label in enumerate(labels):
        sel = idx == k
        rk = r_dec[sel]
        stats = bootstrap_regime_stats(
            rk, n_boot=bootstrap_n_boot, block_len=bootstrap_block_len,
            ci_level=bootstrap_ci_level, seed=bootstrap_seed + k, min_obs=bootstrap_min_obs,
        )
        degenerate = k < len(edd) and edd[k] < degenerate_duration_days
        entry: dict[str, dict] = {}
        for name, (point, lo, hi) in stats.items():
            entry[name] = {
                "value": point if np.isfinite(point) else 0.0,
                "ci_low": None if degenerate else lo,
                "ci_high": None if degenerate else hi,
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
    bootstrap_degenerate_duration_days: float = 0.0,
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
    # E[D_k] = 1/(1-p_kk): se calcula una vez y se reutiliza para el bloque de
    # regimen y para decidir que regimenes son degenerados en conditional_stats
    # (les suprimimos el IC, F4).
    durations = expected_duration_days(P)

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
            "expected_duration_days": durations,
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
            expected_durations=durations,
            degenerate_duration_days=bootstrap_degenerate_duration_days,
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

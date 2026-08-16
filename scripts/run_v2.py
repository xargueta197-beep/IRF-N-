"""Orquestador de V2: capa de sorpresa calendarizada (indice de sorpresa SI_t).

    python scripts/run_v2.py            # corrida completa V2
    python scripts/run_v2.py --quick    # menos arranques (humo, no R6)

Orden (CLAUDE.md, sesion V2 2026-07-12):
  a) descargar calendario actualizado (capture_consensus.py + data/calendar.py)
  b) calcular z_i y w_i (features/surprise.py)
  c) construir SI_t (eventos ponderados)
  d) re-estimar el modelo completo con la covariable (models/estimate.py, R6)
  e) correr walk-forward completo (>=6 bloques)
  f) calcular atribucion precio/noticias
  g) correr ablacion M3 vs M4
  h) publicar artifacts/latest/irfn.json version "V2"
  i) generar reports/ablation_news.md

Bloqueantes de datos de ESTA sesion (reports/data_audit.md secciones 4 y 7;
CLAUDE.md V1 "macro_blocker"): CERO dias de consenso historico capturados
(Trading Economics es de pago sin TRADING_ECONOMICS_API_KEY configurada; Forex
Factory fue auditada y descartada por ToS -- ver data_audit.md seccion 7) y
CERO ALFRED_API_KEY (M3/macro ya estaba bloqueado desde V1). Este script hace
TODO lo de arriba de verdad, con datos reales cuando existen; donde los datos
no alcanzan, documenta el bloqueante (R8) en vez de fingir un resultado. Nada
se resuelve escondiendo el problema o imputando consenso (config/news.yaml).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from irfn.audit.pit import (  # noqa: E402
    block_reestimation_check,
    consensus_vintage_ledger,
    expanding_window_check,
    lag_ledger,
    prefix_invariance_check,
    v1_feature_registry,
)
from irfn.config import BaseConfig, NewsConfig  # noqa: E402
from irfn.data import calendar as calendar_data  # noqa: E402
from irfn.data.prices import load_close  # noqa: E402
from irfn.features import surprise as surprise_mod  # noqa: E402
from irfn.features.technical import technical_features  # noqa: E402
from irfn.models.estimate import fit  # noqa: E402
from irfn.models.tvtp import transition_matrix_at  # noqa: E402
from irfn.outputs.publish import build_payload, publish  # noqa: E402
from irfn.pipeline import _filter_frame, regime_labels, run_pipeline, standardize_with, train_scaler  # noqa: E402
from irfn.validation.ablation import full_ladder_specs, run_ablation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("run_v2")

ARTIFACTS = ROOT / "artifacts" / "latest"
RUNS = ROOT / "artifacts" / "runs"
REPORTS = ROOT / "reports"
CONFIG_DIR = ROOT / "config"
VERSION = "V2"

# Activo de la corrida (globals de modulo, fijados por _configure_asset() en
# main() antes de correr full_run()). Default = SPY, identico al historico.
ASSET_NAME = None
ASSET_SOURCE = "yfinance"
ASSET_IDENTIFIER = None
ASSET_PRICES_START = None
ASSET_SAMPLE_START = None
REPORT_SUFFIX = ""


def _configure_asset(base: BaseConfig, asset_override: str | None) -> None:
    """Ver docstring gemelo en run_v1.py. Sin --asset: comportamiento identico al
    historico (SPY, artifacts/latest/, reports/ablation_news.md). Con --asset
    NAME: artifacts/<name>/ + reports/ablation_news_<name>.md, aditivo."""
    global ARTIFACTS, RUNS, ASSET_NAME, ASSET_SOURCE, ASSET_IDENTIFIER
    global ASSET_PRICES_START, ASSET_SAMPLE_START, REPORT_SUFFIX
    if asset_override is None:
        ASSET_NAME = base.v0.anchor_asset
        ASSET_SOURCE = "yfinance"
        ASSET_IDENTIFIER = base.v0.anchor_asset
        ASSET_PRICES_START = base.v1.prices_start
        ASSET_SAMPLE_START = base.v0.returns_start
        REPORT_SUFFIX = ""
        ARTIFACTS = ROOT / "artifacts" / "latest"
        RUNS = ROOT / "artifacts" / "runs"
        return
    from irfn.config import AssetsConfig

    asset_cfg = AssetsConfig.load().get(asset_override)
    slug = asset_cfg.name.lower()
    ASSET_NAME = asset_cfg.name
    ASSET_SOURCE = asset_cfg.source
    ASSET_IDENTIFIER = asset_cfg.identifier
    ASSET_PRICES_START = asset_cfg.start_date
    ASSET_SAMPLE_START = asset_cfg.start_date
    REPORT_SUFFIX = f"_{slug}"
    ARTIFACTS = ROOT / "artifacts" / slug / "latest"
    RUNS = ROOT / "artifacts" / slug / "runs"


PRE_REGISTERED_COMMITMENT = (
    "Compromiso pre-registrado: si M4 no mejora la log-loss de forma "
    "distinguible (DM p < 0.10), el indicador se publica sin capa de sorpresa."
)


# --------------------------------------------------------------------------- #
# Identidad reproducible (misma logica que run_v0/run_v1)
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=ROOT, capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "nogit"


def _config_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(CONFIG_DIR.glob("*.yaml")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _run_id(config_hash: str, git_commit: str, seed: int) -> str:
    return hashlib.sha256(f"{config_hash}|{git_commit}|{seed}|{VERSION}".encode()).hexdigest()[:12]


def _dump(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    log.info("escrito %s", path.relative_to(ROOT))


def _fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# --------------------------------------------------------------------------- #
# a) Ingesta del calendario
# --------------------------------------------------------------------------- #
def ingest_calendar(news_cfg: NewsConfig) -> tuple[pd.DataFrame, dict]:
    """Captura el snapshot de hoy (idempotente, no sobreescribe) y carga TODO lo
    acumulado en data/raw/consensus_calendar/ (data/calendar.load_local_snapshots).
    Nunca lanza por un fallo de captura: el hueco es informacion (docstring de
    capture_consensus.py), no un motivo para detener el orquestador."""
    try:
        from capture_consensus import capture_snapshot

        capture_snapshot()
    except Exception as exc:  # noqa: BLE001 -- ver docstring: nunca detiene el orquestador
        log.warning("captura de hoy no disponible (se documenta, no bloquea): %s", exc)

    indicators = [i.name for i in news_cfg.indicators]
    cal = calendar_data.load_local_snapshots(indicators)
    coverage = calendar_data.coverage_summary(cal, indicators)
    log.info("calendario: %d eventos de interes acumulados (%s).",
             len(cal), ", ".join(f"{k}={v['n_releases']}" for k, v in coverage.items()))
    return cal, coverage


# --------------------------------------------------------------------------- #
# b) z_i, w_i -- regresion de impacto (R7). Plan B sin intradia (surprise.py
#    lo documenta): |retorno DIARIO| del activo ancla alrededor del release.
# --------------------------------------------------------------------------- #
def _abs_daily_return(returns_pct: pd.Series) -> pd.Series:
    s = returns_pct.abs().copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s


def compute_z_and_weights(
    cal: pd.DataFrame, news_cfg: NewsConfig, anchor_returns_pct: pd.Series
) -> tuple[pd.DataFrame, dict]:
    if cal.empty:
        return pd.DataFrame(), {}
    z_wide, _si_coverage = surprise_mod.expanding_surprise_z(cal, min_obs=news_cfg.sigma_min_obs)
    abs_ret = _abs_daily_return(anchor_returns_pct)
    abs_z: dict[str, pd.Series] = {}
    for ind in z_wide.columns:
        z = z_wide[ind].abs().dropna()
        z.index = pd.DatetimeIndex(z.index).tz_localize(None).normalize()
        abs_z[ind] = z
    weights = surprise_mod.estimate_impact_weights(
        abs_ret, abs_z, t_threshold=news_cfg.impact_t_threshold, min_events=3, use_intraday=False,
    )
    return z_wide, weights


def _events_table(z_wide: pd.DataFrame, weights: dict) -> list[dict]:
    """Un renglon por release con consenso (z_i estimable), para la serie de
    puntos de pantalla 3 (tamano ∝ |z_i|, color = signo). Independiente de si
    la capa esta activa: existe en cuanto haya al menos un z_i valido."""
    if z_wide.empty:
        return []
    rows = []
    for ind in z_wide.columns:
        w = weights.get(ind, {}).get("w", float("nan"))
        w_val = float(w) if np.isfinite(w) else None
        for ts, z in z_wide[ind].dropna().items():
            rows.append({
                "fecha": str(pd.Timestamp(ts).date()),
                "indicator": ind,
                "z": float(z),
                "w": w_val,
                "contribution": (w_val * float(z) if w_val is not None else None),
            })
    rows.sort(key=lambda r: r["fecha"])
    return rows


def _indicator_weight_rows(news_cfg: NewsConfig, weights: dict, coverage: dict) -> list[dict]:
    """Fila por CADA indicador de config/news.yaml, exista o no en `weights`
    (sin datos, entra con w=None y distinguible_de_cero=False: la tabla de
    pantalla 3 muestra los 6 indicadores siempre, no solo los que tuvieron suerte)."""
    rows = []
    for ind_cfg in news_cfg.indicators:
        name = ind_cfg.name
        w = weights.get(name)
        cov = coverage.get(name, {})
        if w is None:
            rows.append({
                "indicator": name, "w": None, "se": None, "t_stat": None,
                "distinguible_de_cero": False, "n_events": int(cov.get("n_with_consensus", 0)),
            })
        else:
            rows.append({
                "indicator": name,
                "w": (float(w["w"]) if np.isfinite(w["w"]) else None),
                "se": (float(w["se"]) if np.isfinite(w["se"]) else None),
                "t_stat": (float(w["t_stat"]) if np.isfinite(w["t_stat"]) else None),
                "distinguible_de_cero": bool(w["distinguible_de_cero"]),
                "n_events": int(w["n_events"]),
            })
    return rows


# --------------------------------------------------------------------------- #
# d) Re-estimacion del modelo completo con delta libre (R7, MLE del modelo
#    completo). delta es un HIPERPARAMETRO DE INGENIERIA DE FEATURES estimado
#    UNA VEZ por MLE de muestra completa (igual que z_window/bb_window son fijos
#    de config, no re-estimados por bloque): una vez fijo, SI_t(delta) entra al
#    walk-forward como CUALQUIER OTRA covariable precomputada, y son los beta_ij
#    del logit -- los parametros de verdad sujetos a R2 -- los que SI se
#    re-estiman desde cero en cada bloque. Decision de diseno documentada aqui
#    para que el director la revise (CLAUDE.md: "si crees que una decision de
#    diseno es incorrecta, detente y dilo"); no se re-deriva delta por bloque
#    porque hoy no hay ni un evento con el que hacerlo (ver mas abajo).
# --------------------------------------------------------------------------- #
def fit_delta_mle(
    events: pd.Series, target_index: pd.DatetimeIndex, X_tech_std: np.ndarray, r: np.ndarray,
    K: int, dist: str, base: BaseConfig,
) -> dict:
    """Intenta el fit conjunto (delta libre via surprise_spec). Devuelve dict
    con delta/delta_se/blocker/fit_result/si_of_delta (None si bloqueado)."""
    min_events = base.v2.delta_mle.min_events_total
    n_events = int(len(events))
    if n_events < min_events:
        return {
            "delta": None, "delta_se": None, "fit_result": None, "si_of_delta": None,
            "blocker": (
                f"solo {n_events} evento(s) valido(s) de consenso acumulados "
                f"(se necesitan >= {min_events}, config v2.delta_mle.min_events_total); "
                "no se intenta el MLE conjunto de delta. Ver reports/data_audit.md."
            ),
        }

    def si_of_delta(delta: float) -> np.ndarray:
        raw = surprise_mod.surprise_feature(events, target_index, delta=float(delta))
        raw = np.nan_to_num(raw.to_numpy(dtype=float), nan=0.0)
        std = raw.std()
        return (raw - raw.mean()) / (std if std > 0 else 1.0)

    fr = fit(
        r, K=K, n_starts=base.v2.delta_mle.n_starts, seed=base.model.seed, compute_se=True,
        X_lagged=X_tech_std, dist=dist, surprise_spec={"si_of_delta": si_of_delta},
    )
    delta_hat = float(fr.params["delta"])
    se_delta = float(fr.se["delta"][0]) if fr.se.get("delta") is not None else None
    return {"delta": delta_hat, "delta_se": se_delta, "fit_result": fr,
            "si_of_delta": si_of_delta, "blocker": None}


# --------------------------------------------------------------------------- #
# Macro (M3): mismo bloqueante que V1 (ALFRED_API_KEY). Se re-intenta aqui
# porque la config pudo cambiar entre sesiones; si sigue sin key, se documenta
# igual que run_v1.py ya lo hacia.
# --------------------------------------------------------------------------- #
def try_macro_covariates(base: BaseConfig, target_index: pd.DatetimeIndex) -> tuple[pd.DataFrame | None, str | None]:
    from irfn.data.alfred import MissingAPIKeyError, fetch_vintage_observations, point_in_time_series
    from irfn.data.recovered import splice_recovered_prefix
    from irfn.features.macro import macro_features

    try:
        pit = {}
        for s in base.macro.series:
            vintages = fetch_vintage_observations(s.series_id)
            if base.macro.use_recovered_prefix:
                # prefijo Wayback para series truncadas por la ventana rodante
                # de FRED (decision del director 2026-07-18, data_audit secc. 3);
                # si no hay snapshot recuperado para la serie, no cambia nada.
                vintages, splice_info = splice_recovered_prefix(s.series_id, vintages)
                if splice_info is not None:
                    print(
                        f"      {s.series_id}: prefijo recuperado "
                        f"{splice_info['prefix_first_obs']}..{splice_info['prefix_last_obs']} "
                        f"({splice_info['prefix_n_obs']} obs) + ALFRED desde "
                        f"{splice_info['alfred_first_obs']}."
                    )
            pit[s.series_id] = point_in_time_series(
                vintages, margin_days=base.macro.availability_margin_days, name=s.series_id,
            )
        X_macro = macro_features(pit, target_index, z_window=base.windows.z_window)
        if X_macro.empty:
            return None, "covariables macro vacias tras alinear."
        # Distinguir warm-up INICIAL (benigno: el z-score movil de z_window dias
        # necesita ~1 anio de arranque, igual que cualquier feature rodante --
        # load_sample ya recorta el de las tecnicas) de huecos INTERNOS (cobertura
        # real insuficiente, que SI debe bloquear, R4). Se rechaza solo lo segundo;
        # el warm-up inicial lo recorta el llamador alineando la muestra al tramo
        # comun sin NaN (dropna) antes de la ablacion, dejando M0..M3 sobre las
        # MISMAS fechas (Diebold-Mariano valido).
        internal_na = 0
        for c in X_macro.columns:
            fv = X_macro[c].first_valid_index()
            lv = X_macro[c].last_valid_index()
            if fv is None:
                return None, f"covariable macro {c} sin ningun valor valido en la muestra."
            internal_na += int(X_macro[c].loc[fv:lv].isna().sum())
        if internal_na > 0:
            return None, (
                f"covariables macro con {internal_na} NaN INTERNOS (huecos de cobertura, "
                "no warm-up); no se usan hasta tener una fuente vintage con cobertura "
                "continua (R4). Ver reports/data_audit.md."
            )
        return X_macro, None
    except MissingAPIKeyError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 -- se documenta como bloqueante, no se detiene el script
        return None, f"fallo construyendo covariables macro: {exc}"


# --------------------------------------------------------------------------- #
# Muestra: retorno del activo ancla + covariables tecnicas M2 (misma receta de
# run_v1.load_sample; se duplica en vez de importar entre scripts/ -- src/ es
# la unica fuente de logica compartida).
# --------------------------------------------------------------------------- #
def load_sample(base: BaseConfig) -> tuple[pd.Series, pd.DataFrame]:
    w = base.windows
    close = load_close(ASSET_IDENTIFIER, ASSET_PRICES_START, source=ASSET_SOURCE)
    r_all = (np.log(close).diff() * 100.0).rename("r")
    X_all = technical_features(
        close, sma_short=w.sma_short, sma_long=w.sma_long,
        bb_window=w.bb_window, bb_k=w.bb_k, z_window=w.z_window,
    )
    covs = base.v1.covariates_ablation_m2
    df = pd.concat([r_all, X_all[covs]], axis=1).loc[ASSET_SAMPLE_START:].dropna()
    return df["r"], df[covs]


def _load_titular_K_dist(base: BaseConfig) -> tuple[int, str]:
    ks_path = ARTIFACTS.parent / "analysis" / "v1_kselect.json"
    if ks_path.exists():
        ks = json.loads(ks_path.read_text(encoding="utf-8"))
        K, dist = ks["winner"]["K"], ks["winner"]["dist"]
        # Se RESPETA el K seleccionado por BIC, INCLUIDO K=1 (decision del director
        # 2026-08-15). Antes esto hacia `if K > 1: return K,dist` y, cuando el BIC
        # elegia K=1, caia al default K=2/normal de abajo -- sustituyendo EN SILENCIO
        # el modelo que los datos eligieron por uno que rechazaron. Para BTC el BIC
        # elige K=1 t (14559.11, 18/20 arranques R6) y el diagnostico de estabilidad
        # confirma que el optimo de K=2 t es un artefacto numerico (1/100 incluso con
        # 100 arranques; reports/diag_btc_k2t_stability.md). Publicar K=2 cuando el
        # BIC eligio K=1 violaba la disciplina de seleccion del proyecto. Con K=1 no
        # hay logit de transicion: el titular es un GARCH de un solo regimen (run_v3
        # lo maneja explicitamente; lambda_N_z/SI_t no aplican como covariables, y el
        # branching ratio de Hawkes se publica como indicador de fragilidad standalone).
        return K, dist
    # El default K=2/normal es SOLO para "falta la seleccion" (kselect ausente), NUNCA
    # para pisar un K=1 legitimamente seleccionado.
    log.warning("v1_kselect.json ausente; usando K=2 dist=normal por defecto (solo por ausencia de seleccion).")
    return 2, "normal"


# --------------------------------------------------------------------------- #
# Corrida completa
# --------------------------------------------------------------------------- #
def full_run(quick: bool = False, n_jobs: int = 1) -> None:
    base = BaseConfig.load()
    news_cfg = NewsConfig.load()
    seed = base.model.seed
    n_starts_today = 8 if quick else base.model.n_multistart

    config_hash, git_commit = _config_hash(), _git_commit()
    run_id = _run_id(config_hash, git_commit, seed)

    # --- a) calendario ---
    log.info("[a] ingesta del calendario macro...")
    cal, coverage = ingest_calendar(news_cfg)
    vintage_df = consensus_vintage_ledger(cal)
    log.info("    vintage ledger: %s (%d eventos)", "VERDE" if vintage_df.attrs["passed"] else "ROJO",
             vintage_df.attrs["n_events"])

    # --- muestra: retorno ancla + tecnicas M2 (siempre disponible) ---
    log.info("    cargando muestra (activo ancla + tecnicas)...")
    returns, X_tech = load_sample(base)
    r = returns.to_numpy(dtype=float)
    Kt, distt = _load_titular_K_dist(base)
    log.info("    titular V1: K=%d dist=%s (de v1_kselect.json si existe).", Kt, distt)

    # --- b) z_i, w_i ---
    log.info("[b] z_i (sigma_i expanding, warm-up=%d) y w_i (regresion de impacto)...",
             news_cfg.sigma_min_obs)
    z_wide, weights = compute_z_and_weights(cal, news_cfg, returns)
    ew_df = expanding_window_check(cal, min_obs=news_cfg.sigma_min_obs)
    log.info("    expanding window check: %s%s", "VERDE" if ew_df.attrs["passed"] else "ROJO",
             " (vacuo: <2 eventos con consenso)" if ew_df.attrs.get("vacuous") else "")

    # --- c) SI_t (eventos ponderados w_i*z_i) ---
    events = surprise_mod.weighted_events_from(z_wide, weights) if len(z_wide.columns) else pd.Series(dtype=float)
    log.info("[c] SI_t: %d eventos con contribucion valida (w_i*z_i).", len(events))

    # --- scaler de tecnicas: TODO el historico hasta asof (V1: "artefacto de
    #     hoy", no es look-ahead usar todo el pasado para publicar hoy). ---
    tech_scaler = train_scaler(X_tech, None)
    X_tech_std = standardize_with(X_tech, tech_scaler)

    # --- d) re-estimacion completa con delta libre (o bloqueante documentado) ---
    log.info("[d] re-estimacion del modelo completo con la covariable de sorpresa...")
    delta_res = fit_delta_mle(events, returns.index, X_tech_std, r, Kt, distt, base)
    news_active = delta_res["blocker"] is None

    if news_active:
        log.info("    delta_hat=%.5f (SE=%s)", delta_res["delta"], _fmt(delta_res["delta_se"], 5))
        si_full = delta_res["si_of_delta"](delta_res["delta"])
        titular_fr = delta_res["fit_result"]
        X_today_std = np.column_stack([X_tech_std, si_full])
        today_cov_names = list(X_tech.columns) + [base.v2.news_covariate_name]
        frame = _filter_frame(returns, titular_fr.params, Kt, X_std=X_today_std)
        titular_params = titular_fr.params
        titular_n_converged, titular_n_starts, titular_hessian_ok = (
            titular_fr.n_converged, titular_fr.n_starts, titular_fr.hessian_ok)
    else:
        log.warning("    BLOQUEADO: %s", delta_res["blocker"])
        si_full = None
        today_run = run_pipeline(returns, K=Kt, seed=seed, n_starts=n_starts_today,
                                 compute_se=True, X=X_tech, dist=distt)
        X_today_std = standardize_with(X_tech, today_run.scaler)
        today_cov_names = list(X_tech.columns)
        frame = today_run.frame
        titular_params = today_run.fit.params
        titular_n_converged, titular_n_starts, titular_hessian_ok = (
            today_run.fit.n_converged, today_run.fit.n_starts, today_run.fit.hessian_ok)

    xi_hist = frame[[f"xi_filtered_{k}" for k in range(Kt)]].to_numpy()
    r_pct_hist = frame["r"].to_numpy()
    argmax_hist = frame["argmax_idx"].to_numpy()

    # --- macro (M3): mismo bloqueante de V1, re-chequeado ---
    log.info("    intentando covariables macro (M3)...")
    X_macro, macro_blocker = try_macro_covariates(base, returns.index)
    if macro_blocker:
        log.warning("    BLOQUEADO (macro/M3): %s", macro_blocker)

    # --- e/g) ablacion completa M0..M4: se declara SIEMPRE (R8); se CORRE solo
    #     lo que los datos permiten hoy. M4 requiere M3 (macro) por diseno de
    #     la escalera (M4 = M3 + surprise_index, acumulativo). ---
    log.info("[e/g] ablacion M0..M4 (walk-forward, >=6 bloques por peldano corrido)...")
    covs_m3 = list(base.v2.covariates_ablation_m3)
    specs_all = full_ladder_specs(
        Kt, distt, covariates_m2=list(X_tech.columns), covariates_m3=covs_m3,
        news_covariate=base.v2.news_covariate_name,
    )
    by_name = {s.name: s for s in specs_all}

    X_all_parts = [X_tech]
    runnable_names = ["M0", "M1", "M2"]
    if X_macro is not None:
        X_all_parts.append(X_macro)
        runnable_names.append("M3")
        if news_active:
            X_all_parts.append(pd.Series(si_full, index=returns.index, name="surprise_index").to_frame())
            runnable_names.append("M4")
    X_all = pd.concat(X_all_parts, axis=1)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()]

    # Alinear la muestra de la ablacion al tramo comun SIN NaN. Con macro, las
    # covariables traen warm-up inicial del z-score (z_window dias); recortarlo
    # es el mismo trato que load_sample da a las tecnicas. Todos los peldanos
    # (M0..) corren sobre las MISMAS fechas -> el DM entre peldanos es valido.
    # Sin macro, X_all no tiene NaN y esto es un no-op (muestra tecnica completa).
    n_before = len(X_all)
    X_all = X_all.dropna()
    returns_abl = returns.loc[X_all.index]
    if len(X_all) < n_before:
        log.info("    ablacion alineada al tramo con macro: %d -> %d filas (%s..%s); "
                 "recortado el warm-up del z-score macro (mismo trato que las tecnicas).",
                 n_before, len(X_all), X_all.index[0].date(), X_all.index[-1].date())

    runnable_specs = [by_name[n] for n in runnable_names]
    n_starts_wf = 6 if quick else base.v0.wf_n_starts
    abl = run_ablation(
        returns_abl, X_all, runnable_specs, seed=seed, n_starts=n_starts_wf,
        train_years=base.walkforward.train_years, test_months=base.walkforward.test_months,
        n_blocks_min=base.walkforward.n_blocks, l1_grid=[0.0],
        checkpoint_dir=ARTIFACTS.parent / "checkpoints" / "v2_ablation",
        n_jobs=n_jobs,
    )
    log.info("    peldanos corridos: %s (n_jobs=%d)", runnable_names, n_jobs)

    # --- f) atribucion "de hoy" (proxy documentado si no hay sensibilidades) ---
    log.info("[f] atribucion precio/noticias de hoy...")
    if news_active and len(xi_hist) >= 2:
        delta_SI_today = float(si_full[-1] - si_full[-2])
        delta_price = X_tech_std[-1] - X_tech_std[-2]
        attr = surprise_mod.attribution(xi_hist[-2], xi_hist[-1], delta_SI_today, delta_price)
    else:
        attr = surprise_mod.attribution(
            xi_hist[-2] if len(xi_hist) >= 2 else xi_hist[-1], xi_hist[-1], None, None,
        )
    log.info("    atribucion hoy: %s", attr)

    # --- h) publicar artifacts/latest/irfn.json version V2 ---
    log.info("[h] publicando artifacts/latest/irfn.json (version V2)...")
    labels = regime_labels(Kt)
    news_layer_params = {
        "active": news_active,
        "delta": delta_res["delta"], "delta_se": delta_res["delta_se"],
        "surprise_start_date": news_cfg.surprise_start_date,
        "indicators": _indicator_weight_rows(news_cfg, weights, coverage),
        "coverage": coverage,
        "blocker": delta_res["blocker"],
    }
    news_block = {
        "surprise_index": float(si_full[-1]) if news_active else 0.0,
        "lambda_N": 0.0,
        "lambda_N_z": 0.0,
        "branching_ratio": 0.0,
        "expected_cascade": 0.0,
        "attribution": {"price": attr.get("price", 1.0), "news": attr.get("surprise", 0.0)},
    }

    if "beta_tvtp" in titular_params:
        P_today = transition_matrix_at(titular_params["d"], titular_params["beta_tvtp"], X_today_std[-1])
    else:
        P_today = titular_params["P"]

    # build_payload ya agrega un warning automatico "Capa de noticias (V2)
    # inactiva: <blocker>" cuando news_layer_params.active es False (ver
    # outputs/publish.build_payload); no se duplica aqui.
    warnings_list: list[str] = []
    if macro_blocker:
        warnings_list.append(
            "M3 (macro) bloqueado: " + macro_blocker + " Por tanto M4 tambien queda bloqueado "
            "(M4 = M3 + surprise_index; ver reports/ablation_news.md)."
        )
    if titular_hessian_ok is False:
        warnings_list.append("Hessiano degenerado en el titular: algunas SE de betas no son fiables.")

    spec = (
        f"MS-GJR-GARCH (Haas et al. 2004), TVTP (tecnico + surprise_index), "
        f"innovaciones {'t de Student' if distt == 't' else 'Normales'}"
        if news_active else
        f"MS-GJR-GARCH (Haas et al. 2004), TVTP (tecnico), "
        f"innovaciones {'t de Student' if distt == 't' else 'Normales'} [sin capa de noticias: ver warnings]"
    )

    payload = build_payload(
        asof=returns.index[-1].date(), version=VERSION, run_id=run_id,
        git_commit=git_commit, config_hash=config_hash, K=Kt, labels=labels,
        xi_history=xi_hist, P=titular_params["P"],
        entropy_mid=base.regime.entropy_mid, entropy_high=base.regime.entropy_threshold,
        seed=seed, n_multistart=titular_n_starts,
        converged=(titular_n_converged >= max(1, titular_n_starts // 3)),
        r_pct=r_pct_hist, argmax_idx=argmax_hist,
        asset=ASSET_NAME, validation_ref=f"reports/ablation_news{REPORT_SUFFIX}.md",
        warnings=warnings_list, spec=spec, tvtp=True, covariates=today_cov_names,
        transition_matrix_today=P_today,
        bootstrap_n_boot=base.v2.bootstrap.n_boot, bootstrap_block_len=base.v2.bootstrap.block_len,
        bootstrap_ci_level=base.v2.bootstrap.ci_level, bootstrap_min_obs=base.v2.bootstrap.min_obs,
        news_layer=([base.v2.news_covariate_name] if news_active else []),
        news_block=news_block, news_layer_params=news_layer_params,
    )
    # Fase 3 (publicacion atomica): escribir SOLO en runs/<run_id>/. A latest/
    # solo se llega con scripts/promote.py (contrato + guardarrail).
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    publish(payload, run_dir / "irfn.json")

    # --- serie de SI_t + eventos individuales, para pantalla 3 (R9: la app solo
    #     lee; este es el unico lugar que ESCRIBE estos artefactos). ---
    events_table = _events_table(z_wide, weights)
    _dump({"events": events_table}, run_dir / "surprise_events.json")
    if news_active:
        si_hist = pd.DataFrame({"fecha": returns.index.astype(str), "surprise_index": si_full})
        si_hist.to_parquet(run_dir / "surprise_history.parquet", index=False)

    # --- auditoria: PIT + ledger + reestimacion + vintage + expanding window ---
    log.info("    auditoria PIT (con registro de features de noticias)...")
    pit_df = prefix_invariance_check(
        returns, K=Kt, seed=seed, n_starts=6, train_len=len(returns) - 260,
        dates_to_test=10, X=X_tech, dist=distt,
    )
    ledger_df = lag_ledger(v1_feature_registry(include_macro=(X_macro is not None), include_news=True))
    reest_blocks = []
    for name in runnable_names:
        if name in abl.wf_results:
            reest_blocks.extend(abl.wf_results[name].blocks)
    reest_df = block_reestimation_check(reest_blocks) if reest_blocks else block_reestimation_check([])

    audit_obj = {
        "run_id": run_id, "generated_at": datetime.now(timezone.utc).isoformat(),
        "entropy_zones": {"mid": base.regime.entropy_mid, "high": base.regime.entropy_threshold},
        "prefix_invariance": {
            "passed": bool(pit_df.attrs["passed"]), "atol": pit_df.attrs["atol"],
            "rows": json.loads(pit_df.assign(fecha=pit_df["fecha"].astype(str)).to_json(orient="records")),
        },
        "lag_ledger": {"passed": bool(ledger_df.attrs["passed"]), "rows": json.loads(ledger_df.to_json(orient="records"))},
        "block_reestimation": {
            "passed": bool(reest_df.attrs["passed"]) if len(reest_df) else True,
            "rows": (
                json.loads(
                    reest_df.assign(train_start=reest_df["train_start"].astype(str),
                                    test_start=reest_df["test_start"].astype(str)).to_json(orient="records")
                ) if len(reest_df) else []
            ),
        },
        "consensus_vintage_ledger": {
            "passed": bool(vintage_df.attrs["passed"]), "n_events": vintage_df.attrs["n_events"],
            "rows": json.loads(vintage_df.to_json(orient="records")),
        },
        "expanding_window_check": {
            "passed": bool(ew_df.attrs["passed"]), "vacuous": bool(ew_df.attrs.get("vacuous", False)),
            "n_events_con_consenso": ew_df.attrs.get("n_events_con_consenso", 0),
            "rows": json.loads(ew_df.to_json(orient="records")),
        },
        "sources": [{"name": ASSET_NAME, "source": ASSET_SOURCE,
                     "last_date": str(returns.index[-1].date()), "n_obs": int(len(returns))}],
    }
    _dump(audit_obj, run_dir / "audit.json")

    # --- i) reports/ablation_news.md (R8: SIEMPRE, sea cual sea el resultado) ---
    log.info("[i] escribiendo reports/ablation_news.md...")
    _write_ablation_report(
        abl, runnable_names, macro_blocker, delta_res["blocker"], news_layer_params,
        Kt, distt, run_id,
    )

    log.info("Listo V2. run_id=%s PIT=%s noticias_activas=%s",
             run_id, "VERDE" if pit_df.attrs["passed"] else "ROJA", news_active)


# --------------------------------------------------------------------------- #
# i) Reporte (R8): se escribe SEA CUAL SEA el resultado
# --------------------------------------------------------------------------- #
def _write_ablation_report(abl, runnable_names, macro_blocker, surprise_blocker,
                            news_layer_params, K, dist, run_id) -> None:
    L: list[str] = []
    L.append("# Ablacion V2: capa de noticias (M3 vs M4)\n")
    L.append(f"generado: {datetime.now(timezone.utc).isoformat()}  |  run_id: `{run_id}`  |  "
             f"K={K}, dist={dist}\n")

    L.append(f"> {PRE_REGISTERED_COMMITMENT}\n")

    m4_ran = "M4" in runnable_names
    if not m4_ran:
        reasons = []
        if macro_blocker:
            reasons.append(f"M3 (macro) bloqueado: {macro_blocker}")
        if surprise_blocker:
            reasons.append(f"capa de sorpresa bloqueada: {surprise_blocker}")
        if not reasons:
            reasons.append("M4 no se corrio por una razon no capturada; revisar log de la corrida.")
        L.append("## Veredicto\n")
        L.append("**M4 NO SE PUDO CORRER.** Siguiendo el compromiso pre-registrado de arriba, "
                 "sin M4 el indicador se publica SIN capa de sorpresa activa "
                 f"(model.news_layer_params.active = {news_layer_params['active']} en irfn.json). "
                 "Motivo(s):\n")
        for r in reasons:
            L.append(f"- {r}")
        L.append("")
        L.append("Esto NO es un resultado negativo de M4 (no perdio un DM); es la ausencia de datos "
                 "para siquiera intentarlo. R8 exige documentar el walk-forward de todas formas -- "
                 "abajo va la tabla de lo que SI corrio esta sesion (M0-M2, y M3 si el bloqueante de "
                 "macro se resuelve antes que este reporte).\n")
    else:
        dm_m4_vs_m3 = next((d for d in abl.dm_pairs if d["model_a"] == "M4" and d["model_b"] == "M3"), None)
        if dm_m4_vs_m3:
            better = dm_m4_vs_m3["dm_stat"] < 0
            distinguible = dm_m4_vs_m3["p_value"] < 0.10
            L.append("## Veredicto\n")
            if distinguible and better:
                L.append(f"**M4 mejora la log-loss de forma distinguible** "
                         f"(DM={_fmt(dm_m4_vs_m3['dm_stat'],3)}, p={_fmt(dm_m4_vs_m3['p_value'],3)} < 0.10): "
                         "la capa de sorpresa se publica activa.\n")
            else:
                L.append(f"**M4 NO mejora la log-loss de forma distinguible** "
                         f"(DM={_fmt(dm_m4_vs_m3['dm_stat'],3)}, p={_fmt(dm_m4_vs_m3['p_value'],3)}): "
                         "siguiendo el compromiso pre-registrado, el indicador se publica SIN capa de "
                         "sorpresa activa, aunque los datos alcanzaron para correr M4.\n")

    L.append("## Ablacion corrida esta sesion\n")
    L.append(f"Peldanos corridos: `{runnable_names}` (de M0..M4 declarados en "
             "validation/ablation.full_ladder_specs).\n")
    L.append("| modelo | descripcion | covs | bloques | n_oos | log-loss OOS/obs |")
    L.append("| :-- | :-- | :-- | --: | --: | --: |")
    for row in abl.table:
        L.append(f"| {row['model']} | {row['description']} | {row['covariates'] or '-'} | "
                 f"{row['n_blocks']} | {row['n_oos']} | {_fmt(row['oos_logloss_per_obs'],4)} |")
    L.append("")

    L.append("## Diebold-Mariano entre peldanos consecutivos (perdida OOS)\n")
    L.append("| A vs B | DM stat | p-value | dif. media |")
    L.append("| :-- | --: | --: | --: |")
    for d in abl.dm_pairs:
        L.append(f"| {d['model_a']} vs {d['model_b']} | {_fmt(d['dm_stat'],3)} | "
                 f"{_fmt(d['p_value'],3)} | {_fmt(d['mean_diff'],5)} |")
    L.append("")
    L.append("> DM<0 => el primer modelo (A) tiene MENOR perdida (mejor). HAC Newey-West + "
             "correccion Harvey-Leybourne-Newbold.\n")

    L.append("## w_i estimados (R7: por regresion de impacto, con SE)\n")
    L.append("| indicador | w_i | SE | t | distinguible de cero | n eventos |")
    L.append("| :-- | --: | --: | --: | :-- | --: |")
    for row in news_layer_params["indicators"]:
        L.append(f"| {row['indicator']} | {_fmt(row['w'],4)} | {_fmt(row['se'],4)} | "
                 f"{_fmt(row['t_stat'],2)} | {row['distinguible_de_cero']} | {row['n_events']} |")
    L.append("")
    L.append(f"delta estimado: {_fmt(news_layer_params['delta'],5)} "
             f"(SE={_fmt(news_layer_params['delta_se'],5)}). "
             f"surprise_start_date (config/news.yaml): {news_layer_params['surprise_start_date']}.\n")
    if news_layer_params["blocker"]:
        L.append(f"> Bloqueante de la capa de sorpresa: {news_layer_params['blocker']}\n")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"ablation_news{REPORT_SUFFIX}.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Orquestador V2 (capa de sorpresa calendarizada).")
    ap.add_argument("--quick", action="store_true", help="Menos arranques (humo, no cumple R6).")
    ap.add_argument("--asset", default=None,
                    help="Nombre del activo en config/assets.yaml (p.ej. BTC). "
                         "Sin este flag, comportamiento identico al historico (SPY).")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Procesos para paralelizar los bloques del walk-forward de la "
                         "ablacion (R2: bloques independientes). 1 = serie (referencia). "
                         "-1 = todos los nucleos menos uno. Solo velocidad; los numeros "
                         "son identicos al serial (n_jobs no entra en el checkpoint).")
    args = ap.parse_args()
    n_jobs = max(1, (os.cpu_count() or 1) - 1) if args.jobs == -1 else max(1, args.jobs)
    _configure_asset(BaseConfig.load(), args.asset)
    full_run(quick=args.quick, n_jobs=n_jobs)


if __name__ == "__main__":
    main()

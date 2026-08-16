"""Orquestador de V3: flujo de titulares -> relevancia s_i (FinBERT) -> Hawkes
-> branching ratio (indicador de fragilidad) -> lambda_N_z como covariable.

    python scripts/run_v3.py            # corrida completa V3
    python scripts/run_v3.py --quick    # menos arranques (humo, no R6)
    python scripts/run_v3.py --no-capture  # sin tocar la red (usa lo que hay en disco)

Orden:
  a) calendario de consenso (V2, sin cambios: sigue bloqueado y se documenta)
  b) titulares: top-up de dias faltantes recientes (GDELT) + carga del corpus
  c) relevancia s_i con FinBERT local (Trampa 4: SOLO relevancia, jamas direccion)
  d) auditoria de timestamps (Trampa 3): titular vs evento + resolucion del feed
  e) MLE de Hawkes (R6 multistart) + branching ratio n = alpha*E[s]/beta + KS
  f) lambda_N diaria -> covariable lambda_N_z (shift(1), R3)
  g) ablacion: escalera M0..M5 declarada SIEMPRE; corre lo que los datos permiten
     + diagnostico pre-declarado M2+H (ver DECISION DE DISENO abajo)
  h) titular V3 + atribucion de 3 vias + publicacion de artefactos
  i) reports/ablation_news.md y reports/validation_v3.md (R8: pase lo que pase)

--------------------------------------------------------------------------------
DECISIONES DE DISENO DE ESTA SESION (para revision del director, CLAUDE.md §3)
--------------------------------------------------------------------------------
1. Los parametros del Hawkes (mu_N, alpha, beta) se estiman UNA VEZ por MLE
   sobre TODO el corpus de titulares, no por bloque de walk-forward. Analogo
   exacto a la decision de V2 con delta (fit_delta_mle): son parametros del
   PROCESO PUNTUAL de titulares (se estiman solo con titulares, sin mirar
   retornos), es decir ingenieria de features; los beta_ij del logit -- los
   parametros de verdad sujetos a R2 -- SI se re-estiman desde cero por bloque.
   lambda_N(t) evaluada en t solo usa titulares <= t (causal por construccion;
   test_hawkes_features lo verifica truncando el feed).

2. La comparacion PRE-REGISTRADA de V3 es M5 vs M4 (flujo de noticias POR
   ENCIMA de la sorpresa calendarizada). Hoy es INFRANQUEABLE aguas arriba por
   M4 (sorpresa): sin consenso historico no hay peldano M4, y M5 depende de M4.
   M3 (macro) NO es el bloqueante -- su estado real se calcula en ejecucion en
   _macro_status() (ALFRED_API_KEY + reports/ablation_m3_l1.json); con la key
   configurada la ablacion M3 ya corrio en negativo (auditoria 2026-08-14,
   reports/auditoria_pre_corrida.md). NO se sustituye en silencio:
   se declara ademas un diagnostico M2+H = M2 + lambda_N_z, que responde una
   pregunta DISTINTA y mas debil ("el flujo aporta por encima del TVTP
   tecnico?"), pre-registrada aqui ANTES de correrla, con este compromiso:
   si M2+H no mejora la log-loss OOS de M2 de forma distinguible (DM p < 0.10),
   lambda_N_z se publica INACTIVA como covariable (el titular queda tecnico) y
   el hallazgo se reporta. El branching ratio como INDICADOR DESCRIPTIVO diario
   no depende de ese veredicto: se publica con su MLE, su KS y su cobertura.

3. El corpus de titulares arranca donde el backfill de GDELT haya llegado
   (data/raw/headlines/, reanudable, newest-first). La cobertura viaja en el
   artefacto; si no alcanza para el walk-forward de M2+H (train 4a + 6 bloques
   de 6m), se documenta el bloqueante en vez de encoger la malla de bloques
   (encogerla cambiaria el diseno pre-registrado del walk-forward, R8).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
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
from irfn.data import headlines as headlines_mod  # noqa: E402
from irfn.features import hawkes_features as hf  # noqa: E402
from irfn.features import surprise as surprise_mod  # noqa: E402
from irfn.features.relevance import RelevanceModelUnavailable, score_headlines  # noqa: E402
from irfn.models import hawkes_mle  # noqa: E402
from irfn.models.tvtp import transition_matrix_at  # noqa: E402
from irfn.outputs.publish import build_payload, default_hawkes_layer_params, publish  # noqa: E402
from irfn.outputs.serialize import write_history_parquet, write_walkforward_json  # noqa: E402
from irfn.pipeline import regime_labels, run_pipeline  # noqa: E402
from irfn.validation import calibration  # noqa: E402
from irfn.validation.ablation import ModelSpec, full_ladder_specs, run_ablation  # noqa: E402
from irfn.validation.walkforward import walk_forward  # noqa: E402

# Helpers de V2 reutilizados tal cual (mismo patron que run_v2 importando
# capture_consensus: scripts/ puede importar scripts/; la LOGICA vive en src/).
import run_v2  # noqa: E402
from run_v2 import (  # noqa: E402
    _fmt,
    _git_commit,
    _config_hash,
    _dump,
    _events_table,
    _indicator_weight_rows,
    _load_titular_K_dist,
    compute_z_and_weights,
    ingest_calendar,
    load_sample,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("run_v3")

ARTIFACTS = ROOT / "artifacts" / "latest"
RUNS = ROOT / "artifacts" / "runs"
REPORTS = ROOT / "reports"
VERSION = "V3"

# Activo de la corrida: delega en run_v2._configure_asset (misma resolucion,
# mismo directorio artifacts/<asset>/latest/ que V1/V2 para el mismo activo,
# necesario porque _load_titular_K_dist/load_sample importados de run_v2 leen
# los globals de ESE modulo). Default = SPY, identico al historico.
ASSET_NAME = None
ASSET_SOURCE = "yfinance"
REPORT_SUFFIX = ""


def _configure_asset(base: BaseConfig, asset_override: str | None) -> None:
    global ARTIFACTS, RUNS, ASSET_NAME, ASSET_SOURCE, REPORT_SUFFIX
    run_v2._configure_asset(base, asset_override)
    ARTIFACTS = run_v2.ARTIFACTS
    RUNS = run_v2.RUNS
    ASSET_NAME = run_v2.ASSET_NAME
    ASSET_SOURCE = run_v2.ASSET_SOURCE
    REPORT_SUFFIX = run_v2.REPORT_SUFFIX

PRE_REGISTERED_M5 = (
    "Compromiso pre-registrado (V3, guia 8.1): la pregunta titular es M5 vs M4 "
    "(el flujo de noticias aporta POR ENCIMA de la sorpresa calendarizada?). "
    "Si M5 no mejora la log-loss OOS de M4 de forma distinguible (DM p < 0.10), "
    "el indicador se publica sin lambda_N_z en el logit."
)
PRE_REGISTERED_M2H = (
    "Compromiso pre-registrado del diagnostico M2+H (esta sesion, ANTES de "
    "correrlo): M2+H = M2 + lambda_N_z responde una pregunta mas debil que M5 "
    "vs M4 (aporte sobre el TVTP tecnico, no sobre la sorpresa). Si M2+H no "
    "mejora la log-loss OOS de M2 de forma distinguible (DM p < 0.10), "
    "lambda_N_z se publica INACTIVA como covariable y el titular queda tecnico. "
    "Este diagnostico NO sustituye a M5 vs M4, que sigue bloqueada aguas arriba."
)


def _run_id(config_hash: str, git_commit: str, seed: int) -> str:
    return hashlib.sha256(f"{config_hash}|{git_commit}|{seed}|{VERSION}".encode()).hexdigest()[:12]


def _macro_status() -> str:
    """Estado REAL de la macro (M3), calculado en tiempo de EJECUCION -- no una
    plantilla heredada (auditoria 2026-08-14, reports/auditoria_pre_corrida.md).

    Comprueba dos hechos, sin inventar ninguno:
      1. si ALFRED_API_KEY esta configurada (mismo mecanismo que data/alfred.py:
         dotenv + entorno), y
      2. si la ablacion M3 con L1 ya se corrio, leyendo su resultado persistido
         (reports/ablation_m3_l1.json). El DM de M3 vs M2 se lee del archivo, NO
         se hardcodea, para que no se desincronice si la ablacion se re-corre.

    Devuelve la frase honesta que va al warning del artefacto y al reporte de
    ablacion. Si la key falta o la ablacion no consta, lo dice tal cual: el
    bloqueante NO se afirma ni se niega sin evidencia."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:  # noqa: BLE001  -- sin dotenv, se cae al entorno del proceso
        pass
    key_set = bool(os.environ.get("ALFRED_API_KEY", "").strip())

    dm_m3 = None
    # Suffijo por activo: para un activo != SPY (REPORT_SUFFIX != "") se busca su
    # propio archivo; si no existe (p.ej. BTC, K=1 sin M3) dm_m3 queda None y el
    # mensaje cae al caso honesto "no consta", no se lee el M3 de OTRO activo.
    m3_json = REPORTS / f"ablation_m3_l1{REPORT_SUFFIX}.json"
    if m3_json.exists():
        try:
            data = json.loads(m3_json.read_text(encoding="utf-8"))
            dm_m3 = next(
                (d for d in data.get("dm_pairs", [])
                 if d.get("model_a") == "M3" and d.get("model_b") == "M2"),
                None,
            )
        except (json.JSONDecodeError, OSError, KeyError):
            dm_m3 = None

    if not key_set:
        # Sin key no hay vintages ALFRED (R4 prohibe caer a FRED revisado): M3
        # genuinamente no puede correr. Es el unico caso en que el aviso original
        # es cierto.
        return (
            "ALFRED_API_KEY sin configurar; M3 = M2 + macro no puede correr, y "
            "M4/M5 dependen de M3/M4 por diseno de la escalera."
        )
    if dm_m3 is not None:
        aporta = dm_m3["dm_stat"] < 0 and dm_m3["p_value"] < 0.10
        verdict = "aporta" if aporta else "NO aporta"
        return (
            f"NO es el bloqueante: ALFRED_API_KEY esta configurada y la ablacion M3 "
            f"con L1 ya se corrio (M3 vs M2 DM={dm_m3['dm_stat']:.3f}, "
            f"p={dm_m3['p_value']:.3f} -> la macro {verdict}; reports/ablation_m3_l1.md). "
            "El bloqueante real de M5 vs M4 es M4 (sorpresa): sin consenso historico. "
            "M5 depende de M4 por diseno de la escalera."
        )
    return (
        "ALFRED_API_KEY configurada, pero la ablacion M3 con L1 no consta "
        f"(reports/{m3_json.name} ausente); correr scripts/run_m3_l1.py para el "
        "veredicto macro. El bloqueante real de M5 vs M4 sigue siendo M4 (sorpresa), "
        "no la macro."
    )


# --------------------------------------------------------------------------- #
# b/c) Corpus de titulares + relevancia
# --------------------------------------------------------------------------- #
def topup_headlines(news_cfg: NewsConfig, *, max_days: int = 5) -> None:
    """Intenta rellenar los dias faltantes MAS RECIENTES (no el backfill entero:
    ese corre aparte, scripts/capture_headlines.py). Nunca lanza: un hueco es
    informacion documentada en la cobertura, no un motivo para detenerse."""
    cfg = news_cfg.headlines
    try:
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        since = yesterday - timedelta(days=max_days - 1)
        headlines_mod.capture_headlines_range(
            query=cfg.query, since=since, until=yesterday,
            max_records=cfg.max_records,
            request_spacing_seconds=cfg.request_spacing_seconds,
            rate_limit_backoff_seconds=cfg.rate_limit_backoff_seconds,
            max_backoff_retries=1,          # top-up ligero; el backfill largo va aparte
            min_window_hours=cfg.min_window_hours,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("top-up de titulares no disponible (se documenta, no bloquea): %s", exc)


def build_hawkes_layer(
    news_cfg: NewsConfig, base: BaseConfig, calendar: pd.DataFrame, *, quick: bool,
) -> dict:
    """Corre b)-f): corpus -> s_i -> auditoria de timestamps -> MLE -> KS ->
    lambda_N diaria. Devuelve un dict con todo lo necesario aguas abajo; si una
    pieza falta, `blocker` lo documenta y el resto queda en None (R8)."""
    cfg = news_cfg.headlines
    out: dict = {
        "headlines": None, "scored": None, "fit": None, "ks": None,
        "lambda_raw": None, "origin": None, "timestamp_audit": None,
        "coverage": {"first_day": None, "last_day": None, "n_days": 0,
                     "n_missing_days": 0, "n_censored_days": 0},
        "blocker": None,
    }
    if not cfg.enabled:
        out["blocker"] = "headlines.enabled=false en config/news.yaml."
        return out

    headlines = headlines_mod.load_headlines()
    cov = headlines.attrs["coverage"]
    out["coverage"] = {
        "first_day": cov["first_day"], "last_day": cov["last_day"], "n_days": cov["n_days"],
        "n_missing_days": len(cov["missing_days"]), "n_censored_days": len(cov["censored_days"]),
    }
    out["headlines"] = headlines
    if headlines.empty:
        out["blocker"] = (
            "corpus de titulares vacio: data/raw/headlines/ sin snapshots. El "
            "backfill de GDELT (scripts/capture_headlines.py) esta limitado a 1 "
            "peticion cada 5 s y es reanudable; correrlo y re-ejecutar run_v3."
        )
        return out

    # d) auditoria de timestamps (Trampa 3) -- corre ANTES del MLE: si esta en
    # rojo, lambda_N no es publicable como covariable intradia-honesta.
    matched = headlines_mod.match_event_times(headlines, calendar,
                                              match_window_hours=cfg.match_window_hours)
    out["timestamp_audit"] = headlines_mod.timestamp_audit(matched)

    # c) relevancia s_i (FinBERT local; Trampa 4 documentada en el modulo)
    try:
        scored = score_headlines(
            headlines, model_id=news_cfg.relevance.finbert_model,
            batch_size=news_cfg.relevance.batch_size,
        )
    except RelevanceModelUnavailable as exc:
        out["blocker"] = f"FinBERT local no disponible: {exc}"
        return out
    out["scored"] = scored

    n_events = int(scored["s"].notna().sum())
    if n_events < cfg.min_events_fit:
        out["blocker"] = (
            f"solo {n_events} titulares puntuados (se requieren >= {cfg.min_events_fit}, "
            "config news.headlines.min_events_fit) para un MLE de Hawkes honesto."
        )
        return out
    if not out["timestamp_audit"]["passed"]:
        out["blocker"] = (
            "auditoria de timestamps de titulares en ROJO (pantalla 6, seccion 6): "
            "con timestamps no confiables, lambda_N no se publica."
        )
        return out

    # e) MLE de Hawkes con multistart (R6) + KS de re-escalamiento
    times, marks, origin = hf.headline_event_times(scored)
    # De-empatar los timestamps cuantizados a 15 min de GDELT (dithering intra-bin,
    # decision del director 2026-08-14): sin esto el Hawkes continuo degenera por
    # los ~83% de eventos empatados (beta->inf). Determinista dado la semilla (R6).
    times, marks = hf.dither_quantized_times(times, marks, seed=base.model.seed)
    # T_span: ventana CALENDARIO (origen -> ultimo dia +1). Antes se ajustaba con
    # esta T, integrando mu_N sobre los dias fantasma -> mu_N sesgada a la baja y n
    # inflada hacia la frontera (auditoria 2026-08-14/15).
    last_day_end = pd.Timestamp(cov["last_day"], tz="UTC") + pd.Timedelta(days=1)
    T_span = float((last_day_end - origin).total_seconds() / 86400.0)
    if len(times):
        T_span = max(T_span, float(np.max(times)))
    # PARTE A (correccion, decision del director 2026-08-15 que REVIERTE la de
    # 2026-08-14 de ajustar sobre span completo): se ajusta sobre el TIEMPO
    # OBSERVADO. Se comprime el reloj excindiendo los dias fantasma; el estimador
    # (verificado) queda intacto, solo recibe una serie bien-relojeada. La feature
    # lambda_N(t) sigue en reloj calendario (params son tasas por dia y la
    # excitacion es intradia: la feature es identica salvo el piso mu_N ya corregido).
    times_obs, T_obs = hf.compress_to_observed_time(times, origin, cov["missing_days"])
    n_starts = 8 if quick else news_cfg.hawkes.n_starts
    fit = hawkes_mle.fit_hawkes_mle(times_obs, marks, T=T_obs, n_starts=n_starts, seed=base.model.seed)
    out["fit"] = fit
    # KS sobre el MISMO reloj del ajuste (tiempo observado), no el calendario.
    out["ks"] = hawkes_mle.rescaling_ks(times_obs, marks, fit.params)
    out["origin"] = origin
    # Diagnostico de TRAZABILIDAD (antes vs despues): se re-ajusta sobre el span
    # calendario SOLO para registrar cuanto sesgaba la ventana inflada. NO se
    # publica; viaja al log y a la seccion de decisiones de validation_v3.
    fit_span = hawkes_mle.fit_hawkes_mle(times, marks, T=T_span, n_starts=n_starts, seed=base.model.seed)
    out["n_before_span"] = fit_span.branching_ratio
    out["n_after_observed"] = fit.branching_ratio
    out["T_span"] = T_span
    out["T_obs"] = T_obs
    out["mu_before_span"] = fit_span.params["mu"]
    out["mu_after_observed"] = fit.params["mu"]
    log.info("    PARTE A (ventana observada): T_span=%.1f d -> T_obs=%.1f d (%d dias fantasma excindidos)",
             T_span, T_obs, out["coverage"]["n_missing_days"])
    log.info("    n ANTES (span, sesgado)=%.4f  mu_N=%.4f  |  n DESPUES (observado)=%.4f  mu_N=%.4f",
             fit_span.branching_ratio, fit_span.params["mu"], fit.branching_ratio, fit.params["mu"])
    log.info("    Hawkes MLE: mu_N=%.3f alpha=%.3f beta=%.3f | n=%.3f [IC95 %.3f, %.3f] E[hijos]=%.2f | "
             "KS p=%s | %d/%d arranques en el optimo",
             fit.params["mu"], fit.params["alpha"], fit.params["beta"],
             fit.branching_ratio, fit.branching_ratio_ci_low, fit.branching_ratio_ci_high,
             fit.expected_cascade, _fmt(out["ks"]["p_value"], 4),
             fit.starts_at_best, fit.n_starts)

    if not fit.stationary:
        # n >= 1: el modelo es explosivo y hay un bug (o el corpus esta roto).
        # NO SE PUBLICA (criterio de aceptacion V3). Se documenta y se corta aqui.
        out["blocker"] = (
            f"branching ratio n = {fit.branching_ratio:.3f} >= 1: proceso explosivo. "
            "Es un bug o un corpus degenerado; el indicador NO se publica hasta "
            "resolverlo (guia 6.6)."
        )
        return out

    out["times"], out["marks"] = times, marks
    return out


# --------------------------------------------------------------------------- #
# Corrida completa
# --------------------------------------------------------------------------- #
def full_run(quick: bool = False, no_capture: bool = False, publish_m1: bool = False) -> None:
    base = BaseConfig.load()
    news_cfg = NewsConfig.load()
    seed = base.model.seed
    config_hash, git_commit = _config_hash(), _git_commit()
    run_id = _run_id(config_hash, git_commit, seed)

    # --- a) calendario de consenso (V2: sigue bloqueado, se documenta igual) ---
    log.info("[a] ingesta del calendario macro (V2)...")
    cal, coverage_cal = ingest_calendar(news_cfg)
    vintage_df = consensus_vintage_ledger(cal)
    ew_df = expanding_window_check(cal, min_obs=news_cfg.sigma_min_obs)

    log.info("    cargando muestra (activo ancla + tecnicas)...")
    returns, X_tech = load_sample(base)
    Kt, distt = _load_titular_K_dist(base)
    log.info("    titular: K=%d dist=%s | muestra %s..%s (%d obs)",
             Kt, distt, returns.index[0].date(), returns.index[-1].date(), len(returns))

    z_wide, weights = compute_z_and_weights(cal, news_cfg, returns)
    events = surprise_mod.weighted_events_from(z_wide, weights) if len(z_wide.columns) else pd.Series(dtype=float)
    if len(events) < base.v2.delta_mle.min_events_total:
        surprise_blocker = (
            f"solo {len(events)} evento(s) valido(s) de consenso acumulados (se necesitan >= "
            f"{base.v2.delta_mle.min_events_total}); sin SI_t no hay capa de sorpresa. "
            "Mismo bloqueante que V2 (reports/data_audit.md)."
        )
    else:
        # V3 no re-estima delta (eso vive en scripts/run_v2.py, fit_delta_mle):
        # si el consenso por fin alcanza, hay que correr run_v2 e integrar SI_t
        # aqui ANTES de declarar la capa activa. No se activa a medias.
        surprise_blocker = (
            "hay consenso suficiente pero SI_t/delta se estiman en scripts/run_v2.py; "
            "correr run_v2 e integrar la covariable en run_v3 antes de activar la capa."
        )
    surprise_active = False  # inactiva en V3 mientras el bloqueante de arriba exista

    # --- b/c/d/e/f) capa de Hawkes ---
    log.info("[b-f] capa de Hawkes (titulares -> s_i -> MLE -> lambda_N)...")
    if not no_capture:
        topup_headlines(news_cfg)
    hawkes = build_hawkes_layer(news_cfg, base, cal, quick=quick)
    hawkes_indicator_active = hawkes["blocker"] is None
    if not hawkes_indicator_active:
        log.warning("    BLOQUEADO (indicador Hawkes): %s", hawkes["blocker"])

    lam_raw = None
    lam_feature = None
    if hawkes_indicator_active:
        cov = hawkes["coverage"]
        lam_raw = hf.lambda_daily(
            hawkes["times"], hawkes["marks"], hawkes["fit"].params, returns.index,
            origin=hawkes["origin"],
            coverage_first=pd.Timestamp(cov["first_day"], tz="UTC"),
            coverage_last=pd.Timestamp(cov["last_day"], tz="UTC"),
        )
        lam_feature = hf.hawkes_feature(lam_raw, z_window=base.windows.z_window)

    # --- g) ablacion: escalera completa declarada SIEMPRE + diagnostico M2+H ---
    log.info("[g] ablacion (escalera M0..M5 declarada; corre lo que los datos permiten)...")
    macro_status = _macro_status()
    single_regime = (Kt == 1)
    sample_returns = returns
    X_all = X_tech.copy()

    if single_regime:
        # K=1 (decision del director 2026-08-15; ver run_v2._load_titular_K_dist).
        # Sin regimenes NO hay logit de transicion: la escalera M0..M5 y el
        # diagnostico M2+H NO APLICAN (no hay transiciones que covariables puedan
        # modular). La seleccion de K ya se dirimio en V1 (BIC K=1 t 18/20 R6 +
        # bootstrap LR + reports/diag_btc_k2t_stability.md: K=2 t es artefacto 1/100).
        # lambda_N_z/surprise_index quedan fuera POR CONSTRUCCION, no por un DM; el
        # branching ratio de Hawkes es independiente de K y SI se publica (fragilidad).
        abl = None
        runnable_names = []
        m2h_blocker = (
            "no aplica: el activo es K=1 (un solo regimen). Sin logit de transicion, "
            "lambda_N_z y surprise_index no tienen donde entrar como covariables. La "
            "seleccion de K se dirimio en V1 (BIC + bootstrap LR + diagnostico de "
            "estabilidad 2026-08-15, reports/diag_btc_k2t_stability.md). El branching "
            "ratio de Hawkes se publica igual, como indicador de fragilidad standalone."
        )
        dm_m2h = None
        covariate_active = False
        covariate_reason = m2h_blocker
        log.info("    K=1: escalera de ablacion NO aplica (sin regimenes); Hawkes standalone se publica.")
    else:
        specs_all = full_ladder_specs(
            Kt, distt, covariates_m2=list(X_tech.columns),
            covariates_m3=list(base.v2.covariates_ablation_m3),
            news_covariate=base.v2.news_covariate_name,
        )
        by_name = {s.name: s for s in specs_all}
        runnable_names = ["M0", "M1", "M2"]

        # Diagnostico M2+H pre-declarado (ver DECISION DE DISENO 2 del docstring).
        m2h_spec = ModelSpec(
            "M2+H", Kt, distt, tuple(list(X_tech.columns) + ["lambda_N_z"]),
            "+lambda_N_z sobre M2 (diagnostico V3, NO es el M5 pre-registrado)",
        )
        m2h_blocker = None
        if hawkes_indicator_active:
            joined = pd.concat([returns.rename("r"), X_tech, lam_feature], axis=1).dropna()
            span_years = (joined.index[-1] - joined.index[0]).days / 365.25
            need_years = base.walkforward.train_years + base.walkforward.n_blocks * base.walkforward.test_months / 12.0
            if span_years >= need_years:
                sample_returns = joined["r"]
                X_all = joined.drop(columns="r")
                runnable_names.append("M2+H")
            else:
                m2h_blocker = (
                    f"cobertura de lambda_N_z insuficiente para el walk-forward pre-registrado: "
                    f"muestra alineada de {span_years:.1f} anios y se requieren >= {need_years:.1f} "
                    f"(train {base.walkforward.train_years}a + {base.walkforward.n_blocks} bloques de "
                    f"{base.walkforward.test_months}m). El backfill de GDELT sigue creciendo hacia "
                    "atras; re-correr cuando alcance. NO se encoge la malla de bloques (R8)."
                )
                log.warning("    BLOQUEADO (M2+H): %s", m2h_blocker)
        else:
            m2h_blocker = f"sin capa de Hawkes activa: {hawkes['blocker']}"

        runnable_specs = [by_name.get(n, m2h_spec if n == "M2+H" else None) for n in runnable_names]
        n_starts_wf = 6 if quick else base.v0.wf_n_starts
        abl = run_ablation(
            sample_returns, X_all, runnable_specs, seed=seed, n_starts=n_starts_wf,
            train_years=base.walkforward.train_years, test_months=base.walkforward.test_months,
            n_blocks_min=base.walkforward.n_blocks, l1_grid=[0.0],
            checkpoint_dir=ARTIFACTS.parent / "checkpoints" / "v3_ablation",
            n_jobs=-1,  # paraleliza por bloque; determinista bit-identico al serial (R2)
        )
        log.info("    peldanos corridos: %s", runnable_names)

        # Veredicto del diagnostico M2+H (compromiso pre-registrado arriba).
        dm_m2h = next((d for d in abl.dm_pairs if d["model_a"] == "M2+H" and d["model_b"] == "M2"), None)
        if dm_m2h is not None:
            covariate_active = bool(dm_m2h["dm_stat"] < 0 and dm_m2h["p_value"] < 0.10)
            covariate_reason = (
                f"M2+H {'mejora' if covariate_active else 'NO mejora'} la log-loss OOS de M2 "
                f"(DM={dm_m2h['dm_stat']:.3f}, p={dm_m2h['p_value']:.3f}; umbral pre-registrado p<0.10)."
            )
        else:
            covariate_active = False
            covariate_reason = m2h_blocker or "diagnostico M2+H no corrido."
    if publish_m1 and not single_regime:
        # A4 (bias-variance): publicar M1 (solo regimenes de volatilidad). Se fuerza
        # lambda_N_z fuera del logit e se ignora el diagnostico M2+H: M1 no lleva
        # NINGUNA covariable de transicion (ni tecnica ni de noticias).
        covariate_active = False
        covariate_reason = "M1 (A4, bias-variance): sin covariables de transicion en el logit."
    log.info("    covariable lambda_N_z activa en el titular: %s (%s)", covariate_active, covariate_reason)

    # --- h) titular V3 ---
    log.info("[h] estimando el titular V3...")
    n_starts_today = 8 if quick else base.model.n_multistart
    if publish_m1 and not single_regime:
        # M1 (A4): K>=2 con matriz de transicion CONSTANTE, SIN covariables de
        # transicion (titular_X=None => tvtp=False, covariates=[]). El indicador de
        # Hawkes standalone se conserva; solo sale del logit.
        titular_returns, titular_X = returns, None
        news_layer_list = []
    elif covariate_active:
        titular_returns, titular_X = sample_returns, X_all[list(X_tech.columns) + ["lambda_N_z"]]
        news_layer_list = ["lambda_N_z"]
    elif single_regime:
        # K=1: titular = GARCH de un solo regimen, SIN covariables de TVTP (no hay
        # logit). Mismo tratamiento que run_v1 para K=1 (tvtp=False).
        titular_returns, titular_X = returns, None
        news_layer_list = []
    else:
        titular_returns, titular_X = returns, X_tech
        news_layer_list = []
    today_run = run_pipeline(titular_returns, K=Kt, seed=seed, n_starts=n_starts_today,
                             compute_se=True, X=titular_X, dist=distt)
    frame = today_run.frame
    titular_params = today_run.fit.params
    from irfn.pipeline import standardize_with
    X_today_std = standardize_with(titular_X, today_run.scaler) if titular_X is not None else None

    xi_hist = frame[[f"xi_filtered_{k}" for k in range(Kt)]].to_numpy()
    r_pct_hist = frame["r"].to_numpy()
    argmax_hist = frame["argmax_idx"].to_numpy()

    # atribucion de 3 vias (proxy documentado: sin sensibilidades del modelo). Con
    # K=1 (sin covariables ni regimenes) no hay atribucion posible: todo a 0.
    if titular_X is not None and len(xi_hist) >= 2:
        n_tech = len(X_tech.columns)
        delta_price = X_today_std[-1, :n_tech] - X_today_std[-2, :n_tech]
        delta_hawkes = (float(X_today_std[-1, -1] - X_today_std[-2, -1])
                        if covariate_active else None)
        attr = surprise_mod.attribution_nway(
            {"price": delta_price, "surprise": None, "hawkes": delta_hawkes},
        )
        attribution = {k: float(attr[k]) for k in ("price", "surprise", "hawkes")}
    else:
        attribution = {"price": 0.0, "surprise": 0.0, "hawkes": 0.0}

    # --- bloques del contrato ---
    fith = hawkes["fit"]
    # PARTE B: censura de la cascada. Si el IC superior de n alcanza el disparador
    # (config news.hawkes.cascade_ci_trigger), 1/(1-n) diverge/pierde sentido cerca
    # de la frontera -> se reporta "no acotada" (cascade_val=None, bounded=False).
    if hawkes_indicator_active:
        cascade_val, cascade_bounded = hawkes_mle.expected_cascade_reported(
            fith.branching_ratio, fith.branching_ratio_ci_high,
            news_cfg.hawkes.cascade_ci_trigger,
        )
    else:
        cascade_val, cascade_bounded = None, None
    if hawkes_indicator_active:
        hawkes_layer_params = {
            "active": True,
            "mu_N": fith.params["mu"], "alpha": fith.params["alpha"], "beta": fith.params["beta"],
            "se_mu_N": _none_if_nan(fith.se["mu"]), "se_alpha": _none_if_nan(fith.se["alpha"]),
            "se_beta": _none_if_nan(fith.se["beta"]),
            "mean_mark": fith.mean_mark, "branching_ratio": fith.branching_ratio,
            "branching_ratio_se": _none_if_nan(fith.branching_ratio_se),
            "branching_ratio_ci_low": _none_if_nan(fith.branching_ratio_ci_low),
            "branching_ratio_ci_high": _none_if_nan(fith.branching_ratio_ci_high),
            "expected_cascade": cascade_val, "expected_cascade_bounded": cascade_bounded,
            "stationary": fith.stationary,
            "ks_stat": _none_if_nan(hawkes["ks"]["ks_stat"]),
            "ks_pvalue": _none_if_nan(hawkes["ks"]["p_value"]),
            "ks_passed": hawkes["ks"]["passed"],
            "n_events": fith.n_events, "n_starts": fith.n_starts,
            "starts_at_best": fith.starts_at_best,
            "coverage": hawkes["coverage"],
            "reflexive_threshold": news_cfg.hawkes.reflexive_threshold,
            "blocker": (None if covariate_active
                        else f"covariable lambda_N_z inactiva en el logit: {covariate_reason}"),
        }
    else:
        hawkes_layer_params = default_hawkes_layer_params(hawkes["blocker"])
        hawkes_layer_params["coverage"] = hawkes["coverage"]

    lam_today = float(lam_raw.dropna().iloc[-1]) if lam_raw is not None and lam_raw.notna().any() else 0.0
    lam_z_today = (float(lam_feature.dropna().iloc[-1])
                   if lam_feature is not None and lam_feature.notna().any() else 0.0)
    news_block = {
        "surprise_index": 0.0,   # capa de sorpresa inactiva (bloqueada aguas arriba)
        "lambda_N": lam_today,
        "lambda_N_z": lam_z_today,
        "branching_ratio": fith.branching_ratio if hawkes_indicator_active else 0.0,
        "branching_ratio_ci_low": (_none_if_nan(fith.branching_ratio_ci_low)
                                   if hawkes_indicator_active else None),
        "branching_ratio_ci_high": (_none_if_nan(fith.branching_ratio_ci_high)
                                    if hawkes_indicator_active else None),
        "expected_cascade": cascade_val if hawkes_indicator_active else 0.0,
        "expected_cascade_bounded": cascade_bounded if hawkes_indicator_active else True,
        "attribution": attribution,
    }
    news_layer_params = {
        "active": surprise_active,
        "delta": None, "delta_se": None,
        "surprise_start_date": news_cfg.surprise_start_date,
        "indicators": _indicator_weight_rows(news_cfg, weights, coverage_cal),
        "coverage": coverage_cal,
        "blocker": surprise_blocker,
    }

    if "beta_tvtp" in titular_params:
        P_today = transition_matrix_at(titular_params["d"], titular_params["beta_tvtp"], X_today_std[-1])
    else:
        P_today = titular_params["P"]

    warnings_list: list[str] = []
    warnings_list.append(
        "M3 (macro): " + macro_status + " La comparacion pre-registrada M5 vs M4 sigue "
        "infranqueable aguas arriba (por M4/sorpresa y la cobertura de lambda_N_z, NO por la "
        "macro); ver reports/ablation_news.md."
    )
    if single_regime:
        warnings_list.append(
            "Activo K=1 (un solo regimen; seleccion V1 por BIC + bootstrap LR + diagnostico de "
            "estabilidad 2026-08-15, reports/diag_btc_k2t_stability.md). Sin logit de transicion: "
            "NO hay TVTP y lambda_N_z/surprise_index no entran como covariables. El branching ratio "
            "de Hawkes se publica como indicador de fragilidad standalone, independiente de K."
        )
    elif m2h_blocker:
        warnings_list.append("Diagnostico M2+H bloqueado: " + m2h_blocker)
    elif not covariate_active:
        warnings_list.append(
            "lambda_N_z INACTIVA como covariable (compromiso pre-registrado M2+H): " + covariate_reason
        )
    if hawkes_indicator_active and hawkes["coverage"]["n_censored_days"]:
        warnings_list.append(
            f"{hawkes['coverage']['n_censored_days']} dia(s) del corpus censurados por el cap "
            "del API de GDELT: lambda_N esta subestimada en esos picos (censura contada, no oculta)."
        )
    if hawkes_indicator_active and hawkes["coverage"]["n_missing_days"]:
        cov = hawkes["coverage"]
        span = cov["n_days"] + cov["n_missing_days"]
        warnings_list.append(
            f"El corpus cubre {cov['n_days']} dias dentro de un span de {span} ({cov['n_missing_days']} "
            f"dias NO capturados, no genuinamente vacios). El Hawkes se ajusta sobre el TIEMPO OBSERVADO "
            f"(origen {cov['first_day']}; los dias fantasma se excinden del reloj), decision del director "
            "(2026-08-15) que REVIERTE la de ajustar sobre el span completo: mu_N y n son estimaciones "
            "sobre soporte observado, NO cotas superiores infladas por integrar mu_N sobre dias fantasma "
            "(trazabilidad antes/despues en validation_v3.md). Caveat vigente: el corpus es una muestra "
            f"acotada (~{cov['n_days']} dias) y el KS rechaza el kernel exponencial (D pequeno, ver aviso "
            "KS), asi que n/lambda_N reflejan ese ajuste exponencial sobre la ventana observada."
        )
    if hawkes_indicator_active:
        warnings_list.append(
            "El seendate de GDELT esta cuantizado a 15 min (~83% de titulares empatados). "
            "Se aplico dithering intra-bin (ruido U(0,15min) por evento, semilla fija) para "
            "identificar el Hawkes continuo, bajo el supuesto de LLEGADA UNIFORME dentro del "
            "bin observado. Decision del director (2026-08-14). Sin el dithering el MLE degenera."
        )
    if hawkes_indicator_active and hawkes["ks"]["passed"] is False:
        ks = hawkes["ks"]
        warnings_list.append(
            f"KS de re-escalamiento RECHAZA el kernel exponencial (D={ks['ks_stat']:.4f}, "
            f"p={ks['p_value']:.1e}, n={ks['n']}): el ajuste del Hawkes es imperfecto y se "
            "reporta tal cual. El TAMANO DE EFECTO D es pequeno; el rechazo lo domina el n "
            "enorme del corpus (~10^5 titulares), no un desajuste grande de forma. Candidato "
            "power-law en version futura (guia 6.6)."
        )
    if quick:
        warnings_list.append(
            "Corrida --quick: multistart reducido, NO cumple R6. Resultados PROVISIONALES."
        )

    if single_regime:
        spec = (
            f"MS-GJR-GARCH (Haas et al. 2004), UN REGIMEN (K=1: sin TVTP, sin logit de "
            f"transicion), innovaciones {'t de Student' if distt == 't' else 'Normales'}. "
            "lambda_N_z fuera por construccion; branching ratio de Hawkes standalone."
        )
    elif publish_m1:
        spec = (
            f"MS-GJR-GARCH (Haas et al. 2004), K={Kt}, matriz de transicion CONSTANTE "
            f"(M1: solo regimenes de volatilidad, SIN covariables de transicion; A4 "
            f"bias-variance), innovaciones {'t de Student' if distt == 't' else 'Normales'}. "
            "Branching ratio de Hawkes standalone (fuera del logit)."
        )
    else:
        spec = (
            f"MS-GJR-GARCH (Haas et al. 2004), TVTP (tecnico"
            + (" + lambda_N_z" if covariate_active else "")
            + f"), innovaciones {'t de Student' if distt == 't' else 'Normales'}"
            + ("" if covariate_active else " [lambda_N_z fuera del logit: ver warnings]")
        )

    payload = build_payload(
        asof=titular_returns.index[-1].date(), version=VERSION, run_id=run_id,
        git_commit=git_commit, config_hash=config_hash, K=Kt, labels=regime_labels(Kt),
        xi_history=xi_hist, P=titular_params["P"],
        entropy_mid=base.regime.entropy_mid, entropy_high=base.regime.entropy_threshold,
        seed=seed, n_multistart=today_run.fit.n_starts,
        converged=(today_run.fit.n_converged >= max(1, today_run.fit.n_starts // 3)),
        r_pct=r_pct_hist, argmax_idx=argmax_hist,
        asset=ASSET_NAME, validation_ref=f"reports/validation_v3{REPORT_SUFFIX}.md",
        warnings=warnings_list, spec=spec, tvtp=(titular_X is not None),
        covariates=(list(titular_X.columns) if titular_X is not None else []),
        transition_matrix_today=P_today,
        bootstrap_n_boot=base.v2.bootstrap.n_boot, bootstrap_block_len=base.v2.bootstrap.block_len,
        bootstrap_ci_level=base.v2.bootstrap.ci_level, bootstrap_min_obs=base.v2.bootstrap.min_obs,
        bootstrap_degenerate_duration_days=base.v2.bootstrap.degenerate_duration_days,
        news_layer=news_layer_list, news_block=news_block,
        news_layer_params=news_layer_params, hawkes_layer_params=hawkes_layer_params,
    )
    # Fase 3 (publicacion atomica): escribir SOLO en runs/<run_id>/. A latest/
    # solo se llega con scripts/promote.py (contrato + guardarrail).
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    publish(payload, run_dir / "irfn.json")
    log.info("    irfn.json escrito en runs/%s (version V3); no publicado a latest/", run_id)

    # --- history.parquet + walkforward.json del MODELO PUBLICADO (Fase 4) ---
    # Condicion 2: el walk-forward se RECALCULA sobre el modelo V3 publicado, no se
    # hereda del V0. Para K>=2 es el peldano de la ablacion que coincide con el
    # titular (M2, o M2+H si lambda_N_z entro al logit); para K=1 se corre un
    # walk-forward explicito del modelo de un solo regimen. El baseline de la
    # calibracion es la climatologia de regimen (honesto), no el favorecedor.
    n_starts_wf_pub = 6 if quick else base.v0.wf_n_starts
    if single_regime:
        wf_pub = walk_forward(
            returns, K=Kt, seed=seed, n_starts=n_starts_wf_pub,
            train_years=base.walkforward.train_years, test_months=base.walkforward.test_months,
            n_blocks_min=base.walkforward.n_blocks, dist=distt,
            checkpoint_path=ARTIFACTS.parent / "checkpoints" / f"v3_wf_pub_k1{REPORT_SUFFIX}.pkl",
            n_jobs=-1,  # paraleliza por bloque; determinista bit-identico al serial (R2)
        )
        published_spec_name = "K=1 (regimen unico)"
    else:
        published_spec_name = "M2+H" if covariate_active else "M2"
        wf_pub = abl.wf_results[published_spec_name]
    oos_pub = wf_pub.oos_frame
    xi_pred_pub = oos_pub[[f"xi_predicted_{k}" for k in range(wf_pub.K)]].to_numpy()
    xi_filt_pub = oos_pub[[f"xi_filtered_{k}" for k in range(wf_pub.K)]].to_numpy()
    calib_pub = calibration.summarize(xi_pred_pub, xi_filt_pub)
    write_walkforward_json(run_dir / "walkforward.json", wf_pub, calib_pub)
    write_history_parquet(run_dir / "history.parquet", oos_pub)
    log.info("    set OOS del modelo publicado (%s): log-loss=%.4f vs climatologia=%.4f "
             "(%d bloques)", published_spec_name, calib_pub["log_loss"],
             calib_pub["log_loss_baseline"], wf_pub.n_blocks)

    # --- artefactos de pantalla 3 (R9: la app solo lee) ---
    _dump({"events": _events_table(z_wide, weights)}, run_dir / "surprise_events.json")
    if hawkes_indicator_active:
        lam_hist = pd.DataFrame({"fecha": lam_raw.index.astype(str),
                                 "lambda_N": lam_raw.to_numpy()}).dropna()
        lam_hist.to_parquet(run_dir / "hawkes_history.parquet", index=False)
        rug = hawkes["scored"][["hora_titular", "s"]].copy()
        rug["hora_titular"] = rug["hora_titular"].dt.tz_convert("UTC").dt.tz_localize(None)
        rug.to_parquet(run_dir / "headline_rug.parquet", index=False)

    # --- auditoria PIT (incluye lambda_N_z si esta en el titular) ---
    log.info("    auditoria PIT...")
    pit_df = prefix_invariance_check(
        titular_returns, K=Kt, seed=seed, n_starts=6,
        train_len=len(titular_returns) - 260, dates_to_test=10, X=titular_X, dist=distt,
    )
    ledger_df = lag_ledger(v1_feature_registry(
        include_macro=False, include_news=True, include_hawkes=True,
    ))
    reest_blocks = []
    for name in runnable_names:
        if name in abl.wf_results:
            reest_blocks.extend(abl.wf_results[name].blocks)
    reest_df = block_reestimation_check(reest_blocks) if reest_blocks else block_reestimation_check([])

    ts_audit = hawkes["timestamp_audit"] or headlines_mod.timestamp_audit(
        headlines_mod.match_event_times(
            hawkes["headlines"] if hawkes["headlines"] is not None else pd.DataFrame(columns=["title", "hora_titular"]),
            cal, match_window_hours=news_cfg.headlines.match_window_hours,
        )
    )

    hl_cov = hawkes["coverage"]
    audit_obj = {
        "run_id": run_id, "generated_at": datetime.now(timezone.utc).isoformat(),
        "entropy_zones": {"mid": base.regime.entropy_mid, "high": base.regime.entropy_threshold},
        "prefix_invariance": {
            "passed": bool(pit_df.attrs["passed"]), "atol": pit_df.attrs["atol"],
            "rows": json.loads(pit_df.assign(fecha=pit_df["fecha"].astype(str)).to_json(orient="records")),
        },
        "lag_ledger": {"passed": bool(ledger_df.attrs["passed"]),
                       "rows": json.loads(ledger_df.to_json(orient="records"))},
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
        "headline_timestamp_check": ts_audit,
        "sources": [
            {"name": ASSET_NAME, "source": ASSET_SOURCE,
             "last_date": str(returns.index[-1].date()), "n_obs": int(len(returns))},
        ] + ([
            {"name": "titulares GDELT", "source": "gdelt doc 2.0 api",
             "last_date": hl_cov["last_day"], "n_obs": int(fith.n_events if hawkes_indicator_active else 0)},
        ] if hl_cov["last_day"] else []),
    }
    _dump(audit_obj, run_dir / "audit.json")

    # --- i) reportes (R8: SIEMPRE) ---
    log.info("[i] escribiendo reports/ablation_news.md y reports/validation_v3.md...")
    ctx = {
        "run_id": run_id, "K": Kt, "dist": distt, "quick": quick,
        "abl": abl, "runnable_names": runnable_names,
        "macro_status": macro_status, "surprise_blocker": surprise_blocker,
        "m2h_blocker": m2h_blocker, "dm_m2h": dm_m2h,
        "covariate_active": covariate_active, "covariate_reason": covariate_reason,
        "hawkes": hawkes, "hawkes_active": hawkes_indicator_active,
        "hawkes_layer_params": hawkes_layer_params, "ts_audit": ts_audit,
        "news_layer_params": news_layer_params, "pit_passed": bool(pit_df.attrs["passed"]),
    }
    _write_ablation_report(ctx)
    _write_validation_report(ctx)

    log.info("Listo V3. run_id=%s PIT=%s hawkes_activo=%s covariable_activa=%s",
             run_id, "VERDE" if pit_df.attrs["passed"] else "ROJA",
             hawkes_indicator_active, covariate_active)


def _none_if_nan(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


# --------------------------------------------------------------------------- #
# i) Reportes
# --------------------------------------------------------------------------- #
def _write_ablation_report(ctx: dict) -> None:
    abl, names = ctx["abl"], ctx["runnable_names"]
    if abl is None:
        # K=1: la escalera de ablacion NO aplica (sin regimenes que ablacionar). Se
        # escribe un reporte honesto en vez de fabricar una tabla vacia (R8).
        L = [
            "# Ablacion de la capa de noticias — NO APLICA (activo K=1)\n",
            f"generado: {datetime.now(timezone.utc).isoformat()}  |  run_id: `{ctx['run_id']}`  |  "
            f"K={ctx['K']}, dist={ctx['dist']}\n",
            "El activo se selecciono como **K=1** (un solo regimen) en V1 (BIC + bootstrap LR + "
            "diagnostico de estabilidad 2026-08-15, `reports/diag_btc_k2t_stability.md`). Sin "
            "regimenes NO hay matriz de transicion que modular: la escalera M0..M5 y el "
            "diagnostico M2+H no aplican (no hay transiciones sobre las que medir el aporte de "
            "covariables tecnicas, macro, sorpresa o lambda_N_z).\n",
            f"- M3 (macro): {ctx['macro_status']}",
            f"- El branching ratio de Hawkes (indicador de fragilidad) SI se publica y es "
            f"independiente de K; ver `reports/validation_v3{REPORT_SUFFIX}.md`.\n",
        ]
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / f"ablation_news{REPORT_SUFFIX}.md").write_text("\n".join(L), encoding="utf-8")
        return
    L: list[str] = []
    L.append("# Ablacion de la capa de noticias: M3 vs M4 (V2) y M4 vs M5 (V3)\n")
    L.append(f"generado: {datetime.now(timezone.utc).isoformat()}  |  run_id: `{ctx['run_id']}`  |  "
             f"K={ctx['K']}, dist={ctx['dist']}"
             + ("  |  **corrida --quick (multistart reducido, NO cumple R6; provisional)**" if ctx["quick"] else "") + "\n")
    L.append(f"> {PRE_REGISTERED_M5}\n")
    L.append(f"> {PRE_REGISTERED_M2H}\n")

    L.append("## Veredicto de la comparacion pre-registrada M4 vs M5\n")
    L.append("**M4 y M5 NO SE PUDIERON CORRER.** No es un resultado negativo de la capa de "
             "noticias (ningun DM se perdio): es la ausencia del peldano previo de la escalera. "
             "El bloqueante VINCULANTE aguas arriba es M4 (sorpresa); M3 (macro) NO bloquea "
             "-- su estado real se reporta abajo:\n")
    L.append(f"- M3 (macro): {ctx['macro_status']}")
    L.append(f"- M4 (sorpresa): {ctx['surprise_blocker'] or 'sin bloqueante (no deberia leerse esto)'}")
    L.append("- M5 = M4 + lambda_N_z depende de M4 por el diseno acumulativo de la escalera "
             "(guia 8.1: una variable a la vez).\n")

    L.append("## Diagnostico pre-declarado M2+H (esta sesion)\n")
    if ctx["dm_m2h"] is not None:
        d = ctx["dm_m2h"]
        L.append(f"M2+H vs M2: DM={_fmt(d['dm_stat'],3)}, p={_fmt(d['p_value'],3)}, "
                 f"dif. media={_fmt(d['mean_diff'],5)}.\n")
        L.append(f"**Decision (compromiso pre-registrado): {ctx['covariate_reason']}** "
                 f"lambda_N_z se publica {'ACTIVA' if ctx['covariate_active'] else 'INACTIVA'} "
                 "como covariable del logit.\n")
    else:
        L.append(f"**M2+H NO corrio.** Motivo: {ctx['m2h_blocker']}\n")
        L.append("El diagnostico queda pre-declarado con su compromiso (arriba) y se corre en "
                 "cuanto la cobertura del backfill alcance.\n")
    L.append("Recordatorio: M2+H responde una pregunta MAS DEBIL que M5 vs M4 (aporte sobre el "
             "TVTP tecnico, no sobre la sorpresa calendarizada). No sustituye la comparacion "
             "pre-registrada; es el minimo honesto para decidir si lambda_N_z entra al titular "
             "mientras M4 siga bloqueada. Decision de diseno para revision del director "
             "(docstring de scripts/run_v3.py).\n")

    L.append("## Ablacion corrida esta sesion\n")
    L.append(f"Peldanos corridos: `{names}` (M0..M5 declarados en validation/ablation.full_ladder_specs; "
             "M2+H declarado en scripts/run_v3.py).\n")
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

    L.append("## w_i estimados (V2, sin cambios: capa de sorpresa bloqueada)\n")
    L.append("| indicador | w_i | SE | t | distinguible de cero | n eventos |")
    L.append("| :-- | --: | --: | --: | :-- | --: |")
    for row in ctx["news_layer_params"]["indicators"]:
        L.append(f"| {row['indicator']} | {_fmt(row['w'],4)} | {_fmt(row['se'],4)} | "
                 f"{_fmt(row['t_stat'],2)} | {row['distinguible_de_cero']} | {row['n_events']} |")
    L.append("")
    if ctx["news_layer_params"]["blocker"]:
        L.append(f"> Bloqueante de la capa de sorpresa: {ctx['news_layer_params']['blocker']}\n")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"ablation_news{REPORT_SUFFIX}.md").write_text("\n".join(L), encoding="utf-8")


def _write_validation_report(ctx: dict) -> None:
    h = ctx["hawkes_layer_params"]
    ks = ctx["hawkes"]["ks"] if ctx["hawkes_active"] else None
    ts = ctx["ts_audit"]
    L: list[str] = []
    L.append(f"# Validacion V3 ({ASSET_NAME}) — Hawkes y branching ratio\n")
    L.append(f"generado: {datetime.now(timezone.utc).isoformat()}  |  run_id: `{ctx['run_id']}`"
             + ("  |  **corrida --quick (provisional, no cumple R6)**" if ctx["quick"] else "") + "\n")

    L.append("## Criterios de aceptacion de V3 (guia, checklist)\n")
    L.append("| criterio | estado |")
    L.append("| :-- | :-- |")
    L.append("| test_hawkes_recovery pasa | ver seccion Tests (suite de pytest) |")
    if ctx["hawkes_active"]:
        ks_line = (f"reportado: stat={_fmt(h['ks_stat'],4)}, p={_fmt(h['ks_pvalue'],4)} — "
                   + ("NO rechaza Exp(1)" if h["ks_passed"] else
                      "RECHAZA Exp(1); se documenta, kernel power-law candidato a futuro"))
        L.append(f"| KS de re-escalamiento reportado (pase o falle) | {ks_line} |")
        L.append(f"| n = alpha*E[s]/beta < 1 verificado | n = {_fmt(h['branching_ratio'],4)} "
                 f"({'estacionario' if h['stationary'] else 'EXPLOSIVO: NO publicable'}) |")
    else:
        L.append("| KS de re-escalamiento reportado | capa bloqueada: sin MLE que testear (ver Bloqueantes) |")
        L.append("| n = alpha*E[s]/beta < 1 verificado | capa bloqueada (ver Bloqueantes) |")
    ts_a = ts["titular_vs_evento"]
    ts_estado = ("VERDE (vacuo: 0 titulares emparejados; el calendario de consenso sigue vacio)"
                 if ts_a.get("vacuous") else ("VERDE" if ts["passed"] else "ROJO"))
    L.append(f"| chequeo de timestamps en pantalla 6 | {ts_estado} |")
    if ctx["K"] == 1:
        L.append("| ablacion M4 vs M5 | NO aplica: activo K=1 (sin regimenes ni logit); "
                 "ver reports/ablation_news.md |")
    else:
        L.append(f"| ablacion M4 vs M5 | bloqueada aguas arriba; diagnostico M2+H "
                 f"{'corrido' if ctx['dm_m2h'] is not None else 'bloqueado por cobertura'} "
                 "(reports/ablation_news.md) |")
    L.append("")

    if ctx["hawkes_active"]:
        L.append("## MLE de Hawkes sobre el corpus real de titulares\n")
        cov = h["coverage"]
        L.append(f"Corpus: {h['n_events']:,} titulares GDELT, {cov['first_day']} a {cov['last_day']} "
                 f"({cov['n_days']} dias; {cov['n_missing_days']} faltantes; "
                 f"{cov['n_censored_days']} censurados por el cap de 250/consulta del API — la "
                 "censura sesga lambda_N A LA BAJA en los picos y esta contada).\n")
        L.append("| parametro | estimado | SE |")
        L.append("| :-- | --: | --: |")
        L.append(f"| mu_N (titulares/dia) | {_fmt(h['mu_N'],4)} | {_fmt(h['se_mu_N'],4)} |")
        L.append(f"| alpha | {_fmt(h['alpha'],4)} | {_fmt(h['se_alpha'],4)} |")
        L.append(f"| beta (1/dia) | {_fmt(h['beta'],4)} | {_fmt(h['se_beta'],4)} |")
        L.append("")
        casc = ("no acotada (censurada: el IC superior de n alcanza el disparador "
                "news.hawkes.cascade_ci_trigger; 1/(1-n) diverge cerca de la frontera)"
                if h["expected_cascade"] is None else f"1/(1-n) = {_fmt(h['expected_cascade'],3)}")
        L.append(f"E[s] = {_fmt(h['mean_mark'],4)} (media de la relevancia FinBERT). "
                 f"**Branching ratio n = alpha*E[s]/beta = {_fmt(h['branching_ratio'],4)}** "
                 f"(IC95 metodo delta [{_fmt(h['branching_ratio_ci_low'],4)}, "
                 f"{_fmt(h['branching_ratio_ci_high'],4)}]; correccion por marcas: NO alpha/beta = "
                 f"{_fmt(h['alpha']/h['beta'],4)}). Cascada esperada E[hijos] = {casc}.\n")
        hk = ctx["hawkes"]
        if hk.get("n_before_span") is not None:
            L.append("### Correccion de la ventana de observacion (Parte A, 2026-08-15)\n")
            L.append(
                "El Hawkes se ajusta sobre el TIEMPO OBSERVADO, no sobre el span calendario "
                "(decision del director que REVIERTE la del 2026-08-14). El compensador "
                "Lambda(T)=mu_N*T integraba mu_N sobre los dias fantasma (rango sin titulares "
                "capturados), sesgando mu_N a la baja e inflando n hacia la frontera. Se comprime "
                "el reloj excindiendo esos dias. Trazabilidad antes vs despues:\n")
            L.append("| ventana | T (dias) | mu_N | n = alpha*E[s]/beta |")
            L.append("| :-- | --: | --: | --: |")
            L.append(f"| ANTES (span calendario, sesgado) | {_fmt(hk['T_span'],1)} | "
                     f"{_fmt(hk['mu_before_span'],4)} | {_fmt(hk['n_before_span'],4)} |")
            L.append(f"| DESPUES (tiempo observado) | {_fmt(hk['T_obs'],1)} | "
                     f"{_fmt(hk['mu_after_observed'],4)} | {_fmt(hk['n_after_observed'],4)} |")
            L.append("")
        L.append(f"Multistart (R6): {h['starts_at_best']}/{h['n_starts']} arranques convergieron "
                 "al mismo optimo (tolerancia 1e-4 en log-verosimilitud)."
                 + (" Pocos arranques en el optimo indicarian un problema de identificacion; "
                    "no es el caso." if h["starts_at_best"] >= max(3, h["n_starts"] // 3) else
                    " **ADVERTENCIA: pocos arranques en el optimo — posible problema de "
                    "identificacion; revisar antes de confiar en los SE.**") + "\n")
        L.append("### Bondad de ajuste (teorema de re-escalamiento temporal)\n")
        if h["ks_passed"]:
            L.append(f"KS sobre interarribos re-escalados vs Exp(1): stat={_fmt(h['ks_stat'],4)}, "
                     f"p={_fmt(h['ks_pvalue'],4)} (n={ks['n']}). **No se rechaza**: el kernel "
                     "exponencial es consistente con el corpus.\n")
        else:
            L.append(f"KS sobre interarribos re-escalados vs Exp(1): stat={_fmt(h['ks_stat'],4)}, "
                     f"p={_fmt(h['ks_pvalue'],6)} (n={ks['n']}). **SE RECHAZA.** La respuesta NO "
                     "es forzar el modelo (guia 6.6): se reporta tal cual y el kernel power-law "
                     "queda como candidato para una version futura. Mientras tanto, lambda_N y n "
                     "se leen como aproximaciones bajo kernel exponencial, con este caveat "
                     "impreso tambien en pantalla 3.\n")
    else:
        L.append("## Bloqueantes de la capa de Hawkes\n")
        L.append(f"- {h['blocker']}\n")

    L.append("## Auditoria de timestamps de titulares (Trampa 3)\n")
    if ts_a.get("vacuous"):
        L.append("titular vs evento: pase VACUO declarado (0 titulares emparejados con releases; "
                 "el calendario de consenso sigue vacio — mismo bloqueante de V2). El chequeo se "
                 "vuelve sustantivo en cuanto ambos feeds acumulen datos.\n")
    else:
        L.append(f"titular vs evento: {ts_a['n_matched']} pares; masa negativa = "
                 f"{_fmt(ts_a['negative_mass'],4)}; mediana = {_fmt(ts_a['median_hours'],2)} h. "
                 f"{'VERDE' if ts_a['passed'] else '**ROJO: timestamps corruptos**'}.\n")
    ts_b = ts["resolucion_feed"]
    if ts_b["n_headlines"]:
        L.append(f"resolucion del feed: {ts_b['n_headlines']:,} titulares; "
                 f"{_fmt(100*ts_b['midnight_exact_frac'],2)}% con timestamp de medianoche exacta "
                 f"({'VERDE: el feed publica horas reales' if ts_b['passed'] else 'ROJO: feed de solo-fechas'}).\n")

    L.append("## Decisiones de diseno para revision del director\n")
    L.append("1. (mu_N, alpha, beta) del Hawkes estimados UNA VEZ sobre todo el corpus, no por "
             "bloque de walk-forward — analogo a delta en V2 (parametros del proceso puntual, "
             "estimados sin mirar retornos; los beta_ij del logit si se re-estiman por bloque, R2). "
             "Detalle en el docstring de scripts/run_v3.py.")
    L.append("2. Diagnostico M2+H pre-declarado como decision intermedia mientras M5 vs M4 este "
             "bloqueada aguas arriba; compromiso pre-registrado en reports/ablation_news.md.")
    L.append("3. s_i = 1 - P(neutral) de FinBERT como relevancia (Trampa 4: jamas se pregunta la "
             "direccion; restriccion documentada en features/relevance.py, con la limitacion "
             "medida de que el score solo es significativo dentro del dominio financiero que "
             "filtra la consulta GDELT).\n")

    L.append("## Pendientes que esta version NO resuelve\n")
    L.append("- Backfill de GDELT hasta headlines.start_date (2017): en curso, reanudable "
             "(scripts/capture_headlines.py); la cobertura actual viaja en el artefacto.")
    L.append("- M5 vs M4: requiere ALFRED_API_KEY (gratis, minutos) y consenso historico "
             "(Trading Economics de pago o acumulacion hacia adelante).")
    L.append(f"- Invarianza de prefijo del titular: {'VERDE' if ctx['pit_passed'] else 'ROJA'} "
             "en esta corrida (pantalla 6).")
    if ctx["quick"]:
        L.append("- Re-correr sin --quick para cumplir R6 (multistart completo) antes de dar "
                 "por cerrada la version.")
    L.append("")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"validation_v3{REPORT_SUFFIX}.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Orquestador V3 (Hawkes y branching ratio).")
    ap.add_argument("--quick", action="store_true", help="Menos arranques (humo, no cumple R6).")
    ap.add_argument("--no-capture", action="store_true", help="No tocar la red de GDELT (usar solo disco).")
    ap.add_argument("--asset", default=None,
                    help="Nombre del activo en config/assets.yaml (p.ej. BTC). "
                         "Sin este flag, comportamiento identico al historico (SPY).")
    ap.add_argument("--publish-m1", action="store_true",
                    help="Publica M1 (K=2, matriz de transicion CONSTANTE, sin covariables "
                         "de transicion; A4 bias-variance) en vez de M2. No cambia la "
                         "estimacion: solo enruta que spec se publica.")
    args = ap.parse_args()
    _configure_asset(BaseConfig.load(), args.asset)
    full_run(quick=args.quick, no_capture=args.no_capture, publish_m1=args.publish_m1)


if __name__ == "__main__":
    main()

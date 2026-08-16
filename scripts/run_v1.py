"""Orquestador de V1: TVTP + seleccion de K (SIN NOTICIAS; eso es V2/V3).

Dos etapas, porque la corrida honesta es cara y el director la pidio escalonada:

    python scripts/run_v1.py kselect     # FASE 1: tabla BIC + bootstrap K vs K-1
                                          #         + fit titular con betas y SE
                                          #         + artefacto "de hoy" V1 (pantalla 1)
                                          #         + reports/validation_v1.md (seleccion de K)
    python scripts/run_v1.py ablation    # FASE 2: walk-forward de V0 vs titular V1
                                          #         + ablacion M0/M1/M2 + Diebold-Mariano
                                          #         (completa validation_v1.md y pantalla 5)

    (--quick reduce arranques/replicas para HUMO, no para reportar; marca PROVISIONAL.)

Alcance V1 (CLAUDE.md): innovaciones t de Student vs Normal (decide el BIC),
matriz de transicion TVTP modulada por covariables TECNICAS rezagadas
(sma_gap, bb_width_z). La covariable macro hy_oas_z queda BLOQUEADA en esta
corrida: no hay ALFRED_API_KEY y el HY OAS de FRED/ALFRED esta limitado a una
ventana rodante de 3 anios (ver reports/data_audit.md), insuficiente para el
walk-forward. Se documenta el bloqueante; jamas se rellena con FRED revisado (R4).

Cada etapa persiste su salida en cuanto la calcula (artifacts/latest/v1_*.json),
para que una corrida larga que se interrumpa no pierda lo ya hecho.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irfn.config import BaseConfig  # noqa: E402
from irfn.data.prices import load_close  # noqa: E402
from irfn.features.technical import technical_features  # noqa: E402
from irfn.models.estimate import cv_l1_lambda  # noqa: E402
from irfn.models.tvtp import transition_matrix_at  # noqa: E402
from irfn.outputs.publish import build_payload, publish  # noqa: E402
from irfn.pipeline import regime_labels, run_pipeline  # noqa: E402
from irfn.validation.ablation import ModelSpec, run_ablation  # noqa: E402
from irfn.validation.tests_stat import bic_table, bootstrap_lr_test, diebold_mariano, pesaran_timmermann  # noqa: E402
from irfn.validation.walkforward import walk_forward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("run_v1")

ARTIFACTS = ROOT / "artifacts" / "latest"
RUNS = ROOT / "artifacts" / "runs"
REPORTS = ROOT / "reports"
CONFIG_DIR = ROOT / "config"
VERSION = "V1"

# Activo de la corrida (globals de modulo, fijados por _configure_asset() en
# main() antes de despachar a kselect/ablation). Default = SPY, identico al
# comportamiento historico.
ASSET_NAME = None
ASSET_SOURCE = "yfinance"
ASSET_IDENTIFIER = None
ASSET_PRICES_START = None
ASSET_SAMPLE_START = None
REPORT_SUFFIX = ""


def _configure_asset(base: BaseConfig, asset_override: str | None) -> None:
    """Fija el activo de la corrida completa. Sin --asset: comportamiento identico
    al historico (SPY via yfinance, artifacts/latest/, reports/validation_v1.md).
    Con --asset NAME: resuelve source/symbol/start_date desde config/assets.yaml y
    escribe en artifacts/<name>/ + reports/..._<name>.md -- aditivo, nunca toca la
    linea SPY existente."""
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
    # Sin una fecha de "pre-calentamiento" separada por activo en assets.yaml, se
    # usa la misma start_date para precios y muestra; dropna() en load_sample ya
    # recorta los ~200 dias que tarda en calentar SMA200/z_window.
    ASSET_PRICES_START = asset_cfg.start_date
    ASSET_SAMPLE_START = asset_cfg.start_date
    REPORT_SUFFIX = f"_{slug}"
    ARTIFACTS = ROOT / "artifacts" / slug / "latest"
    RUNS = ROOT / "artifacts" / slug / "runs"


# --------------------------------------------------------------------------- #
# Identidad reproducible (misma logica que run_pipeline.py V0)
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


# --------------------------------------------------------------------------- #
# Datos: precio ancla -> features tecnicas rezagadas -> muestra alineada (R3)
# --------------------------------------------------------------------------- #
def load_sample(base: BaseConfig) -> tuple[pd.Series, pd.DataFrame]:
    """Devuelve (returns, X) YA alineados y sin NaN, indexados por fecha.

    El precio se descarga desde v1.prices_start (2010) para calentar SMA200 y el
    z-score; la muestra del modelo arranca en v0.returns_start (2013), cuando las
    ventanas de features ya no son NaN. X trae las covariables tecnicas con su
    .shift(1) explicito aplicado en features/technical.py (R3): fila t = x_{t-1}.
    """
    w = base.windows
    close = load_close(ASSET_IDENTIFIER, ASSET_PRICES_START, source=ASSET_SOURCE)
    r_all = (np.log(close).diff() * 100.0).rename("r")
    X_all = technical_features(
        close, sma_short=w.sma_short, sma_long=w.sma_long,
        bb_window=w.bb_window, bb_k=w.bb_k, z_window=w.z_window,
    )
    covs = base.v1.covariates_ablation_m2   # tecnico-solo: [sma_gap, bb_width_z]
    df = pd.concat([r_all, X_all[covs]], axis=1).loc[ASSET_SAMPLE_START:].dropna()
    returns = df["r"]
    X = df[covs]
    log.info("muestra: %d obs, %s a %s, covs=%s",
             len(returns), returns.index[0].date(), returns.index[-1].date(), covs)
    return returns, X


# --------------------------------------------------------------------------- #
# FASE 1 -- seleccion de K + titular
# --------------------------------------------------------------------------- #
def stage_kselect(quick: bool) -> None:
    base = BaseConfig.load()
    seed = base.model.seed
    v1 = base.v1
    returns, X = load_sample(base)
    r = returns.to_numpy()
    covs = list(X.columns)

    bic_starts = 6 if quick else v1.ktest.bic_n_starts
    boot_n = 5 if quick else v1.ktest.n_boot
    boot_starts = 3 if quick else v1.ktest.boot_n_starts

    out: dict = {
        "version": VERSION,
        "quick_provisional": quick,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asof": str(returns.index[-1].date()),
        "n_obs": int(len(r)),
        "sample_start": str(returns.index[0].date()),
        "covariates_titular": covs,
        "macro_blocker": (
            "hy_oas_z EXCLUIDA: sin ALFRED_API_KEY y HY OAS (BAMLH0A0HYM2) limitado a "
            "ventana rodante de 3 anios en FRED/ALFRED, insuficiente para el walk-forward "
            "(reports/data_audit.md). Titular corre tecnico-solo; jamas se rellena con "
            "FRED revisado (R4). Retomar cuando haya fuente vintage con cobertura."
        ),
        "config": {
            "k_candidates": v1.k_candidates, "dist_candidates": v1.dist_candidates,
            "bic_n_starts": bic_starts, "boot_n": boot_n, "boot_starts": boot_starts,
            "seed": seed,
        },
    }

    ckpt_dir = ARTIFACTS.parent / "checkpoints"

    # --- 1) Tabla BIC completa: K x dist. No se esconden los perdedores. ---
    log.info("=== BIC: K=%s x dist=%s (%d arranques) ===", v1.k_candidates, v1.dist_candidates, bic_starts)
    t0 = time.time()
    bic = bic_table(r, k_candidates=v1.k_candidates, dists=v1.dist_candidates,
                    n_starts=bic_starts, seed=seed,
                    checkpoint_path=ckpt_dir / "bic_table.pkl")
    out["bic_table"] = bic
    out["bic_seconds"] = round(time.time() - t0, 1)
    winner = min(bic, key=lambda row: row["bic"])
    out["winner"] = {"K": winner["K"], "dist": winner["dist"], "bic": winner["bic"]}
    log.info("BIC ganador: K=%d dist=%s bic=%.1f", winner["K"], winner["dist"], winner["bic"])
    _dump(out, ARTIFACTS.parent / "analysis" / "v1_kselect.json")   # persiste ya

    # --- 2) Bootstrap parametrico K vs K-1 para la dist ganadora (Hansen es
    #        prohibitivo; decision documentada en validation/tests_stat.py). ---
    dist_w = winner["dist"]
    ladder = [K for K in v1.k_candidates if K >= 2]   # 2v1, 3v2, 4v3
    boots = []
    for K in ladder:
        log.info("=== bootstrap K=%d vs %d (dist=%s, %d replicas) ===", K, K - 1, dist_w, boot_n)
        t0 = time.time()
        res = bootstrap_lr_test(
            r, K_null=K - 1, K_alt=K, dist=dist_w,
            n_boot=boot_n, n_starts_data=bic_starts, n_starts_boot=boot_starts, seed=seed,
            checkpoint_path=ckpt_dir / f"bootstrap_K{K}vs{K-1}.pkl", checkpoint_every=5,
        )
        res["seconds"] = round(time.time() - t0, 1)
        boots.append(res)
        out["bootstrap_ladder"] = boots
        _dump(out, ARTIFACTS.parent / "analysis" / "v1_kselect.json")   # persiste tras cada comparacion
        log.info("  LR_obs=%.2f p=%.3f (%.0fs)", res["lr_obs"], res["p_value"], res["seconds"])

    # --- 3) Fit titular: K*, dist*, TVTP tecnico. lambda L1 por CV DENTRO de la
    #        muestra (el "train de hoy" = todo el historico hasta asof; no hay
    #        test aqui, es el artefacto publicado). Betas por MLE con SE (R7). ---
    Kt, distt = winner["K"], winner["dist"]
    log.info("=== titular: K=%d dist=%s TVTP%s ===", Kt, distt, covs)
    if Kt == 1:
        # Sin regimenes no hay matriz de transicion que modular: el titular TVTP
        # no aplica. Se reporta y se degrada al modelo ganador P-constante.
        out["titular"] = {"note": "BIC eligio K=1: sin TVTP posible; titular = GARCH un regimen."}
        _dump(out, ARTIFACTS.parent / "analysis" / "v1_kselect.json")
        _publish_v1_artifact(base, returns, X, Kt, distt, l1_lambda=0.0, tvtp=False)
    else:
        cv_starts = 3 if quick else v1.tvtp.cv_n_starts
        lam, cv_rows = cv_l1_lambda(
            r, X.to_numpy(), Kt, dist=distt, l1_grid=v1.tvtp.l1_grid,
            val_frac=v1.tvtp.cv_val_frac, n_starts=cv_starts, seed=seed,
            l1_smooth_eps=v1.tvtp.l1_smooth_eps,
        )
        log.info("  lambda L1 elegido por CV = %g", lam)
        titular = run_pipeline(
            returns, K=Kt, seed=seed, n_starts=bic_starts, compute_se=True,
            X=X, dist=distt, l1_lambda=lam, l1_smooth_eps=v1.tvtp.l1_smooth_eps,
        )
        fr = titular.fit
        out["titular"] = {
            "K": Kt, "dist": distt, "covariates": covs,
            "l1_lambda": lam, "l1_cv_table": cv_rows,
            "loglik": fr.loglik, "bic": fr.bic, "n_converged": fr.n_converged,
            "n_starts": fr.n_starts, "hessian_ok": fr.hessian_ok,
            "d_intercepts": fr.params["d"].tolist(),
            "beta_tvtp": fr.params["beta_tvtp"].tolist(),        # (K, K-1, n_cov)
            "beta_tvtp_se": (fr.se.get("beta_tvtp").tolist() if fr.se.get("beta_tvtp") is not None else None),
            "nu": (fr.params["nu"].tolist() if distt == "t" else None),
            "v": fr.params["v"].tolist(),
            "labels": regime_labels(Kt),
        }
        _dump(out, ARTIFACTS.parent / "analysis" / "v1_kselect.json")
        _publish_v1_artifact(base, returns, X, Kt, distt, l1_lambda=lam, tvtp=True,
                             titular=titular)

    # --- 4) Reporte de seleccion de K (R8). La interpretacion de signos se
    #        genera de los numeros REALES; el director la refina despues. ---
    _write_kselect_report(out)
    log.info("FASE 1 lista. run kselect -> %s", (REPORTS / f"validation_v1{REPORT_SUFFIX}.md"))


def _publish_v1_artifact(base, returns, X, K, dist, *, l1_lambda, tvtp, titular=None):
    """Publica artifacts/latest/irfn.json en version V1: regimen de hoy + matriz
    de transicion CONDICIONAL evaluada en x_asof (la ultima covariable rezagada
    disponible; contrato en models/tvtp.py). La pantalla 1 lee esta matriz y el
    regimen actual para el texto 'probabilidad de pasar a risk-off'."""
    seed = base.model.seed
    labels = regime_labels(K)
    config_hash, git_commit = _config_hash(), _git_commit()
    run_id = _run_id(config_hash, git_commit, seed)

    if titular is None:   # caso K=1: fit P-constante para el artefacto
        titular = run_pipeline(returns, K=K, seed=seed, n_starts=base.v1.ktest.bic_n_starts,
                               compute_se=False, dist=dist)
    frame = titular.frame
    xi_hist = frame[[f"xi_filtered_{k}" for k in range(K)]].to_numpy()
    P_const = titular.fit.P

    if tvtp:
        # x_asof = ultima covariable disponible, estandarizada con el scaler de la
        # muestra (train_scaler con train_len=None = todo el pasado hasta asof; no
        # es look-ahead para el artefacto de hoy). transition_matrix_at evalua el
        # softmax del logit en ese punto: la matriz condicional "de hoy".
        mean, std = titular.scaler
        x_asof = (X.to_numpy()[-1] - mean) / std
        P_today = transition_matrix_at(titular.fit.params["d"], titular.fit.params["beta_tvtp"], x_asof)
        spec = (f"MS-GJR-GARCH (Haas et al. 2004), TVTP (logit sobre {list(X.columns)} rezagadas), "
                f"innovaciones {'t de Student' if dist == 't' else 'Normales'}")
    else:
        P_today = P_const
        spec = f"MS-GJR-GARCH (Haas et al. 2004), P constante, innovaciones {'t de Student' if dist=='t' else 'Normales'}"

    warnings = []
    if titular.fit.hessian_ok is False:
        warnings.append("Hessiano degenerado en el titular: algunas SE de betas no son fiables.")
    warnings.append("Walk-forward de V1 (ablacion + DM vs V0) es la etapa 'ablation'; "
                    "hasta correrla, pantalla 5 muestra el walk-forward de V0.")

    payload = build_payload(
        asof=returns.index[-1].date(), version=VERSION, run_id=run_id,
        git_commit=git_commit, config_hash=config_hash, K=K, labels=labels,
        xi_history=xi_hist, P=P_const,
        entropy_mid=base.regime.entropy_mid, entropy_high=base.regime.entropy_threshold,
        seed=seed, n_multistart=titular.fit.n_starts,
        converged=(titular.fit.n_converged >= max(1, titular.fit.n_starts // 3)),
        r_pct=frame["r"].to_numpy(), argmax_idx=frame["argmax_idx"].to_numpy(),
        asset=ASSET_NAME, validation_ref=f"reports/validation_v1{REPORT_SUFFIX}.md",
        warnings=warnings, spec=spec, tvtp=tvtp, covariates=list(X.columns) if tvtp else [],
        transition_matrix_today=P_today,
        bootstrap_n_boot=base.v2.bootstrap.n_boot,
        bootstrap_block_len=base.v2.bootstrap.block_len,
        bootstrap_ci_level=base.v2.bootstrap.ci_level,
        bootstrap_min_obs=base.v2.bootstrap.min_obs,
    )
    # Fase 3: escribir SOLO en runs/<run_id>/; a latest/ solo por scripts/promote.py.
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    publish(payload, run_dir / "irfn.json")
    log.info("artefacto V1 escrito en runs/%s (run_id=%s, tvtp=%s); no publicado a latest/",
             run_id, run_id, tvtp)


# --------------------------------------------------------------------------- #
# FASE 2 -- ablacion walk-forward + Diebold-Mariano contra V0
# --------------------------------------------------------------------------- #
def stage_ablation(quick: bool, test_months: int | None, n_jobs: int = 1) -> None:
    base = BaseConfig.load()
    seed = base.model.seed
    v1 = base.v1
    returns, X = load_sample(base)
    covs = list(X.columns)

    # Ganador de la seleccion de K (de la fase 1). Si no existe, cae a K=2/t.
    ks_path = ARTIFACTS.parent / "analysis" / "v1_kselect.json"
    if ks_path.exists():
        ks = json.loads(ks_path.read_text(encoding="utf-8"))
        Kt, distt = ks["winner"]["K"], ks["winner"]["dist"]
    else:
        log.warning("no hay v1_kselect.json; usando K=2 dist=t por defecto para la ablacion.")
        Kt, distt = 2, "t"
    if Kt == 1:
        Kt = 2   # la ablacion necesita >=2 regimenes para M1/M2; se documenta
        log.warning("BIC eligio K=1; la ablacion usa K=2 para ilustrar el aporte de regimenes.")

    n_starts = 6 if quick else base.v0.wf_n_starts
    tmonths = test_months or base.walkforward.test_months
    n_blocks_min = base.walkforward.n_blocks
    l1_grid = [0.0, 2.0, 8.0] if quick else v1.tvtp.l1_grid

    log.info("=== ablacion: K=%d dist=%s, test=%dm, %d arranques/bloque ===", Kt, distt, tmonths, n_starts)

    # Escalera: M0 un regimen, M1 HMM K P-constante, M2 +TVTP tecnico. La dist es
    # la ganadora del BIC en M1/M2 (M0 usa la misma para comparar en igualdad).
    specs = [
        ModelSpec("M0", 1, distt, None, "GARCH un solo regimen (piso)"),
        ModelSpec("M1", Kt, distt, None, f"HMM K={Kt} P constante"),
        ModelSpec("M2", Kt, distt, tuple(covs), f"+TVTP tecnico {covs}"),
    ]
    ckpt_dir = ARTIFACTS.parent / "checkpoints" / "ablation"
    abl = run_ablation(
        returns, X, specs, seed=seed, n_starts=n_starts,
        train_years=base.walkforward.train_years, test_months=tmonths,
        n_blocks_min=n_blocks_min, l1_grid=l1_grid,
        cv_val_frac=v1.tvtp.cv_val_frac, cv_n_starts=(3 if quick else v1.tvtp.cv_n_starts),
        l1_smooth_eps=v1.tvtp.l1_smooth_eps,
        checkpoint_dir=ckpt_dir,
        n_jobs=n_jobs,
    )

    # V0 explicito para el DM pedido en el criterio de aceptacion: K=2 Normal
    # P-constante, sobre la MISMA muestra y malla de bloques (perdidas emparejadas).
    log.info("=== walk-forward V0 (K=2 Normal P-const) sobre la misma muestra ===")
    wf_v0 = walk_forward(returns, K=2, seed=seed, n_starts=n_starts,
                         train_years=base.walkforward.train_years, test_months=tmonths,
                         n_blocks_min=n_blocks_min, dist="normal",
                         checkpoint_path=ckpt_dir / "wf_V0.pkl", n_jobs=n_jobs)

    loss_v0 = (-wf_v0.oos_frame["loglik_obs"])
    loss_titular = abl.loss_series["M2"]
    common = loss_v0.index.intersection(loss_titular.index)
    dm_vs_v0 = diebold_mariano(loss_titular.loc[common].to_numpy(), loss_v0.loc[common].to_numpy())
    dm_vs_v0.update({"model_a": "M2 (titular V1)", "model_b": "V0 (K=2 Normal P-const)"})

    # Pesaran-Timmermann direccional del titular: signo del retorno realizado vs
    # signo de la media predictiva (media del retorno bajo xi predicho). Como el
    # modelo no predice media condicional distinta por regimen mas alla de mu_k,
    # se usa E[r_{t}] = sum_k xi_pred_k * mu_k. Se toma del oos_frame de M2.
    pt = _directional_pt(abl.wf_results["M2"], Kt)

    out = {
        "version": VERSION, "quick_provisional": quick,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "K": Kt, "dist": distt, "covariates": covs,
        "test_months": tmonths, "n_starts_block": n_starts,
        "ablation_table": abl.table,
        "dm_ladder": abl.dm_pairs,
        "dm_vs_v0": dm_vs_v0,
        "pesaran_timmermann_titular": pt,
        "n_blocks": abl.wf_results["M2"].n_blocks,
    }
    _dump(out, ARTIFACTS.parent / "analysis" / "v1_ablation.json")
    _write_v1_walkforward_json(abl, wf_v0)
    _append_ablation_report(out)
    log.info("FASE 2 lista. DM(titular vs V0) stat=%.3f p=%.3f", dm_vs_v0["dm_stat"], dm_vs_v0["p_value"])


def _directional_pt(wf, K: int) -> dict:
    """Pesaran-Timmermann del titular: signo del retorno realizado OOS vs signo de
    la media predictiva un-paso-adelante E[r_t|F_{t-1}] = sum_k xi_pred_k(t) mu_k.

    Desde V4 esa media se persiste como columna `r_pred_mean` del oos_frame
    (walkforward.py: xi_p_test @ params["mu"]) -- es EXACTAMENTE el nivel que
    Pesaran-Timmermann espera (retorno realizado vs media predictiva; los signos
    se toman dentro del test). Antes se usaba un proxy debil (0.5 - P(alta vol))
    porque mu_k por fecha no estaba disponible en el oos_frame; ese proxy queda
    OBSOLETO y se elimina.

    Nota honesta (R8): en un modelo de VOLATILIDAD los mu_k apenas difieren entre
    regimenes, asi que r_pred_mean puede tener signo casi constante y el test
    degenerar a NaN. Se reporta tal cual -- no se vuelve al proxy para fabricar
    un numero. Un NaN aqui es el resultado correcto, no un fallo."""
    oos = wf.oos_frame
    if "r_pred_mean" not in oos.columns:
        raise KeyError(
            "oos_frame no tiene 'r_pred_mean' (walkforward.py:262). El artefacto "
            "es anterior a V4; re-corre el walk-forward para persistir la columna."
        )
    r_real = oos["r"].to_numpy()
    r_pred = oos["r_pred_mean"].to_numpy()
    return pesaran_timmermann(r_real, r_pred)


# --------------------------------------------------------------------------- #
# Reportes (R8): se escriben SEA CUAL SEA el resultado
# --------------------------------------------------------------------------- #
def _fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _interpret_betas(titular: dict) -> list[str]:
    """Genera el parrafo de interpretacion de los signos de los betas a partir de
    los numeros REALES estimados. Clave estructural: la columna de referencia del
    logit es el ULTIMO regimen (mayor varianza = 'risk-off', beta_{i,K}=0). Por
    tanto un beta>0 sobre la columna de un regimen CALMO significa 'esta covariable
    sube las probabilidades del regimen calmo FRENTE a risk-off' = BAJA la prob de
    risk-off. Los priors economicos se contrastan con el signo estimado; si no
    cuadran, se dice (no se maquilla)."""
    if "beta_tvtp" not in titular:
        return ["- (K=1: sin betas de transicion que interpretar.)"]
    beta = np.array(titular["beta_tvtp"])          # (K, K-1, n_cov)
    se = titular.get("beta_tvtp_se")
    se = np.array(se) if se is not None else None
    covs = titular["covariates"]
    labels = titular["labels"]
    K = beta.shape[0]
    risk_off = labels[-1]
    # prior economico por covariable: signo esperado sobre la PROB DE RISK-OFF.
    prior_riskoff = {
        "sma_gap": "-",       # tendencia alcista (SMA20>>SMA200) => MENOS risk-off
        "bb_width_z": "+",    # bandas anchas (vol reciente alta) => MAS risk-off
        "hy_oas_z": "+",      # spreads HY altos (estres crediticio) => MAS risk-off
    }
    lines = [
        f"Referencia del logit: el regimen de mayor varianza ('{risk_off}') tiene beta=0 por "
        f"identificacion. Un beta POSITIVO sobre la columna de un regimen calmo sube las "
        f"probabilidades de ESE regimen frente a risk-off, es decir BAJA la probabilidad de "
        f"transitar a risk-off (y viceversa).\n",
    ]
    for i in range(K):        # estado de origen
        for j in range(K - 1):  # columna de destino (regimen no-referencia)
            for c, cov in enumerate(covs):
                b = float(beta[i, j, c])
                s = float(se[i, j, c]) if se is not None else None
                z = (b / s) if (s and s > 0) else None
                sig = ("significativo" if (z is not None and abs(z) >= 1.96)
                       else "no significativo" if z is not None else "SE no fiable")
                # efecto sobre prob de risk-off = signo OPUESTO al de b (destino calmo)
                dir_riskoff = "baja" if b > 0 else "sube"
                prior = prior_riskoff.get(cov, "?")
                # coherencia: prior sobre risk-off vs efecto estimado sobre risk-off
                est_sign_riskoff = "-" if b > 0 else "+"
                coh = ("coherente con el prior" if prior == est_sign_riskoff
                       else f"CONTRARIO al prior ({prior})" if prior in "+-" else "sin prior")
                lines.append(
                    f"- desde '{labels[i]}', hacia '{labels[j]}', covariable `{cov}`: "
                    f"beta={_fmt(b,3)}" + (f" (SE={_fmt(s,3)}, z={_fmt(z,2)}, {sig})" if z is not None else f" ({sig})")
                    + f". Efecto sobre prob. de risk-off: la {dir_riskoff}; {coh}."
                )
    return lines


def _write_kselect_report(out: dict) -> None:
    L: list[str] = []
    prov = " -- CORRIDA PROVISIONAL (--quick, NO cumple R6)" if out.get("quick_provisional") else ""
    L.append(f"# Validacion V1 - IRF-N ({ASSET_NAME}){prov}\n")
    L.append(f"generado: {out['generated_at']}  |  asof: {out['asof']}  |  n_obs: {out['n_obs']} "
             f"(desde {out['sample_start']})\n")
    L.append("Alcance V1: TVTP + seleccion de K + innovaciones t de Student. SIN NOTICIAS "
             "(surprise/Hawkes son V2/V3).\n")
    L.append(f"> **Bloqueante macro documentado (R4):** {out['macro_blocker']}\n")
    L.append("Titular tecnico-solo: covariables del TVTP = "
             f"`{out['covariates_titular']}`.\n")

    # --- Tabla BIC ---
    L.append("## 1. Seleccion de K: tabla BIC completa (no se esconden los perdedores)\n")
    L.append("Multistart R6 = %d arranques por celda. BIC in-sample; la separacion "
             "fuera de muestra se dirime en la ablacion walk-forward (fase 2).\n"
             % out["config"]["bic_n_starts"])
    L.append("| K | dist | loglik | k_params | BIC | AIC | conv/arranques | nu |")
    L.append("| --: | :-- | --: | --: | --: | --: | :-- | :-- |")
    win = out["winner"]
    for row in sorted(out["bic_table"], key=lambda r: (r["dist"], r["K"])):
        star = " **<-**" if (row["K"] == win["K"] and row["dist"] == win["dist"]) else ""
        nu = ", ".join(_fmt(x, 1) for x in row["nu"]) if row.get("nu") else "-"
        L.append(f"| {row['K']} | {row['dist']} | {_fmt(row['loglik'],1)} | {row['k_free']} | "
                 f"{_fmt(row['bic'],1)}{star} | {_fmt(row['aic'],1)} | "
                 f"{row['n_converged']}/{row['n_starts']} | {nu} |")
    L.append("")
    L.append(f"**Ganador por BIC: K={win['K']}, dist={win['dist']}** (BIC menor = mejor).\n")

    # --- Bootstrap ---
    L.append("## 2. Test de numero de regimenes: bootstrap parametrico K vs K-1\n")
    L.append("Por que NO el LR con chi2: bajo H0 (K-1) los parametros del regimen extra no "
             "estan identificados (Davies; parametros de molestia solo bajo la alternativa), "
             "asi que 2*(llK - llK-1) NO es chi2. Hansen (1992) exacto es prohibitivo por "
             "costo (malla multidimensional x optimizacion por punto). Se usa bootstrap "
             "parametrico: se simula la distribucion nula del LR desde el modelo K-1 ajustado. "
             "Decision y costo documentados en validation/tests_stat.py.\n")
    L.append("| K vs K-1 | LR_obs | p-value (boot) | replicas ok/tot | LR boot q50 | q95 |")
    L.append("| :-- | --: | --: | :-- | --: | --: |")
    for b in out.get("bootstrap_ladder", []):
        L.append(f"| {b['K_alt']} vs {b['K_null']} | {_fmt(b['lr_obs'],2)} | {_fmt(b['p_value'],3)} | "
                 f"{b['n_boot_ok']}/{b['n_boot']} | {_fmt(b['lr_boot_q50'],2)} | {_fmt(b['lr_boot_q95'],2)} |")
    L.append("")
    L.append("> p pequeno => hay evidencia de que K regimenes mejora sobre K-1. Se reporta la "
             "escalera completa; ningun K se esconde.\n")

    # --- Apuesta del director ---
    L.append("## 3. Contraste con la apuesta a priori del director\n")
    L.append("> Apuesta: K=3 le gana a K=4 fuera de muestra porque el 4o estado casi nunca "
             "tiene suficientes observaciones para estimar su GARCH. **El resultado manda.**\n")
    bic_by = {(r["K"], r["dist"]): r["bic"] for r in out["bic_table"]}
    dw = win["dist"]
    b3 = bic_by.get((3, dw)); b4 = bic_by.get((4, dw))
    if b3 is not None and b4 is not None:
        verdict = "K=3 le gana a K=4 en BIC (coherente con la apuesta)" if b3 < b4 else \
                  "K=4 le gana a K=3 en BIC (CONTRA la apuesta; el resultado manda)"
        L.append(f"- BIC(K=3,{dw})={_fmt(b3,1)} vs BIC(K=4,{dw})={_fmt(b4,1)}: {verdict}.")
    L.append("- La prueba definitiva es fuera de muestra (ablacion walk-forward, fase 2), no el "
             "BIC in-sample. Ver seccion de DM cuando la fase 2 corra.\n")

    # --- Titular + betas ---
    L.append("## 4. Modelo titular V1 y betas del logit de transicion (R7: por MLE con SE)\n")
    tit = out.get("titular", {})
    if "beta_tvtp" in tit:
        L.append(f"Titular: K={tit['K']}, dist={tit['dist']}, covariables `{tit['covariates']}`, "
                 f"lambda L1 (elegido por CV DENTRO de la muestra, jamas mirando test) = "
                 f"{_fmt(tit['l1_lambda'],3)}. loglik={_fmt(tit['loglik'],1)}, "
                 f"convergencia {tit['n_converged']}/{tit['n_starts']}, "
                 f"hessiano_ok={tit['hessian_ok']}.\n")
        if tit.get("nu"):
            L.append(f"Grados de libertad t por regimen (nu): {', '.join(_fmt(x,1) for x in tit['nu'])}. "
                     f"nu bajo => colas gordas en ese regimen.\n")
        L.append("### Interpretacion de los signos (generada de los numeros reales)\n")
        L += _interpret_betas(tit)
        L.append("")
        L.append("Tabla del CV de lambda (no se esconden los lambdas perdedores):\n")
        L.append("| lambda | val loglik/obs | train loglik/obs | conv |")
        L.append("| --: | --: | --: | :-- |")
        for row in tit.get("l1_cv_table", []):
            L.append(f"| {_fmt(row['l1_lambda'],2)} | {_fmt(row['val_loglik_per_obs'],4)} | "
                     f"{_fmt(row['train_loglik_per_obs'],4)} | {row['n_converged']} |")
        L.append("")
    else:
        L.append(f"{tit.get('note','(sin titular TVTP)')}\n")

    L.append("## 5. Pendiente de la fase 2 (ablacion walk-forward + Diebold-Mariano vs V0)\n")
    L.append("Correr `python scripts/run_v1.py ablation`. Completara: log-loss OOS de M0/M1/M2, "
             "DM(titular V1 vs V0), Pesaran-Timmermann direccional, y el veredicto fuera de muestra. "
             "Hasta entonces la seleccion de K de arriba es IN-SAMPLE (BIC + bootstrap).\n")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"validation_v1{REPORT_SUFFIX}.md").write_text("\n".join(L), encoding="utf-8")
    log.info("escrito reports/validation_v1%s.md", REPORT_SUFFIX)


def _append_ablation_report(out: dict) -> None:
    L: list[str] = ["\n\n---\n"]
    prov = " (PROVISIONAL --quick)" if out.get("quick_provisional") else ""
    L.append(f"# Fase 2: walk-forward fuera de muestra{prov}\n")
    L.append(f"generado: {out['generated_at']}. K={out['K']}, dist={out['dist']}, "
             f"covariables `{out['covariates']}`. Bloques de test: {out['n_blocks']} "
             f"(>=6, R2/R8), test={out['test_months']} meses, {out['n_starts_block']} arranques/bloque (R6). "
             f"Metrica primaria: log-loss predictiva del RETORNO fuera de muestra (menor es mejor).\n")

    L.append("## Ablacion M0/M1/M2 (un aporte a la vez)\n")
    L.append("| modelo | descripcion | K | dist | covs | bloques | n_oos | log-loss OOS/obs |")
    L.append("| :-- | :-- | --: | :-- | :-- | --: | --: | --: |")
    for row in out["ablation_table"]:
        L.append(f"| {row['model']} | {row['description']} | {row['K']} | {row['dist']} | "
                 f"{row['covariates'] or '-'} | {row['n_blocks']} | {row['n_oos']} | "
                 f"{_fmt(row['oos_logloss_per_obs'],4)} |")
    L.append("")

    L.append("## Diebold-Mariano entre peldanos consecutivos (perdida predictiva OOS)\n")
    L.append("| A vs B | DM stat | p-value | dif. media |")
    L.append("| :-- | --: | --: | --: |")
    for d in out["dm_ladder"]:
        L.append(f"| {d['model_a']} vs {d['model_b']} | {_fmt(d['dm_stat'],3)} | "
                 f"{_fmt(d['p_value'],3)} | {_fmt(d['mean_diff'],5)} |")
    L.append("")
    L.append("> DM<0 => el primer modelo (A) tiene MENOR perdida (mejor). HAC Newey-West + "
             "correccion de muestra pequena Harvey-Leybourne-Newbold.\n")

    dm0 = out["dm_vs_v0"]
    L.append("## Criterio de aceptacion: Diebold-Mariano titular V1 vs V0\n")
    better = "MEJOR que V0" if dm0["dm_stat"] < 0 else "no mejor que V0"
    sig = "significativo (p<0.05)" if dm0["p_value"] < 0.05 else "no significativo (p>=0.05)"
    L.append(f"- **DM(M2 titular V1 vs V0) = {_fmt(dm0['dm_stat'],3)}, p = {_fmt(dm0['p_value'],3)}**: "
             f"el titular es {better}, {sig}.")
    L.append(f"  - V0 = K=2 Normal P-constante, re-corrido sobre la MISMA muestra y malla de "
             f"bloques que V1 (perdidas emparejadas por fecha). dif. media de perdida = "
             f"{_fmt(dm0['mean_diff'],5)}.\n")

    pt = out["pesaran_timmermann_titular"]
    L.append("## Pesaran-Timmermann (precision direccional del titular)\n")
    L.append(f"- PT stat = {_fmt(pt['pt_stat'],3)}, p = {_fmt(pt['p_value'],3)}, "
             f"aciertos = {_fmt(pt['hit_rate'],3)} vs esperado bajo independencia "
             f"{_fmt(pt['expected_hit_rate'],3)}. {pt['note']}\n")
    L.append("> Nota honesta: la senal direccional del retorno usa un proxy debil (dominancia del "
             "regimen calmo); la senal direccional fuerte es tarea de la capa de noticias (V2). "
             "Si el test degenera por signo constante, se reporta NaN, no se maquilla.\n")

    with (REPORTS / f"validation_v1{REPORT_SUFFIX}.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(L))
    log.info("apendice de fase 2 escrito en reports/validation_v1%s.md", REPORT_SUFFIX)


def _write_v1_walkforward_json(abl, wf_v0) -> None:
    """walkforward_v1.json para pantalla 5: metricas por bloque del titular (M2) +
    calibracion sobre xi predicho. Mantiene el formato que la app ya sabe leer."""
    from irfn.validation import calibration
    wf = abl.wf_results["M2"]
    K = wf.K
    oos = wf.oos_frame
    xi_pred = oos[[f"xi_predicted_{k}" for k in range(K)]].to_numpy()
    xi_filt = oos[[f"xi_filtered_{k}" for k in range(K)]].to_numpy()
    calib = calibration.summarize(xi_pred, xi_filt)
    blocks = []
    for b in wf.blocks:
        blocks.append({
            "block_id": b.block_id, "train_start": str(b.train_start.date()),
            "test_start": str(b.test_start.date()), "test_end": str(b.test_end.date()),
            "n_train": b.n_train, "n_test": b.n_test,
            "loglik_train_per_obs": b.loglik_train_per_obs,
            "loglik_test_per_obs": b.loglik_test_per_obs,
            "n_converged": b.n_converged, "kappa": b.kappa.tolist(),
            "P": b.P.tolist(), "l1_lambda": b.l1_lambda,
        })
    obj = {
        "n_blocks": wf.n_blocks, "K": K, "seed": wf.seed, "n_starts": wf.n_starts,
        "dist": wf.dist, "covariates": wf.covariates,
        "block_boundaries": [str(d.date()) for d in wf.block_boundaries],
        "blocks": blocks, "calibration": calib,
    }
    _dump(obj, ARTIFACTS.parent / "analysis" / "walkforward_v1.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Orquestador V1 (TVTP + seleccion de K).")
    sub = ap.add_subparsers(dest="stage", required=True)
    p1 = sub.add_parser("kselect", help="Fase 1: BIC + bootstrap + titular + artefacto de hoy.")
    p1.add_argument("--quick", action="store_true", help="Arranques/replicas reducidos (HUMO, no R6).")
    p1.add_argument("--asset", default=None,
                    help="Nombre del activo en config/assets.yaml (p.ej. BTC). "
                         "Sin este flag, comportamiento identico al historico (SPY).")
    p2 = sub.add_parser("ablation", help="Fase 2: ablacion walk-forward + DM vs V0.")
    p2.add_argument("--quick", action="store_true")
    p2.add_argument("--test-months", type=int, default=None,
                    help="meses de test por bloque (mas grande = menos bloques = mas rapido).")
    p2.add_argument("--asset", default=None,
                    help="Nombre del activo en config/assets.yaml (p.ej. BTC). "
                         "Sin este flag, comportamiento identico al historico (SPY).")
    p2.add_argument("--jobs", type=int, default=1,
                    help="Procesos para paralelizar los bloques del walk-forward (R2: "
                         "bloques independientes). 1 = serie (referencia). -1 = todos los "
                         "nucleos menos uno. Solo velocidad; numeros identicos al serial.")
    args = ap.parse_args()

    _configure_asset(BaseConfig.load(), args.asset)

    if args.stage == "kselect":
        stage_kselect(quick=args.quick)
    elif args.stage == "ablation":
        n_jobs = max(1, (os.cpu_count() or 1) - 1) if args.jobs == -1 else max(1, args.jobs)
        stage_ablation(quick=args.quick, test_months=args.test_months, n_jobs=n_jobs)


if __name__ == "__main__":
    main()

"""Un comando, todo el pipeline de V0: ingesta -> filtro de Hamilton ->
walk-forward -> calibracion -> auditoria PIT -> artefactos + reporte.

    python scripts/run_pipeline.py            # corrida completa V0 (descarga SPY)
    python scripts/run_pipeline.py --dry-run  # imprime el plan, sin red ni disco
    python scripts/run_pipeline.py --quick    # menos arranques, para iterar rapido
    python scripts/run_pipeline.py --asset BTC --quick  # segunda linea sobre otro
                                               # activo de config/assets.yaml;
                                               # NO toca artifacts/latest/ (SPY)

V0 no tiene TVTP ni noticias (eso es V1+): la fase "modelo" es MS-GJR-GARCH K=2
con matriz de transicion constante e innovaciones Normales. El walk-forward
re-estima por bloque (R2) y documenta el resultado sea cual sea (R8).

Escribe (todo lo que la app lee; R9 = la app solo lee esto):
  artifacts/latest/irfn.json        -- contrato de salida (outputs/schema.py)
  artifacts/latest/audit.json       -- invarianza de prefijo, ledger de rezagos, R2
  artifacts/latest/walkforward.json -- metricas por bloque + calibracion + fronteras
  artifacts/latest/history.parquet  -- ξ filtrado out-of-sample para la vista Historico
  artifacts/runs/<run_id>/          -- copia inmutable de la corrida
  reports/validation_v0.md          -- reporte de walk-forward (R8)

Si IRFN_DEV_MODE=true escribe ademas data/interim/diagnostic_smoother.parquet
(el smoother de Kim, SOLO diagnostico). Nunca entra a artifacts/ (R1); publish.py
lo bloquea aunque alguien lo intente.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irfn.audit.pit import (  # noqa: E402
    block_reestimation_check,
    lag_ledger,
    prefix_invariance_check,
)
from irfn.config import AssetsConfig, BaseConfig, NewsConfig  # noqa: E402
from irfn.data.prices import load_returns  # noqa: E402
from irfn.outputs.publish import build_payload, publish  # noqa: E402
from irfn.outputs.serialize import write_history_parquet, write_walkforward_json  # noqa: E402
from irfn.pipeline import regime_labels, run_pipeline  # noqa: E402
from irfn.validation import calibration  # noqa: E402
from irfn.validation.walkforward import walk_forward  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
CONFIG_DIR = ROOT / "config"
INTERIM = ROOT / "data" / "interim"
VERSION = "V0"

PIPELINE_PHASES = [
    ("ingesta", "Descarga precios del activo ancla (SPY via yfinance). V1+: BTC, macro, noticias."),
    ("features", "V0 no tiene covariables (TVTP y noticias son V1+). El unico input es el retorno."),
    ("modelo", "Estima MS-GJR-GARCH K=2, P constante, Normal, por bloque de walk-forward (R2, R6)."),
    ("filtro", "Recursion de Hamilton: xi_predicted, xi_filtered, loglik. Nunca smoother fuera de @diagnostic_only (R1)."),
    ("validacion", "Walk-forward (>=6 bloques), calibracion sobre xi_predicted, auditoria PIT."),
    ("artefactos", "Serializa contra el contrato y publica via publish.py (guardian anti-smoother)."),
]


# --------------------------------------------------------------------------- #
# Utilidades de identidad reproducible
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
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
    raw = f"{config_hash}|{git_commit}|{seed}".encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def _dev_mode() -> bool:
    return os.environ.get("IRFN_DEV_MODE", "").lower() in {"1", "true", "yes"}


# --------------------------------------------------------------------------- #
# Plan (--dry-run)
# --------------------------------------------------------------------------- #
def print_plan() -> None:
    base = BaseConfig.load()
    assets = AssetsConfig.load()
    news = NewsConfig.load()
    print("Plan de ejecucion - IRF-N pipeline (V0)")
    print("=" * 60)
    print(f"K (regimenes)          : {base.model.K}")
    print(f"Semilla                : {base.model.seed}")
    print(f"Multistart (hoy)       : {base.model.n_multistart}")
    print(f"Multistart (WF/bloque) : {base.v0.wf_n_starts}")
    print(f"Bloques walk-forward   : >= {base.walkforward.n_blocks}")
    print(f"Activo ancla V0        : {base.v0.anchor_asset} desde {base.v0.returns_start}")
    print(f"Activos (roadmap)      : {', '.join(a.name for a in assets.assets)}")
    print(f"Indicadores noticia    : {', '.join(i.name for i in news.indicators)} (V1+)")
    print("=" * 60)
    for i, (name, desc) in enumerate(PIPELINE_PHASES, 1):
        print(f"\n[{i}/{len(PIPELINE_PHASES)}] {name}\n    {desc}")


# --------------------------------------------------------------------------- #
# Corrida completa V0
# --------------------------------------------------------------------------- #
def _resolve_asset(base: BaseConfig, asset_override: str | None):
    """Resuelve (asset, source, identifier, returns_start, artifacts_root, report_name).

    Sin --asset: comportamiento identico al historico (SPY via yfinance,
    artifacts/latest/, reports/validation_v0.md) -- cero cambios de ruta.
    Con --asset NAME: busca NAME en config/assets.yaml (source/symbol/ticker/
    start_date reales) y escribe en artifacts/<name>/ + reports/..._<name>.md,
    aditivo, sin tocar nada de la linea SPY.
    """
    if asset_override is None:
        return (base.v0.anchor_asset, "yfinance", base.v0.anchor_asset,
                base.v0.returns_start, ARTIFACTS, "validation_v0.md")
    asset_cfg = AssetsConfig.load().get(asset_override)
    slug = asset_cfg.name.lower()
    return (asset_cfg.name, asset_cfg.source, asset_cfg.identifier,
            asset_cfg.start_date, ARTIFACTS / slug, f"validation_v0_{slug}.md")


def full_run(quick: bool = False, asset_override: str | None = None) -> None:
    base = BaseConfig.load()
    K = base.model.K
    seed = base.model.seed
    n_starts_today = 12 if quick else base.model.n_multistart
    n_starts_wf = 8 if quick else base.v0.wf_n_starts
    asset, source, identifier, returns_start, artifacts_root, report_name = _resolve_asset(
        base, asset_override
    )
    labels = regime_labels(K)

    config_hash = _config_hash()
    git_commit = _git_commit()
    run_id = _run_id(config_hash, git_commit, seed)
    dev = _dev_mode()

    print(f"[1/6] Ingesta: descargando {asset} ({identifier} via {source}) desde {returns_start}...")
    prices = load_returns(identifier, returns_start, source=source)
    returns = prices.dropna()
    print(f"      {len(returns)} retornos, {returns.index[0].date()} a {returns.index[-1].date()}")

    # --- Auditoria PIT: la joya de la corona. Sobre una submuestra reciente. ---
    print("[2/6] Auditoria PIT: invarianza de prefijo...")
    pit_df = prefix_invariance_check(
        returns, K=K, seed=seed, n_starts=6,
        train_len=len(returns) - 260, dates_to_test=10,
    )
    print(f"      invarianza de prefijo: {'VERDE' if pit_df.attrs['passed'] else 'ROJA'} "
          f"(max_abs_diff={pit_df['max_abs_diff'].max():.2e})")
    ledger_df = lag_ledger()

    # --- Walk-forward: re-estimacion por bloque (R2), >= 6 bloques (R8). ---
    print(f"[3/6] Walk-forward: re-estimando por bloque ({n_starts_wf} arranques c/u)...")
    wf = walk_forward(
        returns, K=K, seed=seed, n_starts=n_starts_wf,
        train_years=base.walkforward.train_years,
        test_months=base.walkforward.test_months,
        n_blocks_min=base.walkforward.n_blocks,
        checkpoint_path=artifacts_root / "checkpoints" / "wf_v0.pkl",
    )
    print(f"      {wf.n_blocks} bloques completados.")
    reest_df = block_reestimation_check(wf.blocks)
    print(f"      re-estimacion por bloque distinta (R2): "
          f"{'VERDE' if reest_df.attrs['passed'] else 'ROJA'}")

    # --- Calibracion sobre el ξ PREDICHO out-of-sample. ---
    print("[4/6] Calibracion sobre xi_predicted out-of-sample...")
    oos = wf.oos_frame
    xi_pred_oos = oos[[f"xi_predicted_{k}" for k in range(K)]].to_numpy()
    xi_filt_oos = oos[[f"xi_filtered_{k}" for k in range(K)]].to_numpy()
    calib = calibration.summarize(xi_pred_oos, xi_filt_oos)
    print(f"      log-loss modelo={calib['log_loss']:.4f} vs climatologia={calib['log_loss_baseline']:.4f}")

    # --- Estimacion "de hoy": todo el historico hasta asof, para el ξ publicado. ---
    print("[5/6] Estimacion de hoy y publicacion de artefactos...")
    today = run_pipeline(returns, K=K, seed=seed, n_starts=n_starts_today, compute_se=True)
    xi_hist = today.frame[[f"xi_filtered_{k}" for k in range(K)]].to_numpy()

    payload = build_payload(
        asof=returns.index[-1].date(),
        version=VERSION,
        run_id=run_id,
        git_commit=git_commit,
        config_hash=config_hash,
        K=K,
        labels=labels,
        xi_history=xi_hist,
        P=today.fit.P,
        entropy_mid=base.regime.entropy_mid,
        entropy_high=base.regime.entropy_threshold,
        seed=seed,
        n_multistart=n_starts_today,
        converged=(today.fit.n_converged >= max(1, n_starts_today // 3)),
        r_pct=today.frame["r"].to_numpy(),
        argmax_idx=today.frame["argmax_idx"].to_numpy(),
        asset=asset,
        validation_ref=f"reports/{report_name}",
        warnings=_build_warnings(today, wf, pit_df, reest_df),
        bootstrap_n_boot=base.v2.bootstrap.n_boot,
        bootstrap_block_len=base.v2.bootstrap.block_len,
        bootstrap_ci_level=base.v2.bootstrap.ci_level,
        bootstrap_min_obs=base.v2.bootstrap.min_obs,
    )

    # Fase 3 (publicacion atomica): la corrida escribe SOLO en runs/<run_id>/.
    # A latest/ solo se llega con scripts/promote.py (unico punto de promocion,
    # con contrato + guardarrail). Aqui no se toca latest/.
    run_dir = artifacts_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # publish() corre el guardian anti-smoother (R1) antes de escribir.
    publish(payload, run_dir / "irfn.json")

    _write_audit_json(run_dir / "audit.json", pit_df, ledger_df, reest_df,
                      _sources_freshness(asset, source, returns), run_id,
                      entropy_zones={"mid": base.regime.entropy_mid,
                                     "high": base.regime.entropy_threshold})
    write_walkforward_json(run_dir / "walkforward.json", wf, calib)
    write_history_parquet(run_dir / "history.parquet", oos)

    # --- Reporte de validacion (R8). ---
    print(f"[6/6] Escribiendo reports/{report_name}...")
    _write_validation_report(wf, calib, pit_df, reest_df, run_id, asset, report_name)

    # Diagnostico DEV: smoother de Kim, jamas en artifacts/ (R1).
    if dev:
        _write_diagnostic_smoother(today, K)
        print("      IRFN_DEV_MODE: smoother de diagnostico en data/interim/ (fuera de artifacts).")

    print(f"\nListo. run_id={run_id}  PIT={'VERDE' if pit_df.attrs['passed'] else 'ROJA'}")
    print(f"Artefacto en {run_dir} (NO publicado). Para promover a latest/:")
    print(f"  python scripts/promote.py runs/{run_id}   # V0 requiere --force-downgrade")


def _build_warnings(today, wf, pit_df, reest_df) -> list[str]:
    w: list[str] = []
    if not pit_df.attrs["passed"]:
        w.append("AUDITORIA PIT EN ROJO: invarianza de prefijo rota. No confiar en el artefacto.")
    if not reest_df.attrs["passed"]:
        w.append("Re-estimacion por bloque en rojo: bloques con parametros iguales (posible R2).")
    kappa = today.fit.params["alpha"] + today.fit.params["gamma"] / 2 + today.fit.params["beta"]
    if np.any(kappa > 0.999):
        reg = int(np.argmax(kappa))
        w.append(f"Regimen {reg} casi integrado (kappa={kappa[reg]:.4f}): estado absorbe-outliers, "
                 f"limitacion conocida de V0 (K=2/Normal), no un bug.")
    if today.fit.hessian_ok is False:
        w.append("Hessiano degenerado en la estimacion de hoy: algunas SE no son fiables.")
    return w


def _sources_freshness(asset: str, source: str, returns: pd.Series) -> list[dict]:
    return [{
        "name": asset,
        "source": source,
        "last_date": str(returns.index[-1].date()),
        "n_obs": int(len(returns)),
    }]


# --------------------------------------------------------------------------- #
# Serializacion de artefactos auxiliares (todo lo que lee la app)
# --------------------------------------------------------------------------- #
def _write_audit_json(path, pit_df, ledger_df, reest_df, sources, run_id, entropy_zones) -> None:
    obj = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entropy_zones": entropy_zones,
        "prefix_invariance": {
            "passed": bool(pit_df.attrs["passed"]),
            "atol": pit_df.attrs["atol"],
            "rows": json.loads(pit_df.assign(fecha=pit_df["fecha"].astype(str)).to_json(orient="records")),
        },
        "lag_ledger": {
            "passed": bool(ledger_df.attrs["passed"]),
            "rows": json.loads(ledger_df.to_json(orient="records")),
        },
        "block_reestimation": {
            "passed": bool(reest_df.attrs["passed"]),
            "rows": json.loads(
                reest_df.assign(
                    train_start=reest_df["train_start"].astype(str),
                    test_start=reest_df["test_start"].astype(str),
                ).to_json(orient="records")
            ),
        },
        "sources": sources,
    }
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _write_diagnostic_smoother(today, K) -> None:
    """SOLO DIAGNOSTICO (R1). Escribe el smoother de Kim FUERA de artifacts/, en
    data/interim/, para que la vista Historico pueda superponerlo bajo DEV_MODE.
    Nunca se publica: publish.py lo rechazaria."""
    from irfn.models.hamilton import kim_smoother

    frame = today.frame
    xi_f = frame[[f"xi_filtered_{k}" for k in range(K)]].to_numpy()
    xi_p = frame[[f"xi_predicted_{k}" for k in range(K)]].to_numpy()
    xi_s = kim_smoother(xi_f, xi_p, today.fit.P)
    out = pd.DataFrame(
        {f"xi_smoothed_{k}": xi_s[:, k] for k in range(K)}, index=frame.index
    )
    out.index.name = "fecha"
    INTERIM.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_parquet(INTERIM / "diagnostic_smoother.parquet", index=False)


# --------------------------------------------------------------------------- #
# Reporte de validacion (R8): se escribe SEA CUAL SEA el resultado
# --------------------------------------------------------------------------- #
def _write_validation_report(wf, calib, pit_df, reest_df, run_id, asset, report_name) -> None:
    ll_model = calib["log_loss"]
    ll_base = calib["log_loss_baseline"]
    br_model = calib["brier"]
    br_base = calib["brier_baseline"]
    beats_ll = ll_model < ll_base
    beats_br = br_model < br_base

    lines = []
    lines.append(f"# Validacion V0 - IRF-N ({asset})\n")
    lines.append(f"run_id: `{run_id}`  |  generado: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}\n")
    lines.append("Especificacion: MS-GJR-GARCH K=2, matriz de transicion CONSTANTE, "
                 "innovaciones Normales. Sin TVTP, sin noticias (V1+).\n")

    lines.append("## Veredicto honesto\n")
    verdict = "SI" if (beats_ll and beats_br) else ("PARCIAL" if (beats_ll or beats_br) else "NO")
    lines.append(f"- El ξ PREDICHO out-of-sample, ¿le gana a la climatologia? **{verdict}**.")
    lines.append(f"  - Log-loss: modelo={ll_model:.4f} vs baseline={ll_base:.4f} "
                 f"({'gana' if beats_ll else 'NO gana'}).")
    lines.append(f"  - Brier:    modelo={br_model:.4f} vs baseline={br_base:.4f} "
                 f"({'gana' if beats_br else 'NO gana'}).")
    lines.append(f"  - ECE (error de calibracion esperado): {calib['ece']:.4f}.")
    lines.append(f"  - Observaciones out-of-sample evaluadas: {calib['n_obs']}.\n")
    lines.append("> Baseline = predecir siempre las frecuencias marginales de regimen "
                 "(climatologia). Es el minimo que un predictor condicional debe batir. "
                 f"El objetivo realizado es un proxy (argmax ξ_{{t|t}}); {calib['target_note']}\n")

    lines.append("## Auditoria point-in-time\n")
    lines.append(f"- Invarianza de prefijo: **{'VERDE' if pit_df.attrs['passed'] else 'ROJA'}** "
                 f"(max_abs_diff = {pit_df['max_abs_diff'].max():.2e}, atol = {pit_df.attrs['atol']:.0e}).")
    lines.append(f"- Re-estimacion por bloque distinta (R2): "
                 f"**{'VERDE' if reest_df.attrs['passed'] else 'ROJA'}**.\n")

    lines.append("## Bloques de walk-forward\n")
    lines.append(f"Total: **{wf.n_blocks}** bloques (minimo exigido: 6; R2, R8). "
                 f"Cada bloque re-estima desde cero sobre su ventana de entrenamiento y "
                 f"arrastra el ultimo ξ_{{t|t}} del entrenamiento al test (no reinicia).\n")
    lines.append("| bloque | train desde | test | n_test | loglik/obs test | kappa (persistencia) |")
    lines.append("| --: | :-- | :-- | --: | --: | :-- |")
    for b in wf.blocks:
        kappa_str = ", ".join(f"{k:.3f}" for k in b.kappa)
        lines.append(
            f"| {b.block_id} | {b.train_start.date()} | "
            f"{b.test_start.date()}..{b.test_end.date()} | {b.n_test} | "
            f"{b.loglik_test_per_obs:.4f} | {kappa_str} |"
        )
    lines.append("")

    mean_ll_test = float(np.mean([b.loglik_test_per_obs for b in wf.blocks]))
    mean_ll_train = float(np.mean([b.loglik_train_per_obs for b in wf.blocks]))
    lines.append(f"- Loglik media por observacion: train={mean_ll_train:.4f}, "
                 f"test={mean_ll_test:.4f}. La brecha train-test mide sobreajuste.\n")

    lines.append("## Limitaciones conocidas de V0\n")
    lines.append("- K=2 con P constante y Normal: un regimen puede colapsar a un estado casi "
                 "integrado (kappa->1) que absorbe outliers en vez de ser un regimen persistente "
                 "de alta vol. No es un bug (recovery lo prueba sobre datos limpios); lo ataca V1 "
                 "con t de Student, TVTP y seleccion de K.")
    lines.append("- Sin covariables ni capa de noticias: la matriz de transicion no reacciona a "
                 "condiciones de mercado. Es V0 por diseno.\n")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / report_name).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline completo de IRF-N (V0).")
    parser.add_argument("--dry-run", action="store_true", help="Imprime el plan sin ejecutar nada.")
    parser.add_argument("--quick", action="store_true", help="Menos arranques (iteracion rapida).")
    parser.add_argument("--asset", default=None,
                        help="Nombre del activo en config/assets.yaml (p.ej. BTC). "
                             "Sin este flag, comportamiento identico al historico (SPY).")
    args = parser.parse_args()

    if args.dry_run:
        print_plan()
        return
    full_run(quick=args.quick, asset_override=args.asset)


if __name__ == "__main__":
    main()

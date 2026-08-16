"""Tests del contrato de publicacion (outputs/contract.py).

Caso negativo real: la carpeta de cuarentena con el artefacto V0 roto de la
auditoria 2026-08-15 debe fallar >= 5 reglas. Caso positivo: un artefacto V3
sintetico y coherente debe ser PROMOVIBLE. Ademas se prueba cada regla en
aislamiento partiendo del artefacto sano y rompiendolo de una en una.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from irfn.outputs.contract import (
    ContractViolation,
    assert_promotable,
    validate_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
QUARANTINE = ROOT / "artifacts" / "quarantine" / "2026-08-15_v0_regression"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write_conforming_v3(d: Path, *, run_id: str = "abc123def456", asof: str = "2026-08-14") -> None:
    """Escribe un artefacto V3 coherente y completo en d (sin manifest todavia)."""
    d.mkdir(parents=True, exist_ok=True)

    # history.parquet con la ultima fecha == asof (sin hueco de frescura)
    dates = pd.date_range(end=asof, periods=10, freq="D")
    hist = pd.DataFrame({"fecha": dates, "xi_filtered_0": 0.6, "xi_filtered_1": 0.4})
    hist.to_parquet(d / "history.parquet", index=False)

    irfn = {
        "run_id": run_id,
        "git_commit": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        "asof": asof,
        "version": "V3",
        "model": {
            "K": 2,
            "tvtp": True,
            "covariates": ["sma_gap", "lambda_N_z"],
            "news_layer": [],
            "estimation": {"multistart": 30, "seed": 42, "converged": True},
            "news_layer_params": {"active": False},
            "hawkes_layer_params": {"active": True},
        },
    }
    (d / "irfn.json").write_text(json.dumps(irfn), encoding="utf-8")
    (d / "audit.json").write_text(json.dumps({"run_id": run_id, "prefix_invariance": {"passed": True}}), encoding="utf-8")
    (d / "walkforward.json").write_text(json.dumps({"n_blocks": 19}), encoding="utf-8")
    (d / "surprise_events.json").write_text(json.dumps({"events": []}), encoding="utf-8")
    hist.head(3).to_parquet(d / "hawkes_history.parquet", index=False)
    hist.head(3).to_parquet(d / "headline_rug.parquet", index=False)


def _write_manifest(d: Path, run_id: str) -> None:
    files = {}
    for p in sorted(d.iterdir()):
        if p.is_file() and p.name != "manifest.json":
            files[p.name] = {"sha256": _sha256(p), "size": p.stat().st_size}
    (d / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "version": "V3", "files": files}), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Caso negativo real: la cuarentena
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not QUARANTINE.is_dir(), reason="carpeta de cuarentena no presente")
def test_cuarentena_falla_al_menos_5_reglas():
    res = validate_artifact(QUARANTINE)
    assert not res.promotable
    assert not res.ok
    assert len(res.violations) >= 5, res.violations
    # toca R1,R2,R5,R6,R7. NO R3: su problema de fechas era la mezcla vieja-historia/
    # nuevo-asof, que ahora atrapa R1 (procedencia). asof-vs-generated_at es fresco
    # (1 dia) y history no adelanta a asof, asi que R3 no dispara -- correcto.
    blob = " ".join(res.violations)
    for marca in ("R1", "R2", "R5", "R6", "R7"):
        assert marca in blob, f"esperaba {marca} en las violaciones: {blob}"


# --------------------------------------------------------------------------- #
# Caso positivo: artefacto V3 sano
# --------------------------------------------------------------------------- #

def test_artefacto_v3_conforme_es_promovible(tmp_path):
    d = tmp_path / "run_ok"
    _write_conforming_v3(d)
    _write_manifest(d, "abc123def456")
    res = validate_artifact(d)
    assert res.promotable, res.violations
    assert res.violations == []
    # assert_promotable no lanza
    assert_promotable(d).run_id == "abc123def456"


# --------------------------------------------------------------------------- #
# Reglas en aislamiento (romper el artefacto sano de una en una)
# --------------------------------------------------------------------------- #

def test_r1_archivo_ajeno(tmp_path):
    d = tmp_path / "run"
    _write_conforming_v3(d)
    (d / "walkforward_v1.json").write_text("{}", encoding="utf-8")  # ajeno + variante WF
    _write_manifest(d, "abc123def456")
    res = validate_artifact(d)
    assert not res.promotable
    assert any("R1" in v and "walkforward_v1.json" in v for v in res.violations)
    assert any("R7" in v for v in res.violations)


def test_r2_v0_no_puede_traer_hawkes(tmp_path):
    d = tmp_path / "run"
    _write_conforming_v3(d)
    # degradar a V0 dejando los parquets Hawkes -> R2 debe protestar
    irfn = json.loads((d / "irfn.json").read_text(encoding="utf-8"))
    irfn["version"] = "V0"
    irfn["model"]["tvtp"] = False
    irfn["model"]["hawkes_layer_params"]["active"] = False
    (d / "irfn.json").write_text(json.dumps(irfn), encoding="utf-8")
    _write_manifest(d, "abc123def456")
    res = validate_artifact(d)
    assert any("R2" in v and "Hawkes" in v for v in res.violations)


def test_r3_cola_walkforward_es_legitima(tmp_path):
    # history que termina ANTES de asof (cola del walk-forward) NO es violacion.
    d = tmp_path / "run"
    _write_conforming_v3(d, asof="2026-08-14")
    dates = pd.date_range(end="2026-07-05", periods=10, freq="D")  # ~40 dias antes
    pd.DataFrame({"fecha": dates}).to_parquet(d / "history.parquet", index=False)
    _write_manifest(d, "abc123def456")
    res = validate_artifact(d)
    assert not any("R3" in v for v in res.violations), res.violations


def test_r3_lookahead_history_despues_de_asof(tmp_path):
    d = tmp_path / "run"
    _write_conforming_v3(d, asof="2026-08-14")
    dates = pd.date_range(end="2026-09-30", periods=10, freq="D")  # DESPUES de asof
    pd.DataFrame({"fecha": dates}).to_parquet(d / "history.parquet", index=False)
    _write_manifest(d, "abc123def456")
    res = validate_artifact(d)
    assert any("R3 look-ahead" in v for v in res.violations)


def test_r3_frescura_asof_rancio(tmp_path):
    d = tmp_path / "run"
    _write_conforming_v3(d, asof="2026-01-01")  # history termina en asof
    irfn = json.loads((d / "irfn.json").read_text(encoding="utf-8"))
    irfn["generated_at"] = "2026-08-15T00:00:00+00:00"  # generado 7 meses despues
    (d / "irfn.json").write_text(json.dumps(irfn), encoding="utf-8")
    _write_manifest(d, "abc123def456")
    res = validate_artifact(d)
    assert any("R3 frescura" in v for v in res.violations)


def test_r4_r6_multistart_bajo_es_duro_pero_provisional_con_flag(tmp_path):
    d = tmp_path / "run"
    _write_conforming_v3(d)
    irfn = json.loads((d / "irfn.json").read_text(encoding="utf-8"))
    irfn["model"]["estimation"]["multistart"] = 12
    (d / "irfn.json").write_text(json.dumps(irfn), encoding="utf-8")
    _write_manifest(d, "abc123def456")

    # sin flag: falla dura -> rechazado
    strict = validate_artifact(d)
    assert not strict.ok
    assert any("multistart" in v for v in strict.violations)

    # con flag: valido pero NO promovible (provisional)
    prov = validate_artifact(d, allow_provisional=True)
    assert prov.ok            # sin fallas duras
    assert not prov.promotable
    assert any("multistart" in p for p in prov.provisional_reasons)


def test_r5_git_commit_nogit(tmp_path):
    d = tmp_path / "run"
    _write_conforming_v3(d)
    irfn = json.loads((d / "irfn.json").read_text(encoding="utf-8"))
    irfn["git_commit"] = "nogit"
    (d / "irfn.json").write_text(json.dumps(irfn), encoding="utf-8")
    _write_manifest(d, "abc123def456")
    res = validate_artifact(d)
    assert any("R5" in v for v in res.violations)


def test_r6_v1_sin_tvtp(tmp_path):
    # Con covariables de transicion DECLARADAS (sma_gap, lambda_N_z) pero tvtp=false,
    # sigue siendo incoherente (A4 solo desvincula tvtp cuando NO hay covariables).
    d = tmp_path / "run"
    _write_conforming_v3(d)
    irfn = json.loads((d / "irfn.json").read_text(encoding="utf-8"))
    irfn["model"]["tvtp"] = False  # covariables presentes + tvtp=false -> incoherente
    (d / "irfn.json").write_text(json.dumps(irfn), encoding="utf-8")
    _write_manifest(d, "abc123def456")
    res = validate_artifact(d)
    assert any("tvtp" in v for v in res.violations)


def test_a4_m1_sin_covariables_tvtp_false_es_valido(tmp_path):
    # A4 (decision del director 2026-08-16): M1 -- K=2, regimenes solos, SIN
    # covariables de transicion -- es un modelo de produccion valido; el contrato ya
    # NO exige tvtp por el solo hecho de ser V1+. covariates=[] + tvtp=false => OK.
    d = tmp_path / "run"
    _write_conforming_v3(d)
    irfn = json.loads((d / "irfn.json").read_text(encoding="utf-8"))
    irfn["model"]["tvtp"] = False
    irfn["model"]["covariates"] = []   # M1: sin covariables de transicion
    (d / "irfn.json").write_text(json.dumps(irfn), encoding="utf-8")
    _write_manifest(d, "abc123def456")
    res = validate_artifact(d)
    assert res.promotable, res.violations
    assert not any("tvtp" in v for v in res.violations)


def test_v3_hawkes_inactivo_no_exige_parquets(tmp_path):
    # Un V3 legitimo con Hawkes inactivo (corpus insuficiente) y sin parquets de
    # Hawkes debe seguir siendo promovible (obligatorios solo si active).
    import os
    d = tmp_path / "run"
    _write_conforming_v3(d)
    os.remove(d / "hawkes_history.parquet")
    os.remove(d / "headline_rug.parquet")
    irfn = json.loads((d / "irfn.json").read_text(encoding="utf-8"))
    irfn["model"]["hawkes_layer_params"]["active"] = False
    irfn["model"]["covariates"] = ["sma_gap"]  # titular tecnico, sin lambda_N_z
    (d / "irfn.json").write_text(json.dumps(irfn), encoding="utf-8")
    _write_manifest(d, "abc123def456")
    res = validate_artifact(d)
    assert res.promotable, res.violations


def test_v3_k1_tvtp_false_y_hawkes_standalone_es_valido(tmp_path):
    # K=1 (linea BTC): sin logit de transicion, tvtp=false es correcto y el Hawkes
    # activo se publica como indicador standalone (covariates vacio). Debe promover.
    d = tmp_path / "run"
    _write_conforming_v3(d)  # base K=2 con Hawkes activo + parquets
    irfn = json.loads((d / "irfn.json").read_text(encoding="utf-8"))
    irfn["model"]["K"] = 1
    irfn["model"]["tvtp"] = False
    irfn["model"]["covariates"] = []
    irfn["model"]["news_layer"] = []
    (d / "irfn.json").write_text(json.dumps(irfn), encoding="utf-8")
    _write_manifest(d, "abc123def456")
    res = validate_artifact(d)
    assert res.promotable, res.violations


def test_manifest_run_id_incoherente(tmp_path):
    d = tmp_path / "run"
    _write_conforming_v3(d)
    _write_manifest(d, "OTRO_run_id")  # no coincide con irfn.run_id
    res = validate_artifact(d)
    assert any("R1" in v and "manifest" in v.lower() for v in res.violations)

"""Tests de los lectores de la app (app/components) tras la Fase 5.

Clave: load_walkforward YA NO cae a walkforward_v1.json (otro modelo, baseline
favorecedor). Si falta el walkforward.json vigente devuelve None (vacio honesto).
Y artifact_coherent detecta mezclas de procedencia.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import components  # noqa: E402


def _point_artifacts_to(tmp: Path):
    """Redirige components.ARTIFACTS a un dir temporal (los loaders leen de ahi)."""
    components.ARTIFACTS = tmp
    tmp.mkdir(parents=True, exist_ok=True)


def test_load_walkforward_sin_fallback_a_v1(tmp_path, monkeypatch):
    monkeypatch.setattr(components, "ARTIFACTS", tmp_path)
    # existe SOLO el favorecedor v1: NO debe leerse
    (tmp_path / "walkforward_v1.json").write_text(json.dumps({"calibration": {"log_loss_baseline": 0.55}}), encoding="utf-8")
    assert components.load_walkforward() is None  # vacio honesto, no cae al V1

    # ahora existe el vigente: se lee ese
    (tmp_path / "walkforward.json").write_text(json.dumps({"n_starts": 20}), encoding="utf-8")
    wf = components.load_walkforward()
    assert wf is not None and wf["n_starts"] == 20


def test_artifact_coherent(tmp_path, monkeypatch):
    monkeypatch.setattr(components, "ARTIFACTS", tmp_path)
    irfn = {"run_id": "aaa"}
    audit = {"run_id": "aaa"}
    manifest = {"run_id": "aaa"}
    assert components.artifact_coherent(irfn, audit, manifest)
    # una mezcla (audit de otra corrida) se detecta
    assert not components.artifact_coherent(irfn, {"run_id": "bbb"}, manifest)
    # None no rompe (se ignora)
    assert components.artifact_coherent(irfn, None, None)


def test_momentum_5d(tmp_path):
    irfn = {"regime": {"labels": ["baja", "alta"], "xi_momentum_5d": [0.021, -0.021]}}
    mom = components.momentum_5d(irfn)
    assert mom == [("baja", 2.1), ("alta", -2.1)]
    # invariante: suma de cambios = 0 (dos distribuciones que suman 1)
    assert abs(sum(pp for _, pp in mom)) < 1e-9
    assert components.momentum_5d(None) == []


def test_load_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(components, "ARTIFACTS", tmp_path)
    assert components.load_manifest() is None
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "zzz"}), encoding="utf-8")
    assert components.load_manifest()["run_id"] == "zzz"

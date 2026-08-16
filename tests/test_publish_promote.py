"""Tests de la promocion atomica (outputs/publish.promote_run + _atomic_swap).

Cubre los criterios de aceptacion de la Fase 3:
  (a) promover el artefacto V0 de hoy (cuarentena) es rechazado.
  (c) si el swap se interrumpe, latest/ nunca queda en estado intermedio.
Ademas: guardarrail anti-downgrade (V<3 requiere force), independencia de
latest/ respecto de runs/, y el registro en publish_log.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from irfn.outputs import publish as pub
from irfn.outputs.contract import ContractViolation

ROOT = Path(__file__).resolve().parents[1]
QUARANTINE = ROOT / "artifacts" / "quarantine" / "2026-08-15_v0_regression"

GIT_SHA = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"


def _mk_run(d: Path, *, version: str = "V3", run_id: str = "run_aaa", asof: str = "2026-08-14",
            multistart: int = 30, hawkes: bool = True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(end=asof, periods=8, freq="D")
    hist = pd.DataFrame({"fecha": dates, "xi_filtered_0": 0.6, "xi_filtered_1": 0.4})
    hist.to_parquet(d / "history.parquet", index=False)
    rank = int(version[1:])
    irfn = {
        "run_id": run_id, "git_commit": GIT_SHA, "asof": asof, "version": version,
        "model": {
            "K": 2, "tvtp": rank >= 1, "covariates": (["lambda_N_z"] if hawkes else []),
            "news_layer": [],
            "estimation": {"multistart": multistart, "seed": 42, "converged": True},
            "news_layer_params": {"active": False},
            "hawkes_layer_params": {"active": hawkes},
        },
    }
    (d / "irfn.json").write_text(json.dumps(irfn), encoding="utf-8")
    (d / "audit.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    (d / "walkforward.json").write_text(json.dumps({"n_blocks": 19}), encoding="utf-8")
    if rank >= 2:
        (d / "surprise_events.json").write_text(json.dumps({"events": []}), encoding="utf-8")
    if rank >= 3:
        hist.head(2).to_parquet(d / "hawkes_history.parquet", index=False)
        hist.head(2).to_parquet(d / "headline_rug.parquet", index=False)
    return d


def test_promueve_v3_conforme_y_latest_es_independiente(tmp_path):
    run_dir = _mk_run(tmp_path / "runs" / "run_aaa", run_id="run_aaa")
    latest = tmp_path / "artifacts" / "latest"
    res = pub.promote_run(run_dir, latest_dir=latest, triggered_by="test")
    assert res.promotable
    assert (latest / "irfn.json").exists()
    assert json.loads((latest / "irfn.json").read_text())["run_id"] == "run_aaa"
    assert (latest / "manifest.json").exists()
    # el log de publicacion tiene una linea
    log = (tmp_path / "artifacts" / "publish_log.jsonl").read_text().strip().splitlines()
    assert len(log) == 1 and json.loads(log[0])["run_id"] == "run_aaa"
    # latest sobrevive al borrado del run (es una copia, no un enlace)
    import shutil
    shutil.rmtree(run_dir)
    assert (latest / "irfn.json").exists()


@pytest.mark.skipif(not QUARANTINE.is_dir(), reason="cuarentena no presente")
def test_promover_v0_de_hoy_es_rechazado(tmp_path):
    # copiar la cuarentena a tmp: promote_run escribe manifest.json en el run_dir,
    # y el backup de cuarentena debe quedar pristino (no se muta un respaldo).
    import shutil
    run_copy = tmp_path / "q_copy"
    shutil.copytree(QUARANTINE, run_copy)
    latest = tmp_path / "artifacts" / "latest"
    with pytest.raises(ContractViolation):
        pub.promote_run(run_copy, latest_dir=latest, triggered_by="test")
    # no se creo ningun latest a medias
    assert not (latest / "irfn.json").exists()


def test_guardarrail_bloquea_v0_completo_salvo_force(tmp_path):
    # V0 que SI pasa el contrato (set completo, sin ajenos) pero rank < V3
    run_dir = _mk_run(tmp_path / "runs" / "run_v0", version="V0", run_id="run_v0", hawkes=False)
    latest = tmp_path / "artifacts" / "latest"
    with pytest.raises(ContractViolation) as exc:
        pub.promote_run(run_dir, latest_dir=latest, triggered_by="test")
    assert "guardarrail" in str(exc.value).lower()
    assert not latest.exists()
    # con force_downgrade explicito si promueve y queda registrado como forzado
    res = pub.promote_run(run_dir, latest_dir=latest, force_downgrade=True, triggered_by="test")
    assert res.version == "V0"
    log = json.loads((tmp_path / "artifacts" / "publish_log.jsonl").read_text().strip())
    assert log["forced"] is True


def test_swap_interrumpido_no_deja_latest_a_medias(tmp_path, monkeypatch):
    latest = tmp_path / "artifacts" / "latest"
    # primer artefacto A promovido normalmente
    run_a = _mk_run(tmp_path / "runs" / "run_a", run_id="run_a")
    pub.promote_run(run_a, latest_dir=latest, triggered_by="test")
    assert json.loads((latest / "irfn.json").read_text())["run_id"] == "run_a"

    # intentar promover B pero forzar un crash en el 2do rename (staging -> latest)
    run_b = _mk_run(tmp_path / "runs" / "run_b", run_id="run_b")
    real_replace = pub.os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:  # el rename que mueve staging -> latest
            raise OSError("crash simulado a mitad del swap")
        return real_replace(src, dst)

    monkeypatch.setattr(pub.os, "replace", flaky_replace)
    with pytest.raises(OSError):
        pub.promote_run(run_b, latest_dir=latest, triggered_by="test")
    monkeypatch.setattr(pub.os, "replace", real_replace)

    # latest sigue siendo A intacto (no una mezcla, no vacio)
    assert (latest / "irfn.json").exists()
    assert json.loads((latest / "irfn.json").read_text())["run_id"] == "run_a"
    # y una promocion posterior de B funciona (recuperacion + swap limpio)
    pub.promote_run(run_b, latest_dir=latest, triggered_by="test")
    assert json.loads((latest / "irfn.json").read_text())["run_id"] == "run_b"

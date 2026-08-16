"""Chequeo espejo del panel publico (F6, sprint de honestidad 2026-08-16).

El panel no puede publicar en silencio una validacion que describe un run
distinto al indicador publicado. `assert_panel_coherent` ABORTA por defecto y
solo deja pasar con allow_stale (y entonces el dato ya lleva stale=true + ambos
run_id). `_validated_run_id` extrae el run que valida el reporte V4.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_panel_data.py"
_spec = importlib.util.spec_from_file_location("export_panel_data", _SCRIPT)
epd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(epd)


def test_validated_run_id_parses_report_line():
    text = "Fecha: 2026-07-14\nrun_id (indicador publicado): `02db03d3d6d3` (V3)\n"
    assert epd._validated_run_id(text) == "02db03d3d6d3"


def test_validated_run_id_none_when_absent():
    assert epd._validated_run_id("sin la linea esperada") is None


def _write_validation(out_dir: Path, validates: str, published: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation.json").write_text(
        json.dumps({
            "generated_at": "2026-07-14", "rows": [], "source": "x",
            "validates_run_id": validates, "published_run_id": published,
            "stale": validates != published,
        }),
        encoding="utf-8",
    )


def test_coherence_aborts_on_mismatch_without_flag(tmp_path):
    _write_validation(tmp_path, validates="OLD123456789", published="NEW123456789")
    with pytest.raises(SystemExit):
        epd.assert_panel_coherent(tmp_path, "NEW123456789", allow_stale=False)


def test_coherence_allows_mismatch_with_flag(tmp_path):
    _write_validation(tmp_path, validates="OLD123456789", published="NEW123456789")
    # con el flag no lanza (el dato ya lleva stale=true para que la pagina avise)
    epd.assert_panel_coherent(tmp_path, "NEW123456789", allow_stale=True)


def test_coherence_ok_when_run_ids_match(tmp_path):
    _write_validation(tmp_path, validates="SAME12345678", published="SAME12345678")
    epd.assert_panel_coherent(tmp_path, "SAME12345678", allow_stale=False)


def test_coherence_skips_when_no_validation_json(tmp_path):
    # activo sin V4 propia (--asset): no hay validation.json => no aborta
    epd.assert_panel_coherent(tmp_path, "ANY123456789", allow_stale=False)

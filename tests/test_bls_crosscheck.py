"""Tests del cliente BLS-QA (@diagnostic_only). El parsing se prueba con una
respuesta enlatada (sin red); la marca @diagnostic_only se verifica para que
nadie lo cablee al modelo por error (R1/R4)."""
from __future__ import annotations

import pandas as pd

from irfn.audit import bls_crosscheck as B


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


_CANNED = {
    "status": "REQUEST_SUCCEEDED",
    "message": [],
    "Results": {
        "series": [
            {
                "seriesID": "CUSR0000SA0",
                "data": [
                    {"year": "2025", "period": "M03", "value": "312.500"},
                    {"year": "2025", "period": "M02", "value": "311.100"},
                    {"year": "2025", "period": "M13", "value": "310.000"},  # anual -> se descarta
                    {"year": "2025", "period": "M01", "value": "310.200"},
                ],
            }
        ]
    },
}


def test_fetch_bls_series_parsing(monkeypatch):
    monkeypatch.setattr(B.requests, "post", lambda *a, **k: _FakeResp(_CANNED))
    out = B.fetch_bls_series(["CUSR0000SA0"], start_year=2025, end_year=2025, api_key=None)
    df = out["CUSR0000SA0"]
    # M13 (promedio anual) descartado; 3 meses reales, ordenados ascendente
    assert list(df["fecha"]) == [pd.Timestamp(2025, 1, 1), pd.Timestamp(2025, 2, 1), pd.Timestamp(2025, 3, 1)]
    assert list(df["value"]) == [310.2, 311.1, 312.5]


def test_fetch_bls_series_raises_on_failure(monkeypatch):
    bad = {"status": "REQUEST_NOT_PROCESSED", "message": ["daily threshold reached"]}
    monkeypatch.setattr(B.requests, "post", lambda *a, **k: _FakeResp(bad))
    try:
        B.fetch_bls_series(["X"], start_year=2025, end_year=2025)
        raised = False
    except RuntimeError as exc:
        raised = "daily threshold" in str(exc)
    assert raised


def test_period_to_month_edges():
    assert B._period_to_month("2025", "M07") == pd.Timestamp(2025, 7, 1)
    assert B._period_to_month("2025", "M13") is None  # promedio anual
    assert B._period_to_month("2025", "Q02") is None  # no mensual


def test_diagnostic_only_marks():
    # R1/R4: estas funciones estan marcadas como diagnostico; nunca al modelo.
    assert getattr(B.fetch_bls_series, "_diagnostic_only", False) is True
    assert getattr(B.crosscheck_against_alfred, "_diagnostic_only", False) is True

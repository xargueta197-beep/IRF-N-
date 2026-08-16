"""Ingesta de titulares (V3): parseo de snapshots, cobertura honesta y la
auditoria de timestamps de la Trampa 3 (hora_titular vs hora_evento).

Sin red: los snapshots se fabrican en tmp_path con el MISMO formato que escribe
capture_headlines_range. Lo que se prueba es el contrato del modulo, no GDELT.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from irfn.data.headlines import load_headlines, match_event_times, timestamp_audit


def _write_snapshot(dirpath, day: str, articles: list[dict], cap_hit: bool = False):
    snap = {
        "captured_at": f"{day}T23:59:00+00:00",
        "source": "gdelt doc 2.0 api",
        "query": "(test)",
        "day": day,
        "n_total": len(articles),
        "cap_hit_any": cap_hit,
        "segments": [{
            "start": f"{day}T00:00:00Z", "end": f"{day}T23:59:59Z",
            "n": len(articles), "cap_hit": cap_hit, "articles": articles,
        }],
    }
    (dirpath / f"{day}.json").write_text(json.dumps(snap), encoding="utf-8")


def _art(seendate: str, title: str, url: str) -> dict:
    return {"seendate": seendate, "title": title, "url": url,
            "domain": "example.com", "sourcecountry": "United States"}


@pytest.fixture
def snapshot_dir(tmp_path):
    d = tmp_path / "headlines"
    d.mkdir()
    _write_snapshot(d, "2026-07-01", [
        _art("20260701T130000Z", "CPI report shows inflation rate above forecasts", "https://a/1"),
        _art("20260701T134500Z", "Markets react to inflation data", "https://a/2"),
        _art("20260701T134500Z", "Markets react to inflation data", "https://a/2"),  # duplicado por URL
    ])
    # 2026-07-02 FALTA a proposito (hueco de cobertura)
    _write_snapshot(d, "2026-07-03", [
        _art("20260703T090000Z", "Treasury yields fall ahead of jobs report", "https://a/3"),
    ], cap_hit=True)
    return d


def test_load_headlines_parses_dedupes_and_reports_coverage(snapshot_dir):
    df = load_headlines(snapshot_dir)
    assert len(df) == 2 + 1                       # el duplicado por URL se elimina
    assert df["hora_titular"].is_monotonic_increasing
    assert str(df["hora_titular"].dt.tz) == "UTC"
    cov = df.attrs["coverage"]
    assert cov["first_day"] == "2026-07-01" and cov["last_day"] == "2026-07-03"
    assert cov["missing_days"] == ["2026-07-02"]  # el hueco se REPORTA, no se esconde
    assert cov["censored_days"] == ["2026-07-03"]  # cap_hit => censura documentada


def test_load_headlines_empty_dir(tmp_path):
    df = load_headlines(tmp_path / "no_existe")
    assert df.empty
    assert df.attrs["coverage"]["n_days"] == 0


def _calendar_with_release(hora_evento_utc: str) -> pd.DataFrame:
    ts = pd.to_datetime([hora_evento_utc], utc=True)
    return pd.DataFrame(
        {"indicator": ["CPI"], "actual": [3.1], "consensus": [3.0],
         "previous": [3.2], "unit": ["%"], "fecha_evento": [str(ts[0].date())],
         "captured_at": ["2026-06-30"]},
        index=pd.DatetimeIndex(ts, name="hora_evento"),
    )


def test_match_and_audit_positive_lag_is_green(snapshot_dir):
    # CPI publicado 12:30 UTC; el titular de CPI llega 13:00 -> lag +0.5h, verde.
    cal = _calendar_with_release("2026-07-01T12:30:00Z")
    matched = match_event_times(load_headlines(snapshot_dir), cal, match_window_hours=36)
    audit = timestamp_audit(matched)
    a = audit["titular_vs_evento"]
    assert a["n_matched"] >= 1 and not a["vacuous"]
    assert a["negative_mass"] == 0.0
    assert audit["passed"]


def test_audit_flags_negative_mass(snapshot_dir):
    # Release DESPUES del titular que lo reporta: masa negativa -> ROJO.
    cal = _calendar_with_release("2026-07-01T14:00:00Z")
    matched = match_event_times(load_headlines(snapshot_dir), cal, match_window_hours=36)
    audit = timestamp_audit(matched)
    a = audit["titular_vs_evento"]
    assert a["n_matched"] >= 1
    assert a["negative_mass"] > 0.0
    assert not audit["passed"]


def test_audit_vacuous_without_calendar(snapshot_dir):
    # Calendario vacio (la condicion de hoy): pase VACUO declarado, no fingido.
    empty_cal = pd.DataFrame(columns=["indicator"])
    matched = match_event_times(load_headlines(snapshot_dir), empty_cal, match_window_hours=36)
    audit = timestamp_audit(matched)
    assert audit["titular_vs_evento"]["vacuous"]
    assert audit["titular_vs_evento"]["passed"]
    # el chequeo de resolucion del feed SI es sustantivo aunque no haya calendario
    assert audit["resolucion_feed"]["n_headlines"] == 3
    assert audit["passed"]


def test_audit_flags_date_only_feed(tmp_path):
    # Feed degradado: todos los timestamps en medianoche exacta -> la alineacion
    # intradia no es creible y la resolucion se marca en rojo.
    d = tmp_path / "h"
    d.mkdir()
    _write_snapshot(d, "2026-07-01", [
        _art("20260701T000000Z", f"stock market headline {i}", f"https://b/{i}")
        for i in range(10)
    ])
    matched = match_event_times(load_headlines(d), pd.DataFrame(columns=["indicator"]),
                                match_window_hours=36)
    audit = timestamp_audit(matched)
    assert audit["resolucion_feed"]["midnight_exact_frac"] == 1.0
    assert not audit["resolucion_feed"]["passed"]
    assert not audit["passed"]

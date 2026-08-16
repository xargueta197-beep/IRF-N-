"""Tests del modulo experimental de sentiment/noticias: contrato del esquema y
la primera fuente (Finnhub) con respuesta enlatada (sin red)."""
from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from irfn.data.sentiment import schema
from irfn.data.sentiment.schema import (
    SENTIMENT_COLUMNS,
    SentimentRecord,
    empty_frame,
    validate_frame,
)
from irfn.data.sentiment.sources import finnhub as F


# --------------------------- esquema --------------------------- #
def test_empty_frame_contract():
    df = empty_frame()
    assert list(df.columns) == SENTIMENT_COLUMNS
    assert str(df["sentiment_score"].dtype) == "float64"
    assert str(df["fecha"].dtype).startswith("datetime64") and "UTC" in str(df["fecha"].dtype)


def test_validate_frame_normalizes_and_preserves_nan():
    df = pd.DataFrame(
        {
            "fecha": ["2026-08-14T18:15:00Z", "2026-08-14T18:30:00+00:00"],
            "fuente": ["finnhub", "gdelt"],
            "ticker_o_moneda": ["AAPL", None],
            "titulo": ["t1", "t2"],
            "texto_resumen": ["r1", None],
            "sentiment_score": [0.5, None],
            "url": ["http://a", "http://b"],
        }
    )
    v = validate_frame(df)
    assert "UTC" in str(v["fecha"].dtype)
    assert v["sentiment_score"].iloc[0] == 0.5
    assert np.isnan(v["sentiment_score"].iloc[1])  # gdelt no puntua -> NaN preservado


def test_validate_frame_rejects_missing_and_extra_columns():
    with pytest.raises(ValueError):
        validate_frame(pd.DataFrame({"fecha": [], "fuente": []}))  # faltan columnas
    full = {c: [] for c in SENTIMENT_COLUMNS}
    full["columna_intrusa"] = []
    with pytest.raises(ValueError):
        validate_frame(pd.DataFrame(full))  # columna de mas


def test_sentiment_record_range_and_utc():
    r = SentimentRecord(
        fecha=datetime(2026, 8, 14, tzinfo=timezone.utc),
        fuente="apitube", titulo="x", url="http://x", sentiment_score=-0.3,
    )
    assert r.sentiment_score == -0.3
    with pytest.raises(ValueError):
        SentimentRecord(fecha=datetime(2026, 8, 14, tzinfo=timezone.utc),
                        fuente="x", titulo="x", url="http://x", sentiment_score=5.0)
    with pytest.raises(ValueError):
        SentimentRecord(fecha=datetime(2026, 8, 14), fuente="x", titulo="x", url="http://x")  # naive


# --------------------------- Finnhub --------------------------- #
_CANNED = [
    {"datetime": 1765730100, "headline": "Fed holds rates", "summary": "resumen 1",
     "related": "SPY", "url": "http://n/1"},
    {"datetime": 1765731000, "headline": "Jobs report beats", "summary": "",
     "related": "SPY", "url": "http://n/2"},
    {"datetime": 1765732000, "headline": "", "summary": "sin titulo -> descartado",
     "related": "SPY", "url": "http://n/3"},  # sin titulo: se descarta
]


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_finnhub_rows_to_frame_maps_schema():
    df = F._rows_to_frame(_CANNED, default_symbol="SPY")
    assert list(df.columns) == SENTIMENT_COLUMNS
    assert len(df) == 2  # el item sin titulo se descarto
    assert (df["fuente"] == "finnhub").all()
    assert df["sentiment_score"].isna().all()  # company-news no trae sentiment
    assert "UTC" in str(df["fecha"].dtype)
    assert pd.isna(df["texto_resumen"].iloc[1])  # summary "" -> ausente (NaN en columna object)


def test_finnhub_fetch_requires_query():
    with pytest.raises(ValueError):
        F.FinnhubSource().fetch(since=date(2026, 1, 1), until=date(2026, 1, 2))


def test_finnhub_fetch_mocked(monkeypatch):
    monkeypatch.setattr(F, "_api_key", lambda: "fake")
    monkeypatch.setattr(F.requests, "get", lambda *a, **k: _FakeResp(_CANNED))
    df = F.FinnhubSource().fetch(since=date(2025, 12, 1), until=date(2025, 12, 31), query="SPY", save=False)
    assert len(df) == 2
    assert list(df.columns) == SENTIMENT_COLUMNS


def test_finnhub_backoff_on_429(monkeypatch):
    calls = {"n": 0}

    def _get(*a, **k):
        calls["n"] += 1
        return _FakeResp({}, status=429) if calls["n"] == 1 else _FakeResp(_CANNED, status=200)

    monkeypatch.setattr(F, "_api_key", lambda: "fake")
    monkeypatch.setattr(F.requests, "get", _get)
    monkeypatch.setattr(F.time, "sleep", lambda *a, **k: None)  # sin esperas reales
    df = F.fetch_finnhub("SPY", date(2025, 12, 1), date(2025, 12, 31), save=False)
    assert calls["n"] == 2  # reintento explicito tras 429
    assert len(df) == 2


def test_finnhub_network_error_isolated(monkeypatch):
    def _boom(*a, **k):
        raise F.requests.Timeout("timed out")
    monkeypatch.setattr(F, "_api_key", lambda: "fake")
    monkeypatch.setattr(F.requests, "get", _boom)
    df = F.fetch_finnhub("SPY", date(2025, 12, 1), date(2025, 12, 31), save=False)
    assert len(df) == 0 and list(df.columns) == SENTIMENT_COLUMNS  # aislado, no tumba


def test_finnhub_market_news_crypto_mocked(monkeypatch, tmp_path):
    monkeypatch.setattr(F, "_api_key", lambda: "fake")
    monkeypatch.setattr(F.requests, "get", lambda *a, **k: _FakeResp(_CANNED))
    df = F.fetch_finnhub_market_news("crypto", save=True, out_dir=tmp_path)
    assert list(df.columns) == SENTIMENT_COLUMNS
    assert len(df) == 2
    assert (df["fuente"] == "finnhub").all()
    assert df["sentiment_score"].isna().all()  # market-news no trae sentiment
    assert list(tmp_path.glob("crypto__*.parquet"))


def test_finnhub_market_news_bad_category():
    with pytest.raises(ValueError):
        F.fetch_finnhub_market_news("stonks")


# --------------------------- GDELT --------------------------- #
from irfn.data.sentiment.sources import gdelt as G  # noqa: E402


class _FakeHTTP:
    def __init__(self, *, status=200, text="{}", payload=None):
        self.status_code = status
        self.text = text
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


_GDELT_CANNED = {
    "articles": [
        {"seendate": "20260814T085517Z", "title": "Fed holds rates", "url": "http://g/1", "domain": "x.com"},
        {"seendate": "20260814T135152Z", "title": "Jobs report", "url": "http://g/2", "domain": "y.com"},
        {"seendate": "20260814T160000Z", "title": "", "url": "http://g/3"},  # sin titulo -> descartado
        {"seendate": "20260814T085517Z", "title": "Fed holds rates", "url": "http://g/1"},  # dup url
    ]
}


def test_gdelt_articles_to_frame_maps_schema():
    df = G._articles_to_frame(_GDELT_CANNED["articles"])
    assert list(df.columns) == SENTIMENT_COLUMNS
    assert len(df) == 2  # sin titulo descartado + url duplicada colapsada
    assert (df["fuente"] == "gdelt").all()
    assert df["sentiment_score"].isna().all()  # artlist no puntua
    assert df["ticker_o_moneda"].isna().all()  # noticia general
    assert "UTC" in str(df["fecha"].dtype)
    # timestamps de resolucion fina (segundos != 0): 08:55:17
    assert df["fecha"].iloc[0].second == 17


def test_gdelt_fetch_success(monkeypatch, tmp_path):
    import json as _json
    resp = _FakeHTTP(status=200, text=_json.dumps(_GDELT_CANNED))  # fetch_gdelt parsea resp.text
    monkeypatch.setattr(G.requests, "get", lambda *a, **k: resp)
    df = G.fetch_gdelt("inflation", date(2026, 8, 1), date(2026, 8, 14), save=True, out_dir=tmp_path)
    assert len(df) == 2
    # guardo Parquet siguiendo la convencion de rutas
    saved = list(tmp_path.glob("*.parquet"))
    assert len(saved) == 1 and saved[0].name.startswith("inflation__")


def test_gdelt_fetch_rate_limit_returns_empty(monkeypatch):
    # HTTP 200 con texto de rate-limit: manejo aislado -> frame vacio, sin excepcion
    resp = _FakeHTTP(status=200, text="Please limit requests to one every 5 seconds.")
    monkeypatch.setattr(G.requests, "get", lambda *a, **k: resp)
    df = G.fetch_gdelt("x", date(2026, 8, 1), date(2026, 8, 2), save=False, max_retries=1)
    assert len(df) == 0 and list(df.columns) == SENTIMENT_COLUMNS


def test_gdelt_fetch_network_error_returns_empty(monkeypatch):
    def _boom(*a, **k):
        raise G.requests.Timeout("timed out")
    monkeypatch.setattr(G.requests, "get", _boom)
    df = G.fetch_gdelt("x", date(2026, 8, 1), date(2026, 8, 2), save=False)
    assert len(df) == 0 and list(df.columns) == SENTIMENT_COLUMNS  # no tumba el pipeline


# --------------------------- CryptoPanic --------------------------- #
from irfn.data.sentiment.sources import cryptopanic as C  # noqa: E402

_CP_CANNED = {
    "results": [
        {"published_at": "2026-08-14T13:00:00Z", "title": "BTC surges", "url": "http://c/1",
         "currencies": [{"code": "BTC"}], "votes": {"positive": 8, "negative": 2}},   # -> 0.6
        {"published_at": "2026-08-13T10:00:00Z", "title": "BTC quiet", "url": "http://c/2",
         "currencies": [{"code": "BTC"}], "votes": {"positive": 0, "negative": 0}},   # -> NaN
        {"published_at": "2026-08-10T10:00:00Z", "title": "BTC old", "url": "http://c/3",
         "currencies": [{"code": "BTC"}], "votes": {"positive": 1, "negative": 1}},   # fuera de rango
    ],
    "next": None,
}


def test_cryptopanic_votes_to_score():
    assert C._votes_to_score({"positive": 8, "negative": 2}) == 0.6
    assert C._votes_to_score({"positive": 0, "negative": 0}) != C._votes_to_score({"positive": 0, "negative": 0}) or \
        np.isnan(C._votes_to_score({"positive": 0, "negative": 0}))  # NaN sin votos
    assert C._votes_to_score(None) != C._votes_to_score(None) or np.isnan(C._votes_to_score(None))
    assert C._votes_to_score({"positive": 1, "negative": 1}) == 0.0


class _FakeHTTP2:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_cryptopanic_fetch_maps_and_filters(monkeypatch, tmp_path):
    import json as _json
    monkeypatch.setattr(C, "_api_key", lambda: "fake")
    monkeypatch.setattr(C.requests, "get", lambda *a, **k: _FakeHTTP2(_json.dumps(_CP_CANNED)))
    df = C.fetch_cryptopanic(date(2026, 8, 12), date(2026, 8, 15), currencies="BTC",
                             save=True, out_dir=tmp_path, max_pages=1)
    assert list(df.columns) == SENTIMENT_COLUMNS
    assert len(df) == 2  # el post del 08-10 queda fuera del rango
    assert (df["fuente"] == "cryptopanic").all()
    assert (df["ticker_o_moneda"] == "BTC").all()
    # score mapeado de votos: 0.6 para el primero, NaN para el sin-votos
    scores = df.sort_values("fecha")["sentiment_score"].tolist()
    assert np.isnan(scores[0]) and scores[1] == 0.6
    assert list(tmp_path.glob("*.parquet"))  # Parquet guardado


def test_cryptopanic_missing_key_raises(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("CRYPTOPANIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        C._api_key()


# --------------------------- APITube --------------------------- #
from irfn.data.sentiment.sources import apitube as A  # noqa: E402

_AT_CANNED = {
    "status": "ok",
    "results": [
        {"published_at": "2026-08-14T13:00:00.000Z", "title": "Bitcoin rallies", "href": "http://a/1",
         "description": "desc1", "sentiment": {"overall": {"score": 0.42, "polarity": "positive"}}},
        {"published_at": "2026-08-13T10:00:00.000Z", "title": "Markets mixed", "href": "http://a/2",
         "description": None, "sentiment": {"overall": {"score": 1.7}}},  # fuera de rango -> clamp 1.0
        {"published_at": "2026-08-12T10:00:00.000Z", "title": "No sentiment", "href": "http://a/3"},  # -> NaN
        {"published_at": "2026-08-12T10:00:00.000Z", "title": "", "href": "http://a/4"},  # sin titulo -> fuera
    ],
}


def test_apitube_extract_score_clamps_and_nan():
    assert A._extract_score({"sentiment": {"overall": {"score": 0.42}}}) == 0.42
    assert A._extract_score({"sentiment": {"overall": {"score": 1.7}}}) == 1.0   # clamp
    assert A._extract_score({"sentiment": {"overall": {"score": -3.0}}}) == -1.0  # clamp
    assert np.isnan(A._extract_score({}))  # sin sentiment -> NaN


def test_apitube_fetch_maps_and_scores(monkeypatch, tmp_path):
    import json as _json
    monkeypatch.setattr(A, "_api_key", lambda: "fake")
    monkeypatch.setattr(A.requests, "get", lambda *a, **k: _FakeHTTP2(_json.dumps(_AT_CANNED)))
    df = A.fetch_apitube("bitcoin", date(2026, 8, 12), date(2026, 8, 14), save=True, out_dir=tmp_path)
    assert list(df.columns) == SENTIMENT_COLUMNS
    assert len(df) == 3  # el item sin titulo se descarto
    s = df.sort_values("fecha", ascending=False)["sentiment_score"].tolist()
    assert s[0] == 0.42 and s[1] == 1.0 and np.isnan(s[2])  # clamp y NaN
    assert list(tmp_path.glob("bitcoin__*.parquet"))


def test_apitube_fetch_status_error_returns_empty(monkeypatch):
    import json as _json
    bad = {"status": "error", "message": "quota exceeded"}
    monkeypatch.setattr(A, "_api_key", lambda: "fake")
    monkeypatch.setattr(A.requests, "get", lambda *a, **k: _FakeHTTP2(_json.dumps(bad)))
    df = A.fetch_apitube("x", date(2026, 8, 1), date(2026, 8, 2), save=False)
    assert len(df) == 0 and list(df.columns) == SENTIMENT_COLUMNS


def test_apitube_missing_key_raises(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("APITUBE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        A._api_key()


# --------------------------- Orquestador --------------------------- #
from irfn.data.sentiment import orchestrator as O  # noqa: E402
from irfn.data.sentiment.base import SentimentSource  # noqa: E402


def _frame(rows):
    return validate_frame(pd.DataFrame(rows, columns=SENTIMENT_COLUMNS))


def _mk_source(frame=None, exc=None, api_key_env=None):
    class _S(SentimentSource):
        pass
    _S.api_key_env = api_key_env

    def fetch(self, *, since, until, query=None, **k):
        if exc is not None:
            raise exc
        return frame

    _S.fetch = fetch
    _S.__abstractmethods__ = frozenset()  # fetch ya esta definido -> instanciable
    return _S


_R1 = _frame([["2026-08-14T10:00:00Z", "a", None, "t1", None, 0.1, "u1"],
              ["2026-08-13T10:00:00Z", "a", None, "t2", None, float("nan"), "u2"]])
_R2 = _frame([["2026-08-12T10:00:00Z", "b", None, "t3", None, -0.2, "u3"],
              ["2026-08-13T10:00:00Z", "b", None, "t2dup", None, 0.0, "u2"]])  # u2 duplicada


def test_ingest_merges_dedups_and_sorts(monkeypatch):
    monkeypatch.setattr(O, "SOURCES", {"a": _mk_source(_R1), "b": _mk_source(_R2)})
    out = O.ingest(date(2026, 8, 1), date(2026, 8, 14), {"a": "q", "b": "q"}, save=False)
    assert list(out.columns) == SENTIMENT_COLUMNS
    assert len(out) == 3  # u1, u2, u3 (u2 deduplicada)
    assert out["fecha"].is_monotonic_increasing  # ordenado por fecha
    assert set(out["url"]) == {"u1", "u2", "u3"}


def test_ingest_isolates_failing_source(monkeypatch):
    fakes = {"good": _mk_source(_R1), "bad": _mk_source(exc=RuntimeError("boom"))}
    monkeypatch.setattr(O, "SOURCES", fakes)
    out = O.ingest(date(2026, 8, 1), date(2026, 8, 14), {"good": "q", "bad": "q"}, save=False)
    assert len(out) == 2  # la fuente que revienta se omite; la buena sobrevive
    assert out.attrs["sources_run"] == ["good"]


def test_ingest_skips_source_without_key(monkeypatch):
    monkeypatch.setattr(O, "SOURCES", {"needs_key": _mk_source(_R1, api_key_env="NOPE_KEY_XYZ")})
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("NOPE_KEY_XYZ", raising=False)
    out = O.ingest(date(2026, 8, 1), date(2026, 8, 14), {"needs_key": "q"}, save=False)
    assert len(out) == 0  # sin key -> omitida

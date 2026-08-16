"""Fuente CryptoPanic — noticias CRIPTO con votos de comunidad (Fase 2, sentiment).

Para la linea BTC del proyecto. Endpoint v1 de posts:
  GET https://cryptopanic.com/api/v1/posts/?auth_token=KEY&currencies=BTC&public=true
Docs: https://cryptopanic.com/developers/api/ . Key: CRYPTOPANIC_API_KEY.

CryptoPanic NO da un score numerico: da VOTOS (positive/negative/...). Se mapean a
`sentiment_score` en [-1, 1] = (positive - negative) / (positive + negative), y NaN
si no hay votos (R7: no inventar sentimiento donde la comunidad no voto).

La API v1 no filtra por rango de fechas en el servidor: se pagina lo reciente
(newest-first) y se filtra por `published_at` en [since, until] del lado cliente,
con tope de paginas para no colgar. Manejo de errores de red AISLADO (frame vacio,
no propaga); la falta de key SI se reporta (es un error de config, no de red).
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from irfn.data.sentiment.base import SentimentSource
from irfn.data.sentiment.schema import empty_frame, validate_frame

log = logging.getLogger("irfn.sentiment.cryptopanic")

ROOT = Path(__file__).resolve().parents[5]
RAW_DIR = ROOT / "data" / "raw" / "sentiment" / "cryptopanic"
POSTS_URL = "https://cryptopanic.com/api/v1/posts/"


def _api_key() -> str:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = (os.environ.get("CRYPTOPANIC_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "CRYPTOPANIC_API_KEY no configurada. Registro gratis en "
            "https://cryptopanic.com/developers/api/ y ponla en irfn/.env."
        )
    return key


def _to_utc(d: date | datetime) -> datetime:
    if isinstance(d, datetime):
        return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _slug(text: str, *, maxlen: int = 40) -> str:
    return (re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()[:maxlen] or "cripto")


def _votes_to_score(votes: dict | None) -> float:
    """Mapea los votos de CryptoPanic a un score en [-1, 1]. NaN si no hay votos
    direccionales (positive+negative == 0): no se inventa sentimiento (R7)."""
    if not isinstance(votes, dict):
        return float("nan")
    pos = int(votes.get("positive", 0) or 0)
    neg = int(votes.get("negative", 0) or 0)
    total = pos + neg
    if total == 0:
        return float("nan")
    return (pos - neg) / total


def _results_to_rows(results: list[dict], *, default_ticker: str | None) -> list[dict]:
    rows: list[dict] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        pub = r.get("published_at")
        title = r.get("title")
        url = r.get("url")
        if not pub or not title or not url:
            continue
        fecha = pd.to_datetime(pub, utc=True, errors="coerce")
        if pd.isna(fecha):
            continue
        currencies = r.get("currencies") or []
        code = currencies[0].get("code") if currencies and isinstance(currencies[0], dict) else None
        rows.append(
            {
                "fecha": fecha,
                "fuente": "cryptopanic",
                "ticker_o_moneda": code or default_ticker,
                "titulo": title,
                "texto_resumen": None,  # los posts v1 no traen cuerpo
                "sentiment_score": _votes_to_score(r.get("votes")),
                "url": url,
            }
        )
    return rows


def fetch_cryptopanic(
    since: date | datetime,
    until: date | datetime,
    *,
    currencies: str = "BTC",
    kind: str = "news",
    max_pages: int = 5,
    timeout: float = 30.0,
    save: bool = True,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Descarga posts de CryptoPanic para `currencies` (p.ej. "BTC") y los filtra a
    [since, until] (UTC), mapeados al esquema comun. Guarda Parquet en
    data/raw/sentiment/cryptopanic/ si `save`. Errores de red -> frame vacio."""
    start, end = _to_utc(since), _to_utc(until)
    params = {
        "auth_token": _api_key(),
        "currencies": currencies,
        "kind": kind,
        "public": "true",
    }

    collected: list[dict] = []
    url: str | None = POSTS_URL
    page = 0
    while url and page < max_pages:
        page += 1
        try:
            resp = requests.get(url, params=params if page == 1 else None, timeout=timeout)
            resp.raise_for_status()
            data = json.loads(resp.text)
        except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
            log.warning("CryptoPanic red/JSON (currencies=%s): %s -> lo obtenido hasta ahora", currencies, exc)
            break

        results = data.get("results", []) if isinstance(data, dict) else []
        rows = _results_to_rows(results, default_ticker=currencies)
        collected.extend(rows)
        # results vienen newest-first: si el mas viejo de la pagina ya es anterior
        # a `since`, no hay razon para seguir paginando.
        if rows and min(r["fecha"] for r in rows) < start:
            break
        url = data.get("next") if isinstance(data, dict) else None

    if not collected:
        return empty_frame()
    df = validate_frame(pd.DataFrame(collected))
    df = df[(df["fecha"] >= start) & (df["fecha"] <= end)].reset_index(drop=True)
    df = df.drop_duplicates(subset="url").reset_index(drop=True)

    if save and len(df):
        path = (out_dir or RAW_DIR) / f"{_slug(currencies)}__{start:%Y%m%d}__{end:%Y%m%d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        log.info("CryptoPanic: %d posts -> %s", len(df), path)
    return df


class CryptopanicSource(SentimentSource):
    name = "cryptopanic"
    api_key_env = "CRYPTOPANIC_API_KEY"

    def fetch(self, *, since: date, until: date, query: str | None = None, **kwargs) -> pd.DataFrame:
        """Delega en fetch_cryptopanic(). `query` = codigo(s) de moneda (default BTC)."""
        return fetch_cryptopanic(since, until, currencies=query or "BTC", **kwargs)

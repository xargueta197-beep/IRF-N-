"""Fuente APITube — News API con sentiment propio (Fase 2, modulo sentiment).

Endpoint: GET https://api.apitube.io/v1/news/everything
Docs: https://apitube.io/ . Key: APITUBE_API_KEY.

APITube corre NLP y devuelve `sentiment.overall.score` (tipicamente en [-1, 1]):
se mapea a `sentiment_score` con clamp defensivo al rango del contrato; NaN si el
articulo no trae score (R7). Filtra por keyword (`title`) y rango de fechas
(`published_at.start`/`.end`) del lado servidor. Errores de red AISLADOS (frame
vacio); la falta de key SI se reporta (error de config, no de red).
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
from irfn.data.sentiment.schema import SENTIMENT_MAX, SENTIMENT_MIN, empty_frame, validate_frame

log = logging.getLogger("irfn.sentiment.apitube")

ROOT = Path(__file__).resolve().parents[5]
RAW_DIR = ROOT / "data" / "raw" / "sentiment" / "apitube"
EVERYTHING_URL = "https://api.apitube.io/v1/news/everything"


def _api_key() -> str:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = (os.environ.get("APITUBE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "APITUBE_API_KEY no configurada. Registro en https://apitube.io/ y ponla "
            "en irfn/.env (APITUBE_API_KEY=...)."
        )
    return key


def _to_utc(d: date | datetime) -> datetime:
    if isinstance(d, datetime):
        return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _slug(text: str, *, maxlen: int = 60) -> str:
    return (re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()[:maxlen] or "query")


def _extract_score(art: dict) -> float:
    """sentiment.overall.score -> [-1, 1] con clamp defensivo; NaN si no hay score."""
    s = art.get("sentiment")
    if isinstance(s, dict):
        overall = s.get("overall")
        if isinstance(overall, dict) and overall.get("score") is not None:
            try:
                score = float(overall["score"])
            except (TypeError, ValueError):
                return float("nan")
            return max(SENTIMENT_MIN, min(SENTIMENT_MAX, score))
    return float("nan")


def _results_to_frame(results: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for a in results:
        if not isinstance(a, dict):
            continue
        pub = a.get("published_at")
        title = a.get("title")
        url = a.get("href") or a.get("url")
        if not pub or not title or not url:
            continue
        fecha = pd.to_datetime(pub, utc=True, errors="coerce")
        if pd.isna(fecha):
            continue
        rows.append(
            {
                "fecha": fecha,
                "fuente": "apitube",
                "ticker_o_moneda": None,  # APITube devuelve noticia, no ticker
                "titulo": title,
                "texto_resumen": a.get("description") or None,
                "sentiment_score": _extract_score(a),
                "url": url,
            }
        )
    if not rows:
        return empty_frame()
    return validate_frame(pd.DataFrame(rows).drop_duplicates(subset="url"))


def fetch_apitube(
    keyword: str,
    since: date | datetime,
    until: date | datetime,
    *,
    language: str = "en",
    per_page: int = 10,
    max_pages: int = 5,
    timeout: float = 30.0,
    save: bool = True,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Noticias de APITube que cumplen `keyword` en [since, until] (UTC), mapeadas al
    esquema comun (con sentiment.overall.score). Guarda Parquet en
    data/raw/sentiment/apitube/. Errores de red -> frame vacio (aislado).

    per_page tope depende del plan (el trial permite <=10); se pagina siguiendo
    `next_page` hasta `max_pages` para juntar mas resultados.
    """
    start, end = _to_utc(since), _to_utc(until)
    params: dict | None = {
        "api_key": _api_key(),
        "title": keyword,
        "language.code": language,
        "published_at.start": start.strftime("%Y-%m-%d"),
        "published_at.end": end.strftime("%Y-%m-%d"),
        "per_page": str(per_page),
    }

    collected: list[dict] = []
    url: str | None = EVERYTHING_URL
    page = 0
    while url and page < max_pages:
        page += 1
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = json.loads(resp.text)
        except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
            log.warning("APITube red/JSON (keyword=%r): %s -> lo obtenido hasta ahora", keyword, exc)
            break
        if isinstance(data, dict) and data.get("status") not in (None, "ok"):
            errs = data.get("errors") or data.get("message")
            log.warning("APITube status=%s (keyword=%r): %s -> lo obtenido hasta ahora",
                        data.get("status"), keyword, str(errs)[:150])
            break
        collected.extend(data.get("results", []) if isinstance(data, dict) else [])
        if isinstance(data, dict) and data.get("has_next_pages") and data.get("next_page"):
            url, params = data["next_page"], None  # next_page ya trae api_key y filtros
        else:
            break

    df = _results_to_frame(collected)

    if save and len(df):
        path = (out_dir or RAW_DIR) / f"{_slug(keyword)}__{start:%Y%m%d}__{end:%Y%m%d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        log.info("APITube: %d noticias -> %s", len(df), path)
    return df


class ApitubeSource(SentimentSource):
    name = "apitube"
    api_key_env = "APITUBE_API_KEY"

    def fetch(self, *, since: date, until: date, query: str | None = None, **kwargs) -> pd.DataFrame:
        """Delega en fetch_apitube(). `query` = keyword de busqueda (obligatorio)."""
        if not query:
            raise ValueError("APITube requiere un keyword en `query`.")
        return fetch_apitube(query, since, until, **kwargs)

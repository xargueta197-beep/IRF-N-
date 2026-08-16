"""Fuente GDELT — DOC 2.0 API, sin key (Fase 2, modulo experimental sentiment).

fetch_gdelt(keyword, since, until) es el ingestor INDEPENDIENTE: consulta el DOC
2.0 API (mode=artlist), mapea al esquema comun y opcionalmente guarda en Parquet
bajo la convencion de rutas del proyecto (data/raw/sentiment/gdelt/).

Endpoint: https://api.gdeltproject.org/api/v2/doc/doc  (no requiere key).

Manejo de errores AISLADO (no tumba el pipeline): timeouts, rate-limit (HTTP 429
o texto "limit requests" con HTTP 200), respuesta vacia o JSON invalido -> se
loguea un warning y se devuelve un frame VACIO del esquema. Un hueco es
informacion documentada, no un motivo para reventar (mismo criterio que el
top-up de headlines.py).

GDELT DOC/artlist NO trae sentiment ni cuerpo -> sentiment_score = NaN,
texto_resumen = None (R7: jamas inventar). La IP compartida de esta maquina sufre
rate-limit agresivo (ver reports/data_audit y memoria del proyecto).
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from irfn.data.sentiment.base import SentimentSource
from irfn.data.sentiment.schema import empty_frame, validate_frame

log = logging.getLogger("irfn.sentiment.gdelt")

ROOT = Path(__file__).resolve().parents[5]
RAW_DIR = ROOT / "data" / "raw" / "sentiment" / "gdelt"
DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MAX_RECORDS = 250  # cap duro del API por consulta


def _to_datetime(d: date | datetime) -> datetime:
    """date -> medianoche UTC; datetime naive -> UTC; datetime aware -> UTC."""
    if isinstance(d, datetime):
        return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _slug(text: str, *, maxlen: int = 60) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return (s[:maxlen] or "query")


def _parquet_path(keyword: str, start: datetime, end: datetime, out_dir: Path | None) -> Path:
    base = out_dir or RAW_DIR
    fname = f"{_slug(keyword)}__{start:%Y%m%d}__{end:%Y%m%d}.parquet"
    return base / fname


def _articles_to_frame(articles: list[dict]) -> pd.DataFrame:
    """Mapea la respuesta artlist de GDELT al esquema comun. Descarta items sin
    seendate/title/url; no fabrica campos obligatorios."""
    rows: list[dict] = []
    for a in articles:
        if not isinstance(a, dict):
            continue
        seen = a.get("seendate")
        title = a.get("title")
        url = a.get("url")
        if not seen or not title or not url:
            continue
        fecha = pd.to_datetime(seen, utc=True, errors="coerce", format="%Y%m%dT%H%M%SZ")
        if pd.isna(fecha):
            continue
        rows.append(
            {
                "fecha": fecha,
                "fuente": "gdelt",
                "ticker_o_moneda": None,  # GDELT es noticia general, sin ticker
                "titulo": title,
                "texto_resumen": None,  # artlist no trae cuerpo
                "sentiment_score": float("nan"),  # artlist no puntua sentiment
                "url": url,
            }
        )
    if not rows:
        return empty_frame()
    return validate_frame(pd.DataFrame(rows).drop_duplicates(subset="url"))


def fetch_gdelt(
    keyword: str,
    since: date | datetime,
    until: date | datetime,
    *,
    max_records: int = GDELT_MAX_RECORDS,
    timeout: float = 30.0,
    save: bool = True,
    out_dir: Path | None = None,
    max_retries: int = 2,
    backoff_seconds: float = 6.0,
) -> pd.DataFrame:
    """Descarga titulares de GDELT que cumplen `keyword` en [since, until] (UTC) y
    los devuelve en el esquema comun. Si `save`, escribe un Parquet en
    data/raw/sentiment/gdelt/ (o `out_dir`). Ante CUALQUIER fallo (red, rate-limit,
    respuesta vacia/invalida) loguea y devuelve un frame VACIO: no propaga, para no
    tumbar el resto del pipeline.
    """
    start_dt, end_dt = _to_datetime(since), _to_datetime(until)
    params = {
        "query": keyword,
        "mode": "artlist",
        "maxrecords": str(min(max_records, GDELT_MAX_RECORDS)),
        "format": "json",
        "sort": "dateasc",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
    }

    df = empty_frame()
    for intento in range(1, max_retries + 1):
        params["cb"] = str(int(time.time() * 1000))  # cache-buster (GDELT cachea por URL)
        try:
            resp = requests.get(DOC_URL, params=params, timeout=timeout)
        except requests.RequestException as exc:
            log.warning("GDELT red/timeout (keyword=%r %s..%s): %s -> frame vacio",
                        keyword, start_dt.date(), end_dt.date(), exc)
            return empty_frame()

        text = resp.text or ""
        if resp.status_code == 429 or "limit requests" in text[:300].lower():
            if intento < max_retries:
                log.warning("GDELT rate-limit (intento %d/%d); espero %.0fs", intento, max_retries, backoff_seconds)
                time.sleep(backoff_seconds)
                continue
            log.warning("GDELT rate-limit persistente (keyword=%r) -> frame vacio", keyword)
            return empty_frame()
        if resp.status_code != 200:
            log.warning("GDELT HTTP %d (keyword=%r): %s -> frame vacio",
                        resp.status_code, keyword, text[:150])
            return empty_frame()

        stripped = text.lstrip()
        if not stripped.startswith("{"):
            # Texto plano que no es rate-limit: error de la consulta (p.ej. frase
            # demasiado corta). No reintentable; se documenta y se sigue.
            log.warning("GDELT rechazo la consulta (keyword=%r): %s -> frame vacio",
                        keyword, text[:150])
            return empty_frame()
        try:
            data = json.loads(stripped)
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("GDELT JSON invalido (keyword=%r): %s -> frame vacio", keyword, exc)
            return empty_frame()

        df = _articles_to_frame(data.get("articles", []) if isinstance(data, dict) else [])
        break

    if save and len(df):
        path = _parquet_path(keyword, start_dt, end_dt, out_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        log.info("GDELT: %d titulares -> %s", len(df), path)
    return df


class GdeltSource(SentimentSource):
    name = "gdelt"
    api_key_env = None  # GDELT no requiere key

    def fetch(self, *, since: date, until: date, query: str | None = None, **kwargs) -> pd.DataFrame:
        """Delega en fetch_gdelt(). `query` es el keyword (obligatorio en GDELT)."""
        if not query:
            raise ValueError("GDELT requiere un keyword en `query`.")
        return fetch_gdelt(query, since, until, **kwargs)

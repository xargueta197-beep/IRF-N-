"""Fuente Finnhub — noticias por ticker (Fase 2, modulo experimental sentiment).

Endpoint: GET https://finnhub.io/api/v1/company-news?symbol=SYM&from=YYYY-MM-DD&to=YYYY-MM-DD&token=KEY
Docs: https://finnhub.io/docs/api/company-news . Key: FINNHUB_API_KEY.

Ventajas para el proyecto:
  - `datetime` en unix SEGUNDOS (resolucion FINA) -> a diferencia de la malla de
    15 min de GDELT, no genera empates masivos que degeneren el Hawkes.
  - noticias asociadas a un ticker (`related`), util para la linea de un activo.

company-news NO trae sentiment -> `sentiment_score = NaN` (R7: jamas inventar).
El free tier limita a 1 anio de historia por simbolo y ~60 llamadas/min.
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from irfn.data.sentiment.base import SentimentSource
from irfn.data.sentiment.schema import empty_frame, validate_frame

log = logging.getLogger("irfn.sentiment.finnhub")

ROOT = Path(__file__).resolve().parents[5]
RAW_DIR = ROOT / "data" / "raw" / "sentiment" / "finnhub"
COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"
MARKET_NEWS_URL = "https://finnhub.io/api/v1/news"  # noticias de mercado por categoria
MARKET_CATEGORIES = ("general", "forex", "crypto", "merger")


def _api_key() -> str:
    """FINNHUB_API_KEY del entorno/.env. Lanza si falta (sin key no hay datos)."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = (os.environ.get("FINNHUB_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "FINNHUB_API_KEY no configurada. Registro gratis en https://finnhub.io/register "
            "y ponla en irfn/.env (FINNHUB_API_KEY=...)."
        )
    return key


def _rows_to_frame(rows: list[dict], *, default_symbol: str | None) -> pd.DataFrame:
    """Mapea la respuesta de company-news al esquema comun. Descarta items sin
    titulo/url/datetime (no se fabrican campos obligatorios)."""
    out: list[dict] = []
    for it in rows:
        ts = it.get("datetime")
        title = it.get("headline")
        url = it.get("url")
        if not ts or not title or not url:
            continue
        fecha = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        resumen = it.get("summary") or None
        out.append(
            {
                "fecha": fecha,
                "fuente": "finnhub",
                "ticker_o_moneda": it.get("related") or default_symbol,
                "titulo": title,
                "texto_resumen": resumen,
                "sentiment_score": float("nan"),  # company-news no puntua sentiment
                "url": url,
            }
        )
    if not out:
        return empty_frame()
    return validate_frame(pd.DataFrame(out))


def _to_utc(d: date | datetime) -> datetime:
    if isinstance(d, datetime):
        return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _slug(text: str, *, maxlen: int = 40) -> str:
    return (re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()[:maxlen] or "symbol")


def _get_with_backoff(url, params, *, timeout, max_retries=4, backoff=2.0):
    """GET que respeta el limite de 60 calls/min de Finnhub: ante HTTP 429 espera
    con backoff exponencial (2,4,8s...) y reintenta; si persiste, propaga el error
    para que el llamador lo aisle. Manejo EXPLICITO de rate limit (no un simple get)."""
    delay = backoff
    resp = None
    for intento in range(1, max_retries + 1):
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 429 and intento < max_retries:
            log.warning("Finnhub 429 rate-limit (intento %d/%d); espero %.0fs", intento, max_retries, delay)
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp
    return resp


def fetch_finnhub(
    symbol: str,
    since: date | datetime,
    until: date | datetime,
    *,
    timeout: float = 30.0,
    save: bool = True,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """company-news de Finnhub para `symbol` (p.ej. 'SPY') en [since, until] (UTC),
    en el esquema comun. Respeta 60 calls/min con backoff explicito en 429; errores
    de red -> frame vacio (aislado); guarda Parquet en data/raw/sentiment/finnhub/.
    company-news no trae sentiment -> sentiment_score = NaN (R7)."""
    if not symbol:
        raise ValueError("Finnhub company-news requiere un simbolo (p.ej. 'SPY').")
    try:
        resp = _get_with_backoff(
            COMPANY_NEWS_URL,
            {
                "symbol": symbol,
                "from": _to_utc(since).strftime("%Y-%m-%d"),
                "to": _to_utc(until).strftime("%Y-%m-%d"),
                "token": _api_key(),
            },
            timeout=timeout,
        )
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("Finnhub company-news red/JSON (symbol=%s): %s -> frame vacio", symbol, exc)
        return empty_frame()
    if not isinstance(data, list):
        log.warning("Finnhub company-news respuesta inesperada (no lista) -> frame vacio")
        return empty_frame()

    df = _rows_to_frame(data, default_symbol=symbol)
    if save and len(df):
        start, end = _to_utc(since), _to_utc(until)
        path = (out_dir or RAW_DIR) / f"company_{_slug(symbol)}__{start:%Y%m%d}__{end:%Y%m%d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        log.info("Finnhub company-news %s: %d -> %s", symbol, len(df), path)
    return df


def fetch_finnhub_market_news(
    category: str = "crypto",
    *,
    since: date | datetime | None = None,
    until: date | datetime | None = None,
    timeout: float = 30.0,
    save: bool = True,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Noticias de mercado de Finnhub por categoria (general/forex/crypto/merger),
    mapeadas al esquema comun. Para la LINEA BTC: category='crypto'.

    El endpoint /news devuelve lo RECIENTE (sin rango de fechas en el servidor); si
    se pasan since/until se filtra del lado cliente. company-news/market-news no
    traen sentiment -> sentiment_score = NaN (R7). Errores de red -> frame vacio
    (aislado, no tumba el pipeline). Guarda Parquet en data/raw/sentiment/finnhub/.
    """
    if category not in MARKET_CATEGORIES:
        raise ValueError(f"category invalida: {category!r}; usar una de {MARKET_CATEGORIES}.")
    try:
        resp = requests.get(
            MARKET_NEWS_URL, params={"category": category, "token": _api_key()}, timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("Finnhub market-news red/JSON (category=%s): %s -> frame vacio", category, exc)
        return empty_frame()
    if not isinstance(data, list):
        log.warning("Finnhub market-news respuesta inesperada (no lista) -> frame vacio")
        return empty_frame()

    df = _rows_to_frame(data, default_symbol=None)
    if since is not None:
        df = df[df["fecha"] >= _to_utc(since)]
    if until is not None:
        df = df[df["fecha"] <= _to_utc(until)]
    df = df.reset_index(drop=True)

    if save and len(df):
        if since or until:
            s = _to_utc(since).strftime("%Y%m%d") if since else "start"
            u = _to_utc(until).strftime("%Y%m%d") if until else "end"
            fname = f"{category}__{s}__{u}.parquet"
        else:
            fname = f"{category}__{pd.Timestamp.now('UTC'):%Y%m%dT%H%M}.parquet"
        path = (out_dir or RAW_DIR) / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        log.info("Finnhub %s: %d noticias -> %s", category, len(df), path)
    return df


class FinnhubSource(SentimentSource):
    name = "finnhub"
    api_key_env = "FINNHUB_API_KEY"

    def fetch(
        self, *, since: date, until: date, query: str | None = None, **kwargs
    ) -> pd.DataFrame:
        """Delega en fetch_finnhub() (company-news por simbolo). `query` = simbolo
        Finnhub (p.ej. "SPY"). Con rate-limit backoff, errores aislados y Parquet."""
        if not query:
            raise ValueError("Finnhub company-news requiere un simbolo en `query` (p.ej. 'SPY').")
        return fetch_finnhub(query, since, until, **kwargs)

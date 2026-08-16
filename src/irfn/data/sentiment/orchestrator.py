"""Orquestador de la ingesta multi-fuente de sentiment/noticias (Fase 2).

`ingest()` corre cada fuente pedida de forma AISLADA (una que falle -o no tenga
key- no tumba las demas), une los resultados en el esquema comun, deduplica por
`url` y guarda un Parquet consolidado en data/raw/sentiment/.

`query` es POR FUENTE (su semantica difiere): simbolo para Finnhub company-news,
keyword para GDELT/APITube, codigo de moneda para CryptoPanic. Por eso se pasa un
dict {fuente: query}, no un unico string.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from irfn.data.sentiment.schema import empty_frame, validate_frame
from irfn.data.sentiment.sources import SOURCES

log = logging.getLogger("irfn.sentiment.ingest")

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data" / "raw" / "sentiment"


def _to_utc(d: date | datetime) -> datetime:
    if isinstance(d, datetime):
        return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def source_available(name: str) -> bool:
    """True si la fuente puede correr: sin key (GDELT) o con su key presente en .env."""
    cls = SOURCES.get(name)
    if cls is None:
        return False
    if cls.api_key_env is None:
        return True
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return bool((os.environ.get(cls.api_key_env) or "").strip())


def available_sources() -> list[str]:
    """Fuentes registradas cuya key esta disponible (o que no necesitan key)."""
    return [n for n in SOURCES if source_available(n)]


def ingest(
    since: date | datetime,
    until: date | datetime,
    queries: dict[str, str],
    *,
    save: bool = True,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Une varias fuentes en un solo DataFrame del esquema comun.

    queries: {nombre_fuente: query}. Solo se corren las fuentes presentes en
    `queries` que ademas esten disponibles (key). Cada fuente se ejecuta con
    save=False (el guardado es el consolidado) y dentro de try/except (aislamiento).
    Devuelve el frame combinado, deduplicado por url y ordenado por fecha.
    """
    frames: list[pd.DataFrame] = []
    corridas: list[str] = []
    for name, q in queries.items():
        if name not in SOURCES:
            log.warning("fuente desconocida %r -> se ignora", name)
            continue
        if not source_available(name):
            log.warning("fuente %s sin key disponible -> se omite", name)
            continue
        try:
            df = SOURCES[name]().fetch(since=since, until=until, query=q, save=False)
            log.info("%s: %d items", name, len(df))
            frames.append(df)
            corridas.append(name)
        except Exception as exc:  # noqa: BLE001  aislamiento: una fuente no tumba el resto
            log.warning("fuente %s fallo (%s) -> se omite", name, exc)

    if not frames:
        out = empty_frame()
        out.attrs["sources_run"] = corridas
        return out

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        validate_frame(combined)
        .drop_duplicates(subset="url")
        .sort_values("fecha")
        .reset_index(drop=True)
    )
    combined.attrs["sources_run"] = corridas

    if save and len(combined):
        start, end = _to_utc(since), _to_utc(until)
        path = (out_dir or RAW_DIR) / f"consolidado__{start:%Y%m%d}__{end:%Y%m%d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(path, index=False)
        log.info("consolidado: %d items de %s -> %s", len(combined), corridas, path)
    return combined

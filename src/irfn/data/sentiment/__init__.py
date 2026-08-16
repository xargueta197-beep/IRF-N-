"""Ingesta multi-fuente de sentiment/noticias bajo un esquema comun (Fase 2, Data).

MODULO EXPERIMENTAL / TEMPORAL para probar features de sentiment. El esquema
(schema.py) es el contrato; las fuentes (sources/) estan en scaffolding, sin
implementar. Ver README.md.
"""
from __future__ import annotations

from irfn.data.sentiment.orchestrator import available_sources, ingest, source_available
from irfn.data.sentiment.schema import (
    SENTIMENT_COLUMNS,
    SENTIMENT_MAX,
    SENTIMENT_MIN,
    SentimentRecord,
    empty_frame,
    validate_frame,
)

__all__ = [
    "SENTIMENT_COLUMNS",
    "SENTIMENT_MIN",
    "SENTIMENT_MAX",
    "SentimentRecord",
    "empty_frame",
    "validate_frame",
    "ingest",
    "available_sources",
    "source_available",
]

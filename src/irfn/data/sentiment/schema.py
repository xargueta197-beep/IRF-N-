"""Esquema comun de la ingesta multi-fuente de sentiment/noticias (Fase 2, Data).

MODULO EXPERIMENTAL / TEMPORAL: sirve para PROBAR features de sentiment/noticias
de varias fuentes bajo un unico contrato de salida. TODAS las fuentes (GDELT,
Finnhub, APITube, CryptoPanic, ...) deben devolver un DataFrame con EXACTAMENTE
estas columnas y estos tipos. El contrato manda: si una fuente no puede llenar
una columna, va NaN/None, nunca un valor inventado (R7).

Contrato de columnas (fijado por el director):
  fecha            datetime UTC, tz-aware (se serializa ISO 8601)
  fuente           str    nombre de la fuente ("gdelt", "finnhub", ...)
  ticker_o_moneda  str    activo referido (ticker de accion o simbolo cripto); None si general
  titulo           str    titular
  texto_resumen    str    resumen/cuerpo corto; None si la fuente no lo da
  sentiment_score  float  sentimiento en [-1, 1]; NaN si la fuente NO lo provee
  url              str    enlace a la nota original

Este esquema NO alimenta el modelo por si mismo: es una zona de PRUEBAS. Cualquier
feature derivada que llegue al pipeline debe pasar despues por la disciplina de
PIT/ablacion (R3/R8) como el resto.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

# El ORDEN y los NOMBRES son parte del contrato: un consumidor puede confiar en
# estas columnas exactas. No agregar columnas por fuente (ver validate_frame).
SENTIMENT_COLUMNS: list[str] = [
    "fecha",
    "fuente",
    "ticker_o_moneda",
    "titulo",
    "texto_resumen",
    "sentiment_score",
    "url",
]

# Rango canonico del score. Una fuente que puntue en otra escala DEBE re-escalar
# a [-1, 1] en su adaptador antes de devolver el DataFrame (no aqui).
SENTIMENT_MIN = -1.0
SENTIMENT_MAX = 1.0


class SentimentRecord(BaseModel):
    """Un item de sentiment/noticia normalizado (validacion por-fila del contrato).

    extra="forbid": un campo de mas (una fuente que arrastra columnas propias sin
    mapearlas) falla ruidosamente, no en silencio. Mismo criterio que el contrato
    de salida del proyecto (outputs/schema.py).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fecha: datetime
    fuente: str
    ticker_o_moneda: Optional[str] = None
    titulo: str
    texto_resumen: Optional[str] = None
    sentiment_score: Optional[float] = None
    url: str

    @field_validator("fecha")
    @classmethod
    def _fecha_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("fecha debe ser tz-aware en UTC (ISO 8601), no naive.")
        return v.astimezone(timezone.utc)

    @field_validator("sentiment_score")
    @classmethod
    def _score_en_rango(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (SENTIMENT_MIN <= v <= SENTIMENT_MAX):
            raise ValueError(f"sentiment_score {v} fuera de [{SENTIMENT_MIN}, {SENTIMENT_MAX}].")
        return v


def empty_frame() -> pd.DataFrame:
    """DataFrame vacio con las columnas y dtypes del contrato. Punto de partida
    honesto de una fuente sin resultados (nunca None ni un frame sin columnas)."""
    df = pd.DataFrame(
        {
            "fecha": pd.Series(dtype="datetime64[ns, UTC]"),
            "fuente": pd.Series(dtype="object"),
            "ticker_o_moneda": pd.Series(dtype="object"),
            "titulo": pd.Series(dtype="object"),
            "texto_resumen": pd.Series(dtype="object"),
            "sentiment_score": pd.Series(dtype="float64"),
            "url": pd.Series(dtype="object"),
        }
    )
    return df[SENTIMENT_COLUMNS]


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Valida y normaliza el DataFrame de una fuente contra el contrato.

    - Exige EXACTAMENTE las columnas de SENTIMENT_COLUMNS (ni de menos ni de mas).
    - `fecha` -> datetime UTC tz-aware.
    - `sentiment_score` -> float con NaN permitido; en [-1, 1] cuando no es NaN.
    Devuelve el frame con columnas en orden canonico e indice reseteado. Cada
    fuente concreta DEBE terminar su fetch() con `return validate_frame(df)`.
    """
    missing = [c for c in SENTIMENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"faltan columnas del contrato: {missing}")
    extra = [c for c in df.columns if c not in SENTIMENT_COLUMNS]
    if extra:
        raise ValueError(f"columnas fuera del contrato: {extra} (una fuente no agrega columnas propias)")

    out = df.copy()
    out["fecha"] = pd.to_datetime(out["fecha"], utc=True, errors="raise")
    out["sentiment_score"] = pd.to_numeric(out["sentiment_score"], errors="coerce").astype("float64")
    fuera = out["sentiment_score"].dropna()
    if ((fuera < SENTIMENT_MIN) | (fuera > SENTIMENT_MAX)).any():
        raise ValueError(f"sentiment_score fuera de [{SENTIMENT_MIN}, {SENTIMENT_MAX}] en alguna fila.")
    for col in ("fuente", "ticker_o_moneda", "titulo", "texto_resumen", "url"):
        out[col] = out[col].astype("object")
    return out[SENTIMENT_COLUMNS].reset_index(drop=True)

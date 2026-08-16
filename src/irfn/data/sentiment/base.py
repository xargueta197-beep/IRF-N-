"""Interfaz comun de una fuente de sentiment/noticias (Fase 2, MODULO EXPERIMENTAL).

Cada fuente concreta en sources/*.py hereda de SentimentSource e implementa
fetch(), que DEBE terminar con `return schema.validate_frame(df)`. Aqui solo esta
el contrato de la interfaz; ninguna fuente esta implementada todavia.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from irfn.data.sentiment import schema


class SentimentSource(ABC):
    """Fuente de items de sentiment/noticias que respeta schema.SENTIMENT_COLUMNS."""

    #: nombre que va en la columna 'fuente' del esquema
    name: str = "base"
    #: variable de entorno con la API key; None si la fuente no necesita key (GDELT)
    api_key_env: str | None = None

    @abstractmethod
    def fetch(self, *, since: date, until: date, query: str | None = None) -> pd.DataFrame:
        """Descarga items en [since, until] (UTC) y los devuelve en el esquema comun.

        Contrato de la implementacion futura:
          - mapear los campos de la fuente a SENTIMENT_COLUMNS (sentiment_score=NaN
            si la fuente no puntua; jamas inventar un score, R7),
          - `return schema.validate_frame(df)`.
        """
        raise NotImplementedError

    def empty(self) -> pd.DataFrame:
        """Frame vacio del esquema (fuente sin resultados en el rango)."""
        return schema.empty_frame()

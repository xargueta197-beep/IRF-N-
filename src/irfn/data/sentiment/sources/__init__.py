"""Registro de fuentes del modulo sentiment.

SOURCES mapea nombre -> clase para que el orquestador futuro itere las fuentes de
forma uniforme. Cada clase respeta la interfaz base.SentimentSource y devuelve el
esquema comun (schema.validate_frame).

Estado: las 4 fuentes IMPLEMENTADAS. `finnhub` (company-news + noticias cripto,
FINNHUB_API_KEY, key activa) y `gdelt` (fetch_gdelt, DOC 2.0, sin key) corren en
vivo; `cryptopanic` (votos->score) y `apitube` (sentiment.overall) tienen el
codigo listo y testeado pero requieren su key (dormidas hasta tenerla).
"""
from __future__ import annotations

from irfn.data.sentiment.base import SentimentSource
from irfn.data.sentiment.sources.apitube import ApitubeSource
from irfn.data.sentiment.sources.cryptopanic import CryptopanicSource
from irfn.data.sentiment.sources.finnhub import FinnhubSource
from irfn.data.sentiment.sources.gdelt import GdeltSource

SOURCES: dict[str, type[SentimentSource]] = {
    GdeltSource.name: GdeltSource,
    FinnhubSource.name: FinnhubSource,
    ApitubeSource.name: ApitubeSource,
    CryptopanicSource.name: CryptopanicSource,
}

__all__ = ["SOURCES", "GdeltSource", "FinnhubSource", "ApitubeSource", "CryptopanicSource"]

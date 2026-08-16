# Modulo `sentiment` — ingesta multi-fuente de sentiment/noticias (Fase 2)

**Estado: EXPERIMENTAL / TEMPORAL. Solo esquema + scaffolding; ninguna fuente
implementada todavia.** Zona de pruebas para features de sentiment/noticias; nada
de aqui alimenta el modelo sin pasar despues por la disciplina PIT/ablacion (R3/R8).

## Contrato de salida (`schema.py`)

Toda fuente devuelve un `DataFrame` con EXACTAMENTE estas columnas:

| columna | tipo | nota |
| :-- | :-- | :-- |
| `fecha` | datetime UTC (tz-aware) | ISO 8601 al serializar |
| `fuente` | str | `"gdelt"`, `"finnhub"`, ... |
| `ticker_o_moneda` | str o None | ticker de accion o simbolo cripto; None si general |
| `titulo` | str | titular |
| `texto_resumen` | str o None | resumen; None si la fuente no lo da |
| `sentiment_score` | float | en `[-1, 1]`; **NaN si la fuente no lo provee** (R7) |
| `url` | str | enlace original |

Helpers: `empty_frame()`, `validate_frame(df)`, modelo por-fila `SentimentRecord`.
`validate_frame` exige las columnas exactas (ni de menos ni de mas), fuerza
`fecha` a UTC y `sentiment_score` a float con NaN permitido.

## Fuentes (scaffolding en `sources/`)

| fuente | key (`.env`) | timestamp | sentiment propio | estado |
| :-- | :-- | :-- | :-- | :-- |
| GDELT | — (sin key) | 15 min (grueso) | no (NaN) | **implementada** (`fetch_gdelt`, DOC 2.0) |
| Finnhub | `FINNHUB_API_KEY` | unix seg (fino) | no en company-news (NaN) | **implementada** |
| APITube | `APITUBE_API_KEY` | ISO 8601 (fino) | `sentiment.overall` -> `[-1,1]` | **implementada** (`fetch_apitube`) |
| CryptoPanic | `CRYPTOPANIC_API_KEY` | ISO 8601 (fino) | votos -> `[-1,1]` | **implementada** (`fetch_cryptopanic`, línea BTC) |

Cada fuente hereda `base.SentimentSource` e implementa `fetch(since, until, query)`
terminando en `return schema.validate_frame(df)`. Finnhub (company-news) ya esta
implementada y testeada (`tests/test_sentiment.py`); las demas lanzan
`NotImplementedError`.

## Estructura

```
sentiment/
  schema.py            contrato comun (columnas, validacion)
  base.py              interfaz SentimentSource (ABC)
  orchestrator.py      ingest(): une fuentes, dedup por url, guarda consolidado
  sources/
    __init__.py        registro SOURCES: nombre -> clase
    gdelt.py           fetch_gdelt (DOC 2.0, sin key)
    finnhub.py         company-news + fetch_finnhub_market_news (crypto)
    apitube.py         fetch_apitube (con sentiment)
    cryptopanic.py     fetch_cryptopanic (dormida: token de pago)
  README.md
```

## Orquestador

```python
from irfn.data.sentiment import ingest, available_sources
# query es POR FUENTE: simbolo (finnhub), keyword (gdelt/apitube), moneda (cryptopanic)
df = ingest(since, until, {"apitube": "bitcoin", "finnhub": "SPY", "gdelt": "bitcoin"})
```
Corre cada fuente disponible de forma aislada, une en el esquema comun, deduplica
por `url`, ordena por fecha y guarda `data/raw/sentiment/consolidado__*.parquet`.

## Estado

Modulo completo y verificado en vivo (543 items consolidados de 3 fuentes).
`tests/test_sentiment.py`: 23 passed. Nada de aqui alimenta el modelo sin pasar
antes por PIT/ablacion (R3/R8).

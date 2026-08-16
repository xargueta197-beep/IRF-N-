"""Ingesta de precios OHLCV: BTC via Binance, SPY/GLD/IEF/DXY via yfinance.

`load_returns` devuelve log-retornos en escala porcentual (log-ret * 100),
indexados por fecha, que es la convencion que usa todo el nucleo (ver
demo_v0.py). Ver reports/data_audit.md para el estado auditado de cada fuente.
"""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

# Cache de precios crudos (R-reproducibilidad V1): yfinance revisa los precios
# ajustados entre llamadas, asi que dos descargas del mismo ticker dan retornos
# ligeramente distintos (documentado en CLAUDE.md ESTADO ACTUAL). Para que una
# corrida de V1 sea reproducible bit a bit se cachea el cierre descargado la
# PRIMERA vez y se reutiliza despues; borrar el archivo fuerza una descarga
# fresca. El cache vive en data/raw/ (inmutable por convencion del repo).
_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

_BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
_BINANCE_PAGE_LIMIT = 1000


class BinanceRateLimited(Exception):
    """Peticion de klines rechazada por limite de tasa (HTTP 429/418), reintentable."""


def _fetch_binance_page(symbol: str, start_ms: int, end_ms: int, timeout: int = 30) -> list[list]:
    import requests

    params = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": _BINANCE_PAGE_LIMIT,
    }
    resp = requests.get(_BINANCE_KLINES_URL, params=params, timeout=timeout)
    if resp.status_code in (429, 418):
        raise BinanceRateLimited(resp.text[:200])
    if resp.status_code != 200:
        raise RuntimeError(f"Binance devolvio HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def _fetch_binance_page_with_retry(
    symbol: str, start_ms: int, end_ms: int, *, backoff_seconds: float = 5.0, max_retries: int = 5
) -> list[list]:
    """Reintenta LA MISMA peticion ante rechazo por tasa.

    Mismo patron que el fetcher de GDELT (src/irfn/data/headlines.py): la IP de
    esta maquina es compartida, asi que un 429/418 puntual no implica que el
    limite siga vigente unos segundos despues -- se reintenta la peticion en
    vez de abortar la descarga completa.
    """
    for attempt in range(max_retries + 1):
        try:
            return _fetch_binance_page(symbol, start_ms, end_ms)
        except BinanceRateLimited:
            if attempt >= max_retries:
                raise
            _time.sleep(backoff_seconds)
    return []  # inalcanzable: el loop siempre retorna o lanza


def _load_close_binance(symbol: str, start: str) -> pd.Series:
    """Cierre diario desde los klines publicos de Binance (sin API key).

    Paginado en bloques de 1000 velas (limite de la API) desde `start` hasta
    ahora. Ver reports/data_audit.md seccion 1 para la auditoria de este
    endpoint (BTCUSDT disponible desde 2017-08-17).
    """
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        page = _fetch_binance_page_with_retry(symbol, cursor, end_ms)
        if not page:
            break
        rows.extend(page)
        next_cursor = int(page[-1][6]) + 1  # close_time (ms) del ultimo kline + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(page) < _BINANCE_PAGE_LIMIT:
            break
        _time.sleep(0.2)  # cortesia entre paginas, no solo ante rechazo

    if not rows:
        raise RuntimeError(f"Binance no devolvio klines para {symbol!r} desde {start!r}.")

    idx = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True).tz_localize(None)
    close = pd.Series([float(r[4]) for r in rows], index=idx, name="close")
    close.index.name = "fecha"
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close


def load_close(
    symbol: str, start: str, *, source: str = "yfinance", use_cache: bool = True
) -> pd.Series:
    """Cierre ajustado indexado por fecha, con cache en disco (ver nota arriba).

    A diferencia de load_returns (que devuelve log-retornos), aqui se necesita el
    NIVEL de precio para las features tecnicas: SMA200, Bollinger y sus z-scores
    se calculan sobre el cierre. El caching hace reproducible el vintage de
    precios entre corridas de V1 (CLAUDE.md pide cachearlo).
    """
    cache = _RAW_DIR / f"close_{symbol}_{start}.parquet"
    if use_cache and cache.exists():
        s = pd.read_parquet(cache)["close"]
        s.index = pd.to_datetime(s.index)
        s.name = "close"
        return s

    if source == "binance":
        close = _load_close_binance(symbol, start)
    elif source == "yfinance":
        import yfinance as yf

        df = yf.download(symbol, start=start, auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise RuntimeError(
                f"yfinance no devolvio datos para {symbol!r} (ver reports/data_audit.md)."
            )
        close = df["Close"].squeeze().astype(float)
        close.name = "close"
        close.index.name = "fecha"
    else:
        raise NotImplementedError(
            f"source={source!r} no implementada; solo yfinance/binance (ver data_audit.md)."
        )

    if use_cache:
        _RAW_DIR.mkdir(parents=True, exist_ok=True)
        close.to_frame().to_parquet(cache)
    return close


def load_returns(symbol: str, start: str, *, source: str = "yfinance") -> pd.Series:
    """Descarga precios y devuelve log-retornos en escala % indexados por fecha.

    Parametros
    ----------
    symbol : ticker (p.ej. "SPY") o par (p.ej. "BTCUSDT") segun `source`.
    start  : fecha inicial "YYYY-MM-DD".
    source : "yfinance" o "binance".
    """
    if source == "binance":
        close = _load_close_binance(symbol, start)
    elif source == "yfinance":
        import yfinance as yf

        df = yf.download(symbol, start=start, auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise RuntimeError(
                f"yfinance no devolvio datos para {symbol!r} (ver reports/data_audit.md)."
            )
        close = df["Close"].squeeze()
    else:
        raise NotImplementedError(
            f"source={source!r} no implementada; solo yfinance/binance (ver data_audit.md)."
        )

    log_ret = np.log(close).diff().dropna() * 100.0    # escala porcentual
    log_ret.name = f"{symbol}_logret_pct"
    return log_ret

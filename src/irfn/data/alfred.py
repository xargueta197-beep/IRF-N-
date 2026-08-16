"""Ingesta de series macro VINTAGE desde ALFRED (R4). Nunca FRED revisado.

Por que ALFRED y no FRED: el M2 de enero que se ve hoy en FRED no es el que se
publico en enero. FRED muestra la serie REVISADA; entrenar con ella le da al
modelo informacion que nadie tenia en su momento. ALFRED es la misma base de
datos pero con el eje temporal de la REALIDAD: cada observacion viaja con su
periodo realtime [realtime_start, realtime_end] = "este valor fue el vigente
entre estas dos fechas". realtime_start ES la fecha de publicacion real.

Este modulo hace tres cosas, todas auditables:

  1. fetch_vintage_observations: descarga TODAS las vintages de una serie
     (realtime_start=EARLIEST_REALTIME .. LATEST_REALTIME) y las cachea en
     data/vintages/<serie>.parquet. El cache es ademas el registro reproducible
     de que datos vio el modelo (la descarga en vivo cambia; el parquet no).

  2. point_in_time_series: reconstruye la serie "tal como se conocia" dia a dia:
     el valor asignado a la fecha t es el de la observacion MAS RECIENTE cuyo
     realtime_start <= t - margen. El margen (config macro.availability_margin_
     days) existe porque el vintage no trae hora de publicacion: asumir que un
     dato publicado el dia t ya era operable el mismo dia t seria optimista.
     El rezago resultante NO es un shift(1) arbitrario: es el lag de publicacion
     REAL de cada observacion, distinto por serie y por fecha.

  3. vintage_ledger_entry: resume, por serie, cuantas observaciones y vintages
     hay y la distribucion del lag de publicacion. La pantalla 6 (auditoria)
     muestra este ledger: es la evidencia de que R4 se cumple con datos, no con
     intenciones.

Hallazgo de data_audit.md que este modulo debe vigilar: desde abril de 2026 FRED
limita las series ICE BofA (BAMLH0A0HYM2) a una ventana rodante de 3 anios en la
vintage ACTUAL. Las vintages HISTORICAS de ALFRED pueden conservar el historico
largo; fetch_vintage_observations pide todas y el ledger reporta la cobertura
real obtenida -- si el historico no alcanza para el walk-forward, se reporta
como bloqueante, no se rellena con FRED revisado.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("irfn.alfred")

ROOT = Path(__file__).resolve().parents[3]
VINTAGES_DIR = ROOT / "data" / "vintages"
API_URL = "https://api.stlouisfed.org/fred/series/observations"

# Limites del protocolo de la API FRED/ALFRED (documentados en su spec, no son
# decisiones de modelado): maximo de filas por pagina y sentinelas del eje
# realtime que significan "todas las vintages".
_PAGE_LIMIT = 100000
EARLIEST_REALTIME = "1776-07-04"
LATEST_REALTIME = "9999-12-31"

# Chunking de respaldo cuando el rango completo excede el cap de vintage-dates
# de FRED (ver VintageCapExceeded): 4 anios da margen bajo el limite duro de
# 2000 aunque CADA dia habil sea un vintage nuevo (series diarias sin revision
# como DGS10/DGS2, ~1050 dias habiles en 4 anios). _CHUNK_FLOOR_YEAR es
# anterior a cualquier serie FRED conocida; los tramos sin datos simplemente
# devuelven count=0 y se descartan, no son un error.
_VINTAGE_CAP_CHUNK_YEARS = 4
_CHUNK_FLOOR_YEAR = 1900


class MissingAPIKeyError(RuntimeError):
    """No hay ALFRED_API_KEY: sin ella no hay datos macro vintage (R4)."""


class VintageCapExceeded(RuntimeError):
    """FRED rechazo la peticion: el rango realtime_start/realtime_end pedido
    tiene mas de 2000 vintage-dates distintas (limite duro del API para este
    file_type, no relacionado con el offset/limit de paginacion de filas).
    fetch_vintage_observations la atrapa y trocea el rango por anios -- no es
    un fallo de la serie ni de la key."""


def _api_key() -> str:
    import os

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = os.environ.get("ALFRED_API_KEY", "").strip()
    if not key:
        raise MissingAPIKeyError(
            "ALFRED_API_KEY no esta configurada. Registra una key gratuita en "
            "https://fredaccount.stlouisfed.org/apikeys y ponla en irfn/.env "
            "(ALFRED_API_KEY=...). R4 prohibe caer a FRED revisado como plan B."
        )
    return key


def _fetch_realtime_window(
    series_id: str, key: str, realtime_start: str, realtime_end: str, *, request_pause_s: float
) -> list[pd.DataFrame]:
    """Pagina por offset/limit TODAS las observaciones de un tramo
    realtime_start/realtime_end fijo. Lanza VintageCapExceeded si el tramo en
    si mismo excede el cap de 2000 vintage-dates (el offset/limit no ayuda ahi:
    el cap se evalua sobre el rango completo antes de paginar)."""
    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        resp = requests.get(
            API_URL,
            params={
                "series_id": series_id,
                "api_key": key,
                "file_type": "json",
                "realtime_start": realtime_start,
                "realtime_end": realtime_end,
                "limit": _PAGE_LIMIT,
                "offset": offset,
            },
            timeout=60,
        )
        if resp.status_code == 400 and "vintage dates" in resp.text.lower():
            raise VintageCapExceeded(resp.text[:300])
        if resp.status_code == 400 and "does not exist in alfred" in resp.text.lower():
            # el tramo realtime_start/realtime_end no tiene NINGUN vintage (p.ej.
            # anterior al inicio real de la serie): FRED devuelve este error en
            # vez de un count=0 vacio. Tramo genuinamente sin datos, no un fallo.
            break
        if resp.status_code != 200:
            raise RuntimeError(
                f"ALFRED devolvio HTTP {resp.status_code} para {series_id}: "
                f"{resp.text[:300]}"
            )
        payload = resp.json()
        obs = payload.get("observations", [])
        if not obs:
            break
        frames.append(pd.DataFrame(obs))
        offset += len(obs)
        if offset >= int(payload.get("count", 0)):
            break
        time.sleep(request_pause_s)
    return frames


def fetch_vintage_observations(
    series_id: str,
    *,
    cache_dir: Path = VINTAGES_DIR,
    refresh: bool = False,
    request_pause_s: float = 0.6,
) -> pd.DataFrame:
    """Todas las vintages de una serie ALFRED, cacheadas en parquet.

    Devuelve DataFrame con columnas:
      obs_date (fecha de la observacion), value (float, NaN si '.'),
      realtime_start (fecha de PUBLICACION de ese valor),
      realtime_end (ultima fecha en que ese valor fue el vigente).

    El cache en data/vintages/ es la fuente de verdad de la corrida: solo se
    golpea la red si no existe o refresh=True. request_pause_s respeta el rate
    limit del tier gratuito (120 req/min) sin acercarse a el.

    Camino rapido: se pide el rango completo (EARLIEST_REALTIME..LATEST_REALTIME)
    en una sola pasada, igual que antes -- funciona para series con pocas
    revisiones (M2SL, BAMLH0A0HYM2, etc.). Si FRED rechaza el rango completo
    por VintageCapExceeded (series diarias sin revision como DGS10/DGS2, donde
    cada dia habil es su propio vintage y el total supera 2000), se cae al
    camino de respaldo: trocear por ventanas de _VINTAGE_CAP_CHUNK_YEARS anios
    desde _CHUNK_FLOOR_YEAR hasta hoy y concatenar. Los tramos sin datos
    (anteriores al inicio real de la serie) devuelven count=0 y se descartan
    silenciosamente, no son un error.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{series_id}.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    key = _api_key()
    try:
        frames = _fetch_realtime_window(
            series_id, key, EARLIEST_REALTIME, LATEST_REALTIME, request_pause_s=request_pause_s
        )
    except VintageCapExceeded as exc:
        logger.info(
            "ALFRED %s: rango completo excede el cap de vintage-dates (%s); "
            "troceando por anios de %d en %d.",
            series_id, exc, _CHUNK_FLOOR_YEAR, _VINTAGE_CAP_CHUNK_YEARS,
        )
        frames = []
        today = pd.Timestamp.now().normalize()
        for start_year in range(_CHUNK_FLOOR_YEAR, today.year + 1, _VINTAGE_CAP_CHUNK_YEARS):
            chunk_start = pd.Timestamp(f"{start_year}-01-01")
            if chunk_start > today:
                break
            # realtime_end no puede superar hoy (salvo que sea el sentinela
            # 9999-12-31): el ultimo tramo se recorta a la fecha real de hoy.
            chunk_end = min(pd.Timestamp(f"{start_year + _VINTAGE_CAP_CHUNK_YEARS}-01-01"), today)
            chunk = _fetch_realtime_window(
                series_id, key,
                chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"),
                request_pause_s=request_pause_s,
            )
            frames.extend(chunk)
            time.sleep(request_pause_s)

    if not frames:
        raise RuntimeError(f"ALFRED no devolvio observaciones para {series_id}.")

    df = pd.concat(frames, ignore_index=True)
    df = pd.DataFrame(
        {
            "obs_date": pd.to_datetime(df["date"]),
            "value": pd.to_numeric(df["value"], errors="coerce"),  # '.' -> NaN
            "realtime_start": pd.to_datetime(df["realtime_start"]),
            "realtime_end": pd.to_datetime(df["realtime_end"]),
        }
    ).dropna(subset=["value"])
    df = df.drop_duplicates(subset=["obs_date", "realtime_start"])
    df = df.sort_values(["realtime_start", "obs_date"]).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    logger.info("ALFRED %s: %d filas vintage cacheadas en %s", series_id, len(df), cache_path)
    return df


def point_in_time_series(
    vintages: pd.DataFrame,
    *,
    margin_days: int,
    name: str,
) -> pd.Series:
    """Serie point-in-time: el valor en la fecha t es el de la observacion mas
    reciente PUBLICADA en o antes de t - margin_days habiles.

    Implementacion: se recorren los eventos de publicacion (realtime_start) en
    orden; se mantiene el "ultimo valor conocido" = valor vigente de la
    observacion con obs_date maximo publicado hasta ese momento (una REVISION de
    esa misma observacion actualiza el valor; una observacion nueva lo avanza;
    una revision de una observacion vieja no toca el ultimo). El resultado se
    indexa por la fecha DESDE la cual el valor es usable: realtime_start
    desplazado margin_days dias habiles hacia adelante.

    Devuelve una Series indexada por fecha de disponibilidad (creciente, sin
    duplicados: si varios eventos caen el mismo dia gana el ultimo estado).
    El consumidor alinea a su calendario con reindex+ffill: el ffill es
    legitimo porque "no hubo publicacion nueva" significa que el ultimo valor
    conocido SIGUE siendo el conocido.
    """
    ev = vintages.sort_values(["realtime_start", "obs_date"])
    last_obs = pd.Timestamp.min
    last_val = np.nan
    dates: list[pd.Timestamp] = []
    vals: list[float] = []
    for pub, obs, val in zip(ev["realtime_start"], ev["obs_date"], ev["value"]):
        if obs >= last_obs:
            last_obs, last_val = obs, float(val)
            dates.append(pub)
            vals.append(last_val)

    s = pd.Series(vals, index=pd.DatetimeIndex(dates), name=name)
    if margin_days > 0:
        # margen de disponibilidad: publicado en t => usable desde t + margen
        # (dias habiles). Es el "shift" honesto de la capa macro (lag_ledger).
        s.index = s.index + pd.tseries.offsets.BusinessDay(margin_days)
    # La dedup va DESPUES del margen: BusinessDay colapsa publicaciones de fin
    # de semana sobre el mismo dia habil (sab y dom + 1 habil = lunes), asi que
    # deduplicar antes dejaria duplicados en el indice de disponibilidad y el
    # reindex del consumidor (features/macro.py) revienta. El offset preserva el
    # orden de publicacion, de modo que keep="last" sigue significando "gana el
    # ultimo estado publicado".
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def vintage_ledger_entry(series_id: str, vintages: pd.DataFrame, margin_days: int) -> dict:
    """Resumen auditable de la cobertura vintage de una serie (pantalla 6)."""
    first_release = vintages.groupby("obs_date")["realtime_start"].min()
    lag = (first_release - first_release.index).dt.days
    return {
        "series_id": series_id,
        "n_obs": int(vintages["obs_date"].nunique()),
        "n_vintage_rows": int(len(vintages)),
        "first_obs": str(vintages["obs_date"].min().date()),
        "last_obs": str(vintages["obs_date"].max().date()),
        "first_pub": str(vintages["realtime_start"].min().date()),
        "last_pub": str(vintages["realtime_start"].max().date()),
        "pub_lag_days_median": float(lag.median()),
        "pub_lag_days_p95": float(lag.quantile(0.95)),
        "pub_lag_days_max": int(lag.max()),
        "availability_margin_days": int(margin_days),
        "note": (
            "valor usable en t = ultima observacion publicada (realtime_start) "
            "en o antes de t menos el margen; rezago = lag de publicacion REAL."
        ),
    }

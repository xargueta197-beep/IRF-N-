"""Calendario macro: releases e indicadores de consenso previo al release.

Fuente y por que (ver reports/data_audit.md, secciones 4 y 7, auditadas en vivo
2026-07-11/12): la unica fuente que resuelve correctamente la Trampa 2 (consenso
PRE-RELEASE, no reescrito ex-post) es Trading Economics, endpoint
`economic_calendar/point-in-time`, de pago sin excepcion (demo publico
`guest:guest` verificado 410 Gone). Forex Factory fue investigado y descartado:
sus terminos de servicio (`forexfactory.com/notices`, leidos en vivo) prohiben
explicitamente tanto el acceso automatizado fuera de su interfaz como la
redistribucion de su historico compilado ("FEED") -- exactamente lo que este
modulo haria al guardar snapshots. Investing.com ya estaba descartado por la
misma razon de ToS. Econoday/FXStreet quedan como "no verificado", pendientes de
una sesion dedicada si Trading Economics de pago no es viable.

Este modulo hace DOS cosas, deliberadamente separadas:

  1. load_local_snapshots: parsea los snapshots DIARIOS que scripts/capture_
     consensus.py ya viene guardando en data/raw/consensus_calendar/ desde la
     Sesion 0 (endpoint regular `/calendar`, gateado por TRADING_ECONOMICS_
     API_KEY -- "guest:guest" muerto hasta que se configure una key real). Esta
     es la via de acumulacion HACIA ADELANTE: cada dia que pasa sin capturar es
     historico perdido para siempre (docstring de capture_consensus.py). Es
     point-in-time POR CONSTRUCCION: para cada release, el consenso que se
     guarda es el de la captura MAS TEMPRANA que lo vio, nunca el de una
     captura posterior (que podria reflejar una revision, no el consenso
     original).

  2. fetch_point_in_time_range: fetcher REAL contra el endpoint pagado
     `economic_calendar/point-in-time`, para cuando exista TRADING_ECONOMICS_
     API_KEY con un plan que lo incluya (backfill de historico ANTERIOR a que
     empezara la captura diaria). El formato exacto de la respuesta paga NO
     esta verificado en vivo esta sesion (sin acceso pagado): la implementacion
     sigue las convenciones documentadas/publicas del resto de la API de
     Trading Economics y esta escrita para ser facil de ajustar en cuanto haya
     una respuesta real que inspeccionar. No se inventa un formato y se declara
     verificado cuando no lo esta.

REGLA (config/news.yaml): sin snapshots validos, `load_local_snapshots` devuelve
un calendario VACIO (columnas presentes, cero filas). No se imputa, no se
inventa, no se rellena. `surprise_start_date` en config/news.yaml documenta
desde cuando hay datos reales; hoy es null.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger("irfn.calendar")

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_DIR = ROOT / "data" / "raw" / "consensus_calendar"

CALENDAR_COLUMNS = ["indicator", "actual", "consensus", "previous", "unit", "fecha_evento", "captured_at"]

# Endpoint de point-in-time (pago). Mismo host que capture_consensus.py.
POINT_IN_TIME_URL = "https://api.tradingeconomics.com/economic_calendar/point-in-time"


class MissingConsensusAPIKeyError(RuntimeError):
    """No hay TRADING_ECONOMICS_API_KEY: sin ella no hay point-in-time pagado."""


def _api_key() -> str | None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = os.environ.get("TRADING_ECONOMICS_API_KEY", "").strip()
    return key or None


# --------------------------------------------------------------------------- #
# Mapeo indicador -> palabras clave de categoria/evento de Trading Economics.
# --------------------------------------------------------------------------- #
# MEJOR ESFUERZO, NO VERIFICADO EN VIVO (sin acceso pagado esta sesion): los
# nombres de "Category"/"Event" que usa la API real de TE para estos 6
# indicadores en Estados Unidos, segun la nomenclatura publica que usa TE en su
# calendario web (economic_calendar/... es del mismo proveedor). Coincidencia
# por subcadena, insensible a mayusculas. Ajustar aqui en cuanto se inspeccione
# una respuesta real -- este diccionario es el UNICO lugar que hace falta tocar.
INDICATOR_KEYWORDS: dict[str, list[str]] = {
    "CPI": ["inflation rate", "cpi"],
    "NFP": ["non farm payrolls", "nonfarm payrolls", "non-farm payrolls"],
    "FOMC": ["fed interest rate decision", "fomc", "federal funds"],
    "PMI": ["ism manufacturing pmi", "manufacturing pmi", "ism pmi"],
    "RETAIL_SALES": ["retail sales"],
    "UNEMPLOYMENT": ["unemployment rate"],
}

US_COUNTRY_NAMES = {"united states", "usa", "us"}


def _target_indicators(indicators: list[str] | None) -> list[str]:
    return list(INDICATOR_KEYWORDS) if indicators is None else list(indicators)


def _match_indicator(category: str, event: str) -> str | None:
    text = f"{category} {event}".lower()
    for name, keywords in INDICATOR_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return name
    return None


def _get_ci(d: dict, *keys: str):
    """Lookup insensible a mayusculas/orden de claves (formato TE no verificado)."""
    lower = {k.lower(): v for k, v in d.items()}
    for key in keys:
        if key.lower() in lower:
            return lower[key.lower()]
    return None


_NUMERIC_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def _to_number(raw) -> float:
    """Parsea valores del calendario ('3.8%', '150K', '1,950B', 21.0) a float.

    Devuelve NaN si no se puede parsear -- eso es "sin dato", nunca se
    substituye por 0 (0 seria afirmar 'sorpresa nula' sin evidencia).
    """
    if raw is None:
        return float("nan")
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().replace(",", "")
    if not s or s in {".", "-"}:
        return float("nan")
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*([KMBT%]?)$", s, re.IGNORECASE)
    if not m:
        return float("nan")
    val = float(m.group(1))
    suffix = m.group(2).upper()
    if suffix in _NUMERIC_SUFFIX:
        val *= _NUMERIC_SUFFIX[suffix]
    return val


def _events_from_raw(data) -> list[dict]:
    """Normaliza el payload crudo de la API TE a una lista de dicts de evento.

    Formato no verificado en vivo: TE documenta que la respuesta es una lista
    JSON de eventos; algunos wrappers de terceros la envuelven en {"data": [...]}
    o {"result": [...]}. Se aceptan ambas formas defensivamente.
    """
    if data is None:
        return []
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    if isinstance(data, dict):
        for key in ("data", "result", "calendar"):
            v = data.get(key)
            if isinstance(v, list):
                return [e for e in v if isinstance(e, dict)]
    return []


def _parse_event(ev: dict, indicators: list[str]) -> dict | None:
    country = str(_get_ci(ev, "Country") or "").strip().lower()
    if country and country not in US_COUNTRY_NAMES:
        return None
    category = str(_get_ci(ev, "Category") or "")
    event_name = str(_get_ci(ev, "Event") or "")
    indicator = _match_indicator(category, event_name)
    if indicator is None or indicator not in indicators:
        return None

    date_raw = _get_ci(ev, "Date", "DateUtc", "date")
    if not date_raw:
        return None
    ts = pd.to_datetime(date_raw, utc=True, errors="coerce")
    if pd.isna(ts):
        return None

    return {
        "indicator": indicator,
        "actual": _to_number(_get_ci(ev, "Actual")),
        "consensus": _to_number(_get_ci(ev, "Forecast", "TEForecast", "Consensus")),
        "previous": _to_number(_get_ci(ev, "Previous")),
        "unit": _get_ci(ev, "Unit") or "",
        "hora_evento": ts,
    }


def _empty_calendar_frame() -> pd.DataFrame:
    df = pd.DataFrame(columns=CALENDAR_COLUMNS)
    df.index = pd.DatetimeIndex([], tz="UTC", name="hora_evento")
    return df


# --------------------------------------------------------------------------- #
# (1) Snapshots locales diarios (capture_consensus.py) -- la via real hoy.
# --------------------------------------------------------------------------- #
def load_local_snapshots(
    indicators: list[str] | None = None,
    *,
    snapshot_dir: Path = SNAPSHOT_DIR,
) -> pd.DataFrame:
    """Calendario point-in-time construido a partir de los snapshots diarios.

    Recorre TODOS los archivos *.json de `snapshot_dir` (uno por dia, escritos
    por scripts/capture_consensus.py). Para cada release identificado por
    (indicador, hora_evento), el consenso/actual/previo que se guarda es el de
    la captura MAS TEMPRANA que lo vio -- nunca el de una captura posterior,
    que podria reflejar una revision y violaria la honestidad point-in-time.

    Snapshots con "result.ok" == False (el caso de hoy: 410 Gone del demo
    muerto) se saltan sin lanzar: son el registro de un hueco, no un error.

    Devuelve DataFrame con DatetimeIndex (hora_evento, UTC) y columnas
    ['indicator','actual','consensus','previous','unit','fecha_evento',
    'captured_at']. Vacio (0 filas, columnas presentes) si no hay ni un
    snapshot valido -- es la condicion de hoy.
    """
    indicators = _target_indicators(indicators)
    if not snapshot_dir.exists():
        return _empty_calendar_frame()

    rows: list[dict] = []
    n_files, n_ok = 0, 0
    for path in sorted(snapshot_dir.glob("*.json")):
        n_files += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("snapshot ilegible %s: %s", path.name, exc)
            continue
        result = payload.get("result", {})
        if not result.get("ok"):
            continue
        n_ok += 1
        captured_at = payload.get("captured_at", path.stem)
        for ev in _events_from_raw(result.get("data")):
            row = _parse_event(ev, indicators)
            if row is None:
                continue
            row["captured_at"] = captured_at
            rows.append(row)

    logger.info("snapshots: %d archivos, %d con result.ok=True, %d eventos de interes.",
                n_files, n_ok, len(rows))
    if not rows:
        return _empty_calendar_frame()

    df = pd.DataFrame(rows).sort_values("captured_at")

    def _first_non_null(s: pd.Series):
        valid = s.dropna()
        return valid.iloc[0] if len(valid) else float("nan")

    grouped = (
        df.groupby(["indicator", "hora_evento"], as_index=False)
        .agg(
            actual=("actual", _first_non_null),
            consensus=("consensus", _first_non_null),
            previous=("previous", _first_non_null),
            unit=("unit", "first"),
            captured_at=("captured_at", "first"),
        )
    )
    grouped["hora_evento"] = pd.to_datetime(grouped["hora_evento"], utc=True)
    grouped["fecha_evento"] = grouped["hora_evento"].dt.date.astype(str)
    grouped = grouped.set_index("hora_evento").sort_index()
    return grouped[["indicator", "actual", "consensus", "previous", "unit", "fecha_evento", "captured_at"]]


def coverage_summary(calendar: pd.DataFrame, indicators: list[str] | None = None) -> dict:
    """Resumen honesto de cobertura por indicador (pantalla 6 / reporte): cuantos
    releases hay, cuantos con consenso, y la fecha del primero -- sin esto, un
    calendario vacio es indistinguible de "no lo revise"."""
    indicators = _target_indicators(indicators)
    out: dict[str, dict] = {}
    for ind in indicators:
        sub = calendar[calendar["indicator"] == ind] if len(calendar) else calendar
        n = int(len(sub))
        n_cons = int(sub["consensus"].notna().sum()) if n else 0
        out[ind] = {
            "n_releases": n,
            "n_with_consensus": n_cons,
            "first_release": (str(sub.index.min().date()) if n else None),
            "last_release": (str(sub.index.max().date()) if n else None),
        }
    return out


# --------------------------------------------------------------------------- #
# (2) Point-in-time pagado -- backfill de historico, cuando haya key.
# --------------------------------------------------------------------------- #
def fetch_point_in_time_range(
    *,
    country: str = "united states",
    d1: str,
    d2: str,
    api_key: str | None = None,
    timeout: int = 60,
) -> dict:
    """Trae el calendario point-in-time (consenso PRE-revision) para [d1, d2].

    Gateado por TRADING_ECONOMICS_API_KEY: sin ella lanza MissingConsensusAPIKeyError
    de inmediato, sin golpear la red (mismo patron que data/alfred.py para ALFRED_
    API_KEY, R4). El formato de parametros (country/d1/d2/c) sigue la convencion
    publica del resto de la API de Trading Economics; el endpoint point-in-time en
    si NO se probo en vivo esta sesion (data_audit.md: requiere plan pago). Ajustar
    en cuanto se inspeccione una respuesta real.
    """
    import requests

    key = api_key or _api_key()
    if not key:
        raise MissingConsensusAPIKeyError(
            "TRADING_ECONOMICS_API_KEY no esta configurada. El demo publico "
            "'guest:guest' esta muerto (410 Gone, ver reports/data_audit.md); "
            "hace falta una key de pago real en irfn/.env."
        )
    response = requests.get(
        f"{POINT_IN_TIME_URL}/country/{country}/{d1}/{d2}",
        params={"c": key, "f": "json"},
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Trading Economics point-in-time devolvio HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )
    return {"ok": True, "status_code": response.status_code, "data": response.json()}

"""Ingesta de titulares (GDELT DOC 2.0 API) con TIMESTAMPS AUDITADOS.

TRAMPA 3 (guia 7.2): la hora de publicacion del feed NO es la hora del evento.
Un titular sobre el FOMC fechado "el dia" cuando el FOMC fue a las 14:00 usaria
informacion de la tarde para explicar la manana. Por eso este modulo:

  1. guarda AMBAS horas por titular: `hora_titular` (seendate del feed, UTC) y
     `hora_evento` (la hora del release macro al que el titular se refiere,
     cuando se puede emparejar contra el calendario de consenso; NaT si no);
  2. reporta la distribucion de (hora_titular - hora_evento). Masa en NEGATIVO
     (titulares "anteriores" al evento que reportan) significa timestamps
     corruptos y se marca en ROJO en pantalla 6 -- hay un problema y se dice;
  3. reporta ademas la RESOLUCION de los timestamps del feed (fraccion de
     titulares con hora exactamente 00:00:00): si el feed diera solo fechas,
     cualquier alineacion intradia seria ficcion, y eso tambien se dice.

Snapshots inmutables (mismo patron que data/raw/consensus_calendar/): un JSON
por dia en data/raw/headlines/, escrito una sola vez. La captura es reanudable
(los dias ya guardados se saltan) porque el API limita a 1 peticion cada 5 s y
un backfill largo puede interrumpirse.

CENSURA DOCUMENTADA, NO SILENCIOSA: el DOC API devuelve a lo sumo `max_records`
(250) articulos por consulta. Si un tramo devuelve exactamente el cap, se parte
en dos y se re-consulta (hasta tramos de `min_window_hours`); si aun asi se
satura, el snapshot registra `cap_hit=True` en ese tramo. Los dias censurados
subestiman la intensidad y el orquestador los cuenta y los reporta -- la
censura sesga lambda_N a la baja en los picos, que es exactamente donde mas
importa, asi que ocultarla seria maquillar el indicador de fragilidad.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger("irfn.headlines")

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_DIR = ROOT / "data" / "raw" / "headlines"

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

HEADLINE_COLUMNS = ["hora_titular", "title", "url", "domain", "sourcecountry", "cap_hit_segment"]

# Un dia que agota los reintentos por tasa NO aborta la corrida entera: se salta
# (queda sin snapshot, asi que la proxima corrida lo reintenta -- reanudable) y
# se sigue. Solo si MUCHOS dias SEGUIDOS chocan con el muro se aborta, porque
# entonces la salida a internet esta genuinamente amurallada y seguir solo quema
# tiempo. Medido en vivo (2026-07-15): con IP compartida el rate-limit es una
# loteria, no un bloqueo permanente -- un `break` al primer dia desafortunado
# tiraba a la basura toda una corrida reanudable de horas.
_MAX_CONSEC_RATE_LIMIT_DAYS = 5


class GdeltRateLimited(RuntimeError):
    """El API devolvio el mensaje de limite de tasa (texto plano, no JSON)."""


class GdeltQueryError(RuntimeError):
    """El API rechazo la CONSULTA (sintaxis, frase corta, etc.). No es
    reintentable: fallara igual para todos los dias hasta arreglar la query."""


def _base_url() -> str:
    """Permite GDELT_BASE_URL de .env como proxy propio (ver .env.example)."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return os.environ.get("GDELT_BASE_URL", "").strip() or GDELT_DOC_URL


# --------------------------------------------------------------------------- #
# Tolerancia a escapes JSON invalidos del feed
# --------------------------------------------------------------------------- #
# Secuencias que un backslash inicia legalmente en JSON (RFC 8259).
_JSON_VALID_ESCAPE = set('"\\/bfnrtu')


def _sanitize_json_escapes(text: str) -> str:
    """Repara escapes invalidos que GDELT emite de vez en cuando (medido en vivo:
    2026-03-18 fallaba con 'Invalid \\escape' por un backslash crudo en una URL/
    titulo). Duplica todo '\\' que NO inicie un escape JSON valido -- asi json lo
    lee como backslash literal -- respetando los escapes ya validos, incluido el
    par '\\\\'. O(n) y solo actua sobre backslashes, que en JSON solo aparecen
    dentro de cadenas, asi que no puede corromper la estructura del documento."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt in _JSON_VALID_ESCAPE:
                out.append(c)
                out.append(nxt)
                i += 2
                continue
            out.append("\\\\")  # escape invalido: duplicar para leerlo literal
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


# --------------------------------------------------------------------------- #
# Fetch crudo de un tramo [start_dt, end_dt)
# --------------------------------------------------------------------------- #
def _fetch_window(query: str, start_dt: datetime, end_dt: datetime, max_records: int,
                  timeout: int = 60) -> list[dict]:
    import time as _time

    import requests

    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": str(max_records),
        "format": "json",
        # dateasc: si el cap trunca, trunca el FINAL del tramo -- determinista,
        # y el splitting adaptativo recupera lo truncado.
        "sort": "dateasc",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        # Cache-buster (medido en vivo 2026-07-14): GDELT cachea la respuesta
        # por URL exacta, INCLUIDO el mensaje de limite de tasa -- sin esto, un
        # reintento del mismo dia devuelve el rechazo cacheado para siempre
        # aunque el limite real ya haya pasado. No aumenta la carga al API: el
        # espaciado entre peticiones se respeta igual.
        "cb": str(int(_time.time() * 1000)),
    }
    resp = requests.get(_base_url(), params=params, timeout=timeout)
    text = resp.text
    # El limite de tasa llega como HTTP 429 o como texto plano con HTTP 200
    # ("Please limit requests to one every 5 seconds..."); ambos son la misma
    # condicion reintentable con backoff, no un error del dia.
    if resp.status_code == 429 or "limit requests" in text[:300].lower():
        raise GdeltRateLimited(text[:200])
    if resp.status_code != 200:
        raise RuntimeError(f"GDELT devolvio HTTP {resp.status_code}: {text[:200]}")
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        # Texto plano que NO es el mensaje de tasa: error de la consulta (p.ej.
        # "The specified phrase is too short."). NO reintentable: reventar con
        # el mensaje para que se arregle la query, no para esperar en vano.
        raise GdeltQueryError(f"GDELT rechazo la consulta: {text[:200]}")
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        # El feed trae un escape invalido: saneamos y reintentamos UNA vez. Si
        # sigue roto, propagamos el error original (el dia queda como hueco
        # documentado, no como un snapshot silenciosamente corrupto).
        data = json.loads(_sanitize_json_escapes(stripped))
    articles = data.get("articles", [])
    return [a for a in articles if isinstance(a, dict)]


def _fetch_with_retry(query: str, start_dt: datetime, end_dt: datetime, *,
                      max_records: int, backoff_seconds: float, max_retries: int) -> list[dict]:
    """Reintenta LA MISMA peticion ante el rechazo por tasa.

    Medido en vivo (2026-07-14): el limitador de GDELT sobre esta salida a
    internet se comporta como una loteria de contencion (IP compartida/CGNAT):
    con espaciado de 12 s pasan ~1 de cada 4 peticiones, sin patron de cooldown.
    Por eso el reintento es POR PETICION con espera corta y fija, no por dia con
    esperas largas: la proxima ranura libre puede llegar en segundos.
    """
    for attempt in range(max_retries + 1):
        try:
            arts = _fetch_window(query, start_dt, end_dt, max_records)
            logger.info("  tramo %s..%s: %d articulos (intento %d)",
                        start_dt.strftime("%m-%d %H:%M"), end_dt.strftime("%H:%M"),
                        len(arts), attempt + 1)
            return arts
        except GdeltRateLimited:
            if attempt >= max_retries:
                raise
            time.sleep(backoff_seconds)
    raise GdeltRateLimited("inalcanzable")  # defensivo; el loop siempre retorna o lanza


def _capture_segments(query: str, start_dt: datetime, end_dt: datetime, *,
                      max_records: int, min_window_hours: int,
                      spacing_seconds: float, backoff_seconds: float,
                      max_retries: int) -> list[dict]:
    """Consulta [start_dt, end_dt) partiendo en dos cada tramo que devuelva
    exactamente el cap, hasta tramos de `min_window_hours`. Devuelve una lista
    de segmentos {start, end, n, cap_hit, articles}."""
    articles = _fetch_with_retry(query, start_dt, end_dt, max_records=max_records,
                                 backoff_seconds=backoff_seconds, max_retries=max_retries)
    hit = len(articles) >= max_records
    window_hours = (end_dt - start_dt).total_seconds() / 3600.0
    kwargs = dict(max_records=max_records, min_window_hours=min_window_hours,
                  spacing_seconds=spacing_seconds, backoff_seconds=backoff_seconds,
                  max_retries=max_retries)
    if hit and window_hours > min_window_hours:
        mid = start_dt + (end_dt - start_dt) / 2
        time.sleep(spacing_seconds)
        left = _capture_segments(query, start_dt, mid, **kwargs)
        time.sleep(spacing_seconds)
        right = _capture_segments(query, mid, end_dt, **kwargs)
        return left + right
    return [{
        "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n": len(articles),
        "cap_hit": bool(hit),      # True aqui = censura real: tramo minimo saturado
        "articles": [
            {
                "seendate": a.get("seendate"),
                "title": a.get("title", "").strip(),
                "url": a.get("url"),
                "domain": a.get("domain"),
                "sourcecountry": a.get("sourcecountry"),
            }
            for a in articles
        ],
    }]


def capture_headlines_range(
    *,
    query: str,
    since: date,
    until: date,
    max_records: int,
    request_spacing_seconds: float,
    rate_limit_backoff_seconds: float,
    max_backoff_retries: int,
    min_window_hours: int,
    snapshot_dir: Path = SNAPSHOT_DIR,
    newest_first: bool = True,
) -> dict:
    """Captura (reanudable) los dias faltantes de [since, until] a un snapshot
    JSON por dia. Los dias ya presentes en disco se SALTAN: los snapshots son
    inmutables, nunca se reescriben (point-in-time, mismo contrato que
    consensus_calendar).

    newest_first=True baja primero los dias recientes: el artefacto "de hoy"
    necesita la ventana reciente completa, y el historico crece hacia atras a
    medida que el backfill avanza. Devuelve un resumen {captured, skipped,
    failed, rate_limited_aborts}.
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(since, until, freq="D")
    if newest_first:
        days = days[::-1]

    captured, skipped, failed = 0, 0, []
    consec_rate_limited = 0   # dias SEGUIDos amurallados por tasa (reset al primer exito)
    aborted = False
    for ts in days:
        day = ts.date()
        out_path = snapshot_dir / f"{day.isoformat()}.json"
        if out_path.exists():
            skipped += 1
            continue
        start_dt = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        end_dt = start_dt + timedelta(days=1)

        segments = None
        try:
            # El reintento ante el limite de tasa vive DENTRO de cada peticion
            # (_fetch_with_retry): ver ahi la medicion que justifica el diseno.
            segments = _capture_segments(
                query, start_dt, end_dt, max_records=max_records,
                min_window_hours=min_window_hours,
                spacing_seconds=request_spacing_seconds,
                backoff_seconds=rate_limit_backoff_seconds,
                max_retries=max_backoff_retries,
            )
        except GdeltRateLimited:
            # Ni con la paciencia por peticion paso este dia. NO se aborta la
            # corrida entera (eso tiraba a la basura horas de progreso reanudable
            # por una sola racha de mala suerte): se salta el dia -- queda sin
            # snapshot, asi que la proxima corrida lo reintenta -- y se sigue tras
            # un cooldown mas largo. Solo se aborta si MUCHOS dias SEGUIDOS chocan
            # con el muro (la salida a internet esta genuinamente amurallada).
            failed.append(str(day))
            consec_rate_limited += 1
            if consec_rate_limited >= _MAX_CONSEC_RATE_LIMIT_DAYS:
                logger.warning(
                    "abortando backfill: %d dias SEGUIDOS agotaron los reintentos de "
                    "tasa (muro persistente). Reanudable: los dias guardados se saltan.",
                    consec_rate_limited)
                aborted = True
                break
            logger.warning(
                "dia %s amurallado por tasa (%d/%d seguidos); se salta y se sigue "
                "tras cooldown.", day, consec_rate_limited, _MAX_CONSEC_RATE_LIMIT_DAYS)
            time.sleep(rate_limit_backoff_seconds * 4)  # cooldown mas largo entre dias amurallados
            continue
        except GdeltQueryError as exc:
            # La query esta mal: fallara para TODOS los dias. Abortar ya.
            logger.error("consulta rechazada por GDELT, se aborta el backfill: %s", exc)
            failed.append(str(day))
            return {"captured": captured, "skipped": skipped, "failed": failed,
                    "query_error": str(exc)}
        except Exception as exc:  # noqa: BLE001 -- un dia fallido es un hueco documentado, no un abort
            logger.warning("fallo capturando %s: %s", day, exc)

        if segments is None:
            failed.append(str(day))
            continue  # hueco de UN dia (error no de tasa): documentado, se sigue

        consec_rate_limited = 0   # exito: el muro no es persistente, reset del contador

        snapshot = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source": "gdelt doc 2.0 api",
            "query": query,
            "day": day.isoformat(),
            "n_total": int(sum(seg["n"] for seg in segments)),
            "cap_hit_any": bool(any(seg["cap_hit"] for seg in segments)),
            "segments": segments,
        }
        out_path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
        captured += 1
        time.sleep(request_spacing_seconds)

    summary = {"captured": captured, "skipped": skipped, "failed": failed, "aborted": aborted}
    logger.info("captura de titulares: %s", summary)
    return summary


# --------------------------------------------------------------------------- #
# Carga de snapshots -> DataFrame de titulares
# --------------------------------------------------------------------------- #
def load_headlines(snapshot_dir: Path = SNAPSHOT_DIR) -> pd.DataFrame:
    """Carga TODOS los snapshots diarios a un DataFrame ordenado por hora_titular.

    Deduplica por URL (el mismo articulo visto en dos tramos por el splitting).
    NO deduplica titulares sindicados en dominios distintos: "una noticia genera
    mas noticias" incluye la sindicacion -- la propagacion entre medios es parte
    de la cascada que Hawkes modela, no ruido a limpiar.

    attrs del DataFrame:
      - "coverage": {first_day, last_day, n_days, missing_days (dentro del rango
        cubierto), censored_days} -- la cobertura es cantidad de primera clase.
    """
    if not snapshot_dir.exists():
        df = pd.DataFrame(columns=HEADLINE_COLUMNS)
        df.attrs["coverage"] = {"first_day": None, "last_day": None, "n_days": 0,
                                "missing_days": [], "censored_days": []}
        return df

    rows: list[dict] = []
    days_seen: list[str] = []
    censored: list[str] = []
    for path in sorted(snapshot_dir.glob("*.json")):
        # Los snapshots de dia se llaman YYYY-MM-DD.json; los archivos auxiliares
        # del backfill/vigia (checkpoints, estado del watchdog) llevan prefijo "_"
        # y viven en el mismo directorio. Saltarlos: no son dias y "day" ausente
        # colaba path.stem (p.ej. "_gdelt_watchdog_state") a la malla de fechas.
        if path.stem.startswith("_"):
            continue
        try:
            snap = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("snapshot de titulares ilegible %s: %s", path.name, exc)
            continue
        days_seen.append(snap.get("day", path.stem))
        if snap.get("cap_hit_any"):
            censored.append(snap.get("day", path.stem))
        for seg in snap.get("segments", []):
            seg_hit = bool(seg.get("cap_hit"))
            for art in seg.get("articles", []):
                ts = pd.to_datetime(art.get("seendate"), utc=True, errors="coerce",
                                    format="%Y%m%dT%H%M%SZ")
                if pd.isna(ts) or not art.get("url") or not art.get("title"):
                    continue
                rows.append({
                    "hora_titular": ts,
                    "title": art["title"],
                    "url": art["url"],
                    "domain": art.get("domain"),
                    "sourcecountry": art.get("sourcecountry"),
                    "cap_hit_segment": seg_hit,
                })

    if rows:
        df = pd.DataFrame(rows).drop_duplicates(subset="url").sort_values("hora_titular")
        df = df.reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=HEADLINE_COLUMNS)

    if days_seen:
        days_dt = pd.to_datetime(sorted(days_seen))
        full = pd.date_range(days_dt.min(), days_dt.max(), freq="D")
        missing = [d.date().isoformat() for d in full.difference(days_dt)]
    else:
        missing = []
    df.attrs["coverage"] = {
        "first_day": min(days_seen) if days_seen else None,
        "last_day": max(days_seen) if days_seen else None,
        "n_days": len(days_seen),
        "missing_days": missing,
        "censored_days": sorted(censored),
    }
    return df


# --------------------------------------------------------------------------- #
# Emparejamiento titular <-> release macro y auditoria de timestamps (Trampa 3)
# --------------------------------------------------------------------------- #
def match_event_times(
    headlines: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    match_window_hours: int,
) -> pd.DataFrame:
    """Asigna `hora_evento` a los titulares que se refieren a un release del
    calendario macro (data/calendar.py): el titular menciona las palabras clave
    del indicador Y existe un release de ese indicador a menos de
    `match_window_hours` de distancia. El resto queda con hora_evento=NaT (un
    titular generico no tiene "hora de evento" definible y no se inventa).

    Se empareja con el release mas CERCANO en el tiempo (pasado o futuro dentro
    de la ventana), a proposito: si el titular quedo fechado ANTES del evento
    que reporta, eso es exactamente la masa negativa que la auditoria debe ver,
    no un caso a descartar.
    """
    from irfn.data.calendar import INDICATOR_KEYWORDS

    out = headlines.copy()
    out["hora_evento"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out["evento_indicator"] = None
    if out.empty or calendar is None or calendar.empty:
        return out

    window = pd.Timedelta(hours=match_window_hours)
    titles_lower = out["title"].str.lower()
    for ind, keywords in INDICATOR_KEYWORDS.items():
        releases = calendar[calendar["indicator"] == ind].index
        if not len(releases):
            continue
        mask = titles_lower.str.contains("|".join(pd.Series(keywords).map(lambda k: k.replace("(", "\\(").replace(")", "\\)"))), regex=True)
        idxs = out.index[mask & out["hora_evento"].isna()]
        if not len(idxs):
            continue
        rel_sorted = pd.DatetimeIndex(releases).sort_values()
        for i in idxs:
            t = out.at[i, "hora_titular"]
            pos = rel_sorted.searchsorted(t)
            candidates = []
            if pos > 0:
                candidates.append(rel_sorted[pos - 1])
            if pos < len(rel_sorted):
                candidates.append(rel_sorted[pos])
            best = min(candidates, key=lambda r: abs(t - r))
            if abs(t - best) <= window:
                out.at[i, "hora_evento"] = best
                out.at[i, "evento_indicator"] = ind
    return out


def timestamp_audit(matched: pd.DataFrame) -> dict:
    """Auditoria de timestamps de titulares para pantalla 6 (Trampa 3).

    Dos chequeos:
      (a) titular vs evento: distribucion de (hora_titular - hora_evento) en
          horas sobre los titulares emparejados. Masa en NEGATIVO => un titular
          quedo fechado antes del evento que reporta: timestamps corruptos,
          ROJO. Con 0 emparejados el chequeo pasa VACUAMENTE y lo declara (el
          calendario de consenso sigue vacio hoy; se activa solo cuando ambos
          feeds acumulen datos).
      (b) resolucion del feed: fraccion de titulares con hora exactamente
          00:00:00 (timestamp "solo fecha"). Si domina, la alineacion intradia
          no es creible y se marca en rojo. GDELT publica seendate a resolucion
          de 15 minutos, asi que una masa grande en medianoche exacta seria un
          sintoma de feed degradado, no de madrugadores.
    """
    n_total = int(len(matched))
    has_event = matched["hora_evento"].notna() if n_total else pd.Series(dtype=bool)
    n_matched = int(has_event.sum()) if n_total else 0

    if n_matched:
        diffs = (matched.loc[has_event, "hora_titular"] - matched.loc[has_event, "hora_evento"])
        diff_hours = diffs.dt.total_seconds() / 3600.0
        neg_mass = float((diff_hours < 0).mean())
        hist_counts, hist_edges = pd.cut(diff_hours, bins=12, retbins=True)[0].value_counts(sort=False), None
        histogram = [
            {"bin": str(interval), "n": int(cnt)}
            for interval, cnt in hist_counts.items()
        ]
        section_a = {
            "n_matched": n_matched,
            "negative_mass": neg_mass,
            "min_hours": float(diff_hours.min()),
            "median_hours": float(diff_hours.median()),
            "max_hours": float(diff_hours.max()),
            "histogram": histogram,
            "passed": bool(neg_mass == 0.0),
            "vacuous": False,
        }
    else:
        section_a = {
            "n_matched": 0, "negative_mass": None, "min_hours": None,
            "median_hours": None, "max_hours": None, "histogram": [],
            # pase vacuo declarado: no hay pares titular-evento que contradigan
            # la propiedad porque no hay pares (calendario de consenso vacio).
            "passed": True, "vacuous": True,
        }

    if n_total:
        t = matched["hora_titular"].dt
        midnight_frac = float(((t.hour == 0) & (t.minute == 0) & (t.second == 0)).mean())
        by_hour = matched["hora_titular"].dt.hour.value_counts().sort_index()
        section_b = {
            "n_headlines": n_total,
            "midnight_exact_frac": midnight_frac,
            "by_hour_utc": {int(h): int(c) for h, c in by_hour.items()},
            # umbral estructural, no de config: >50% en el minuto 00:00 exacto
            # significa que el feed da fechas, no horas -- resolucion rota.
            "passed": bool(midnight_frac < 0.5),
        }
    else:
        section_b = {"n_headlines": 0, "midnight_exact_frac": None, "by_hour_utc": {}, "passed": True}

    return {
        "titular_vs_evento": section_a,
        "resolucion_feed": section_b,
        "passed": bool(section_a["passed"] and section_b["passed"]),
    }

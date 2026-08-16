"""Prefijo historico recuperado (Wayback) para series con vintages truncadas.

Caso que motiva este modulo (data_audit.md seccion 3): desde abril de 2026 FRED
limita las series ICE BofA (BAMLH0A0HYM2) a una ventana rodante de 3 anios y el
recorte alcanza tambien al archivo de vintages de ALFRED (verificado en vivo:
solo desde 2023-07-17). El historico largo se recupero de un snapshot de
Wayback Machine del CSV de fredgraph (1996-2025) y esta guardado en
data/raw/recovered/ junto a su .provenance.json (fuente, QA contra ALFRED en el
solape: diferencia maxima 0.0 sobre 603 dias).

Por que es defendible bajo R4 pese a no ser una vintage: la serie es de mercado,
diaria y NO se revisa tras publicarse (misma categoria que DGS2/DGS10, ver
data_audit seccion 3), asi que el valor de la fecha t en un snapshot tardio es
identico al que se conocia poco despues de t. El rezago de publicacion del
prefijo NO se inventa: se mide de las vintages reales de ALFRED de la misma
serie (mediana de realtime_start - obs_date; para BAMLH0A0HYM2 es 0 dias, la
disponibilidad la aporta entero macro.availability_margin_days, igual que en el
tramo ALFRED).

DECISION DEL DIRECTOR (2026-07-18): integracion aprobada, gateada por
config macro.use_recovered_prefix. La sesion que recupero el snapshot
(2026-07-15) la dejo explicitamente pendiente de esta aprobacion.

El empalme es un PREFIJO ESTRICTO: solo entran observaciones anteriores a la
primera observacion del tramo ALFRED. En el solape manda ALFRED siempre (es la
fuente vintage canonica); el QA del solape vive en el .provenance.json, no se
re-decide aqui.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("irfn.recovered")

ROOT = Path(__file__).resolve().parents[3]
RECOVERED_DIR = ROOT / "data" / "raw" / "recovered"

# Sentinela de "valor aun vigente" del eje realtime de FRED/ALFRED (misma
# convencion que data/alfred.py). Para una serie no revisada cada observacion
# sigue vigente para siempre.
_REALTIME_END_SENTINEL = pd.Timestamp("9999-12-31")


def recovered_snapshot_path(series_id: str, *, recovered_dir: Path = RECOVERED_DIR) -> Path | None:
    """Ruta del snapshot recuperado mas reciente para la serie, o None si no hay.

    Convencion de nombre: <series_id>_wayback_<YYYYMMDD>.parquet con su
    .provenance.json al lado. Si hay varios snapshots gana el de nombre mayor
    (fecha de snapshot mas reciente).
    """
    if not recovered_dir.exists():
        return None
    candidates = sorted(recovered_dir.glob(f"{series_id}_wayback_*.parquet"))
    return candidates[-1] if candidates else None


def splice_recovered_prefix(
    series_id: str,
    alfred_vintages: pd.DataFrame,
    *,
    recovered_dir: Path = RECOVERED_DIR,
) -> tuple[pd.DataFrame, dict | None]:
    """Antepone el historico recuperado a las vintages de ALFRED como prefijo.

    Devuelve (vintages_empalmadas, info). Si no hay snapshot recuperado para la
    serie devuelve (alfred_vintages sin tocar, None) -- el llamador no necesita
    distinguir el caso.

    El prefijo se convierte al MISMO formato de vintages que consume
    point_in_time_series (obs_date, value, realtime_start, realtime_end), con
    realtime_start = obs_date + lag de publicacion MEDIDO en las vintages
    ALFRED de esta misma serie (mediana de realtime_start - obs_date por
    primera publicacion; no es un numero inventado, R7). Solo entran
    observaciones con obs_date estrictamente anterior a la primera obs_date de
    ALFRED: en cualquier solape manda ALFRED.

    info (para warnings/auditoria del artefacto): rango del prefijo, n_obs,
    lag aplicado, ruta del snapshot y su provenance si existe.
    """
    path = recovered_snapshot_path(series_id, recovered_dir=recovered_dir)
    if path is None:
        return alfred_vintages, None

    recovered = pd.read_parquet(path)
    first_alfred_obs = alfred_vintages["obs_date"].min()
    prefix = recovered[recovered["obs_date"] < first_alfred_obs].copy()
    if prefix.empty:
        # ALFRED ya cubre todo lo recuperado: no hay nada que anteponer.
        return alfred_vintages, None

    # Lag de publicacion medido de la propia serie en ALFRED (primera
    # publicacion de cada observacion). Mediana en dias calendario; para las
    # series diarias no revisadas de esta capa es 0 y el margen de
    # disponibilidad (config) hace el resto, igual que en el tramo ALFRED.
    first_release = alfred_vintages.groupby("obs_date")["realtime_start"].min()
    lag_days = int((first_release - first_release.index).dt.days.median())

    prefix_vintages = pd.DataFrame(
        {
            "obs_date": prefix["obs_date"],
            "value": prefix["value"].astype(float),
            "realtime_start": prefix["obs_date"] + pd.Timedelta(days=lag_days),
            "realtime_end": _REALTIME_END_SENTINEL,
        }
    )

    spliced = (
        pd.concat([prefix_vintages, alfred_vintages], ignore_index=True)
        .sort_values(["realtime_start", "obs_date"])
        .reset_index(drop=True)
    )

    provenance_path = path.parent / (path.stem + ".provenance.json")
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.exists()
        else None
    )

    info = {
        "series_id": series_id,
        "source": "wayback_recovered_prefix",
        "snapshot_file": path.name,
        "prefix_first_obs": str(prefix["obs_date"].min().date()),
        "prefix_last_obs": str(prefix["obs_date"].max().date()),
        "prefix_n_obs": int(len(prefix)),
        "alfred_first_obs": str(first_alfred_obs.date()),
        "publication_lag_days_applied": lag_days,
        "provenance": provenance,
        "note": (
            "prefijo estricto anterior al primer obs_date de ALFRED; en el "
            "solape manda ALFRED. Serie no revisada: snapshot tardio = valor "
            "publicado (data_audit seccion 3). Aprobado por el director "
            "2026-07-18."
        ),
    }
    logger.info(
        "%s: prefijo recuperado %s..%s (%d obs, lag aplicado %d d) + ALFRED desde %s.",
        series_id, info["prefix_first_obs"], info["prefix_last_obs"],
        info["prefix_n_obs"], lag_days, info["alfred_first_obs"],
    )
    return spliced, info

"""Acceso a artefactos y utilidades compartidas de la interfaz.

REGLA R9: la app SOLO LEE de artifacts/. Cero logica de modelo aqui. Si algun dia
te encuentras importando de src/irfn/models/ dentro de app/, estas violando R9.
Este modulo carga JSON/parquet de artifacts/ y nada mas; no calcula probabilidades,
no filtra, no suaviza. El unico calculo permitido es cosmetico y sin memoria (dar
color a un valor ya calculado), nunca sobre la serie temporal (un rolling().mean()
"para que se vea mejor" es look-ahead disfrazado de diseno; prohibido).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "latest"
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"

# Paleta fija por rol (no por marca): el indice 0 es el regimen de menor varianza.
REGIME_COLORS = ["#2E7D32", "#C62828", "#F9A825", "#1565C0", "#6A1B9A"]

# Umbral de frescura de fuentes (presentacion, no modelo). 4 dias tolera fines de
# semana y feriados de un activo diario; el intento del contrato es "reciente".
FRESHNESS_STALE_DAYS = 4


def _read_json(name: str) -> dict | None:
    path = ARTIFACTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_irfn() -> dict | None:
    """Contrato de salida artifacts/latest/irfn.json, o None si no hay artefactos."""
    return _read_json("irfn.json")


def load_audit() -> dict | None:
    return _read_json("audit.json")


def load_walkforward() -> dict | None:
    """Walk-forward mas reciente disponible: prefiere el de V1 (walkforward_v1.json,
    escrito por scripts/run_v1.py ablation) y cae al de V0 si aun no existe."""
    return _read_json("walkforward_v1.json") or _read_json("walkforward.json")


def load_ablation() -> dict | None:
    """Resultado de la fase 2 de V1 (ablacion + DM vs V0), o None si no ha corrido."""
    return _read_json("v1_ablation.json")


def load_surprise_events() -> list[dict]:
    """Eventos individuales (release + z_i + w_i) escritos por scripts/run_v2.py,
    para la serie de puntos de pantalla 3. Lista vacia si no hay artefacto o si
    el calendario todavia no tiene ni un release con consenso (la condicion de
    hoy, ver reports/data_audit.md)."""
    obj = _read_json("surprise_events.json")
    return (obj or {}).get("events", [])


def load_surprise_history() -> pd.DataFrame | None:
    """Serie temporal de SI_t (indexada por fecha), escrita por scripts/run_v2.py
    SOLO cuando la capa de noticias esta activa (model.news_layer_params.active).
    None si no existe -- pantalla 3 lo muestra con honestidad, no con un grafico
    vacio disfrazado de dato."""
    path = ARTIFACTS / "surprise_history.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.set_index("fecha")


def load_hawkes_history() -> pd.DataFrame | None:
    """Serie diaria de lambda_N (intensidad de Hawkes) escrita por
    scripts/run_v3.py SOLO cuando la capa de Hawkes esta activa
    (model.hawkes_layer_params.active). None si no existe."""
    path = ARTIFACTS / "hawkes_history.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.set_index("fecha")


def load_headline_rug() -> pd.DataFrame | None:
    """Titulares individuales (hora_titular + relevancia s) para el rug plot de
    pantalla 3, escritos por scripts/run_v3.py. None si no existen. La app solo
    decide CUANTO enseña (ventana de display); jamas recalcula nada."""
    path = ARTIFACTS / "headline_rug.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["hora_titular"] = pd.to_datetime(df["hora_titular"])
    return df


def load_ablation_news() -> str | None:
    """Reporte de ablacion M3/M4 (reports/ablation_news.md, R8: existe SIEMPRE
    que se corrio scripts/run_v2.py, con el compromiso pre-registrado y el
    veredicto, sea cual sea)."""
    path = REPORTS / "ablation_news.md"
    return path.read_text(encoding="utf-8") if path.exists() else None


def load_history() -> pd.DataFrame | None:
    path = ARTIFACTS / "history.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"])
        df = df.set_index("fecha")
    return df


def load_diagnostic_smoother() -> pd.DataFrame | None:
    """Smoother de Kim, SOLO diagnostico (R1). Vive FUERA de artifacts/, en
    data/interim/, y solo lo escribe el pipeline bajo IRFN_DEV_MODE. Devuelve None
    si no existe. Nunca se publica: aqui solo se LEE un archivo ya generado."""
    path = INTERIM / "diagnostic_smoother.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"])
        df = df.set_index("fecha")
    return df


def load_validation_report() -> str | None:
    """Reporte de validacion mas reciente: prefiere el de la version mas alta."""
    for name in ("validation_v3.md", "validation_v1.md", "validation_v0.md"):
        path = REPORTS / name
        if path.exists():
            return path.read_text(encoding="utf-8")
    return None


def pit_is_green(audit: dict | None) -> bool:
    """True solo si la invarianza de prefijo paso. Si no hay auditoria, NO es
    verde (ausencia de prueba no es prueba de ausencia de look-ahead)."""
    if not audit:
        return False
    return bool(audit.get("prefix_invariance", {}).get("passed", False))


def dev_mode() -> bool:
    return os.environ.get("IRFN_DEV_MODE", "").lower() in {"1", "true", "yes"}

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

# Paleta del dashboard: fuente unica de verdad en theme/. El indice 0 es el
# regimen de menor varianza (extremo BAJO del indice IRF-N). Verde/rojo quedan
# reservados al signo del precio (theme.DIRECTIONAL); los regimenes salen de la
# escala viridis del indice. Importar theme registra ademas el template oscuro
# de Plotly (se aplica a toda figura de las pantallas).
from theme import REGIME_COLORS, apply_layout  # noqa: E402,F401

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
    """Walk-forward del artefacto VIGENTE: SOLO artifacts/latest/walkforward.json,
    que la promocion atomica garantiza que es el del modelo publicado (mismo run_id).

    Fase 5: se elimino el fallback a walkforward_v1.json. Ese archivo era de una
    corrida V1 distinta con un baseline mas favorable; dejarlo como respaldo hacia
    que la pantalla de Validacion mostrara metricas de OTRO modelo. Si falta el
    walkforward.json vigente, se devuelve None y la pantalla muestra vacio honesto,
    NUNCA el de otra corrida."""
    return _read_json("walkforward.json")


def load_ablation() -> dict | None:
    """Ablacion V1 (DM vs V0). Ya NO vive en artifacts/latest/ (es analisis V1, no
    parte del set publicado; la promocion atomica solo trae el set del modelo
    vigente). Devuelve None salvo que exista: la pantalla de Validacion oculta el
    bloque en vez de mostrar el DM de una version que no es la publicada."""
    return _read_json("v1_ablation.json")


def load_manifest() -> dict | None:
    """manifest.json del set vigente (run_id + version + sha256 por archivo). Lo
    escribe la promocion; es la fuente de verdad de 'que corrida es esta'."""
    return _read_json("manifest.json")


def artifact_coherent(irfn: dict | None, audit: dict | None, manifest: dict | None) -> bool:
    """True si irfn/audit/manifest declaran el MISMO run_id. Con promocion atomica
    siempre deberia serlo; si no, es que alguien escribio latest/ a mano (la
    regresion que estamos matando) y la app debe avisar en vez de dibujar una
    mezcla."""
    ids = {o.get("run_id") for o in (irfn, audit, manifest) if o}
    return len(ids) <= 1


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


def render_header(irfn: dict | None = None, audit: dict | None = None) -> None:
    """Cabecera comun de las 7 pantallas: identidad del artefacto vigente
    (version, run_id, git, asof) + coherencia de procedencia. Fase 5: si irfn,
    audit y manifest no comparten run_id, se avisa en TODAS las pantallas -- una
    mezcla de procedencia (la regresion V0) no se dibuja en silencio."""
    import streamlit as st

    irfn = irfn if irfn is not None else load_irfn()
    audit = audit if audit is not None else load_audit()
    manifest = load_manifest()
    if irfn is None:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.caption(f"**version** {irfn.get('version', '?')}")
    c2.caption(f"**run_id** {irfn.get('run_id', '?')}")
    c3.caption(f"**git** {irfn.get('git_commit', '?')}")
    c4.caption(f"**asof** {irfn.get('asof', '?')}")
    if not artifact_coherent(irfn, audit, manifest):
        st.error(
            "Procedencia INCOHERENTE: irfn/audit/manifest no comparten run_id. "
            "El set de artifacts/latest/ es una MEZCLA de corridas (no se promovio "
            "de forma atomica). No confies en lo que se dibuja hasta re-promover con "
            "scripts/promote.py."
        )
    if manifest is None:
        st.warning(
            "Sin manifest.json en artifacts/latest/: el set no se promovio con "
            "scripts/promote.py (posible escritura manual). Procedencia no verificable."
        )
    # Fase 6: los avisos del artefacto viajan a TODAS las pantallas, no solo Home.
    warnings = irfn.get("warnings", [])
    if warnings:
        with st.expander(f"Avisos del artefacto ({len(warnings)})", expanded=False):
            for w in warnings:
                st.warning(w)


# Umbral (presentacion) para avisar del hueco entre el fin de la serie OOS y asof.
# La serie del walk-forward termina antes que asof por su cola natural (el ultimo
# tramo no forma un bloque de test completo); por encima de esto se avisa para que
# nadie lea el Historico como "hasta hoy".
FRESHNESS_GAP_DAYS = 10


def se_unreliable(irfn: dict | None) -> bool:
    """True si el artefacto avisa de Hessiano degenerado (SE/IC no fiables)."""
    if not irfn:
        return False
    return any("hessiano degenerado" in w.lower() for w in irfn.get("warnings", []))


# Duracion esperada (dias) por debajo de la cual un regimen es un "absorbe-outliers"
# (E[D]=1/(1-p_kk) ~ 1 dia): no persiste, no es interpretable como un estado de
# mercado ni su retorno condicional como un retorno esperado.
DEGENERATE_DURATION_DAYS = 2.0


def momentum_5d(irfn: dict | None) -> list[tuple[str, float]]:
    """(label, delta en puntos porcentuales) del cambio en la probabilidad filtrada
    a 5 dias habiles: regime.xi_momentum_5d = ξ_{t|t}(k) − ξ_{t−5|t−5}(k) (lo calcula
    el pipeline, P3-13). Se expresa en PUNTOS PORCENTUALES de probabilidad (x100).
    Propiedad: la suma sobre regimenes es 0 (ambos son distribuciones que suman 1),
    salvo redondeo -- la app solo formatea el numero publicado, no lo recalcula (R9)."""
    if not irfn:
        return []
    reg = irfn.get("regime", {})
    labels = reg.get("labels", [])
    mom = reg.get("xi_momentum_5d", [])
    return [(labels[k], 100.0 * float(mom[k])) for k in range(min(len(labels), len(mom)))]


def degenerate_regimes(irfn: dict | None) -> list[tuple[str, float]]:
    """(label, E[D]) de los regimenes con duracion esperada < DEGENERATE_DURATION_DAYS.
    Disparado por regime.expected_duration_days del artefacto (P1-4), no hardcodeado."""
    if not irfn:
        return []
    reg = irfn.get("regime", {})
    labels = reg.get("labels", [])
    edd = reg.get("expected_duration_days", [])
    return [(labels[k], float(edd[k])) for k in range(min(len(labels), len(edd)))
            if edd[k] < DEGENERATE_DURATION_DAYS]


def render_freshness_gap(irfn: dict | None = None) -> None:
    """Banner si la serie OOS (history.parquet) termina bastante antes que asof
    (P0-3). Disparado por datos del artefacto, no hardcodeado."""
    import pandas as pd
    import streamlit as st

    irfn = irfn if irfn is not None else load_irfn()
    hist = load_history()
    if irfn is None or hist is None or hist.empty:
        return
    asof_raw = irfn.get("asof")
    if not asof_raw:
        return
    last = pd.to_datetime(hist.index.max()).date()
    asof = pd.to_datetime(asof_raw).date()
    gap = (asof - last).days
    if gap > FRESHNESS_GAP_DAYS:
        st.info(
            f"La serie histórica out-of-sample llega hasta **{last}**, mientras que "
            f"el estado de hoy es de **{asof}** ({gap} días después). No es un dato "
            "viejo: el walk-forward evalúa fuera de muestra por bloques y ese último "
            "tramo aún no forma un bloque de test completo, así que no se grafica. "
            "El 'Régimen hoy' sí usa toda la muestra hasta asof."
        )

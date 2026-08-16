"""Sala de control. Semaforo de auditoria PIT, frescura de fuentes, boton
'Correr pipeline'.

R9: esta interfaz solo lee de artifacts/. Cero logica de modelo aqui. El semaforo
PIT manda: si esta en rojo, el resto de la app se muestra deshabilitado a proposito
(un artefacto con look-ahead no se mira "con cuidado", no se mira).
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from components import (  # noqa: E402
    FRESHNESS_STALE_DAYS,
    dev_mode,
    load_audit,
    load_irfn,
    pit_is_green,
    render_freshness_gap,
    render_header,
)

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="IRF-N - Sala de control", layout="wide")


def _semaforo_pit(audit: dict | None) -> bool:
    verde = pit_is_green(audit)
    if verde:
        diff = audit["prefix_invariance"].get("atol", 1e-10)
        st.success(
            "Auditoria point-in-time: VERDE. La invarianza de prefijo pasa "
            f"(sin informacion futura filtrandose, tolerancia {diff:.0e}).",
            icon=None,
        )
    elif audit is None:
        st.warning(
            "Auditoria point-in-time: sin datos. Todavia no hay artefactos. "
            "Corre el pipeline con el boton de abajo."
        )
    else:
        st.error(
            "Auditoria point-in-time: ROJA. La invarianza de prefijo FALLA: hay "
            "informacion futura filtrandose. El resto de la app queda deshabilitado. "
            "No se publica un indicador con look-ahead."
        )
    return verde


def _correr_pipeline_ui() -> None:
    st.subheader("Correr pipeline")
    st.caption(
        "Ejecuta scripts/run_pipeline.py en un subproceso. Descarga SPY, re-estima "
        "el walk-forward y regenera los artefactos. Puede tardar varios minutos."
    )
    quick = st.checkbox("Modo rapido (menos arranques, para probar)", value=True)
    if st.button("Correr pipeline ahora", type="primary"):
        cmd = [sys.executable, str(ROOT / "scripts" / "run_pipeline.py")]
        if quick:
            cmd.append("--quick")
        log_box = st.empty()
        lines: list[str] = []
        with st.status("Corriendo pipeline...", expanded=True):
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:  # type: ignore[union-attr]
                lines.append(line.rstrip())
                log_box.code("\n".join(lines[-30:]))
            proc.wait()
        if proc.returncode == 0:
            st.success("Pipeline terminado. Recarga la pagina para ver los artefactos nuevos.")
        else:
            st.error(
                f"El pipeline termino con codigo {proc.returncode}. Revisa el log de arriba: "
                "la causa mas comun es que yfinance no devolvio datos (ver reports/data_audit.md)."
            )


def _frescura(irfn: dict | None, audit: dict | None) -> None:
    st.subheader("Frescura de fuentes de datos")
    sources = (audit or {}).get("sources", [])
    if not sources:
        st.info("No hay fuentes registradas todavia.")
        return
    now = datetime.now(timezone.utc).date()
    cols = st.columns(len(sources))
    for col, s in zip(cols, sources):
        last = datetime.fromisoformat(s["last_date"]).date()
        age = (now - last).days
        stale = age > FRESHNESS_STALE_DAYS
        with col:
            st.metric(
                label=f"{s['name']} ({s['source']})",
                value=s["last_date"],
                delta=f"hace {age} dias" + (" - REVISAR" if stale else ""),
                delta_color="inverse" if stale else "normal",
            )


def main() -> None:
    st.title("IRF-N - Sala de control")
    st.caption(
        "Indice de Regimen Filtrado con Noticias. La version y la especificacion "
        "del modelo vigentes son las del artefacto (tarjetas de abajo y "
        "model.spec en irfn.json); esta pantalla no las decide."
    )
    if dev_mode():
        st.caption("IRFN_DEV_MODE activo: la vista Historico puede superponer el smoother de diagnostico.")

    irfn = load_irfn()
    audit = load_audit()
    render_header(irfn, audit)
    render_freshness_gap(irfn)

    verde = _semaforo_pit(audit)
    st.divider()

    if irfn is None:
        st.info(
            "Todavia no hay artefactos. Corre el pipeline con el boton de abajo "
            "(o `python scripts/run_pipeline.py` en la terminal)."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("asof", str(irfn["asof"]))
        c2.metric("run_id", irfn["run_id"])
        c3.metric("git commit", irfn["git_commit"])
        c4.metric("version", irfn["version"])

        _frescura(irfn, audit)

        st.subheader("Avisos del artefacto")
        warnings = irfn.get("warnings", [])
        if warnings:
            for w in warnings:
                st.warning(w)
        else:
            st.success("Sin avisos.")

        if not verde:
            st.error(
                "El artefacto existe pero la auditoria PIT esta en rojo. Las vistas de "
                "regimen e historico no son fiables hasta corregir el look-ahead."
            )

    st.divider()
    _correr_pipeline_ui()


main()

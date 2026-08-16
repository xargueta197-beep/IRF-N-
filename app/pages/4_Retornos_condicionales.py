"""Retornos condicionales: estadisticas por regimen (activo x regimen).

R9: solo lee de artifacts/. Las estadisticas las calcula el pipeline
(outputs/publish.conditional_stats); aqui solo se tabulan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import (  # noqa: E402
    degenerate_regimes,
    load_audit,
    load_irfn,
    pit_is_green,
    render_header,
    se_unreliable,
)

st.set_page_config(page_title="IRF-N - Retornos condicionales", layout="wide")


def main():
    st.title("Retornos condicionales por regimen")
    irfn = load_irfn()
    audit = load_audit()
    render_header(irfn, audit)
    if irfn is None:
        st.info("No hay artefactos todavia. Corre el pipeline desde la Sala de control.")
        return
    if not pit_is_green(audit):
        st.error("La auditoria PIT esta en rojo. Vista deshabilitada.")
        return

    st.caption(
        "Estadisticas sobre ventanas donde argmax(ξ) = k. Con ξ filtrada, no suavizada."
    )

    # Fase 6 (P1-4): banner sobre regimenes degenerados (absorbe-outliers). El
    # retorno condicional de un estado que dura ~1 dia NO es un retorno esperado.
    degen = degenerate_regimes(irfn)
    if degen:
        detalle = ", ".join(f"**{lbl}** (dura ~{edd:.0f} día)" for lbl, edd in degen)
        st.warning(
            f"Ojo: {detalle}. Es un estado **absorbe-outliers**, no un régimen "
            "persistente: aparece un día suelto para capturar un movimiento extremo y "
            "desaparece (E[D] = 1/(1−p_kk) ≈ 1 día). Sus estadísticas de abajo "
            "(retorno anual, Sharpe, caída máxima) **no son interpretables como el "
            "rendimiento esperado** de un régimen — son el promedio de puñados de días "
            "atípicos anualizados. Limitación conocida del modelo (K=2/Normal), avisada "
            "en el artefacto; no es un error de datos."
        )
    if se_unreliable(irfn):
        st.caption(
            "Nota: el artefacto avisa de Hessiano degenerado en la estimación de hoy — "
            "los intervalos de confianza de abajo heredan esa limitación (SE no fiables)."
        )

    def _fmt_cell(m: dict, pct: bool) -> str:
        v = m["value"]
        val_s = f"{v:.1%}" if pct else f"{v:.2f}"
        if m["ci_low"] is None or m["ci_high"] is None:
            return val_s
        lo, hi = m["ci_low"], m["ci_high"]
        ci_s = f"[{lo:.1%}, {hi:.1%}]" if pct else f"[{lo:.2f}, {hi:.2f}]"
        return f"{val_s} {ci_s}"

    metric_cols = [
        ("mean_ann", "retorno anual", True),
        ("vol_ann", "vol anual", True),
        ("sharpe", "Sharpe", False),
        ("maxdd", "max caida", True),
    ]

    for asset, regimes in irfn["conditional_stats"].items():
        st.subheader(asset)
        rows = {}
        for regime, metrics in regimes.items():
            rows[regime] = {label: _fmt_cell(metrics[key], pct) for key, label, pct in metric_cols}
        df = pd.DataFrame(rows).T
        st.dataframe(df, width="stretch")
    st.caption(
        "IC al 90% (config v2.bootstrap.ci_level) por bootstrap estacionario "
        "(Politis-Romano, validation/bootstrap.py). Celda sin corchetes = menos "
        "observaciones que v2.bootstrap.min_obs en ese regimen: se muestra el "
        "punto, no se inventa un intervalo."
    )


main()

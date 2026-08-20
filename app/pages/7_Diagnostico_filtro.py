"""Diagnostico interno del filtro: consistencia predicho(t|t-1) vs concluido(t|t).

R9: solo lee de artifacts/. NO es una pantalla de Validacion: y_t = argmax xi_{t|t}
es un PROXY generado por el propio modelo, no una verdad de terreno exogena. Estas
metricas miden si el filtro es auto-consistente consigo mismo, no si predice algo
observable. Movidas aqui desde Validacion (hallazgo F2.b, auditoria 2026-08-18):
estaban presentadas con jerarquia de metrica de desempeno y no lo son.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import load_irfn, load_walkforward, render_header  # noqa: E402

st.set_page_config(page_title="IRF-N - Diagnostico interno del filtro", layout="wide")


def _reliability_fig(rows):
    df = pd.DataFrame(rows).dropna(subset=["mean_confidence", "empirical_accuracy"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color="#8B93A5", dash="dash"), name="autoconsistencia perfecta"))
    fig.add_trace(go.Scatter(
        x=df["mean_confidence"], y=df["empirical_accuracy"], mode="markers+lines",
        marker=dict(size=8 + df["count"] / df["count"].max() * 16, color="#5B8DEF"),
        name="modelo",
    ))
    fig.update_layout(
        height=420, xaxis=dict(title="confianza media predicha (t|t-1)", range=[0, 1]),
        yaxis=dict(title="acierto empirico vs proxy (t|t)", range=[0, 1]),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def main():
    irfn = load_irfn()
    st.title("Diagnostico interno del filtro")
    render_header(irfn)
    wf = load_walkforward()

    if wf is None:
        st.info("No hay artefactos todavia. Corre el pipeline desde la Sala de control.")
        return

    st.warning(
        "Esto NO es una pantalla de Validacion. El objetivo y_t = argmax ξ_{t|t} es un "
        "PROXY que produce el propio modelo (usa r_t, ya visto). Estas metricas miden "
        "consistencia entre lo que el filtro predijo ayer (ξ_{t|t-1}) y lo que concluyo "
        "hoy con el dato de hoy (ξ_{t|t}) -- NO miden habilidad predictiva sobre algo "
        "observable de forma independiente. Para eso, ver Validacion (targets exogenos: "
        "Pesaran-Timmermann, Diebold-Mariano, Sharpe out-of-sample)."
    )

    calib = wf["calibration"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("bloques walk-forward", wf["n_blocks"])
    c2.metric("log-loss de consistencia", f"{calib['log_loss']:.4f}",
              f"climatologia pooled {calib['log_loss_baseline']:.4f}")
    c3.metric("Brier de consistencia", f"{calib['brier']:.4f}",
              f"climatologia pooled {calib['brier_baseline']:.4f}")
    c4.metric("ECE de consistencia", f"{calib['ece']:.4f}")
    st.caption(
        "La 'climatologia pooled' de arriba se estima UNA VEZ sobre toda la muestra "
        "out-of-sample agregada (los 19 bloques juntos), no por bloque a partir de solo "
        "su propio train -- es un baseline informativo, no un test de habilidad "
        "predictiva fuera de muestra en el sentido estricto de R2 (hallazgo F2.a, "
        "auditoria 2026-08-18)."
    )

    st.subheader("Diagrama de autoconsistencia (sobre ξ predicho)")
    st.plotly_chart(_reliability_fig(calib["reliability_curve"]), width="stretch")
    st.caption(calib["target_note"])


main()

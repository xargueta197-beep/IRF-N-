"""Validacion: reporte de walk-forward y diagrama de fiabilidad.

R9: solo lee de artifacts/ y reports/. El reporte lo genera el pipeline (R8); aqui
se muestra tal cual, gane o pierda el modelo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import load_ablation, load_irfn, load_validation_report, load_walkforward  # noqa: E402

st.set_page_config(page_title="IRF-N - Validacion", layout="wide")


def _reliability_fig(rows):
    df = pd.DataFrame(rows).dropna(subset=["mean_confidence", "empirical_accuracy"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color="#B0BEC5", dash="dash"), name="calibracion perfecta"))
    fig.add_trace(go.Scatter(
        x=df["mean_confidence"], y=df["empirical_accuracy"], mode="markers+lines",
        marker=dict(size=8 + df["count"] / df["count"].max() * 16, color="#1565C0"),
        name="modelo",
    ))
    fig.update_layout(
        height=420, xaxis=dict(title="confianza media predicha", range=[0, 1]),
        yaxis=dict(title="acierto empirico", range=[0, 1]),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def main():
    irfn = load_irfn()
    version = (irfn or {}).get("version", "V0")
    st.title(f"Validacion {version}")
    wf = load_walkforward()
    report = load_validation_report()
    abl = load_ablation()

    if wf is None:
        st.info("No hay artefactos todavia. Corre el pipeline desde la Sala de control.")
        return

    # Titular del criterio de aceptacion de V1: Diebold-Mariano contra V0.
    if abl and abl.get("dm_vs_v0"):
        dm = abl["dm_vs_v0"]
        better = "mejor que V0" if dm["dm_stat"] < 0 else "no mejor que V0"
        st.subheader("Diebold-Mariano: titular V1 vs V0")
        d1, d2, d3 = st.columns(3)
        d1.metric("DM stat", f"{dm['dm_stat']:.3f}", better)
        d2.metric("p-value", f"{dm['p_value']:.3f}")
        d3.metric("bloques OOS", abl.get("n_blocks", "-"))
        st.caption("DM<0 => el titular V1 tiene MENOR perdida predictiva fuera de muestra que V0 "
                   "(K=2 Normal P-constante), sobre la misma muestra y malla de bloques.")

    calib = wf["calibration"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("bloques walk-forward", wf["n_blocks"])
    c2.metric("log-loss modelo", f"{calib['log_loss']:.4f}", f"baseline {calib['log_loss_baseline']:.4f}")
    c3.metric("Brier modelo", f"{calib['brier']:.4f}", f"baseline {calib['brier_baseline']:.4f}")
    c4.metric("ECE", f"{calib['ece']:.4f}")

    st.subheader("Diagrama de fiabilidad (sobre ξ predicho)")
    st.plotly_chart(_reliability_fig(calib["reliability_curve"]), width="stretch")
    st.caption(calib["target_note"])

    st.subheader("Reporte de walk-forward")
    if report:
        st.markdown(report)
    else:
        st.info("No se encontro reports/validation_v1.md ni validation_v0.md.")


main()

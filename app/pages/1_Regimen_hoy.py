"""Regimen hoy: barras de ξ filtrado, medidor de entropia, duracion esperada y
matriz de transicion.

R9: solo lee de artifacts/. El punto de esta pantalla es lo que ningun modulo
tradicional hace: cuando la entropia cae en la zona alta, NO se muestra un regimen
ganador; se dice "Sin senal clara". Es el corazon del indicador.
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import (  # noqa: E402
    REGIME_COLORS,
    load_audit,
    load_irfn,
    momentum_5d,
    pit_is_green,
    render_freshness_gap,
    render_header,
)

st.set_page_config(page_title="IRF-N - Regimen hoy", layout="wide")


def _barras_xi(labels, xi):
    colors = [REGIME_COLORS[i % len(REGIME_COLORS)] for i in range(len(labels))]
    fig = go.Figure()
    fig.add_bar(
        x=xi, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:.1%}" for v in xi], textposition="auto",
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1], tickformat=".0%", title="probabilidad filtrada"),
        height=90 + 60 * len(labels), margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


def _medidor_entropia(H, H_max, mid, high):
    norm = H / H_max if H_max > 0 else 0.0
    # Zonas de entropia en la paleta del dashboard (verde/rojo quedan reservados
    # al signo del precio): confiado = teal oscuro del indice bajo, medio = gris
    # sin-dato, sin senal = amarillo oscuro del indice alto. El indicador viridis.
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=norm, number={"valueformat": ".2f", "font": {"color": "#D7DCE5"}},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": "#8B93A5"},
            "bar": {"color": "#5B8DEF"},
            "bordercolor": "#262C38",
            "steps": [
                {"range": [0, mid], "color": "#17323A"},
                {"range": [mid, high], "color": "#2A2F3B"},
                {"range": [high, 1], "color": "#3E3A17"},
            ],
        },
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=10))
    return fig


def _heatmap_P(labels, P):
    fig = go.Figure(go.Heatmap(
        z=P, x=labels, y=labels, colorscale="Blues", zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in P],
        texttemplate="%{text}", showscale=False,
    ))
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="hacia", yaxis_title="desde", yaxis_autorange="reversed",
    )
    return fig


def main():
    st.title("Regimen hoy")
    irfn = load_irfn()
    audit = load_audit()
    render_header(irfn, audit)

    if irfn is None:
        st.info("No hay artefactos todavia. Corre el pipeline desde la Sala de control.")
        return
    if not pit_is_green(audit):
        st.error(
            "La auditoria PIT esta en rojo. Esta vista queda deshabilitada: un ξ con "
            "look-ahead no describe el presente, describe el futuro que ya paso."
        )
        return

    reg = irfn["regime"]
    labels = reg["labels"]
    xi = reg["xi_filtered"]
    H = reg["entropy"]
    H_max = reg["entropy_max"]
    confidence = reg["confidence"]
    argmax_idx = int(max(range(len(xi)), key=lambda i: xi[i]))

    # Zonas del medidor: se leen del artefacto (audit.json), no se hardcodean.
    zones = (audit or {}).get("entropy_zones", {"mid": 0.55, "high": 0.85})
    mid, high = zones["mid"], zones["high"]

    st.caption(f"asof {irfn['asof']}  ·  run_id {irfn['run_id']}  ·  {irfn['version']}")

    single_regime = len(labels) == 1

    col_izq, col_der = st.columns([3, 2])
    with col_izq:
        st.subheader("Estado del mercado")
        if single_regime:
            # K=1 (elegido por BIC, p.ej. BTC): no hay estructura de regimenes. El
            # 100%/entropia 0 son por construccion, no una senal. Se dice la verdad.
            st.markdown(
                "<div style='font-size:2.4rem;font-weight:700;'>Sin estructura de regimenes</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Este activo se ajusto con UN SOLO regimen (K=1, elegido por BIC): sus "
                "colas gordas explican la turbulencia, no un cambio de regimen. El indice "
                "de regimen es trivialmente 100% y la entropia 0 -- no hay nada que "
                "distinguir. Lo informativo aqui es la volatilidad condicional y la capa "
                "de noticias (Hawkes), no el regimen."
            )
        elif confidence == "el modelo no distingue":
            st.markdown(
                "<div style='font-size:2.4rem;font-weight:700;color:#FDE725;'>Sin senal clara</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "La entropia esta en la zona alta: el modelo reparte la probabilidad de "
                "forma demasiado pareja para declarar un regimen. Declararlo seria inventar "
                "una senal que no existe."
            )
        else:
            st.markdown(
                f"<div style='font-size:2.4rem;font-weight:700;'>{labels[argmax_idx]}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Confianza: {confidence}.")

        if not single_regime:
            st.plotly_chart(_barras_xi(labels, xi), width="stretch")
            dur = reg["expected_duration_days"][argmax_idx]
            st.info(f"Este regimen dura en promedio **{dur:.0f} dias** mas (E[D] = 1/(1-p_kk)).")

        # P3-13: momentum 5d de la probabilidad filtrada (publicado en el artefacto).
        mom = momentum_5d(irfn)
        if mom:
            partes = ", ".join(f"{lbl} {pp:+.1f} pp" for lbl, pp in mom)
            st.caption(
                f"Momentum 5 días hábiles (cambio de la probabilidad filtrada): {partes}. "
                "En puntos porcentuales; la suma es 0 (lo que gana un régimen lo pierde otro)."
            )

    with col_der:
        st.subheader("Entropia (confianza)")
        st.plotly_chart(_medidor_entropia(H, H_max, mid, high), width="stretch")
        st.caption(
            f"H = {H:.3f} de H_max = {H_max:.3f}. Verde: confianza alta · "
            "naranja: media · rojo: el modelo no distingue."
        )

    tvtp = bool(irfn.get("model", {}).get("tvtp", False))
    P_today = irfn["transition_matrix_today"]

    # Texto de transicion condicional (V1). Riesgo-off = regimen de mayor
    # varianza = ultimo indice (orden estructural v_1<...<v_K, R5). Esto es
    # LECTURA del artefacto + formato, no calculo de modelo: la matriz ya viene
    # evaluada en x_asof desde el pipeline (R9 respetado).
    if tvtp and len(labels) >= 2:
        risk_off = len(labels) - 1
        p_riskoff = P_today[argmax_idx][risk_off]
        st.subheader("Transicion condicional a las condiciones de hoy")
        if argmax_idx == risk_off:
            st.warning(
                f"El regimen mas probable YA es **{labels[risk_off]}** (risk-off). "
                f"Probabilidad de permanecer manana: **{P_today[risk_off][risk_off]:.0%}**."
            )
        else:
            st.markdown(
                f"Con las condiciones de hoy, la probabilidad de pasar a **{labels[risk_off]}** "
                f"(risk-off) es **{p_riskoff:.1%}**."
            )
        st.caption(
            "La matriz de transicion ya no es constante: se evalua en las covariables "
            "rezagadas mas recientes (x_asof). Fila = desde, columna = hacia."
        )
    else:
        st.subheader("Matriz de transicion (constante en V0)")

    st.plotly_chart(_heatmap_P(labels, P_today), width="stretch")

    # Fase 6 (P2-8): si las filas de la matriz son casi identicas, el regimen de
    # manana depende muy poco del de hoy (matriz casi de rango 1). Disparado por
    # los valores de la matriz publicada, no hardcodeado.
    if len(labels) >= 2 and P_today:
        maxdiff = max(
            abs(P_today[i][j] - P_today[0][j])
            for i in range(len(P_today)) for j in range(len(labels))
        )
        if maxdiff < 0.05:
            st.caption(
                f"Nota: las filas de la matriz son casi idénticas (diferencia máxima "
                f"{maxdiff:.2f}). El régimen de mañana depende muy poco del de hoy: la "
                "matriz es casi de rango 1 (las transiciones apenas reaccionan al estado "
                "actual)."
            )

    render_freshness_gap(irfn)


main()

"""Historico: ξ filtrado out-of-sample apilado, precio ancla coloreado por regimen,
franja de entropia y fronteras de bloques de walk-forward.

R9: solo lee de artifacts/. La serie que se muestra es la PROBABILIDAD FILTRADA
(ξ_{t|t}) out-of-sample: ninguna informacion futura. La etiqueta lo dice, siempre
visible. El toggle de diagnostico (solo DEV_MODE) superpone el smoother, que vive
FUERA de artifacts/ (data/interim/) porque jamas se publica (R1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import (  # noqa: E402
    REGIME_COLORS,
    dev_mode,
    load_audit,
    load_diagnostic_smoother,
    load_history,
    load_irfn,
    load_walkforward,
    pit_is_green,
    render_freshness_gap,
    render_header,
)

st.set_page_config(page_title="IRF-N - Historico", layout="wide")


def main():
    st.title("Historico")
    st.caption("Probabilidad filtrada — ninguna informacion futura")

    irfn = load_irfn()
    audit = load_audit()
    render_header(irfn, audit)
    hist = load_history()
    wf = load_walkforward()

    if irfn is None or hist is None:
        st.info("No hay artefactos todavia. Corre el pipeline desde la Sala de control.")
        return
    if not pit_is_green(audit):
        st.error("La auditoria PIT esta en rojo. Vista deshabilitada hasta corregir el look-ahead.")
        return

    render_freshness_gap(irfn)

    labels = irfn["regime"]["labels"]
    K = len(labels)
    xi_cols = [f"xi_filtered_{k}" for k in range(K)]
    boundaries = [np.datetime64(d) for d in (wf or {}).get("block_boundaries", [])]

    # Banda "tramo aun no evaluable": entre el fin de la serie OOS y asof. NO es
    # dato faltante -- ese ultimo tramo no forma un bloque de test completo, asi
    # que el walk-forward no lo grafica (mismo motivo que render_freshness_gap).
    # Se dibuja desde datos ya publicados (fin de hist + asof); la app no calcula.
    import pandas as pd  # noqa: PLC0415  (import local: solo para esta banda)
    _asof_raw = irfn.get("asof")
    _oos_end = np.datetime64(pd.to_datetime(hist.index.max()).date())
    _asof_dt = np.datetime64(pd.to_datetime(_asof_raw).date()) if _asof_raw else None
    _show_oos_band = _asof_dt is not None and _asof_dt > _oos_end

    def _add_oos_band(target_fig, **vrect_kwargs):
        """Sombra + etiqueta del tramo OOS incompleto, si aplica."""
        if not _show_oos_band:
            return
        target_fig.add_vrect(
            x0=_oos_end, x1=_asof_dt,
            fillcolor="#90A4AE", opacity=0.10, line_width=0, layer="below",
            annotation_text="tramo aun no evaluable (OOS incompleto)",
            annotation_position="top left",
            annotation=dict(font=dict(size=10, color="#90A4AE")),
            **vrect_kwargs,
        )

    # --- Figura 1: ξ filtrado apilado + franja de entropia ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        subplot_titles=("Probabilidad filtrada por regimen (out-of-sample)", "Entropia (donde es alta, el modelo no sabia)"),
    )
    for k in range(K):
        fig.add_trace(
            go.Scatter(
                x=hist.index, y=hist[xi_cols[k]], name=labels[k],
                mode="lines", stackgroup="xi",
                line=dict(width=0.5, color=REGIME_COLORS[k % len(REGIME_COLORS)]),
            ),
            row=1, col=1,
        )
    # franja de entropia: donde H es alta, se hachura visualmente con relleno
    fig.add_trace(
        go.Scatter(
            x=hist.index, y=hist["entropy_norm"], name="entropia H/ln(K)",
            mode="lines", line=dict(color="#455A64", width=1),
            fill="tozeroy", fillpattern=dict(shape="/"),
        ),
        row=2, col=1,
    )
    for b in boundaries:
        fig.add_vline(x=b, line=dict(color="#90A4AE", width=1, dash="dot"))
    _add_oos_band(fig)
    fig.update_yaxes(range=[0, 1], row=1, col=1, tickformat=".0%")
    fig.update_yaxes(range=[0, 1], row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig, width="stretch")
    st.caption("Lineas punteadas verticales: fronteras de bloques de walk-forward.")

    # --- Figura 2: precio ancla coloreado por argmax(xi) ---
    st.subheader(f"Precio de {list(irfn['conditional_stats'].keys())[0]} (equity), fondo por regimen dominante")
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(
        x=hist.index, y=hist["equity"], mode="lines",
        line=dict(color="#D7DCE5", width=1.2), name="equity (base 1.0)",
    ))
    # bandas de fondo por argmax
    am = hist["argmax_idx"].to_numpy()
    changes = np.where(np.diff(am) != 0)[0]
    seg_starts = np.concatenate([[0], changes + 1])
    seg_ends = np.concatenate([changes, [len(am) - 1]])
    for s, e in zip(seg_starts, seg_ends):
        price_fig.add_vrect(
            x0=hist.index[s], x1=hist.index[e],
            fillcolor=REGIME_COLORS[am[s] % len(REGIME_COLORS)], opacity=0.12,
            line_width=0, layer="below",
        )
    for b in boundaries:
        price_fig.add_vline(x=b, line=dict(color="#90A4AE", width=1, dash="dot"))
    _add_oos_band(price_fig)
    price_fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(price_fig, width="stretch")

    # --- Toggle de diagnostico: smoother, SOLO DEV_MODE ---
    if dev_mode():
        st.divider()
        st.subheader("Diagnostico (solo DEV_MODE)")
        show_smoother = st.checkbox("Superponer smoother de Kim (punteado)", value=False)
        if show_smoother:
            sm = load_diagnostic_smoother()
            if sm is None:
                st.warning(
                    "No hay smoother de diagnostico. Corre el pipeline con IRFN_DEV_MODE=true "
                    "para generarlo en data/interim/ (nunca entra a artifacts/, R1)."
                )
            else:
                sm_aligned = sm.reindex(hist.index).dropna()
                dfig = go.Figure()
                for k in range(K):
                    dfig.add_trace(go.Scatter(
                        x=hist.index, y=hist[xi_cols[k]], mode="lines",
                        name=f"{labels[k]} (filtrado)",
                        line=dict(color=REGIME_COLORS[k % len(REGIME_COLORS)], width=1),
                    ))
                    if f"xi_smoothed_{k}" in sm_aligned.columns:
                        dfig.add_trace(go.Scatter(
                            x=sm_aligned.index, y=sm_aligned[f"xi_smoothed_{k}"], mode="lines",
                            name=f"{labels[k]} (smoother)",
                            line=dict(color=REGIME_COLORS[k % len(REGIME_COLORS)], width=1.5, dash="dot"),
                        ))
                dfig.update_yaxes(range=[0, 1], tickformat=".0%")
                dfig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                                   legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(dfig, width="stretch")
                st.warning(
                    "Smoother — solo diagnostico. Nunca se publica. Nota lo limpio que se ve; "
                    "ese es exactamente el problema: usa el futuro para pintar el pasado."
                )


main()

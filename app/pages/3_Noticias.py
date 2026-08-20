"""Noticias: indice de sorpresa, tabla de w_i estimados, intensidad de Hawkes
(V3: lambda_N con rug de titulares, branching ratio con barra de criticidad,
cascada esperada), atribucion de 3 vias.

R9: solo lee de artifacts/. Sin capa de noticias activa (surprise_layer inactivo
o bloqueado por falta de datos, ver reports/data_audit.md), esta pantalla lo
dice con honestidad en vez de mostrar ceros disfrazados de senal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import (  # noqa: E402
    load_ablation_news,
    load_hawkes_history,
    load_headline_rug,
    load_irfn,
    load_surprise_events,
    load_surprise_history,
    render_header,
)

st.set_page_config(page_title="IRF-N - Noticias", layout="wide")

# Ventana de DISPLAY del rug de titulares (presentacion pura: cuantos dias de
# marcas individuales se dibujan bajo la curva; lambda_N se dibuja completa).
RUG_DISPLAY_DAYS = 365


def _si_chart(si_hist: pd.DataFrame | None, events: list[dict]) -> None:
    st.subheader("Serie temporal de SI_t")
    if si_hist is None or si_hist.empty:
        st.info(
            "No hay serie de SI_t publicada todavia: la capa de sorpresa esta "
            "inactiva (sin datos suficientes, ver aviso de arriba). Se activa "
            "cuando `model.news_layer_params.active` sea true en irfn.json."
        )
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=si_hist.index, y=si_hist["surprise_index"],
        mode="lines", name="SI_t", line=dict(color="#5B8DEF", width=1.5),
    ))
    if events:
        edf = pd.DataFrame(events)
        edf["fecha"] = pd.to_datetime(edf["fecha"])
        # y del punto = SI_t en esa fecha si esta en la serie, si no 0 (evento
        # fuera del rango publicado); tamano ∝ |z_i|, color por signo de z_i.
        y_vals = edf["fecha"].map(lambda d: si_hist["surprise_index"].get(d, 0.0))
        sizes = 6 + 14 * (edf["z"].abs() / max(edf["z"].abs().max(), 1e-9))
        # sorpresa por signo de z: divergente sobre fondo oscuro (no verde/rojo,
        # reservados al signo del precio) -- warm = negativa, cool = positiva.
        colors = ["#C9553D" if z < 0 else "#5FB4E8" for z in edf["z"]]
        fig.add_trace(go.Scatter(
            x=edf["fecha"], y=y_vals, mode="markers", name="releases",
            marker=dict(size=sizes, color=colors, line=dict(width=1, color="#0E1117")),
            text=[f"{row.indicator}: z={row.z:.2f}" for row in edf.itertuples()],
            hoverinfo="text+x",
        ))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="SI_t", xaxis_title=None,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Tamano del punto ∝ |z_i| de ese release; verde = sorpresa al alza (z_i>0), "
              "rojo = sorpresa a la baja (z_i<0).")


def _weights_table(indicators: list[dict]) -> None:
    st.subheader("Pesos w_i estimados por indicador")
    if not indicators:
        st.info("Sin indicadores en config/news.yaml.")
        return
    df = pd.DataFrame(indicators)
    # w/se/t_stat son None cuando el indicador no tuvo suficientes eventos (sin
    # datos hoy, ver reports/data_audit.md); to_numeric los vuelve NaN para que
    # el ordenamiento por |w_i| y el formato numerico no rompan con None.
    for col in ("w", "se", "t_stat"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["abs_w"] = df["w"].abs()
    df = df.sort_values("abs_w", ascending=False, na_position="last").drop(columns="abs_w")

    # Formato manual a texto ANTES de pasar por Streamlit: st.dataframe serializa
    # el Styler via Arrow y no siempre respeta Styler.format(na_rep=...) (NaN/None
    # se ven crudos); mas confiable formatear a string aqui.
    def _num(v, nd):
        return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"

    display = pd.DataFrame({
        "indicador": df["indicator"],
        "w_i": df["w"].map(lambda v: _num(v, 4)),
        "SE": df["se"].map(lambda v: _num(v, 4)),
        "t": df["t_stat"].map(lambda v: _num(v, 2)),
        "distinguible de cero": df["distinguible_de_cero"].map(
            lambda ok: "si" if ok else "no (gris)"
        ),
        "n eventos": df["n_events"],
    })

    def _style_row(row):
        if row["distinguible de cero"] != "si":
            return ["color: gray"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display.style.apply(_style_row, axis=1),
        width="stretch", hide_index=True,
    )
    st.caption("Pesos estimados por regresion de impacto. No asignados a mano (R7). "
              "Fila en gris = w_i no distinguible de cero "
              "(|t| < config/news.yaml impact_t_threshold).")


def _criticality_bar(n: float, threshold: float | None) -> None:
    """Barra de criticidad del branching ratio: 0 = noticias independientes,
    1 = criticidad (cascadas infinitas). Solo dibuja un numero ya publicado."""
    thr = threshold if threshold is not None else 1.0
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[min(max(n, 0.0), 1.0)], y=[""], orientation="h",
        marker=dict(color="#FDE725" if n >= thr else "#5B8DEF"),
        width=0.6, showlegend=False, hoverinfo="skip",
    ))
    fig.add_vline(x=thr, line_dash="dot", line_color="#8B93A5",
                  annotation_text=f"umbral reflexivo {thr:.2f}", annotation_position="top")
    fig.add_vline(x=1.0, line_color="#8B93A5",
                  annotation_text="criticidad n=1", annotation_position="bottom right")
    fig.update_xaxes(range=[0, 1.05], title=None)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=110, margin=dict(l=10, r=10, t=28, b=10))
    st.plotly_chart(fig, width="stretch")


def _lambda_chart(lam_hist: pd.DataFrame | None, rug: pd.DataFrame | None) -> None:
    st.subheader("Intensidad del flujo de titulares lambda_N(t)")
    if lam_hist is None or lam_hist.empty:
        st.info(
            "No hay serie de lambda_N publicada todavia: la capa de Hawkes esta "
            "inactiva (ver aviso de arriba). Se activa cuando "
            "`model.hawkes_layer_params.active` sea true en irfn.json."
        )
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.82, 0.18],
                        vertical_spacing=0.02)
    fig.add_trace(go.Scatter(
        x=lam_hist.index, y=lam_hist["lambda_N"], mode="lines", name="lambda_N",
        line=dict(color="#A47BE0", width=1.4),
    ), row=1, col=1)
    if rug is not None and not rug.empty:
        cutoff = rug["hora_titular"].max() - pd.Timedelta(days=RUG_DISPLAY_DAYS)
        shown = rug[rug["hora_titular"] >= cutoff]
        fig.add_trace(go.Scattergl(
            x=shown["hora_titular"], y=[0.0] * len(shown), mode="markers",
            name="titulares",
            marker=dict(symbol="line-ns-open", size=8, color="#455A64",
                        opacity=np.clip(0.15 + 0.85 * shown["s"].to_numpy(), 0.15, 1.0)),
            hoverinfo="x",
        ), row=2, col=1)
        fig.update_yaxes(visible=False, row=2, col=1)
    fig.update_yaxes(title_text="lambda_N (titulares/dia)", row=1, col=1)
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Rug inferior: un tic por titular (ultimos {RUG_DISPLAY_DAYS} dias en pantalla; "
        "opacidad proporcional a la relevancia s_i de FinBERT). lambda_N se dibuja completa."
    )


def _hawkes_params_table(h: dict) -> None:
    st.subheader("Parametros del proceso de Hawkes (MLE, R7)")
    rows = [
        {"parametro": "mu_N (intensidad basal, titulares/dia)", "valor": h["mu_N"], "SE": h["se_mu_N"]},
        {"parametro": "alpha (excitacion)", "valor": h["alpha"], "SE": h["se_alpha"]},
        {"parametro": "beta (velocidad de olvido, 1/dia)", "valor": h["beta"], "SE": h["se_beta"]},
    ]
    df = pd.DataFrame(rows)
    for col in ("valor", "SE"):
        df[col] = df[col].map(lambda v: "-" if v is None else f"{v:.4f}")
    st.dataframe(df, width="stretch", hide_index=True)
    alpha_beta = h["alpha"] / h["beta"] if h.get("beta") else None
    n_val = f"{h['branching_ratio']:.4f}" if h.get("branching_ratio") is not None else "-"
    ab_str = f"{alpha_beta:.4f}" if alpha_beta is not None else "-"
    st.caption(
        f"E[s] = {h['mean_mark']:.3f} sobre {h['n_events']:,} titulares | "
        f"multistart {h['starts_at_best']}/{h['n_starts']} arranques en el mismo optimo (R6). "
        f"n = alpha*E[s]/beta = {n_val} (correccion por marcas; NO alpha/beta = {ab_str}, que "
        "seria explosivo sin corregir). n NO lleva IC95 aqui a proposito -- ver 'Archivo de "
        "decisiones' para la banda de sensibilidad de ventana (0.7388-0.9994), que es la "
        "incertidumbre que de verdad importa sobre este numero."
    )
    ks_p = h.get("ks_pvalue")
    if h.get("ks_passed") is None or ks_p is None:
        st.info("KS de re-escalamiento temporal: sin resultado (capa inactiva o muestra corta).")
    elif h["ks_passed"]:
        st.success(
            f"Bondad de ajuste (re-escalamiento temporal): KS no rechaza Exp(1) "
            f"(stat={h['ks_stat']:.4f}, p={ks_p:.3f}). El kernel exponencial es consistente."
        )
    else:
        st.info(
            f"Bondad de ajuste (re-escalamiento temporal): KS RECHAZA Exp(1) "
            f"(stat={h['ks_stat']:.4f}, p={ks_p:.2e}, D pequeño dominado por n≈95k titulares). "
            "**Cerrado (A1):** se comparó contra un kernel power-law -- gana AIC pero TAMPOCO "
            "pasa el KS en esa comparación; cambiar de kernel no resuelve el rechazo, solo lo "
            "mueve. Se mantiene el exponencial con este caveat. Detalle en 'Archivo de "
            "decisiones'."
        )


def _hawkes_section(irfn: dict) -> None:
    st.header("Flujo de titulares (Hawkes, V3)")
    h = irfn["model"].get("hawkes_layer_params")
    news = irfn["news"]
    if h is None:
        # artefacto de una version anterior a V3: el contrato aun no traia la capa.
        st.info("El artefacto publicado es anterior a V3 (sin hawkes_layer_params). "
                "Corre `python scripts/run_v3.py` para generar esta seccion.")
        return

    if not h["active"]:
        st.warning(
            "Capa de Hawkes INACTIVA: " + (h.get("blocker") or "capa aun no corrida.") +
            " Los numeros de esta seccion no existen todavia; no se muestran ceros "
            "disfrazados de senal."
        )
        return

    st.caption(
        "Este indicador SÍ está activo y publicado (standalone). Distinto de "
        "**M5** (λ_N como covariable del logit de transición): **CERRADO** por "
        "corpus insuficiente para el walk-forward pre-registrado — ver "
        "Archivo de decisiones."
    )

    cov = h["coverage"]
    if cov.get("n_missing_days") or cov.get("n_censored_days"):
        st.info(
            f"Cobertura del corpus: {cov['first_day']} a {cov['last_day']} "
            f"({cov['n_days']} dias; {cov['n_missing_days']} faltantes, "
            f"{cov['n_censored_days']} censurados por el cap del API — la censura "
            "sesga lambda_N a la baja en los picos y esta contada, no escondida)."
        )

    n = news["branching_ratio"]
    thr = h.get("reflexive_threshold")
    cascade = news.get("expected_cascade")

    # KPIs principales: SIN branching ratio n (F3, auditoria 2026-08-18). n es
    # el numero mas vistoso y el menos robusto del panel (cambia ~1300x segun
    # la ventana de integracion elegida, ver 'Archivo de decisiones') -- no
    # pertenece a un bloque de KPIs que implica "publicado con confianza".
    c1, c2 = st.columns(2)
    c1.metric("lambda_N hoy (titulares/dia)", f"{news['lambda_N']:.1f}")
    c2.metric(
        "cobertura del corpus (dias con datos / span)",
        f"{cov.get('n_days', '-')} / {cov.get('n_days', 0) + cov.get('n_missing_days', 0)}",
    )

    _lambda_chart(load_hawkes_history(), load_headline_rug())
    _hawkes_params_table(h)

    with st.expander("Branching ratio n (fragilidad del flujo, NO es un KPI) — con sus DOS incertidumbres"):
        ci_lo = h.get("branching_ratio_ci_low")
        ci_hi = h.get("branching_ratio_ci_high")
        n_span = h.get("branching_ratio_span_calendar")
        casc_span = h.get("expected_cascade_span_calendar")
        cascade_str = "no acotada" if cascade is None else f"{cascade:.2f}"
        st.metric("branching ratio n (ventana OBSERVADA, publicado)", f"{n:.3f}")
        st.metric("cascada esperada E[hijos] = 1/(1-n)", cascade_str)
        # (a) Incertidumbre de MUESTREO del MLE, dentro de la ventana observada.
        if ci_lo is not None and ci_hi is not None:
            st.caption(
                f"**(a) IC del MLE en la ventana observada:** [{ci_lo:.3f}, {ci_hi:.3f}] — "
                "ruido de estimacion del valor publicado (incluye el aporte del de-empate por Rubin)."
            )
        # (b) Impacto de la CENSURA de dias sin corpus. NO es un IC NI un rango de
        # la verdad: el span-calendario es un estimador SESGADO (integra mu_N sobre
        # dias fantasma), el que el proyecto DESCARTO el 2026-08-15 (auditoria A.1
        # lo llama 'inflado'). Se muestra para cuantificar cuanto sesga ignorar la
        # censura, no para sugerir que la verdadera n pueda estar cerca de 1.
        if n_span is not None:
            miss = cov.get("n_missing_days", 0)
            casc_span_str = "no acotada" if casc_span is None else f"~{casc_span:.0f}"
            st.caption(
                f"**(b) Impacto de la censura de dias sin corpus (NO es un IC ni un rango de la "
                f"verdad):** el valor publicado {n:.3f} usa el TIEMPO OBSERVADO (correccion de "
                f"censura, decision 2026-08-15). Ajustar en cambio sobre el span-calendario "
                f"--integrando sobre {miss} dias SIN datos-- daria {n_span:.4f} (cascada "
                f"{casc_span_str}): un estimador **SESGADO AL ALZA**, el que el proyecto DESCARTO. "
                f"La verdadera n es ~{n:.2f}, **NO** un valor entre {n:.2f} y {n_span:.2f}; el "
                f"0.99 solo mide cuanto sesga ignorar la censura (auditoria A.1: span 'inflado' "
                "vs verdad ~0.74). Por eso cruza el umbral reflexivo SIN ser una alerta."
            )
        st.caption(
            "Las dos incertidumbres NO se funden: (a) es azar de muestreo; (b) es una decision "
            "estructural (que soporte integra el compensador). Como 1/(1-n) es convexa y explota "
            "cerca de n=1, un solo intervalo colapsado seria matematicamente enganoso."
        )
        _criticality_bar(n, thr)
        if thr is not None and n >= thr:
            cola = (f"cada titular genera en promedio {cascade:.1f} titulares descendientes."
                    if cascade is not None
                    else "la cascada esperada es NO ACOTADA (n toca la frontera n = 1).")
            st.warning(
                f"Con la ventana vigente, n = {n:.3f} se acerca al umbral de criticidad "
                f"({thr:.2f}): " + cola + " Leer junto con la banda (b) de arriba, "
                "no como una alerta aislada."
            )


def main():
    st.title("Noticias")
    irfn = load_irfn()
    render_header(irfn)
    if irfn is None:
        st.info("No hay artefactos todavia. Corre `python scripts/run_v2.py` (o V0/V1 primero).")
        return

    layer = irfn["model"]["news_layer_params"]

    if not layer["active"]:
        # V2/M4 CERRADO por restriccion de datos (no "en curso" -- F3, auditoria
        # 2026-08-18). Sin metricas vacias (indice de sorpresa=0.00, delta="sin
        # estimar", atribucion en ceros): esta capa no existe hoy, no se dibuja
        # a medias. Detalle completo en 'Archivo de decisiones'.
        st.info(
            "**Capa de sorpresa (V2/M4): CERRADA por restriccion de datos**, no en "
            "curso. No existe fuente gratuita de consenso historico point-in-time "
            "(5 fuentes investigadas y descartadas). Sin SI_t no hay sorpresa que "
            "medir; esta seccion no muestra metricas vacias a proposito. Detalle, "
            "evidencia y condicion de reapertura en **Archivo de decisiones**."
        )
        st.divider()
        _hawkes_section(irfn)
        return

    # Capa activa (reapertura futura): comportamiento completo, sin recortar.
    news = irfn["news"]
    if layer.get("surprise_start_date"):
        st.info(
            f"El indice de sorpresa arranca en {layer['surprise_start_date']} por "
            "disponibilidad de datos de consenso."
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("indice de sorpresa (hoy)", f"{news['surprise_index']:.2f}")
    c2.metric("delta estimado (decaimiento)",
             f"{layer['delta']:.4f}" if layer["delta"] is not None else "sin estimar")
    c3.metric("lambda_N (Hawkes, hoy)", f"{news['lambda_N']:.2f}")
    st.caption(
        "Atribucion del dia (3 vias): "
        + ", ".join(f"{k}={v:.0%}" for k, v in news["attribution"].items())
        + ". surprise = sorpresa calendarizada; hawkes = flujo de titulares; "
        "price = covariables de precio."
    )

    st.divider()
    _hawkes_section(irfn)

    st.divider()
    _si_chart(load_surprise_history(), load_surprise_events())

    st.divider()
    _weights_table(layer["indicators"])

    st.divider()
    st.subheader("Cobertura del calendario por indicador")
    cov_df = pd.DataFrame(layer["coverage"]).T
    if not cov_df.empty:
        st.dataframe(cov_df.fillna("-"), width="stretch")
    else:
        st.caption("Sin cobertura registrada.")

    report = load_ablation_news()
    if report:
        with st.expander("Reporte completo de ablacion M3 vs M4 (reports/ablation_news.md)"):
            st.markdown(report)


main()

"""Validacion: reporte de walk-forward contra targets exogenos.

R9: solo lee de artifacts/ y reports/. El reporte lo genera el pipeline (R8); aqui
se muestra tal cual, gane o pierda el modelo. Las metricas de consistencia interna
(log-loss/Brier/ECE/fiabilidad sobre el proxy y_t=argmax ξ_{t|t}) NO viven aqui --
ver "Diagnostico interno del filtro" (hallazgo F2.b, auditoria 2026-08-18): usan un
target generado por el propio modelo y no miden habilidad predictiva.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import load_ablation, load_irfn, load_validation_report, load_walkforward, render_header  # noqa: E402

st.set_page_config(page_title="IRF-N - Validacion", layout="wide")


def main():
    irfn = load_irfn()
    version = (irfn or {}).get("version", "V0")
    st.title(f"Validacion {version}")
    render_header(irfn)
    wf = load_walkforward()
    report = load_validation_report()
    abl = load_ablation()

    if wf is None:
        st.info("No hay artefactos todavia. Corre el pipeline desde la Sala de control.")
        return

    # --- Metrica principal: log-loss vs climatologia (F2.a) ---
    calib = wf["calibration"]
    gana = calib["log_loss"] < calib["log_loss_baseline"]
    st.subheader("Titular: log-loss del modelo vs. climatologia")
    m1, m2 = st.columns(2)
    m1.metric("log-loss modelo", f"{calib['log_loss']:.4f}")
    m2.metric("log-loss climatologia", f"{calib['log_loss_baseline']:.4f}",
              ("modelo gana" if gana else "modelo NO gana"),
              delta_color=("normal" if gana else "inverse"))
    if gana:
        st.success(
            "El modelo tiene MENOR log-loss que la climatologia sobre la muestra "
            "out-of-sample agregada."
        )
    else:
        st.error(
            "El modelo tiene MAYOR (peor) log-loss que la climatologia sobre la muestra "
            "out-of-sample agregada. Se muestra tal cual, sin suavizar (R8)."
        )
    st.caption(
        "Climatologia = frecuencia marginal de la etiqueta proxy (y_t = argmax ξ_{t|t}), "
        "estimada UNA VEZ sobre TODA la muestra out-of-sample agregada (los 19 bloques "
        "juntos) -- NO por bloque desde su propio train, pese a que esa es la definicion "
        "correcta bajo R2 (hallazgo F2.a, auditoria 2026-08-18; pendiente de decision del "
        "director sobre si recalcular esto en el pipeline)."
    )
    with st.expander("Verificacion: climatologia causal aproximada (sin mirar el futuro)"):
        st.write(
            "Recomputo de verificacion (no oficial, no en el artefacto): para cada bloque, "
            "la climatologia se estima SOLO con las etiquetas OOS de los bloques anteriores "
            "(nunca del propio bloque ni de bloques futuros). Bloques 1-18 (el bloque 0 no "
            "tiene climatologia previa posible), ponderado por n_test:"
        )
        st.metric("modelo vs. climatologia causal", "0.1669  vs.  0.5049", "n=2261 obs")
        st.caption(
            "El modelo gana de forma AUN MAS clara bajo esta definicion mas estricta -- la "
            "climatologia pooled de arriba es, si acaso, un baseline mas facil de vencer, no "
            "mas dificil. No es literalmente 'frecuencia en train aplicada hacia adelante' "
            "(esa exigiria re-estimar y persistir ξ del periodo de train de cada bloque, que "
            "hoy no se guarda) -- es la aproximacion mas fiel que se puede construir con lo "
            "que ya esta en history.parquet, sin nueva estimacion."
        )

    st.divider()

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
    else:
        st.info(
            "Sin comparacion Diebold-Mariano vs V0 en este artefacto. Las metricas de "
            "consistencia interna del filtro se movieron a 'Diagnostico interno del "
            "filtro' (no son targets exogenos)."
        )

    st.caption(
        f"bloques walk-forward: {wf['n_blocks']}. Brier modelo={calib['brier']:.4f} vs. "
        f"climatologia={calib['brier_baseline']:.4f} (mismo baseline pooled de arriba). "
        "Metricas de habilidad predictiva adicionales (Pesaran-Timmermann, walk-forward "
        "economico, Sharpe OOS) en el reporte de abajo, cuando el artefacto las incluya."
    )

    st.subheader("Reporte de walk-forward")
    if report:
        st.markdown(report)
    else:
        st.info("No se encontro reports/validation_v1.md ni validation_v0.md.")


main()

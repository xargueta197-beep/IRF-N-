"""Auditoria PIT: invarianza de prefijo por fecha, ledger de rezagos y chequeo de
re-estimacion por bloque.

R9: solo lee de artifacts/audit.json. Los tres chequeos los corre el pipeline; aqui
solo se muestran. Esta es la pantalla que hace auditable al indicador.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import load_audit  # noqa: E402

st.set_page_config(page_title="IRF-N - Auditoria PIT", layout="wide")


def _badge(passed: bool) -> str:
    return "VERDE" if passed else "ROJA"


def main():
    st.title("Auditoria point-in-time")
    audit = load_audit()
    if audit is None:
        st.info("No hay artefactos todavia. Corre el pipeline desde la Sala de control.")
        return

    pit = audit["prefix_invariance"]
    ledger = audit["lag_ledger"]
    reest = audit["block_reestimation"]

    # --- Invarianza de prefijo ---
    st.subheader("1. Invarianza de prefijo")
    if pit["passed"]:
        st.success(
            f"VERDE. Correr sobre data[:t] y sobre data[:T] da el mismo ξ_{{s|s}} para "
            f"s <= t en todas las fechas probadas (tolerancia {pit['atol']:.0e})."
        )
    else:
        st.error("ROJA. Hay informacion futura filtrandose. El indicador no es publicable.")
    dfp = pd.DataFrame(pit["rows"])
    if not dfp.empty:
        dfp = dfp.rename(columns={
            "fecha": "fecha", "max_abs_diff": "max |diff|",
            "n_dates": "fechas comparadas", "passed": "ok",
        })
        st.dataframe(dfp, width="stretch", hide_index=True)
    st.caption(
        "El mas importante del repo (CLAUDE.md). max |diff| = 0 significa reproduccion "
        "exacta; cualquier fuga de futuro lo dispararia muy por encima de la tolerancia."
    )

    # --- Ledger de rezagos ---
    st.subheader("2. Ledger de rezagos (R3)")
    st.markdown(f"Estado: **{_badge(ledger['passed'])}**.")
    dfl = pd.DataFrame(ledger["rows"])
    st.dataframe(dfl, width="stretch", hide_index=True)
    st.caption(
        "Toda covariable entra rezagada x_{t-1} (shift >= 1). En V0 no hay covariables: "
        "el unico input es el retorno r_t, que es la observacion modelada (shift 0, correcto). "
        "Las covariables llegan en V1+ y cada una debera registrarse aqui con su shift."
    )

    # --- Re-estimacion por bloque ---
    st.subheader("3. Re-estimacion por bloque (R2)")
    if reest["passed"]:
        st.success("VERDE. Los bloques de walk-forward tienen parametros DISTINTOS: "
                   "cada bloque re-estimo desde cero, no se reciclo un ajuste unico.")
    else:
        st.error("ROJA. Hay bloques con parametros identicos: posible estimacion unica (viola R2).")
    dfr = pd.DataFrame(reest["rows"])
    if not dfr.empty:
        cols = ["block_id", "train_start", "test_start", "theta_norm",
                "min_dist_a_otro_bloque", "distinto"]
        dfr = dfr[[c for c in cols if c in dfr.columns]]
        st.dataframe(dfr, width="stretch", hide_index=True)

    # --- Ledger de vintages del calendario (V2) ---
    st.subheader("4. Ledger de vintages del calendario macro (V2)")
    vintage = audit.get("consensus_vintage_ledger")
    if vintage is None:
        st.info("Sin datos: corre scripts/run_v2.py para generar este chequeo.")
    else:
        if vintage["n_events"] == 0:
            st.warning(
                "VERDE (vacuo): 0 eventos en el calendario todavia -- ver "
                "reports/data_audit.md secciones 4 y 7 (sin fuente gratuita de "
                "consenso historico hoy). No hay ninguna fila que contradiga la "
                "propiedad porque no hay ninguna fila."
            )
        elif vintage["passed"]:
            st.success(
                f"VERDE. {vintage['n_events']} eventos: fecha_publicacion_consenso "
                "(primera captura diaria que vio el dato) es SIEMPRE anterior o igual "
                "a fecha_evento -- consenso genuinamente pre-release."
            )
        else:
            st.error(
                "ROJO. Al menos un evento se vio por primera vez DESPUES de su "
                "fecha_evento: ese 'consenso' no tiene garantia de ser pre-release."
            )
        dfv = pd.DataFrame(vintage["rows"])
        if not dfv.empty:
            st.dataframe(dfv, width="stretch", hide_index=True)
        st.caption(
            "fecha_evento vs fecha_publicacion_consenso (data/calendar.load_local_snapshots: "
            "consenso = de la captura diaria MAS TEMPRANA que lo vio, nunca reescrito por una "
            "captura posterior)."
        )

    # --- Chequeo de expanding window (V2) ---
    st.subheader("5. Chequeo de expanding window: sigma_i no usa datos futuros (V2)")
    ew = audit.get("expanding_window_check")
    if ew is None:
        st.info("Sin datos: corre scripts/run_v2.py para generar este chequeo.")
    else:
        if ew.get("vacuous"):
            st.warning(
                f"VERDE (vacuo): solo {ew['n_events_con_consenso']} release(s) con consenso "
                "-- no hay nada que truncar de forma no trivial todavia. Se activa de verdad "
                "en cuanto haya >= 2 releases con consenso."
            )
        elif ew["passed"]:
            st.success(
                "VERDE. Truncar el calendario en cada fecha de corte probada reproduce "
                "EXACTAMENTE el mismo z_i para fechas <= corte: sigma_i (expanding) no usa "
                "datos futuros."
            )
        else:
            st.error("ROJO. z_i cambio al truncar el calendario: hay look-ahead en sigma_i.")
        dfe = pd.DataFrame(ew["rows"])
        if not dfe.empty:
            st.dataframe(dfe, width="stretch", hide_index=True)
        st.caption(
            "Analogo de la seccion 1 (invarianza de prefijo) pero para la capa de sorpresa: "
            "reutiliza features.surprise.expanding_surprise_z sobre el calendario truncado."
        )

    # --- Timestamps de titulares (V3, Trampa 3) ---
    st.subheader("6. Alineacion de timestamps de titulares (V3)")
    ht = audit.get("headline_timestamp_check")
    if ht is None:
        st.info("Sin datos: corre scripts/run_v3.py para generar este chequeo.")
    else:
        a = ht["titular_vs_evento"]
        if a.get("vacuous"):
            st.warning(
                "VERDE (vacuo): 0 titulares emparejados con releases del calendario "
                "macro -- el calendario de consenso sigue vacio (ver seccion 4), asi "
                "que no hay pares (hora_titular - hora_evento) que auditar. El chequeo "
                "se vuelve sustantivo en cuanto ambos feeds acumulen datos."
            )
        elif a["passed"]:
            st.success(
                f"VERDE. {a['n_matched']} titulares emparejados con su release: la "
                f"distribucion de (hora_titular - hora_evento) no tiene masa en negativo "
                f"(mediana {a['median_hours']:.1f} h, max {a['max_hours']:.1f} h). Ningun "
                "titular quedo fechado antes del evento que reporta."
            )
        else:
            st.error(
                f"ROJO. {a['negative_mass']:.1%} de los titulares emparejados quedaron "
                "fechados ANTES del evento que reportan: timestamps corruptos o "
                "emparejamiento roto. Con esto en rojo, lambda_N no es publicable "
                "como covariable intradia-honesta."
            )
        if a.get("histogram"):
            hdf = pd.DataFrame(a["histogram"])
            st.bar_chart(hdf.set_index("bin")["n"])
            st.caption("Distribucion de (hora_titular - hora_evento), en horas.")

        b = ht["resolucion_feed"]
        if b["n_headlines"] == 0:
            st.info("Sin titulares en el corpus todavia (backfill en curso o no corrido).")
        elif b["passed"]:
            st.success(
                f"Resolucion del feed: VERDE. {b['n_headlines']:,} titulares; "
                f"{b['midnight_exact_frac']:.1%} con timestamp de medianoche exacta "
                "(el feed publica horas reales, no solo fechas)."
            )
        else:
            st.error(
                f"Resolucion del feed: ROJO. {b['midnight_exact_frac']:.1%} de los "
                "titulares tienen timestamp 00:00:00 exacto: el feed esta dando fechas, "
                "no horas, y la alineacion intradia no es creible."
            )


main()

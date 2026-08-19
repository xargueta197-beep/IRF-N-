"""Archivo de decisiones: modulos CERRADOS con causa documentada.

R9: solo lee de artifacts/ y reports/ (texto estatico curado a partir de reportes
ya publicados -- ninguna cifra de aqui se calcula en la app). M3 (macro), V2/M4
(sorpresa) y M5 (GDELT como covariable del logit) no son deuda pendiente: son
decisiones tomadas y documentadas (README "Estado de modulos"). Esta pantalla
existe para que un lector no las redescubra como si fueran trabajo en curso
(hallazgo F3, auditoria 2026-08-18).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components import load_irfn, render_header  # noqa: E402

st.set_page_config(page_title="IRF-N - Archivo de decisiones", layout="wide")


def _decision(titulo: str, intento: str, evidencia: str, motivo: str, reapertura: str) -> None:
    st.subheader(titulo)
    st.markdown(f"**Qué se intentó.** {intento}")
    st.markdown(f"**Qué evidencia se obtuvo.** {evidencia}")
    st.markdown(f"**Por qué se cerró.** {motivo}")
    st.markdown(f"**Qué reabriría el eje.** {reapertura}")


def main():
    st.title("Archivo de decisiones")
    render_header(load_irfn())
    st.caption(
        "Estos tres ejes están CERRADOS, no en curso. 'Cerrado' significa que la "
        "pregunta se hizo, se corrió el análisis correspondiente y se documentó "
        "el resultado -- sea cual sea. No hay tarea pendiente aquí salvo que se "
        "cumpla la condición de reapertura de cada uno."
    )

    st.divider()
    _decision(
        "M3 — Covariables macro en el logit de transición",
        "Ablación M3 (`slope_2s10y`, `hy_oas_z` sobre `sma_gap`+`bb_width_z` de M2) "
        "con regularización L1 correcta y walk-forward de 17 bloques "
        "(`reports/ablation_m3_l1.md`).",
        "Diebold-Mariano M3 vs. M2: **DM=1.038, p=0.299** (diferencia media "
        "+0.00468 log-loss OOS/obs — M3 ligeramente PEOR, no distinguible de "
        "cero). Sin L1 la macro se veía peor todavía, pero era artefacto de "
        "sobreajuste; con regularización adecuada el resultado es un empate "
        "estadístico, no una mejora.",
        "La macro no aporta valor predictivo distinguible fuera de muestra "
        "sobre M2, ni con regularización correcta. El único aporte robusto de "
        "toda la escalera M0-M5 sigue siendo M1 (regímenes) sobre M0.",
        "Nueva evidencia de que condicionar la transición con macro aporta en "
        "otra especificación, ventana o conjunto de indicadores. No hay trabajo "
        "de datos pendiente — `ALFRED_API_KEY` ya está configurada.",
    )

    st.divider()
    _decision(
        "V2 / M4 — Índice de sorpresa (consenso económico)",
        "Investigación de 5 fuentes de consenso histórico point-in-time gratuito: "
        "Investing.com, Forex Factory, Econoday, FXStreet, Finnhub economic "
        "calendar (`reports/data_audit.md`, secciones 4, 7-9).",
        "Las 5 descartadas con evidencia directa: términos de servicio que "
        "prohíben scraping/redistribución (Forex Factory, Investing.com), reto "
        "anti-bot activo, o error HTTP en vivo (Finnhub 403, Trading Economics "
        "demo 410/401). `capture_consensus.py` corre en modo acumulación hacia "
        "adelante desde la Sesión 0 — 0 eventos válidos capturados hasta hoy.",
        "No existe ninguna fuente gratuita y honesta de consenso histórico. Sin "
        "consenso no hay sorpresa (z_i = (actual − consenso)/σ) que estimar — "
        "esto no es un bug ni una covariable mal calculada, es una ausencia real "
        "de datos.",
        "Pagar la key de Trading Economics (point-in-time), o acumular "
        "≥30 eventos válidos hacia adelante (el cron ya corre; hoy 0, sigue "
        "activo, pasivo).",
    )

    st.divider()
    _decision(
        "M5 — Hawkes (λ_N) como covariable del logit de transición",
        "Backfill del corpus de titulares GDELT hacia los ~7 años de historia "
        "(train 4a + 6 bloques de 6m) que exige el walk-forward pre-registrado "
        "M5 vs. M4 (`reports/ablation_news.md`).",
        "Corpus detenido en **240 de 998 días** de span (758 días no "
        "capturados). Cobertura alineada real ≈1.7 años; el walk-forward "
        "pre-registrado exige ≥7.0 años. Cuello de botella: rate limit de "
        "GDELT sobre IP compartida (CGNAT), no cómputo.",
        "Decisión explícita del director/usuario (2026-08-12): NO perseguir el "
        "100% del histórico 2017→hoy. 240 días alcanza para AJUSTAR e inspeccionar "
        "el Hawkes (≥200 eventos, ver más abajo) pero no para la ablación "
        "walk-forward M5-vs-M4 pre-registrada — encoger la malla de bloques para "
        "que \"quepa\" sería trampa (R8). **El indicador Hawkes SÍ se publica** "
        "como indicador standalone (branching ratio, cascada, KS) en la pantalla "
        "Noticias — eso es independiente de esta decisión, que es solo sobre "
        "usarlo como covariable λ_N_z del logit.",
        "Retomar el backfill (script reanudable, `scripts/capture_headlines.py` "
        "+ `RESUME_GDELT.md`) hasta cubrir ~7 años de corpus alineado.",
    )

    st.divider()
    _decision(
        "A1 — Kernel del Hawkes: exponencial vs. power-law",
        "Comparación directa AIC + KS de re-escalamiento entre kernel exponencial "
        "y power-law sobre la misma ventana de prueba (`_compare_kernels.log`, "
        "`src/irfn/models/hawkes_powerlaw.py`).",
        "Power-law gana AIC (ΔAIC=+179.6 a su favor: −110228.8 exp vs. "
        "−110408.4 power-law) pero **ninguno de los dos pasa el KS** en esa "
        "ventana (exp p=3.4e−17, power-law p=6.7e−13) — ambos rechazan la "
        "hipótesis de re-escalamiento correcto. El branching ratio también "
        "cambia de kernel a kernel (n=0.695 exp vs. n=0.836 power-law, misma "
        "ventana).",
        "Se mantiene el kernel exponencial: más simple, con el mismo problema "
        "de bondad de ajuste que el candidato (cambiar de kernel no resuelve el "
        "rechazo del KS, solo lo cambia de forma). El rechazo se reporta con "
        "honestidad (tamaño de efecto D pequeño, dominado por n≈95k titulares, "
        "no un desajuste grande de forma) en vez de maquillarse.",
        "Un kernel que sí pase KS, o evidencia de que el rechazo es puramente "
        "un artefacto del tamaño de muestra (n≈10^5) y no un desajuste real de "
        "forma.",
    )

    st.divider()
    st.subheader("Nota aparte: por qué el branching ratio (n) salió de los KPIs")
    st.warning(
        "n es el número más vistoso del panel y el menos robusto: cambia por un "
        "factor ~1300x según qué ventana temporal se integre. Esa decisión se "
        "tomó (span calendario, n=0.9994, 2026-08-14) y se revirtió al día "
        "siguiente (tiempo observado, n=0.7388, 2026-08-15). Un IC95 "
        "[0.731, 0.747] sobre ese número transmite una precisión que el propio "
        "historial de la decisión contradice — el IC captura el ruido del MLE "
        "dentro de UNA elección de ventana, no la sensibilidad a esa elección."
    )
    st.markdown(
        "**Banda de sensibilidad de ventana (no un IC estadístico):** "
        "**n ∈ [0.7388, 0.9994]** según se integre μ_N sobre el tiempo "
        "observado (241 días, decisión vigente) o sobre el span calendario "
        "completo (998 días, decisión revertida). Ver `reports/validation_v3.md`, "
        "sección \"Corrección de la ventana de observación\"."
    )


main()

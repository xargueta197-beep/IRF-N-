# Nota de decisiones para el director — Franja 2 (rigor matemático)

**Fecha:** 2026-08-19
**Para:** Xavier (director de proyecto)
**De:** Claude Code (implementación)
**Regla (CLAUDE.md §3, R3):** esta nota **propone y cuantifica**; no implementa nada. Ninguna
palanca se toca sin tu OK explícito.
**Documentos de referencia consultados (obligatorio antes de proponer metodología):**
- *Guía de Implementación* — leída en vivo (Google Doc, 2026-08-19). Citas verbatim abajo.
- *Reglas Absolutas* — vía la reproducción canónica R1–R9 de `CLAUDE.md` (R3, R7, R8, R9 aplican aquí).

> **Alcance.** La nota previa `nota_decisiones_director_2026-08-16.md` cubría A1 (kernel),
> A4 (M1 vs M2) y A6 (degenerado). **A4 YA SE EJECUTÓ**: M1 es el publicado (`9d30960c94cc`).
> Esta nota cierra las **tres palancas restantes** de la Franja 2, cada una re-verificada con
> números en vivo del artefacto publicado `9d30960c94cc`:
> - **#4 (M-1) Kernel del Hawkes** — exponencial vs power-law.
> - **#3 (M-3) Cómo reportar `n`** — separar dos incertidumbres (NUEVO, no estaba en la nota previa).
> - **#5 (M-2) Régimen degenerado** — documentar vs reparametrizar.

---

## #4 · Kernel del Hawkes — el KS rechaza el exponencial

### Hechos verificados (artefacto `9d30960c94cc`, corpus completo, exponencial)

| Cantidad | Valor |
| :-- | :-- |
| n_events | 95 234 |
| branching ratio `n` | **0.7385**  (IC95 [0.7311, 0.7465]) |
| umbral reflexivo | 0.8  (⇒ `n` publicado está **por debajo**) |
| KS de re-escalamiento | **D = 0.03059**, p = 7.5e-78, `ks_passed = False` |

**Interpretación del KS (auditoría MATH-D7, S5).** El rechazo lo **domina el tamaño muestral**,
no un desajuste grande de forma:
- D_crít(5%) ≈ 1.36/√n ≈ 1.36/√95234 ≈ **0.0044** ⇒ D_obs/D_crít ≈ **6.9**. A n≈95k, un D
  minúsculo rechaza por construcción.
- Es un test tipo **Lilliefors** (parámetros estimados de la misma muestra) ⇒ el p asintótico
  **no es exacto** (sobreestima el rechazo). El p literal no es interpretable; el **tamaño de
  efecto** (D≈0.03, pequeño) sí.

**Guía (verbatim):** *"Si el KS falla, la respuesta **no** es forzar el modelo. Es reportarlo y
considerar un kernel power-law en V3+."* → **Reportarlo ya está hecho** (warning del artefacto
con D, p, n). "Considerar power-law" es opcional, no un mandato.

### El power-law YA se comparó — y falla el criterio pre-registrado

`scripts/compare_hawkes_kernels.py` ajustó ambos kernels sobre la **misma** ventana contigua
(2026-05-01..06-01; 10 704 eventos; el power-law es O(n²) ⇒ **inviable exacto** sobre los 95k):

| Kernel | `n` | KS p | AIC | arranques | numérica |
| :-- | :-: | :-: | :-: | :-: | :-- |
| Exponencial | 0.6946 | 3.42e-17 | −110 228.8 | 20/20 | limpia |
| Power-law | 0.8359 | 6.70e-13 | **−110 408.4** | **10/20** | **overflow en el compensador** |

**Criterio de aceptación pre-registrado** (nota 2026-08-16, declarado ANTES de mirar): adoptar
power-law solo si **(a)** ΔAIC > 0 **y (b)** D_pl < D_exp en **≥ 20 %** **y (c)** n_pl < 1.

Evaluación (D estimado por inversión asintótica de Kolmogorov, p ≈ 2·e^(−2nD²), n=10 704):
- D_exp ≈ **0.0425**, D_pl ≈ **0.0366** ⇒ reducción **13.7 %**.

| Criterio | Umbral | Resultado | ¿Cumple? |
| :-- | :-- | :-- | :-: |
| (a) ΔAIC > 0 | > 0 | +179.6 | **Sí** |
| (b) D_pl < D_exp en ≥20% | ≥ 20 % | **13.7 %** | **NO** |
| (c) n_pl < 1 | < 1 | 0.836 | Sí |

**Por la propia regla pre-registrada del director, el power-law NO se adopta:** falla (b). El
AIC mejora, pero el power-law **reduce el desajuste de forma solo un 13.7 %** — no lo suficiente
para justificar cambiar el titular, y **sigue sin pasar el KS**.

*Caveat de rigor:* los D salen de la **inversión asintótica** del p (aproximada, más aún siendo
Lilliefors). El margen (13.7 % vs 20 %) más la fragilidad numérica (10/20 arranques, overflow)
más la inviabilidad O(n²) sobre el corpus completo hacen la conclusión robusta. Si quieres el D
**exacto** antes de decidir, re-corro `compare_hawkes_kernels.py` (~22 min).

### Opciones y recomendación

- **Opción 1 (recomendada): mantener exponencial + caveat KS.** Es lo publicado. Coherente con
  la Guía ("reportarlo"), con MATH-D7 (efecto pequeño, rechazo n-driven) y con el criterio
  pre-registrado (b falla). Cero cómputo, cero riesgo. **El power-law no compra honestidad
  (tampoco pasa el KS), solo AIC.**
- **Opción 2: adoptar power-law como titular.** Mejor AIC, pero: (i) mueve el número que se
  comunica hacia 0.84 (más cerca del umbral reflexivo 0.8) sobre una ventana de 31 días **no
  comparable** al corpus completo; (ii) requiere resolver O(n²) + la inestabilidad numérica para
  ajustar sobre los 95k; (iii) re-correr walk-forward (R2) + R6; (iv) **sigue fallando el KS**.

> ☐ **DECISIÓN #4:** ___ Opción 1 (exponencial + caveat) ___ Opción 2 (power-law) ___ re-correr para D exacto antes de decidir

---

## #3 · Cómo reportar `n` — dos incertidumbres que NO se pueden colapsar

La Guía dice que `n` es un **indicador de fragilidad publicado a diario** (*"Cuando n → 1...
al borde de la criticidad. Ese número, publicado a diario, es un indicador de fragilidad"*).
Por eso importa cómo se comunica su incertidumbre. Hay **tres fuentes, matemáticamente
distintas** (verificadas en vivo):

**Fuente 1 — muestreo del MLE, dentro de una ventana fija.**
SE(`n`) = 0.0039, IC95 **[0.7311, 0.7465]** (por delta, hessiano). Es incertidumbre
**aleatoria**, condicional a una elección de ventana y kernel.

**Fuente 2 — de-empate (dithering), imputación múltiple (Rubin).**
`reports/dithering_sensitivity_v3.md`: `n` agrupado = 0.73991; varianza total T = 1.621e-05 ⇒
SE 0.00394 → **0.00403 (+2.1 %)**. **Segundo orden.** El multistart es **unimodal** (5 semillas,
logL bit-idéntica a 6 decimales, 30/30). El dithering NO domina lo que se publica.

**Fuente 3 — ELECCIÓN de ventana (soporte del compensador): tiempo observado vs span calendario.**

| Ventana | `n` | cascada E[hijos] = 1/(1−n) |
| :-- | :-: | :-: |
| tiempo observado (**publicado**) | 0.7385 | 3.82 |
| span calendario (diagnóstico) | 0.9994 | ≈ 1667 |

Esto es incertidumbre **estructural/definicional de PRIMER orden**, **no** un intervalo de
confianza. Y es enorme por una razón matemática precisa:

1. **La cascada es convexa y explota cerca de n=1.** 1/(1−n) pasa de 3.82 a ~1667 (≈ **436×**)
   por un cambio de definición. Por Jensen, `cascada(media de n) ≠ media de cascada(n)`: **un
   intervalo único colapsado sobre la cascada es matemáticamente mal definido** cerca de n=1.
2. **La elección de ventana cruza el umbral reflexivo (0.8).** Observado n=0.74 está **por
   debajo** (no reflexivo); calendario n=0.9994 está **por encima** (criticidad). Colapsar las
   dos en un solo número **borraría la distinción exacta que la Guía dice publicar**.

Colapsar Fuente 1–2 (ruido de muestreo) con Fuente 3 (una decisión de definición) es un **error
de categoría**: presentaría una elección estructural como si fuera azar de muestreo.

### Propuesta (cambio de REPORTE ⇒ R3/R7, requiere tu OK)

Publicar `n` (y la cascada) con **dos incertidumbres etiquetadas y separadas**, nunca fundidas:
- **(a) IC del MLE en la ventana observada:** [0.7311, 0.7465], incluyendo el aporte de dithering
  por Rubin (+2.1 %). Es la incertidumbre honesta del **valor publicado**.
- **(b) Banda de sensibilidad de ventana:** {observado 0.7385 (publicado) · span-calendario 0.9994
  (cota superior diagnóstica)}, con la nota de que la cascada correspondiente va de 3.82 a ~1667.

Esto **formaliza en el artefacto** lo que el aviso #12 ya dice en prosa, y protege el veredicto
reflexivo. Coste: bajo, pero toca `schema.py`/`publish.py` ⇒ implica **re-publicar**.

> ☐ **DECISIÓN #3:** ___ adoptar el reporte de dos bandas ___ mantener prosa actual (n fuera de KPIs)

---

## #5 · Régimen degenerado (E[D] ≈ 1 día) — documentar vs reparametrizar

### Hechos verificados

- Régimen "alta volatilidad": **E[D] = 1.17 d** (< umbral 2.0), n_obs = 106 (~2–3 % de días).
  Ya se publica con IC apagado (F4) y **punto suprimido** (F2.c, recién publicado): "no
  anualizable — 106 obs".
- **Es el óptimo global, no un artefacto** (`reports/diag_degenerate_regime.md`, 50 arranques >
  R6): 17/50 convergen a ese óptimo con el régimen degenerado. **Student-t no lo corrige**
  (E[D]=1.10, 1/40); **K=3 no lo corrige limpiamente** (los TRES E[D]<2, 1/40, superficie
  multimodal).
- **Selección de modelo (BIC, menor = mejor):** K=2 normal = **8195.83** (elegido) < K=2 t 8206.81
  < K=3 normal 8220.47. **ΔBIC(K=3 − K=2) = +24.6** ⇒ evidencia **"muy fuerte"** contra K=3
  (Kass–Raftery: ΔBIC>10). Cambiar a K=3 **contradice** la selección de modelo ya validada.
- **Prueba de que es de los DATOS, no del estimador (AUDIT_MATH_v1, Fase A.2):** en sintético
  bien condicionado el MS-GARCH **recupera rango-2 de P** (razón de valores singulares 0.859 est
  vs 0.88 verdadera; clasificación hit 0.823, AUC 0.866). El estimador NO está sesgado hacia
  rango-1; el rango-1 en SPY real es una **propiedad de los datos**.

**Guía (verbatim, Parte 10):** *"Los regímenes son latentes, no reales... una aproximación
conveniente, no una verdad del mundo."* Forzar que el segundo régimen "persista" sería **imponer
una verdad del mundo** que la Guía advierte explícitamente que no existe.

### Opciones y recomendación

- **Opción 1 (recomendada): documentar y cerrar.** El estado absorbe-outliers ya se reporta con
  honestidad total (IC apagado + punto suprimido + banner). Es lo coherente con "los regímenes
  son latentes".
- **Opción 2: reparametrizar** (piso de persistencia sobre p_kk / componente de salto / mezcla de
  colas). Metodología estructural (R5). **Criterio pre-registrado:** adoptar solo si **E[D] > 5 d
  SIN perder log-loss OOS**. Evidencia previa **en contra** (Student-t y K=3 ya fallaron). Alto
  coste (toca parametrización + re-verificar `test_hamilton_recovery`/`test_label_ordering` +
  R6 + walk-forward), alta probabilidad de no aportar.

> ☐ **DECISIÓN #5:** ___ documentar y cerrar ___ abrir línea de reparametrización (criterio E[D]>5d sin perder log-loss)

---

## Resumen

| # | Palanca | Hecho matemático decisivo | Recomendación |
| :-: | :-- | :-- | :-- |
| **#4** | Kernel Hawkes | Power-law falla el criterio pre-registrado (b): reduce D solo 13.7 % (< 20 %), frágil numéricamente, O(n²) inviable, **sigue fallando KS** | **Exponencial + caveat** |
| **#3** | Reporte de `n` | Fuente 3 (ventana) es estructural y cruza el umbral reflexivo 0.8; colapsarla con el ruido de muestreo es error de categoría y la cascada (convexa) queda mal definida | **Dos bandas separadas** |
| **#5** | Régimen degenerado | Óptimo global; K=2 gana BIC por ΔBIC=24.6 ("muy fuerte" vs K=3); el estimador recupera rango-2 en sintético ⇒ es de los datos | **Documentar y cerrar** |

**Nada se implementa hasta tu OK.** #3 y (si eligieras #4-Opción 2) implican re-publicar; #5
Opción 1 no requiere cómputo. El runbook de re-publicación está en `artifacts/README.md`.

---

## RESOLUCIÓN DEL DIRECTOR (2026-08-19)

| # | Decisión | Acción |
| :-: | :-- | :-- |
| **#4 Kernel** | **Mantener exponencial + caveat KS** | Sin cambios: ya es el estado publicado. Power-law queda como diagnóstico comparativo (`_compare_kernels.log`). |
| **#3 Reporte de `n`** | **Dos bandas separadas** | Implementar: `schema.py`/`publish.py` emiten (a) IC del MLE en ventana observada y (b) banda de sensibilidad de ventana {observado · span-calendario}; app/panel las muestran. Requiere re-publicar. |
| **#5 Degenerado** | **Documentar y cerrar** | Sin cómputo: ya se reporta con IC apagado + punto suprimido + banner. Se cierra la línea de reparametrización (no se reabre salvo prioridad nueva). |

# Validación IRF-N — V4

Fecha: 2026-08-20
run_id (indicador publicado): `75f650b1d59d` (V3, **modelo M1**)
Modelo publicado: **M1** — MS-GJR-GARCH K=2, matriz de transición **CONSTANTE**, SIN
covariables de transición (tvtp=false, covariates=[]). Migración A4 (bias-variance):
ninguna covariable de transición añade valor OOS distinguible (Test 7), así que M1
tiene el mismo sesgo que M2 con menor varianza. La capa Hawkes se publica como
indicador **standalone** (fuera del logit).
run_id (walk-forward / calibración / económico / direccional consumidos): `75f650b1d59d`
(walk-forward autocontenido de `run_v3 --publish-m1`, R6, multistart 20)
run_id (selección de K / Hansen): `v1_kselect.json` (`artifacts/analysis/`, R6, compartido)
run_id (ablación M0/M2/M3, Tests 4 y 7): análisis de **selección de modelo** (estructural,
`reports/ablation_news.md` / `ablation_m3_l1.md`, sobre `3b4f1e39b59c`) — ver nota abajo.
Activo ancla: SPY
asof: 2026-08-19 (cola OOS 2017-01-02..2026-07-01, n=2386)

> **Re-validación 2026-08-20 (cierre de F6).** Este reporte se recomputa sobre el run
> PUBLICADO `75f650b1d59d`. La versión previa describía `3b4f1e39b59c` (mismo modelo M1,
> datos ~5 días más antiguos) y quedó `stale=true` en el panel. Tests 2, 3, 5 y 6 se
> recomputan sobre los artefactos del run vivo (calibración de `walkforward.json`,
> `history.parquet`, `validation.json` del económico). Tests 1, 4 y 7 son análisis
> **estructurales** (selección de K y selección de modelo): no se re-generan por run —
> igual que la selección de K nunca se re-corre por cambios de 5 días en la cola. El
> modelo ganador de esa ablación (M1) es el que está publicado. Baseline del Test 2:
> climatología MARGINAL. La capa de noticias (M4/M5) sigue cerrada por datos.

---

## Resumen ejecutivo

| Test | Resultado en una frase | Veredicto |
| :-- | :-- | :-- |
| 1. Número de regímenes | BIC elige K=2 normal (8195.8, mínimo claro); Hansen R6 (`v1_kselect.json`, B=49): K=2 vs K=1 p=0.02. Estructural. | ✓ |
| 2. Calibración | Bien calibrado y sub-confiado; contra la climatología MARGINAL gana en Brier (0.069 vs 0.296) y log-loss (0.165 vs 0.472). | ✓ |
| 3. Precisión direccional (PT) | Cruza el 5% unilateral (hit 55.5% vs 54.4%, p=0.032) pero **frágil**: efecto de 1.1 pp, predice "sube" el ~90% de los días (deriva), en el run previo daba p=0.086. No robusto. | ⚠ |
| 4. Comparación con benchmark (DM) | M1 vence al GARCH de un régimen (M1 vs M0, DM=−3.21, p=0.001); añadir TVTP técnico (M2 vs M1) NO aporta (DM=+1.62, p=0.106) — por eso se publica M1. Estructural. | ✓/⚠ |
| 5. Walk-forward económico | Regla pre-registrada de des-riesgo: Sharpe de la diferencia vs buy-and-hold −0.503, IC95 [−1.133, 0.040] → **NO supera**. Reduce drawdown levemente (32.5% vs 33.7%). | ⚠ |
| 6. Sharpe real (bootstrap) | El IC del Sharpe del estado predicho baja-vol excluye 0 [0.02, 1.45] pero es cercano al buy-and-hold SPY (0.78 [0.16, 1.42]): sin Sharpe atribuible. | ⚠ |
| 7. Ablación | M0→M1 aporta (regímenes, DM p=0.001, robusto). M2 (TVTP técnico) no aporta (p=0.106); M3 (macro) p=0.299. Único aporte robusto: los regímenes → publicar M1. Estructural. | ⚠ |

Lectura de una línea: **el valor del modelo está en la varianza condicional
(regímenes de volatilidad), no en covariables de transición ni en dirección ni en un
Sharpe atribuible.** M1 (regímenes solos) es el modelo de producción por bias-variance.

---

## Test 1 — Número de regímenes

Ganador por BIC: **K=2, normal** (BIC=8195.8), mínimo de la tabla
(`artifacts/analysis/v1_kselect.json`, análisis compartido; la selección de K es
estructural, no se re-genera por run). Hansen R6 (B=49): K=2 vs K=1 **p=0.02** → se
rechaza K=1. K=2 confirmado por BIC y bootstrap.

---

## Test 2 — Calibración

M1, N=2386 OOS (`75f650b1d59d`, `walkforward.json::calibration`). Evaluado sobre
ξ_{t|t-1} PREDICHA:

| métrica | modelo M1 | climatología MARGINAL |
| :-- | --: | --: |
| Brier multiclase | **0.0689** | 0.2960 |
| Log-loss multiclase | **0.1651** | 0.4724 |
| ECE | 0.0627 | — |

Fiabilidad (top-label; bins con datos): (0.547→acc 0.474, n=19), (0.646→0.893, n=28),
(0.763→0.939, n=49), (0.864→0.970, n=631), (0.936→0.975, n=1659).

> **NOTA DE BASELINE.** El baseline es la **climatología MARGINAL** (frecuencias
> marginales constantes del régimen OOS, `validation/calibration.py::log_loss_baseline`).
> Es DISTINTA de la "climatología de régimen" de versiones muy antiguas del reporte
> (log-loss 0.177): NO comparar números entre definiciones sin fijar el baseline.

**Interpretación.** M1 está bien calibrado (sub-confiado en los bins poblados) y gana
claramente a la climatología marginal en ambas métricas.

---

## Test 3 — Precisión direccional (Pesaran-Timmermann)

M1, n=2386 (`history.parquet`, `r` vs `r_pred_mean`): hit rate **55.5%** vs 54.4%
esperado bajo independencia, PT=1.848, **p=0.032** (unilateral).

**Cruza el 5% unilateral pero NO es una señal robusta.** El tamaño de efecto es de solo
1.1 pp; el signo predicho es positivo el **~90% de los días** (la predicción está
dominada por la deriva incondicional del SPY, no por información de régimen); y el mismo
test sobre el run previo `3b4f1e39b59c` daba **p=0.086** — un cruce del umbral que se
mueve con 5 días de datos no es robusto. En un modelo de regímenes de **volatilidad** la
dirección es de segundo orden. Se reporta el número tal cual (⚠), sin leerlo como
capacidad direccional del modelo.

---

## Test 4 — Comparación con benchmark (Diebold-Mariano)

> **Análisis estructural de selección de modelo** (`reports/ablation_news.md`, 19
> bloques, n=2386; log-loss/obs de densidad; run de ablación `3b4f1e39b59c`). No se
> re-genera con la cola de 5 días de `75f650b1d59d`: la comparación M0/M1/M2 es una
> decisión de arquitectura (qué modelo publicar), del mismo tipo que la selección de K
> del Test 1, y su ganador (M1) es el que está publicado.

| Comparación | log-loss/obs | DM stat | p-valor | ¿mejor? |
| :-- | --: | --: | --: | :-- |
| M0 (1 régimen) | 1.2940 | — | — | — |
| **M1 (K=2, publicado) vs M0** | 1.2583 | **−3.210** | **0.001** | **Sí** |
| M2 (+TVTP técnico) vs M1 | 1.2672 | +1.617 | 0.106 | No aporta |

**Interpretación.** M1 le gana contundentemente al GARCH de un régimen (densidad del
retorno; p=0.001) — el aporte está en la varianza condicional. Añadir el TVTP técnico
(M2) no mejora distinguible (p=0.106) y **empeora** la log-loss puntual (1.2672 >
1.2583): por eso se publica M1 (A4, bias-variance).

---

## Test 5 — Walk-forward económico

M1, re-corrido 2026-08-20 (`run_economic_v4` sobre `75f650b1d59d`): **NO SUPERA** el
criterio pre-registrado (⚠, negativo limpio). Regla congelada
(`docs/preregistro_regla_trading.md`): des-riesgo cuando P(alta vol, ξ de t−1) > 0.5,
costo 2 pb; éxito = IC95 del Sharpe de la diferencia vs buy-and-hold excluye 0 por arriba.

| Métrica (2 pb) | Estrategia | Buy & hold |
| :-- | --: | --: |
| Sharpe anualizado | 0.624 | 0.781 |
| Max drawdown | 32.5% | 33.7% |
| Exposición | 81.9% | 100% |
| Cambios de posición | 132 | 0 |

**Sharpe de la diferencia: −0.503, IC95 [−1.133, 0.040] → incluye 0, NO SUPERA.**
Sensibilidad: 0 pb −0.468 [−1.095, 0.075]; 5 pb −0.556 [−1.187, −0.004] (a 5 pb el IC
excluye 0 por **abajo**: la estrategia es peor, nunca mejor — el veredicto no cambia con
el costo). El valor del indicador es informacional, no de estrategia.
Artefacto: `artifacts/runs/75f650b1d59d/validation.json`. BTC no aplica (K=1).

---

## Test 6 — Sharpe real (Bootstrap Politis-Romano)

M1, n=2386, Sharpe por estado predicho (info a t−1; `history.parquet`, 2000 réplicas,
semilla 42):

| serie | n | Sharpe | IC 95% | ¿incluye 0? |
| :-- | --: | --: | :-- | :-: |
| Estado predicho = baja vol | 1991 | 0.691 | [0.017, 1.454] | No |
| Estado predicho = alta vol | 395 | 1.213 | [−0.374, 2.843] | Sí |
| **SPY OOS completo (buy & hold)** | 2386 | **0.781** | **[0.158, 1.424]** | No |

**Interpretación — sin maquillar.** El cubo "baja vol" excluye el 0 (0.691) pero es
**cercano** al buy-and-hold (0.781): la condición de régimen no genera Sharpe
atribuible. El cubo "alta vol" (395 días) incluye el 0. Sharpe descriptivo del activo,
no de una estrategia ejecutable.

---

## Test 7 — Ablación (escalera de covariables)

> **Análisis estructural de selección de modelo** (`reports/ablation_news.md` +
> `ablation_m3_l1.md`; run de ablación `3b4f1e39b59c`), del mismo tipo que el Test 1.
> No se re-genera con la cola de 5 días de `75f650b1d59d`; el ganador (M1) es el
> publicado. M3 con L1 en `ablation_m3_l1.md`.

| Modelo | Especificación | log-loss/obs | DM vs anterior |
| :-- | :-- | --: | --: |
| M0 | GARCH 1 régimen | 1.2940 | — |
| **M1** | **HMM K=2, P constante (PUBLICADO)** | **1.2583** | DM=−3.210, **p=0.001** (mejor) |
| M2 | + TVTP técnico | 1.2672 | DM=+1.617, p=0.106 (no aporta) |
| M3 | + macro, L1 | — | DM=+1.04, p=0.299 (`ablation_m3_l1.md`) |
| M4 | + sorpresa | — | CERRADO: sin consenso histórico |
| M5 | + λ_N(t) Hawkes | — | CERRADO: corpus GDELT insuficiente |

**VEREDICTO.** El único aporte robusto es **M1 sobre M0** (regímenes, DM p=0.001). Ni
el TVTP técnico (M2) ni la macro (M3) añaden valor OOS distinguible. **Se publica M1**
(A4, bias-variance): mismo sesgo que M2, menor varianza. M4/M5 cerrados por datos.

---

## Limitaciones

- **R6:** walk-forward, calibración y económico R6 (multistart 20, `run_v3 --publish-m1`).
  Selección de K: `v1_kselect.json` (R6, compartido). Ablación M0/M2/M3 (Tests 4/7):
  análisis estructural sobre `3b4f1e39b59c` (multistart 30), no re-corrido en la cola de
  `75f650b1d59d` (misma especificación M1, decisión de arquitectura).
- **Baseline del Test 2:** climatología MARGINAL (no comparar con definiciones viejas).
- **Test 3:** cruza el 5% unilateral (p=0.032) pero frágil (efecto 1.1 pp, deriva,
  p=0.086 en el run previo); dirección de segundo orden, no señal robusta.
- **Test 5:** no supera (negativo limpio); valor informacional, no de estrategia.
- **A6 (régimen degenerado):** el 2º régimen es absorbe-outliers (E[D]≈1 d), óptimo
  global; se reporta con IC condicional suprimido + banner (decisión informada, sin parches).
- **A1 (Hawkes):** kernel exponencial (KS rechaza, D pequeño); sesgo de días fantasma
  cerrado (soporte observado). Hawkes es standalone, fuera del logit de M1.
- **El "estado realizado" es un proxy** (argmax ξ_{t|t}): Tests 2, 4-BM1 y 5 miden
  consistencia predicho vs. concluido.

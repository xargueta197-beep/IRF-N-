# Validación IRF-N — V4

Fecha: 2026-08-16
run_id (indicador publicado): `3b4f1e39b59c` (V3, **modelo M1**)
Modelo publicado: **M1** — MS-GJR-GARCH K=2, matriz de transición **CONSTANTE**, SIN
covariables de transición (tvtp=false, covariates=[]). Migración A4 (bias-variance):
ninguna covariable de transición añade valor OOS distinguible (Test 7), así que M1
tiene el mismo sesgo que M2 con menor varianza. La capa Hawkes se publica como
indicador **standalone** (fuera del logit).
run_id (walk-forward / ablación / económico consumidos): `3b4f1e39b59c` (walk-forward
autocontenido de `run_v3 --publish-m1`, R6, multistart 30)
run_id (selección de K / Hansen): `v1_kselect.json` (`artifacts/analysis/`, R6, compartido)
Activo ancla: SPY
asof: 2026-08-14 (cola OOS 2017-01-03..2026-07-01, n=2386)

> Re-validación 2026-08-16 tras la migración M2→M1 (A4). Todos los tests se
> recomputan sobre el run publicado M1 `3b4f1e39b59c`. La versión previa de este
> reporte describía M2 (`7c44a7fac16d`). Baseline del Test 2: climatología MARGINAL
> (ver nota). La capa de noticias (M4/M5) sigue cerrada por datos.

---

## Resumen ejecutivo

| Test | Resultado en una frase | Veredicto |
| :-- | :-- | :-- |
| 1. Número de regímenes | BIC elige K=2 normal (8195.8, mínimo claro); Hansen R6 (`v1_kselect.json`, B=49): K=2 vs K=1 p=0.02. Estructural. | ✓ |
| 2. Calibración | Bien calibrado y sub-confiado; contra la climatología MARGINAL gana en Brier (0.071 vs 0.232) y log-loss (0.169 vs 0.394). | ✓ |
| 3. Precisión direccional (PT) | Marginal, sin señal al 5%: hit 55.2% vs 54.3% esperado, p=0.086. El modelo predice "sube" la mayoría de días; dirección de segundo orden. | ⚠ |
| 4. Comparación con benchmark (DM) | M1 vence al GARCH de un régimen (M1 vs M0, DM=−3.21, p=0.001); añadir TVTP técnico (M2 vs M1) NO aporta (DM=+1.62, p=0.106) — por eso se publica M1. | ✓/⚠ |
| 5. Walk-forward económico | Regla pre-registrada de des-riesgo: Sharpe de la diferencia vs buy-and-hold −0.323, IC95 [−1.070, 0.274] → **NO supera**. Reduce drawdown levemente (32.5% vs 33.7%). | ⚠ |
| 6. Sharpe real (bootstrap) | El IC del Sharpe del estado predicho baja-vol excluye 0 [0.15, 1.48] pero es cercano al buy-and-hold SPY (0.78 [0.17, 1.40]): sin Sharpe atribuible. | ⚠ |
| 7. Ablación | M0→M1 aporta (regímenes, DM p=0.001, robusto). M2 (TVTP técnico) no aporta (p=0.106); M3 (macro) p=0.299. Único aporte robusto: los regímenes → publicar M1. | ⚠ |

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

M1, N=2386 OOS (`3b4f1e39b59c`). Evaluado sobre ξ_{t|t-1} PREDICHA:

| métrica | modelo M1 | climatología MARGINAL |
| :-- | --: | --: |
| Brier multiclase | **0.0707** | 0.2323 |
| Log-loss multiclase | **0.1688** | 0.3941 |
| ECE | 0.0661 | — |

Fiabilidad (top-label; bins con datos): (0.55→acc 0.48, n=23), (0.64→0.91, 32),
(0.76→0.93, 54), (0.87→0.97, 646), (0.93→0.98, 1631).

> **NOTA DE BASELINE.** El baseline es la **climatología MARGINAL** (frecuencias
> marginales constantes del régimen OOS, `validation/calibration.py::log_loss_baseline`).
> Es DISTINTA de la "climatología de régimen" de versiones muy antiguas del reporte
> (log-loss 0.177): NO comparar números entre definiciones sin fijar el baseline.

**Interpretación.** M1 está bien calibrado (sub-confiado en los bins poblados) y gana
claramente a la climatología marginal en ambas métricas.

---

## Test 3 — Precisión direccional (Pesaran-Timmermann)

M1, n=2386: hit rate **55.2%** vs 54.3% esperado bajo independencia, **p=0.086**
(unilateral). **Marginal, sin señal al 5%.** El signo predicho es positivo la gran
mayoría de los días: en un modelo de regímenes de volatilidad la dirección es de
segundo orden. (M2 saliente daba hit 52.9%, p=0.90; M1 queda algo más alto pero sigue
sin cruzar el 5% — coherente con "no robusto/segundo orden".)

---

## Test 4 — Comparación con benchmark (Diebold-Mariano)

Ablación (`reports/ablation_news.md`, 19 bloques, n=2386; log-loss/obs de densidad):

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

M1, re-corrido 2026-08-16 (`run_economic_v4` sobre `3b4f1e39b59c`): **NO SUPERA** el
criterio pre-registrado (⚠, negativo limpio). Regla congelada
(`docs/preregistro_regla_trading.md`): des-riesgo cuando P(alta vol, ξ de t−1) > 0.5,
costo 2 pb; éxito = IC95 del Sharpe de la diferencia vs buy-and-hold excluye 0 por arriba.

| Métrica (2 pb) | Estrategia | Buy & hold |
| :-- | --: | --: |
| Sharpe anualizado | 0.711 | 0.781 |
| Max drawdown | 32.5% | 33.7% |
| Exposición | 86.6% | 100% |
| Cambios de posición | 132 | 0 |

**Sharpe de la diferencia: −0.323, IC95 [−1.070, 0.274] → incluye 0, NO SUPERA.**
Sensibilidad: 0 pb −0.286 [−1.023, 0.304]; 5 pb −0.380 [−1.137, 0.212] (el veredicto
no cambia con el costo). El valor del indicador es informacional, no de estrategia.
Artefacto: `artifacts/runs/3b4f1e39b59c/validation.json`. BTC no aplica (K=1).

---

## Test 6 — Sharpe real (Bootstrap Politis-Romano)

M1, n=2386, Sharpe por estado predicho (info a t−1):

| serie | n | Sharpe | IC 95% | ¿incluye 0? |
| :-- | --: | --: | :-- | :-: |
| Estado predicho = baja vol | 2110 | 0.709 | [0.154, 1.481] | No |
| Estado predicho = alta vol | 276 | 1.271 | [−0.524, 3.183] | Sí |
| **SPY OOS completo (buy & hold)** | 2386 | **0.781** | **[0.168, 1.396]** | No |

**Interpretación — sin maquillar.** El cubo "baja vol" excluye el 0 (0.709) pero es
**cercano** al buy-and-hold (0.781): la condición de régimen no genera Sharpe
atribuible. El cubo "alta vol" (276 días) incluye el 0. Sharpe descriptivo del activo,
no de una estrategia ejecutable.

---

## Test 7 — Ablación (escalera de covariables)

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

- **R6:** walk-forward, ablación y económico R6 (multistart 30, `run_v3 --publish-m1`).
  Selección de K: `v1_kselect.json` (R6, compartido).
- **Baseline del Test 2:** climatología MARGINAL (no comparar con definiciones viejas).
- **Test 3:** marginal (p=0.086), sin señal robusta; dirección de segundo orden.
- **Test 5:** no supera (negativo limpio); valor informacional, no de estrategia.
- **A6 (régimen degenerado):** el 2º régimen es absorbe-outliers (E[D]≈1 d), óptimo
  global; se reporta con IC condicional suprimido + banner (decisión informada, sin parches).
- **A1 (Hawkes):** kernel exponencial (KS rechaza, D pequeño); sesgo de días fantasma
  cerrado (soporte observado). Hawkes es standalone, fuera del logit de M1.
- **El "estado realizado" es un proxy** (argmax ξ_{t|t}): Tests 2, 4-BM1 y 5 miden
  consistencia predicho vs. concluido.

# Validación IRF-N — V4

Fecha: 2026-08-16
run_id (indicador publicado): `7c44a7fac16d` (V3)
run_id (walk-forward / ablación / económico consumidos): `7c44a7fac16d` (walk-forward
autocontenido de `run_v3`, **R6, multistart 30**)
run_id (selección de K / Hansen consumidos): `v1_kselect.json` en `artifacts/analysis/`
(análisis compartido de selección de K, R6, corrida completa 2026-07-22 — la selección
de K es estructural, no se re-genera por run)
Activo ancla: SPY
asof: 2026-08-14 (cola OOS del walk-forward: 2017-01-03..2026-07-01, n=2386)

> Re-validación 2026-08-16 sobre el run vivo `7c44a7fac16d`. Tests 2, 3, 4, 6 y 7
> (M0-M2) se reescriben desde el walk-forward/ablación/history ya generados por
> `run_v3` (R6). Test 5 (económico) se re-corrió con `scripts/run_economic_v4.py`
> sobre este run. La versión anterior de este reporte consumía un walk-forward
> **superado** (V1/ablación, `--quick` y R6 del 2026-07-29) con una definición de
> baseline distinta — ver la NOTA DE BASELINE en el Test 2. La capa de noticias
> (M4/M5) sigue bloqueada por datos, no por diseño.

---

## Resumen ejecutivo

| Test | Resultado en una frase | Veredicto |
| :-- | :-- | :-- |
| 1. Número de regímenes | BIC elige K=2 normal de forma inequívoca (8195.8, mínimo claro); Hansen R6 (`v1_kselect.json`, B=49): K=2 vs K=1 p=0.02 — se rechaza K=1. Estructural, compartido entre runs. | ✓ |
| 2. Calibración | Bien calibrado y **sub-confiado** (acierto empírico > confianza en todos los bins). Contra la **climatología marginal** (frecuencias marginales de régimen) gana en Brier (0.125 vs 0.356) y log-loss (0.243 vs 0.541). Baseline DISTINTO al del reporte viejo — ver nota. | ✓/⚠ |
| 3. Precisión direccional (PT) | Sin señal: hit 52.9% vs 53.8% esperado, PT=−1.27, **p=0.90**. El modelo predice "sube" el 84% de los días; la dirección es de segundo orden en un modelo de volatilidad. (El p=0.048 viejo venía de un walk-forward superado.) | ⚠ |
| 4. Comparación con benchmark (DM) | Vence al GARCH de un régimen en densidad predictiva (M1 vs M0, DM=−3.21, **p=0.001**); el TVTP técnico (M2 vs M1) no es distinguible (DM=+1.62, p=0.106). | ✓/⚠ |
| 5. Walk-forward económico | Regla pre-registrada de des-riesgo: Sharpe de la diferencia vs buy-and-hold −0.502, IC95 [−1.301, 0.031] → **NO supera** (a 5 pb el IC ya excluye 0 por abajo). Reduce drawdown (23.3% vs 33.7%). | ⚠ |
| 6. Sharpe real (bootstrap) | El IC 95% del Sharpe del estado predicho baja-vol excluye 0 [0.20, 1.62] pero es cercano al de comprar y mantener SPY (0.78 [0.17, 1.40]): el modelo no aporta Sharpe atribuible. | ⚠ |
| 7. Ablación de noticias | M0→M1 aporta (regímenes, DM p=0.001, robusto). M2 (TVTP técnico) no distinguible (p=0.106). M3 (macro) con L1 NO aporta (DM=+1.04, p=0.299). Único aporte robusto: los regímenes. M4/M5 bloqueados por datos. | ⚠ |

Lectura de una línea: **el valor del modelo está en la varianza condicional
(densidad predictiva, regímenes de volatilidad), no en una "llamada" direccional ni
en un Sharpe atribuible al modelo.** La capa de noticias sigue sin poder evaluarse.

---

## Test 1 — Número de regímenes

**Resultado.** Ganador por BIC: **K=2, innovaciones normales** (BIC=8195.8), el
mínimo de la tabla (fuente: `artifacts/analysis/v1_kselect.json`, análisis compartido
de selección de K; es estructural y no se re-genera por run):

| K | dist | loglik | k_params | BIC | conv/arranques |
| --: | :-- | --: | --: | --: | :-- |
| 1 | normal | −4201.6 | 5 | 8443.9 | 6/6 |
| **2** | **normal** | **−4049.1** | **12** | **8195.8 ←** | 4/6 |
| 3 | normal | −4027.8 | 21 | 8226.3 | 1/6 |
| 4 | normal | −4017.5 | 32 | 8295.1 | 1/6 |
| 1 | t | −4083.4 | 6 | 8215.6 | 6/6 |
| 2 | t | −4048.0 | 14 | 8209.8 | 1/6 |

**Test de Hansen (1992) por bootstrap paramétrico (R6, B=49, 2026-07-22):** K=2 vs
K=1 **p=0.02** → se rechaza K=1. K=2 confirmado por BIC **y** bootstrap. (El LR
estándar no aplica por el problema de Davies; se simula la nula desde el modelo K−1.)

---

## Test 2 — Calibración

**Resultado** (evaluado sobre ξ_{t|t-1} PREDICHA, nunca la filtrada; el `assert` de
`calibration_metrics` lo garantiza). N=2386 observaciones OOS del run `7c44a7fac16d`:

| métrica | modelo | climatología MARGINAL |
| :-- | --: | --: |
| Brier multiclase | **0.1251** | 0.3561 |
| Log-loss multiclase | **0.2430** | 0.5414 |
| ECE | 0.0891 | — |

Diagrama de fiabilidad (top-label; bins con datos):

| confianza predicha (bin) | confianza media | acierto empírico | n |
| --: | --: | --: | --: |
| 0.5–0.6 | 0.549 | 0.713 | 101 |
| 0.6–0.7 | 0.661 | 0.875 | 168 |
| 0.7–0.8 | 0.757 | 0.935 | 429 |
| 0.8–0.9 | 0.854 | 0.955 | 557 |
| 0.9–1.0 | 0.948 | 0.973 | 1131 |

> **NOTA DE BASELINE (importante para comparaciones futuras).** El baseline de este
> Test es la **climatología MARGINAL**: predecir en todo t las frecuencias
> marginales constantes de cada régimen sobre el OOS (`validation/calibration.py::
> log_loss_baseline` / `brier_baseline`). Es una definición DISTINTA de la
> "**climatología de régimen**" que usaba la versión anterior de este reporte
> (log-loss 0.177), que era un baseline más fuerte. **Los números de ambas versiones
> NO son comparables entre sí**: el "modelo no gana en log-loss (0.179 vs 0.177)" del
> reporte viejo y el "modelo gana (0.243 vs 0.541)" de este miden contra bases
> distintas. Cualquier comparación futura debe fijar primero la definición de baseline.

**Interpretación.** El modelo está **bien calibrado y sub-confiado**: en todos los
bins el acierto empírico supera a la confianza declarada (lo contrario del
sobreajuste). Contra la climatología marginal gana claramente en ambas métricas. El
veredicto de fondo del proyecto no cambia: el valor está en la densidad condicional,
no en una "llamada" que supere a un baseline de régimen fuerte (Test 4-BM1).

---

## Test 3 — Precisión direccional (Pesaran-Timmermann)

**Resultado (run vivo `7c44a7fac16d`, n=2386): SIN señal direccional.** Hit rate
**52.85%** vs 53.79% esperado bajo independencia, PT=**−1.266**, **p=0.897**
(unilateral). El signo predicho es positivo el **84.2%** de los días.

| Métrica | Valor |
| :-- | :-- |
| Hit rate direccional | 52.85% |
| Hit rate esperado bajo independencia | 53.79% |
| Estadístico PT | −1.266 |
| p-valor (unilateral) | 0.897 |

**Interpretación.** No hay evidencia de precisión direccional — exactamente lo que la
nota metodológica siempre anticipó: en un modelo de regímenes de **volatilidad** los
μ_k difieren poco y la dirección es de segundo orden; el test PT es en gran parte una
comparación contra "predecir siempre subida". **Reconciliación con el número viejo
(p=0.048):** ese valor venía de un walk-forward **superado** (V1/ablación, R6 del
2026-07-29, hit 55.28%); el reporte mismo lo calificaba de "marginal y sensible al
multistart, no robusto". El run vivo (walk-forward autocontenido de `run_v3`) da
p=0.90 — la lectura honesta y estable (tres runs recientes coinciden bit a bit).
Diagnóstico completo: la diferencia NO viene de la capa Hawkes (A1) — el modelo
publicado usa `covariates=[sma_gap, bb_width_z]`, `news_layer=[]`, `lambda_N_z`
inactiva; Hawkes no entra al logit y no toca `r_pred_mean`.

Nota BTC (línea secundaria, K=1): el test degenera (signo predicho positivo el 100%
de los días); no informativo.

---

## Test 4 — Comparación con benchmark (Diebold-Mariano)

Benchmarks definidos ANTES de ver resultados:

- **BM1**: predicción constante = frecuencia marginal de cada régimen (climatología
  marginal). Pérdida: log-loss de régimen sobre el proxy y_t = argmax ξ_{t|t}.
- **BM2**: GARCH de un solo régimen (M0 de la ablación). Pérdida: log-densidad
  predictiva del retorno por observación (la métrica primaria del proyecto).

**Resultado (run vivo, ablación M0/M1/M2 de `reports/ablation_news.md`, 19 bloques,
n=2386; log-loss/obs de la densidad):**

| Comparación | log-loss/obs | DM stat | p-valor | ¿modelo mejor? |
| :-- | --: | --: | --: | :-- |
| M0 (1 régimen) | 1.2940 | — | — | — |
| **M1 (K=2) vs M0** | 1.2583 | **−3.210** | **0.001** | **Sí** |
| M2 (+TVTP técnico) vs M1 | 1.2672 | +1.617 | 0.106 | No (indistinguible) |

(DM<0 ⇒ el primer modelo pierde menos. HAC Newey-West + corrección
Harvey-Leybourne-Newbold.)

**Interpretación.** El modelo le gana de forma contundente al GARCH de un solo
régimen en densidad del retorno (M1 vs M0, DM=−3.21, p=0.001): el aporte está en la
**varianza condicional** (dos regímenes de volatilidad). El TVTP técnico (M2) no es
distinguible de M1 (p=0.106). Consistente con el reporte viejo (M1 vs M0 −3.15,
p=0.002).

---

## Test 5 — Walk-forward económico

**Resultado (run vivo `7c44a7fac16d`, re-corrido 2026-08-16 con
`scripts/run_economic_v4.py`): NO SUPERA el criterio pre-registrado (⚠, negativo
limpio).**

Regla pre-registrada (`docs/preregistro_regla_trading.md`, congelada y aprobada por
el director ANTES de correr, R8): des-riesgo binario — salir a efectivo cuando
P(alta vol, ξ filtrada de t−1) > 0.5, costo 2 pb/cambio; éxito = IC95 bootstrap
(Politis-Romano) del Sharpe de la diferencia vs buy-and-hold excluye 0 por arriba.

| Métrica (2 pb) | Estrategia | Buy & hold |
| :-- | --: | --: |
| Sharpe anualizado | 0.601 | 0.781 |
| Max drawdown | 23.3% | 33.7% |
| Exposición | 76.8% | 100% |
| Cambios de posición | 220 | 0 |

**Sharpe de la diferencia: −0.502, IC95 [−1.301, 0.031] → incluye 0 (por arriba), NO
SUPERA.** Sensibilidad: a 0 pb la diferencia es −0.466 [−1.250, 0.090]; a 5 pb −0.556
[−1.386, −0.028] (a 5 pb el IC ya excluye 0 por abajo: la estrategia es peor). La
hipótesis nula del pre-registro se confirma: **el valor del indicador es
informacional (densidad predictiva), no de estrategia.** Observación informativa (sin
umbral pre-registrado): reduce el drawdown (23.3% vs 33.7%), dirección esperable de
una regla de des-riesgo pero no acreditada estadísticamente. Artefacto:
`artifacts/runs/7c44a7fac16d/validation.json`. La línea BTC no aplica (K=1, regla
vacua). **Diferencia vs el reporte viejo (diff −0.03):** el viejo consumía otro
walk-forward (`16d4190d17e2`, --quick); el veredicto (NO SUPERA) es el mismo, la
magnitud negativa es mayor en este run.

---

## Test 6 — Sharpe real (Bootstrap Politis-Romano)

IC 95% del Sharpe con `sharpe_ci` (bootstrap estacionario, bloque por Politis-White
2004). Serie: retornos OOS de SPY por estado **predicho** (info a t−1). Run vivo,
n=2386:

| serie | n | Sharpe | IC 95% | bloque | ¿incluye 0? |
| :-- | --: | --: | :-- | --: | :-: |
| Estado predicho = baja vol | 1905 | 0.893 | [0.195, 1.623] | 1 | No |
| Estado predicho = alta vol | 481 | 0.660 | [−0.561, 2.413] | 14 | Sí |
| **SPY OOS completo (comprar y mantener)** | 2386 | **0.781** | **[0.168, 1.396]** | 4 | No |

**Interpretación — sin maquillar.** El IC del cubo "baja vol predicha" excluye el
cero (0.893), pero es **cercano** al de comprar y mantener SPY (0.781): la condición
de régimen no genera un Sharpe claramente distinto del mercado. El cubo "alta vol
predicha" (481 días) tiene IC que incluye el cero — no dice nada direccional. Es un
Sharpe **descriptivo del activo**, no de una estrategia ejecutable. (Nota: el split
predicho baja/alta vol es 80%/20% en este run, distinto del ~96%/4% del run viejo —
consecuencia del re-ajuste R6, coherente con el cambio del Test 3.)

---

## Test 7 — Ablación (escalera de covariables)

**Run vivo (M0-M2, `reports/ablation_news.md`, 19 bloques, n=2386):**

| Modelo | Especificación | log-loss/obs | DM vs anterior |
| :-- | :-- | --: | --: |
| M0 | GARCH 1 régimen | 1.2940 | — |
| M1 | HMM K=2, P fija | **1.2583** | DM=−3.210, **p=0.001** (mejor) |
| M2 | + TVTP técnico (sma_gap, bb_width_z) | 1.2672 | DM=+1.617, p=0.106 (no dist.) |
| M3 | + macro (slope_2s10y, hy_oas_z), L1 | — | DM=+1.04, p=0.299 (no aporta; `ablation_m3_l1.md`) |
| M4 | + SI_t (sorpresa) | — | congelado: sin consenso histórico |
| M5 | + λ_N(t) (Hawkes) | — | bloqueado: corpus GDELT insuficiente para el WF |

**VEREDICTO (⚠).**
1. **Regímenes: aportan, robusto.** M1 mejora a M0 de forma distinguible (DM=−3.21,
   p=0.001), consistente en todas las corridas. El hallazgo sólido.
2. **TVTP técnico (M2): no aporta distinguible** (DM=+1.62, p=0.106 en este run).
3. **Macro (M3): NO aporta** (DM=+1.04, p=0.299, con L1; eje cerrado,
   `ablation_m3_l1.md`).

Cierre del eje de covariables de transición: **el único aporte robusto es M1 sobre M0
(regímenes); ni el TVTP técnico ni la macro añaden valor OOS distinguible.** M4
(sorpresa) congelado por decisión del director (sin fuente gratuita de consenso
histórico); M5 (Hawkes) espera corpus GDELT. Ninguno es un DM perdido: es ausencia de
datos aguas arriba (R8).

---

## Limitaciones

- **R6:** el walk-forward, la ablación (M0-M2) y el económico de esta re-validación
  son R6 (multistart 30, walk-forward autocontenido de `run_v3`). La selección de K
  (Test 1) consume el análisis compartido `v1_kselect.json` (R6, 2026-07-22).
- **Baseline del Test 2:** climatología MARGINAL (frecuencias marginales), distinta de
  la "climatología de régimen" del reporte viejo. No comparar números entre versiones
  sin fijar la definición (ver nota en Test 2).
- **Test 3 sin señal (p=0.90):** es la lectura del walk-forward vigente; el p=0.048
  viejo era de un walk-forward superado y ya se calificaba de no robusto.
- **Test 5 no supera:** resultado negativo limpio, confirma el pre-registro. El valor
  del indicador es informacional, no de estrategia.
- **Test 6 no es un Sharpe de estrategia:** descriptivo del activo condicionado al
  régimen predicho, sin costos ni lado corto.
- **El "estado realizado" es un proxy** (argmax ξ_{t|t}), no una etiqueta verdadera:
  Tests 2, 4-BM1 y 5 miden consistencia predicho vs. concluido.
- **Capa de noticias (M4/M5):** bloqueada por datos, no por diseño. A1 (kernel Hawkes)
  no afecta al modelo publicado (`lambda_N_z` inactiva).

# Validación IRF-N — V4

Fecha: 2026-07-14
run_id (indicador publicado): `02db03d3d6d3` (V3)
run_id (walk-forward / BIC consumidos): `c1a8e85ca408` (titular V1, **corrida `--quick`, provisional, NO cumple R6**)
Activo ancla: SPY

> Todos los artefactos consumidos se generaron en modo `--quick` (multistart
> reducido): los números son **provisionales** hasta re-correr con multistart
> completo (R6). Se reportan igual (R8). La capa de noticias (M3/M4/M5) está
> bloqueada por datos, no por diseño; se documenta dónde corresponde.

---

## Resumen ejecutivo

| Test | Resultado en una frase | Veredicto |
| :-- | :-- | :-- |
| 1. Número de regímenes | BIC elige K=2 normal de forma inequívoca (8195.8, mínimo claro). Bootstrap de Hansen **ya con potencia (R6, corrida completa 2026-07-22, `v1_kselect.json`, B=49): K=2 vs K=1 p=0.02** — se rechaza K=1. K=2 confirmado por BIC **y** bootstrap. | ✓ |
| 2. Calibración | Bien calibrado y ligeramente **sub-confiado**; gana a la climatología en Brier (0.079 vs 0.082) pero **no** en log-loss (0.179 vs 0.177). | ⚠ |
| 3. Precisión direccional (PT) | **R6 (2026-07-29)**: hit rate 55,28% vs 54,17% esperado, PT=1,66, **p=0,048** — señal direccional **marginalmente significativa** al 5% (cruza el umbral por poco; sensible al multistart, era p=0,082 en `--quick`). | ⚠ |
| 4. Comparación con benchmark (DM) | **R6 (2026-07-29)**: vence al GARCH de un régimen en densidad predictiva (M1 vs M0, DM=−3.15, **p=0.002**, robusto); el TVTP técnico con L1 (M2 vs M1) es nominalmente mejor pero **no distinguible** (DM=−1.14, p=0.253); **no** vence a la climatología de régimen en log-loss. | ⚠ |
| 5. Walk-forward económico | **Económico ejecutado 2026-07-18** (regla pre-registrada de des-riesgo, aprobada por el director): Sharpe de la diferencia vs buy-and-hold −0.03, IC95 [−0.70, 0.53] → **NO supera el criterio**, como declaraba la hipótesis del pre-registro. El **estadístico**: 19 bloques (R2 ✓), el modelo bate a la climatología solo en 6/19 bloques, **concentrados en los de estrés**. | ⚠ |
| 6. Sharpe real (bootstrap) | El IC 95% del Sharpe condicional excluye el cero [0.18, 1.42], pero es **casi idéntico** al de comprar y mantener SPY: el modelo no aporta al Sharpe. | ⚠ |
| 7. Ablación de noticias | **R6 + L1, cerrada (2026-07-29)**: M0→M1 aporta (regímenes, DM p=0.001, robusto). **M3 (macro) con L1: NO aporta** sobre M2 (DM=+1.04, p=0.299). El TVTP técnico (M2) es marginal y sensible a la muestra. Único aporte robusto: los regímenes. M4 congelado (consenso de paga), M5 esperando GDELT. | ⚠ |

Lectura de una línea: **el valor del modelo está en la varianza condicional
(densidad predictiva calibrada, regímenes de volatilidad), no en una "llamada"
de régimen que le gane a la climatología ni en un Sharpe atribuible al modelo.**
La capa de noticias sigue sin poder evaluarse.

---

## Test 1 — Número de regímenes

**Resultado.** Ganador por BIC: **K=2, innovaciones normales** (BIC=8195.8), el
mínimo de la tabla completa (fuente: `artifacts/latest/v1_kselect.json`, consumida
sin recalcular):

| K | dist | loglik | k_params | BIC | conv/arranques |
| --: | :-- | --: | --: | --: | :-- |
| 1 | normal | −4201.6 | 5 | 8443.9 | 6/6 |
| **2** | **normal** | **−4049.1** | **12** | **8195.8 ←** | 4/6 |
| 3 | normal | −4027.8 | 21 | 8226.3 | 1/6 |
| 4 | normal | −4017.5 | 32 | 8295.1 | 1/6 |
| 1 | t | −4083.4 | 6 | 8215.6 | 6/6 |
| 2 | t | −4048.0 | 14 | 8209.8 | 1/6 |
| 3 | t | −4025.5 | 24 | 8246.1 | 1/6 |
| 4 | t | −4013.2 | 36 | 8319.2 | 1/6 |

**Test de Hansen (1992) por bootstrap paramétrico.** El LR estándar no aplica:
bajo H0 (K−1) los parámetros del régimen extra no están identificados (problema
de Davies), así que 2·(llK − llK−1) no es χ². Se simula la distribución nula del
LR desde el modelo K−1 ajustado (documentado en `validation/tests_stat.py`).
Escalera observada (consumida del artefacto):

| K vs K−1 | LR_obs | p (boot) | réplicas ok/tot |
| :-- | --: | --: | :-- |
| 2 vs 1 | 305.0 | 0.167 | 5/5 |
| 3 vs 2 | 42.7 | 0.167 | 5/5 |
| 4 vs 3 | 20.6 | 0.167 | 5/5 |

**Interpretación.** El BIC es inequívoco (K=2). El bootstrap de Hansen, en cambio,
**no es concluyente por falta de potencia**: con B=5 réplicas (`--quick`) el
p-valor mínimo posible es 1/6 ≈ 0.167, que es justo el que sale en los tres
peldaños. No se puede leer nada del p-valor con esa granularidad. `config/base.yaml`
fija `v1.ktest.n_boot: 49` para la corrida real; hasta correrla, la selección de
K descansa **solo en el BIC**. No se inventa un p-valor. **K=2 se acepta por BIC.**

---

## Test 2 — Calibración

**Resultado** (evaluado sobre ξ_{t|t-1} PREDICHA, nunca sobre la filtrada; el
`assert` de `calibration_metrics` lo garantiza). N=2387 observaciones OOS:

| métrica | modelo | climatología (frecuencia marginal) |
| :-- | --: | --: |
| Brier multiclase | **0.0792** | 0.0818 |
| Log-loss multiclase | 0.1789 | **0.1765** |
| ECE | 0.0688 | — |

Diagrama de fiabilidad (top-label; bins con datos):

| confianza predicha (bin) | confianza media | acierto empírico | n |
| --: | --: | --: | --: |
| 0.5–0.6 | 0.545 | 0.681 | 47 |
| 0.6–0.7 | 0.645 | 0.941 | 68 |
| 0.7–0.8 | 0.760 | 0.933 | 75 |
| 0.8–0.9 | 0.874 | 0.963 | 644 |
| 0.9–1.0 | 0.933 | 0.977 | 1553 |

**Interpretación.** El modelo está **bien calibrado y más bien sub-confiado**: en
todos los bins el acierto empírico supera a la confianza declarada (la diagonal se
cruza por encima), lo contrario del sobreajuste. Gana a la climatología en Brier,
pero **no** en log-loss (0.1789 > 0.1765). El motivo es honesto: el estado
realizado-proxy es 95.7% "baja volatilidad", y una climatología que siempre
predice esa mezcla es un rival duro en log-loss. El modelo no domina esa métrica;
domina donde la penalización cuadrática (Brier) premia acertar los episodios raros
de alta volatilidad. Veredicto ambiguo (⚠): calibración sólida, superioridad sobre
la climatología solo parcial.

---

## Test 3 — Precisión direccional (Pesaran-Timmermann)

**Resultado R6 (2026-07-29): señal direccional marginalmente significativa al 5%
(⚠, al borde).** Re-corrido con multistart completo (n_starts=20, `run_v1 ablation
--jobs -1`, 19 bloques, n=2386): hit rate 55,28% vs 54,17% esperado, PT=1,663,
**p=0,048**. Cruza el umbral del 5% por poco. Es un **cambio respecto al `--quick`**
(p=0,082 abajo): el multistart R6 halla mejores óptimos → `r_pred_mean` algo distinto
→ el hit rate sube de 55,0% a 55,28%. Lectura honesta: el resultado es **marginal y
sensible a la calidad del multistart** (p justo por debajo de 0,05); no es una señal
direccional robusta, y la nota metodológica de abajo (regímenes de volatilidad ⇒
señal direccional de segundo orden; signo predicho positivo la gran mayoría de los
días) sigue aplicando. Se reporta tal cual (R8).

<details><summary>Resultado provisional anterior (`--quick`, 2026-07-18) — para el registro</summary>

Historia del bloqueo, para el registro: en la fecha original de este reporte
(2026-07-14) el test no era ejecutable porque `r_pred_mean` no se persistía en
`history.parquet`. La sesión del 2026-07-15/16 regeneró el walk-forward
(`run_id=16d4190d17e2`, 19 bloques, mismo diseño) con la columna ya persistida
(`walkforward.py:262`), lo que desbloqueó el insumo. El test se corrió el
2026-07-18 sobre ese artefacto con la `pesaran_timmermann` ya implementada y
testeada (`tests/test_tests_stat.py`). **Advertencia R6: esa corrida también es
`--quick` (n_starts=8) — número provisional, como el resto del reporte.**

Resultado (SPY, n=2387 días OOS):

| Métrica | Valor |
| :-- | :-- |
| Hit rate direccional | 55,0% |
| Hit rate esperado bajo independencia | 54,1% |
| Estadístico PT | 1,390 |
| p-valor (unilateral) | 0,082 |

**No hay evidencia de que el modelo prediga el signo mejor que el azar al nivel
del 5%.** Es exactamente lo que la nota metodológica de la versión anterior de
este reporte anticipaba: en un modelo de regímenes de **volatilidad** los μ_k
difieren poco y la señal direccional es de segundo orden. El signo predicho es
positivo el 87,3% de los días (el μ del régimen dominante de baja volatilidad es
positivo), así que el test PT es en gran parte una comparación contra "predecir
siempre subida". Se reporta tal cual (R8): el valor del modelo, si lo hay, está
en la densidad predictiva (Test 4), no en la dirección.

</details>

Nota BTC (línea secundaria, `artifacts/btc/latest`, 9 bloques): el test
**degenera** — el signo predicho es positivo el 100% de los días (coherente con
que el BIC de BTC elige K=1) y el PT no es interpretable (hit rate 49,7% =
esperado). Se registra como no informativo, no como aprobado ni suspendido.

---

## Test 4 — Comparación con benchmark (Diebold-Mariano)

Benchmarks definidos ANTES de ver resultados:

- **BM1**: predicción constante = frecuencia histórica de cada régimen
  (climatología). Pérdida: log-loss de régimen por observación sobre el proxy
  y_t = argmax ξ_{t|t}. Espacio de clases común (K=2) ⇒ comparación válida.
- **BM2**: GARCH de un solo régimen (M0 de la ablación). Pérdida: log-densidad
  predictiva del retorno por observación (la única comparable entre modelos con
  distinto K; es la métrica primaria del proyecto).

**Resultado.**

| Comparación | pérdida | DM stat | p-valor | ¿modelo mejor? |
| :-- | :-- | --: | --: | :-- |
| Modelo vs **BM1** (climatología) | log-loss de régimen | +0.20 | 0.84 | **No** |
| Modelo (M1, regímenes) vs **BM2** (M0, 1 régimen) | log-densidad del retorno | **−3.29** | **0.001** | **Sí** |

(La fila BM2 se consume de la ablación ya corrida, `reports/ablation_news.md`:
DM(M1 vs M0). DM<0 ⇒ el primer modelo pierde menos.)

**Interpretación.** Dos benchmarks, dos veredictos, y juntos cuentan la historia
real: el modelo **no** le gana a "casi siempre está en calma" cuando la métrica es
la etiqueta de régimen (BM1: DM=0.20, p=0.84), pero **sí** le gana de forma
contundente al GARCH de un solo régimen cuando la métrica es la densidad del
retorno (BM2: DM=−3.29, p=0.001). Es decir: el aporte del modelo está en la
**varianza condicional** (dos regímenes de volatilidad capturan la densidad mejor
que uno), no en producir una "llamada" de régimen que supere a la climatología.

---

## Test 5 — Walk-forward económico

**Resultado (parte económica): ejecutado 2026-07-18 — NO SUPERA el criterio
pre-registrado (⚠, resultado negativo limpio).**

Historia para el registro: en la fecha original de este reporte no existía
regla de trading (decisión reservada al director). El 2026-07-18 el director
aprobó la regla pre-registrada en `docs/preregistro_regla_trading.md` —
des-riesgo binario: salir a efectivo cuando P(régimen alta vol., ξ filtrada del
día anterior) > 0.5, costo 2 pb por cambio de posición, criterio de éxito = IC
95% bootstrap (Politis-Romano, bloque automático) del Sharpe de la diferencia
vs comprar-y-mantener excluye 0 por arriba. El documento congeló regla,
parámetros e hipótesis ANTES de correr nada (R8); la hipótesis declarada era
que NO superaría el criterio (consistente con el Test 6). Corrida:
`scripts/run_economic_v4.py` sobre `run_id=16d4190d17e2` (2387 días OOS,
19 bloques; **`--quick`, provisional hasta R6**). Artefacto:
`artifacts/latest/validation.json`.

| Métrica (2 pb) | Estrategia | Buy & hold |
| :-- | --: | --: |
| Sharpe anualizado | 0.797 | 0.780 |
| Retorno anualizado | 14.2% | 14.3% |
| Volatilidad anualizada | 17.7% | 18.3% |
| Max drawdown | 31.4% | 33.7% |
| Exposición | 95.7% | 100% |
| Cambios de posición | 132 | 0 |

**Sharpe de la diferencia: −0.030, IC95 [−0.704, 0.530] → incluye 0, NO
SUPERA.** Sensibilidad: a 0 pb la diferencia es +0.031 [−0.633, 0.582]; a 5 pb,
−0.122 [−0.831, 0.463] — el veredicto no cambia con el costo. La hipótesis nula
declarada en el pre-registro se confirma: **el valor del indicador es
informacional (densidad predictiva, Tests 4-5 estadístico), no de estrategia.**
Observación informativa (sin umbral pre-registrado, no cambia el veredicto): la
estrategia reduce el drawdown máximo (31.4% vs 33.7%) con solo 132 cambios en
9.5 años — dirección esperable de una regla de des-riesgo, pero no acreditada
estadísticamente. La línea BTC no aplica: su artefacto es K=1 (sin régimen de
alta volatilidad que evitar; regla vacua por construcción).

Lo anterior deja el eje económico CERRADO con resultado negativo documentado.
El resto de esta sección es el walk-forward **estadístico** original:

Lo que **sí** corrió y se consume es el walk-forward **estadístico**
(`walkforward.json`, run `c1a8e85ca408`): **19 bloques** de test con
re-estimación desde cero por bloque (R2 cumplida; mínimo 6). Métrica por bloque =
log-densidad predictiva del retorno por observación, y consistencia contra la
climatología de régimen:

| blk | ventana de test | n | loglik_test/obs | reg-LL modelo | reg-LL base | bate base |
| --: | :-- | --: | --: | --: | --: | :-: |
| 0 | 2017-01…2017-07 | 125 | −0.629 | 0.097 | 0.093 | n |
| 1 | 2017-07…2018-01 | 127 | −0.482 | 0.179 | 0.093 | n |
| 2 | 2018-01…2018-07 | 125 | −1.315 | 0.231 | 0.218 | n |
| 3 | 2018-07…2019-01 | 126 | −1.259 | 0.181 | 0.118 | n |
| 4 | 2019-01…2019-07 | 125 | −1.095 | 0.147 | 0.118 | n |
| 5 | 2019-07…2020-01 | 127 | −0.983 | 0.113 | 0.093 | n |
| **6** | **2020-01…2020-07 (COVID)** | 126 | **−2.123** | 0.199 | 0.241 | **Y** |
| 7 | 2020-07…2021-01 | 126 | −1.345 | 0.146 | 0.241 | Y |
| 8 | 2021-01…2021-07 | 126 | −1.210 | 0.252 | 0.389 | Y |
| 9 | 2021-07…2022-01 | 126 | −1.077 | 0.230 | 0.241 | Y |
| 10 | 2022-01…2022-07 | 125 | −1.919 | 0.199 | 0.143 | n |
| 11 | 2022-07…2023-01 | 126 | −1.874 | 0.188 | 0.167 | n |
| 12 | 2023-01…2023-07 (SVB) | 124 | −1.342 | 0.389 | 0.520 | Y |
| 13 | 2023-07…2024-01 | 127 | −1.101 | 0.151 | 0.093 | n |
| 14 | 2024-01…2024-07 | 125 | −0.991 | 0.195 | 0.143 | n |
| 15 | 2024-07…2025-01 | 127 | −1.343 | 0.116 | 0.117 | Y |
| 16 | 2025-01…2025-07 | 123 | −1.606 | 0.113 | 0.094 | n |
| 17 | 2025-07…2026-01 | 127 | −0.968 | 0.137 | 0.117 | n |
| 18 | 2026-01…2026-07 | 124 | −1.237 | 0.137 | 0.119 | n |

**Interpretación.** No es un desempeño promedio uniforme: **el modelo bate a la
climatología de régimen en 6 de 19 bloques, y esos 6 son los de estrés** (COVID
2020, la salida de 2020-21, SVB 2023) — justo donde un modelo condicional debe
aportar. En los tramos tranquilos la climatología 95.7% no se supera en log-loss
(coherente con el Test 2/4). No hay ningún bloque donde el modelo sea
*catastróficamente* peor que el azar; los peores en densidad predictiva
(bloques 6, 10, 11, 16) coinciden con shocks de volatilidad, no con fallos del
modelo. El veredicto es ambiguo (⚠) porque el eje económico no se pudo testear;
en el eje estadístico el modelo es consistente y su valor se concentra en el estrés.

---

## Test 6 — Sharpe real (Bootstrap Politis-Romano)

El IC 95% del Sharpe se computa con `sharpe_ci` (bootstrap estacionario,
longitud de bloque elegida por **Politis-White 2004**; ver `validation/bootstrap.py`).
Serie primaria: retornos OOS de SPY en los días en que el modelo **predijo** (info
a t−1, sin look-ahead) el régimen de baja volatilidad.

**IC 95% del Sharpe (estado predicho = baja volatilidad, n=2343):
Sharpe = 0.75, IC [0.18, 1.42], longitud de bloque = 11. El intervalo EXCLUYE el
cero (`includes_zero = False`).**

| serie | n | Sharpe | IC 95% | bloque | ¿incluye 0? |
| :-- | --: | --: | :-- | --: | :-: |
| Estado predicho = baja vol | 2343 | 0.75 | [0.18, 1.42] | 11 | No |
| Estado predicho = alta vol | 44 | 2.21 | [−2.55, 7.92] | 1 | Sí |
| **SPY OOS completo (comprar y mantener)** | 2387 | **0.78** | **[0.18, 1.43]** | 4 | No |

**Interpretación — sin maquillar.** El IC excluye el cero, pero eso **no es
evidencia de que el modelo genere Sharpe**: el Sharpe del cubo "baja volatilidad
predicha" (0.75) es **prácticamente idéntico** al de comprar y mantener SPY sin
modelo alguno (0.78), con IC casi calcados. La condición de régimen no mueve el
Sharpe. El cubo "alta volatilidad predicha" tiene solo 44 días y su IC es tan
ancho que incluye el cero — no dice nada. Además, esto es un Sharpe **descriptivo
del activo**, no de una estrategia ejecutable (no hay costo del lado corto/plano;
CLAUDE.md: no es una señal de trading). Conclusión honesta: el Sharpe positivo del
período es el del mercado alcista 2017-2026, no un producto del indicador (⚠).

---

## Test 7 — Ablación de noticias

**Actualización 2026-07-29 (R6, cerrada).** Escalera M0-M3 corrida con R6
(n_starts=20) y **con la L1 canónica** (λ por CV dentro del train de cada bloque,
grid [0, 0.5, 2, 8, 32]), sobre la muestra alineada a la cobertura macro (2135 obs
OOS, 17 bloques, 2013-12-27..2026-07-10; alinear recorta ~11 meses de warm-up del
z-score de `hy_oas_z`, R4, para que los cuatro peldaños compartan fechas y el DM
sea válido). Fuente: `scripts/run_m3_l1.py`, `reports/ablation_m3_l1.md`.

| Modelo | Especificación | log-loss OOS/obs | DM vs anterior |
| :-- | :-- | --: | --: |
| M0 | GARCH 1 régimen | 1.3713 | — |
| M1 | HMM K=2, P fija | **1.3321** | DM=−3.40, **p=0.001** (mejor) |
| M2 | + TVTP técnico (sma_gap, bb_width_z), L1 | 1.3369 | DM=+2.25, p=0.025 (peor) |
| M3 | + macro (slope_2s10y, hy_oas_z), L1 | 1.3416 | **DM=+1.04, p=0.299 (no dist.)** |
| M4 | + SI_t (sorpresa) | — | — (congelado: consenso de paga) |
| M5 | + λ_N(t) (Hawkes) | — | — (esperando corpus GDELT) |

**VEREDICTO (⚠).**
1. **Regímenes: aportan, robusto.** M1 mejora a M0 de forma distinguible
   (DM=−3.40, p=0.001), y el resultado se repite en TODAS las corridas y muestras.
   Es el hallazgo sólido: el valor está en la varianza condicional de dos regímenes.
2. **Macro (M3): NO aporta, ahora con test justo.** Con la L1 canónica, la macro
   ya **no degrada catastróficamente** (M3=1.3416, muy por encima del piso M0 y
   cerca de M2; el 1.3790 "peor que M0" de la corrida sin L1 era sobreajuste no
   regularizado). Pero **tampoco mejora a M2** (DM=+1.04, p=0.299, indistinguible).
   La 2s10y y el HY OAS no añaden densidad predictiva OOS sobre el bloque técnico.
3. **TVTP técnico (M2): marginal y sensible a la muestra.** En esta muestra M2 es
   peor que M1 (p=0.025); en la muestra completa (`run_v1`, 3400 obs) M2 era
   nominalmente mejor que M1 (1.2566 < 1.2629, p=0.253). No es robusto en ningún
   sentido: el aporte de "modular" las transiciones (técnicas o macro) es, en el
   mejor de los casos, de segundo orden frente al de tener regímenes.

Cierre del eje de covariables de transición: **el único aporte robusto es M1 sobre
M0 (regímenes); ni el TVTP técnico ni la macro añaden valor OOS distinguible.**
Resultado negativo limpio y ahora JUSTO (con L1), documentado (R8).

**M4/M5 siguen sin ejecutarse, por razones distintas y explícitas:** M4
(sorpresa) está **congelado** por decisión del director (no hay fuente gratuita
de consenso histórico; solo Trading Economics de paga). M5 (Hawkes) espera el
backfill de titulares de GDELT, en curso. Ninguno es un DM perdido: es ausencia
de datos aguas arriba, documentada (R8). El compromiso pre-registrado (publicar
sin capa de noticias si no aporta) sigue en pie.

**M4/M5 siguen sin ejecutarse, por razones distintas y explícitas:** M4
(sorpresa) está **congelado** por decisión del director (no hay fuente gratuita
de consenso histórico; solo Trading Economics de paga). M5 (Hawkes) espera el
backfill de titulares de GDELT, en curso. Ninguno es un DM perdido: es ausencia
de datos aguas arriba, documentada (R8). El compromiso pre-registrado (publicar
sin capa de noticias si no aporta) sigue en pie.

---

## Limitaciones

- **Estado R6 (actualización 2026-07-29): Tests 1, 3, 4 y 7(M0-M2) YA son R6**
  (multistart completo, n_starts=20), gracias a la paralelización por bloques del
  walk-forward (`--jobs`, resultado idéntico al serial). **Siguen `--quick` los
  Tests 5 (económico) y 6 (Sharpe)**: consumen el walk-forward anterior; pasarlos
  a R6 es re-correr sus scripts sobre el nuevo `walkforward_v1.json` (pendiente,
  job corto). El Test 2 (calibración) tiene ya cifras R6 en `walkforward_v1.json`
  (Brier 0.0817, log-loss 0.1823, ECE 0.069).
- **Test 1 (Hansen): RESUELTO con R6.** La corrida completa 2026-07-22
  (`v1_kselect.json`, B=49) da K=2 vs K=1 **p=0.02**: se rechaza K=1. K=2 ya no
  descansa solo en el BIC.
- **Test 3 (direccional): R6 marginal.** p=0.048 (cruza el 5% por poco); sensible
  al multistart (era p=0,082 en `--quick`). No es una señal direccional robusta.
- **Test 5 (económico): ejecutado el 2026-07-18** con la regla pre-registrada
  aprobada por el director (`docs/preregistro_regla_trading.md`). NO supera el
  criterio de éxito (resultado negativo limpio, confirmando la hipótesis
  declarada); `artifacts/latest/validation.json` existe ahora. **Aún `--quick`**;
  re-correr sobre el walk-forward R6 para cerrarlo.
- **Test 6 no es un Sharpe de estrategia:** es descriptivo del activo condicionado
  al régimen predicho, sin costos ni lado corto; no acredita valor económico del
  modelo. **Aún `--quick`.**
- **Test 7: M3 CERRADO con test justo (R6 + L1, 2026-07-29).** M3 (macro) con la
  malla L1 por bloque NO aporta sobre M2 (DM=+1.04, p=0.299); ya no degrada como
  en la corrida sin L1 (eso era sobreajuste). Único aporte robusto: los regímenes
  (M1>M0). M4 (sorpresa) CONGELADO por decisión del director (sin fuente de
  consenso histórico). M5 (Hawkes) espera el corpus GDELT.
- **El "estado realizado" es un proxy** (argmax ξ_{t|t}), no una etiqueta
  verdadera: los Tests 2, 4-BM1 y 5 miden consistencia predicho vs. concluido, no
  la existencia objetiva de los regímenes.

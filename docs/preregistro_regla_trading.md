# Pre-registro — Regla de trading para el walk-forward económico (Test 5 de V4)

Estado: **BORRADOR PARA APROBACIÓN DEL DIRECTOR — NADA DE ESTO SE HA CORRIDO.**
Fecha del diseño: 2026-07-18.
Disciplina: R8 (el test se diseña y se congela ANTES de correrlo; el resultado
se reporta igual si sale negativo). Este documento es el congelamiento.

## 1. Qué pregunta responde (y qué no)

Pregunta única: **¿la probabilidad filtrada de régimen aporta valor económico
frente a comprar y mantener el activo ancla, después de costos?**

No responde: si el modelo "sirve para tradear" en general, ni optimiza ninguna
regla. El proyecto sigue sin ser una señal de trading (CLAUDE.md §2); esto es
un test de validación económica de la información del indicador, con una regla
deliberadamente simple y fijada de antemano.

## 2. La regla (fijada antes de ver un solo número económico)

**Regla de des-riesgo binaria por régimen de alta volatilidad:**

```
w_{t+1} = 1                                  si  P_t(alta vol) <= p_umbral
w_{t+1} = w_reducido                         si  P_t(alta vol) >  p_umbral
```

donde:

- `P_t(alta vol)` = componente de ξ_{t|t} del régimen de mayor varianza
  incondicional (el último índice bajo R5), **filtrada, no suavizada** (R1) y
  **conocida al cierre del día t** — la posición rige el retorno de t+1.
  En `history.parquet` esto es `xi_filtered_K-1` en t aplicado a `r` en t+1
  (un `shift(1)` explícito, R3).
- `p_umbral = 0.5` — el clasificador natural de argmax binario. No se optimiza,
  no se escanea una rejilla: un solo valor, elegido por ser el único punto
  focal no arbitrario de una probabilidad. Va a config, no al código (R7).
- `w_reducido = 0.0` (salir a efectivo, retorno 0 en esos días, sin
  apalancamiento ni cortos). Es el caso más simple y más interpretable;
  cualquier valor intermedio introduciría un parámetro más que justificar.

Sin señal direccional en la regla: el Test 3 (PT, ejecutado 2026-07-18)
confirmó que el modelo no predice signo. La regla solo usa lo que la
validación dice que el modelo sabe: **cuándo el régimen es de alta
volatilidad**.

## 3. Costos de transacción

- Costo por cambio de posición: **2 pb (0.02%) por unidad de turnover**
  (|Δw| = 1 en cada entrada o salida completa). Para SPY (ETF más líquido del
  mundo) 2 pb es conservador para el spread + impacto de una orden pequeña.
- Se reporta además la sensibilidad a 0 pb y 5 pb (solo como columnas
  adicionales, sin cambiar el veredicto principal, que es a 2 pb).
- Los tres valores van a `config/base.yaml`, no al código (R7).

## 4. Datos y alineación PIT

- Insumo: `artifacts/latest/history.parquet` del walk-forward OOS vigente
  (hoy `run_id=16d4190d17e2`, 19 bloques, 2387 días; **`--quick`, así que el
  resultado será PROVISIONAL hasta re-correr con R6 completo** — se dirá en el
  reporte con la misma prominencia que el resultado).
- `r` está en % (log-retornos × 100); la estrategia se computa en la misma
  unidad y el equity con `exp(cumsum(r_estrategia/100))`.
- El primer día de cada bloque OOS usa la ξ filtrada del último día del bloque
  anterior (la inicialización arrastrada ya está en el artefacto); el primer
  día de todo el período OOS no tiene posición previa definida y se inicia en
  w=1 (comprado), documentado.
- Prohibido: xi_smoothed (R1), cualquier columna futura, cualquier
  re-estimación con datos del período evaluado.

## 5. Métricas y criterios de éxito (congelados)

Comparación: estrategia vs **comprar y mantener SPY** sobre el MISMO período
OOS concatenado.

| Métrica | Cómo | Criterio |
| :-- | :-- | :-- |
| Sharpe anualizado de la diferencia | IC 95% por bootstrap estacionario (Politis-Romano, `optimal_block_length`) sobre la serie diaria `r_estrategia − r_buyhold` | **Éxito** solo si el IC del Sharpe de la diferencia excluye 0 por arriba, a 2 pb de costo |
| Max drawdown | ambas series | informativo (sin umbral) |
| Retorno anualizado, vol anualizada | ambas series | informativo |
| % de días expuesto, nº de cambios de posición, turnover total | estrategia | informativo (sanidad: una regla que cambia de posición cada 2 días es ruido) |
| Desglose por bloque | Sharpe de la diferencia en cada uno de los 19 bloques | informativo — se espera (hipótesis declarada) que el valor, si existe, se concentre en los 6 bloques de estrés que ya identificó el Test 5 estadístico |

**Hipótesis nula esperada y declarada:** dado que el Test 6 de V4 mostró un
Sharpe condicional indistinguible del de comprar y mantener, lo más probable es
que esta regla NO supere el criterio de éxito. Si eso ocurre, el reporte lo
dirá tal cual y el proyecto documentará que su valor es informacional (densidad
predictiva) y no de estrategia — un resultado negativo limpio cierra la
pregunta igual de bien (R8).

## 6. Qué está prohibido después de correr

- Cambiar `p_umbral`, `w_reducido` o los costos después de ver resultados.
- Añadir filtros ("solo si la entropía es baja", "solo en bloques X") post-hoc.
  Si el director quiere variantes, son un SEGUNDO pre-registro, corrido y
  reportado por separado y etiquetado como exploratorio.
- No reportar alguna de las métricas de la tabla.

## 7. Implementación prevista (tras aprobación)

- `src/irfn/validation/economic.py`: `economic_walkforward(history_df, p_threshold, w_reduced, cost_bps)` — puro, testeable, sin I/O.
- Config: sección `v4.economic` en `config/base.yaml` (umbral, w, costos).
- Tests: alineación del shift (la posición de t+1 no puede ver ξ de t+1),
  costos cargados solo en cambios de posición, equity reproducible a mano en
  un caso de 5 días.
- Salida: `artifacts/latest/validation.json` (el que el panel ya espera) +
  sección nueva en `reports/validation_v4.md` reemplazando el "no existe" del
  Test 5 económico, con el mismo run_id del artefacto consumido.
- La línea BTC NO se corre en esta pasada: su artefacto es K=1 (sin régimen de
  alta vol que evitar) — se documentará como no aplicable.

Estimación: 3–5 horas de implementación + tests; la corrida en sí es trivial
(opera sobre el parquet ya existente, sin re-estimar nada).

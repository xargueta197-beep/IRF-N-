# SPY vs BTC — nota comparativa (segunda línea de activo)

generado: 2026-07-16  |  ambas líneas corridas `--quick` (PROVISIONAL, NO cumplen R6)

## Alcance de esta nota

Hasta esta sesión, IRF-N **nunca había corrido sobre BTC**: `config/base.yaml`
(`v0.anchor_asset`) fijaba SPY como único activo ancla desde V0, y así se
mantuvo sin cambios hasta V4 (`reports/validation_v0.md` … `validation_v4.md`).
BTC solo existía como fuente de datos auditada (`reports/data_audit.md`
sección 1), nunca como activo modelado — y su descarga real vía Binance
tampoco estaba implementada (`src/irfn/data/prices.py` lanzaba
`NotImplementedError` para cualquier fuente que no fuera yfinance).

Esta sesión implementó la descarga real de Binance y corrió el pipeline
completo V0→V3 sobre BTC (`BTCUSDT`, klines diarios desde 2017-08-17), como
**segunda línea independiente, en paralelo a SPY**, sin tocar ningún artefacto
ni reporte validado de SPY. Ver `reports/validation_v0_btc.md`,
`validation_v1_btc.md`, `ablation_news_btc.md`, `validation_v3_btc.md` para el
detalle completo por versión. Esta nota solo resume los contrastes.

**No están en el alcance de esta nota:** una validación estadística formal
tipo V4 para BTC (bootstrap del Sharpe, Pesaran-Timmermann, Diebold-Mariano
cruzado entre activos) — eso requeriría su propio diseño y no se ha corrido.
Tampoco se re-corrió nada de esto sin `--quick` (R6): ambas líneas siguen
PROVISIONALES, SPY incluida.

## V0: ¿el ξ predicho le gana a la climatología?

| | SPY | BTC |
| :-- | :-- | :-- |
| Log-loss modelo vs baseline | 0.1786 vs 0.1765 (NO gana) | 0.4501 vs 0.4240 (NO gana) |
| Brier modelo vs baseline | 0.0791 vs 0.0818 (gana) | 0.2762 vs 0.2561 (NO gana) |
| Veredicto | PARCIAL | **NO** |
| Bloques walk-forward | 19 | 9 |
| Observaciones OOS | 2387 | 1645 |

BTC parte de menos historia (2017-08-17 vs 2013-01-01 de SPY), de ahí los 9
bloques frente a 19 — ambos superan el mínimo de 6 (R2/R8). En ninguno de los
dos activos el ξ predicho le gana de forma clara a predecir siempre las
frecuencias marginales de régimen; en BTC el modelo pierde en ambas métricas,
más claramente que en SPY.

## Persistencia de régimen (kappa por bloque, V0)

- **SPY:** el régimen de baja volatilidad es consistentemente persistente
  (kappa 0.90–0.99 en los 19 bloques). El régimen de alta volatilidad es
  **inestable entre bloques**: a veces persistente (kappa≈1.0) y a veces casi
  no-persistente (kappa≈0.30–0.62 en varios bloques) — el limitante conocido
  de V0 documentado en `CLAUDE.md` (un régimen puede colapsar a "absorbe
  outliers" en vez de ser un estado de alta vol persistente).
- **BTC:** **ambos** regímenes son consistentemente persistentes en los 9
  bloques (kappa 0.90–0.99 en las dos columnas, sin el patrón errático que
  muestra SPY en su régimen de alta vol). No es necesariamente una ventaja
  metodológica — con solo 9 bloques hay menos oportunidad de observar la
  inestabilidad que SPY sí muestra en 19; es una observación, no una
  conclusión.

## V1: selección de K (BIC + bootstrap)

| | SPY | BTC |
| :-- | :-- | :-- |
| Ganador BIC | K=2, dist=normal | **K=1, dist=t** |
| Bootstrap 2 vs 1 (p-value) | 0.167 | 0.167 |
| DM(titular V1 vs V0) | stat=-1.144, p=0.253 (mejor, no significativo) | stat=0.209, p=0.835 (no mejor, no significativo) |

**BTC eligió K=1** (un solo régimen, sin TVTP que evaluar) — divergencia real
frente a SPY (K=2). El bootstrap V1 corrió con solo 5 réplicas (`--quick`), lo
que cuantiza el p-value a fracciones de 1/6 ≈ 0.167/0.333/0.5; que ambos
activos muestren el mismo 0.167 es un artefacto de esa cuantización con pocas
réplicas, no evidencia de que la fuerza estadística sea idéntica — no
comparable hasta correr con el `n_boot` completo (config `v1.ktest.n_boot`).
En ninguno de los dos activos el titular de V1 le gana a V0 de forma
significativa.

## V2 (sorpresa) y V3 (Hawkes): mismo bloqueante en ambos activos

Ninguna de las dos capas de noticias está activa en SPY ni en BTC, y por la
misma razón — **no es un resultado de BTC, es un hueco de datos compartido**:

- **V2 (sorpresa calendarizada):** 0 eventos de consenso histórico
  acumulados en ambos (`TRADING_ECONOMICS_API_KEY` sin resolver, ver
  `reports/data_audit.md` secciones 4 y 7). `news_layer_params.active=false`
  en los dos.
- **V3 (Hawkes):** el corpus de titulares GDELT es **compartido** entre
  activos (no es específico de BTC ni de SPY) y hoy solo tiene 115 titulares
  puntuados, por debajo del mínimo de 200 (`news.headlines.min_events_fit`)
  para un MLE honesto — bloqueado en ambos por la misma razón.

## Hallazgo aparte (no de BTC, aplica a ambos activos por igual)

Al correr V2 para BTC se descubrió que `ALFRED_API_KEY` **ya está
configurada** en `.env` (contradice el estado documentado en `CLAUDE.md`,
donde M3/macro se atribuye a la ausencia de esa key). Con la key presente, la
descarga de vintages ALFRED funciona, pero `features/macro.py` falla con
`cannot reindex on an axis with duplicate labels` al construir las
covariables macro — un bug real y nuevo, no relacionado con BTC, que ahora
bloquea M3 en **ambos** activos por una razón distinta a la documentada
hasta hoy. No se investigó ni se corrigió en esta sesión (fuera de alcance);
queda anotado para que el director decida si se prioriza.

## Qué NO resuelve esta nota

- No hay walk-forward económico para BTC (ninguna regla de trading definida,
  igual que para SPY).
- No se corrió V4 (validación estadística formal) para BTC.
- Ninguna corrida de esta nota cumple R6 (multistart completo); todo sigue
  `--quick`, incluidas las líneas de SPY que ya existían.

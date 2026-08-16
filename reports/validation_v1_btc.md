# Validacion V1 - IRF-N (BTC) -- CORRIDA PROVISIONAL (--quick, NO cumple R6)

generado: 2026-07-16T20:02:53.611419+00:00  |  asof: 2026-07-16  |  n_obs: 2987 (desde 2018-05-13)

Alcance V1: TVTP + seleccion de K + innovaciones t de Student. SIN NOTICIAS (surprise/Hawkes son V2/V3).

> **Bloqueante macro documentado (R4):** hy_oas_z EXCLUIDA: sin ALFRED_API_KEY y HY OAS (BAMLH0A0HYM2) limitado a ventana rodante de 3 anios en FRED/ALFRED, insuficiente para el walk-forward (reports/data_audit.md). Titular corre tecnico-solo; jamas se rellena con FRED revisado (R4). Retomar cuando haya fuente vintage con cobertura.

Titular tecnico-solo: covariables del TVTP = `['sma_gap', 'bb_width_z']`.

## 1. Seleccion de K: tabla BIC completa (no se esconden los perdedores)

Multistart R6 = 6 arranques por celda. BIC in-sample; la separacion fuera de muestra se dirime en la ablacion walk-forward (fase 2).

| K | dist | loglik | k_params | BIC | AIC | conv/arranques | nu |
| --: | :-- | --: | --: | --: | --: | :-- | :-- |
| 1 | normal | -7603.7 | 5 | 15247.5 | 15217.5 | 6/6 | - |
| 2 | normal | -7284.8 | 12 | 14665.6 | 14593.5 | 3/6 | - |
| 3 | normal | -7238.2 | 21 | 14644.5 | 14518.5 | 2/6 | - |
| 4 | normal | -7216.7 | 32 | 14689.4 | 14497.4 | 1/6 | - |
| 1 | t | -7255.6 | 6 | 14559.1 **<-** | 14523.1 | 6/6 | 3.1 |
| 2 | t | -7233.1 | 14 | 14578.3 | 14494.3 | 2/6 | 2.3, 3.5 |
| 3 | t | -7220.0 | 24 | 14632.1 | 14488.1 | 1/6 | 1271.2, 3.1, 5.7 |
| 4 | t | -7208.1 | 36 | 14704.4 | 14488.3 | 1/6 | 2.0, 2.4, 3.1, 4.5 |

**Ganador por BIC: K=1, dist=t** (BIC menor = mejor).

## 2. Test de numero de regimenes: bootstrap parametrico K vs K-1

Por que NO el LR con chi2: bajo H0 (K-1) los parametros del regimen extra no estan identificados (Davies; parametros de molestia solo bajo la alternativa), asi que 2*(llK - llK-1) NO es chi2. Hansen (1992) exacto es prohibitivo por costo (malla multidimensional x optimizacion por punto). Se usa bootstrap parametrico: se simula la distribucion nula del LR desde el modelo K-1 ajustado. Decision y costo documentados en validation/tests_stat.py.

| K vs K-1 | LR_obs | p-value (boot) | replicas ok/tot | LR boot q50 | q95 |
| :-- | --: | --: | :-- | --: | --: |
| 2 vs 1 | 44.81 | 0.167 | 5/5 | 10.12 | 15.50 |
| 3 vs 2 | 26.22 | 0.167 | 5/5 | 10.18 | 10.71 |
| 4 vs 3 | 23.80 | 0.500 | 5/5 | 20.18 | 31.62 |

> p pequeno => hay evidencia de que K regimenes mejora sobre K-1. Se reporta la escalera completa; ningun K se esconde.

## 3. Contraste con la apuesta a priori del director

> Apuesta: K=3 le gana a K=4 fuera de muestra porque el 4o estado casi nunca tiene suficientes observaciones para estimar su GARCH. **El resultado manda.**

- BIC(K=3,t)=14632.1 vs BIC(K=4,t)=14704.4: K=3 le gana a K=4 en BIC (coherente con la apuesta).
- La prueba definitiva es fuera de muestra (ablacion walk-forward, fase 2), no el BIC in-sample. Ver seccion de DM cuando la fase 2 corra.

## 4. Modelo titular V1 y betas del logit de transicion (R7: por MLE con SE)

BIC eligio K=1: sin TVTP posible; titular = GARCH un regimen.

## 5. Pendiente de la fase 2 (ablacion walk-forward + Diebold-Mariano vs V0)

Correr `python scripts/run_v1.py ablation`. Completara: log-loss OOS de M0/M1/M2, DM(titular V1 vs V0), Pesaran-Timmermann direccional, y el veredicto fuera de muestra. Hasta entonces la seleccion de K de arriba es IN-SAMPLE (BIC + bootstrap).


---

# Fase 2: walk-forward fuera de muestra (PROVISIONAL --quick)

generado: 2026-07-16T20:34:57.141405+00:00. K=2, dist=t, covariables `['sma_gap', 'bb_width_z']`. Bloques de test: 8 (>=6, R2/R8), test=6 meses, 6 arranques/bloque (R6). Metrica primaria: log-loss predictiva del RETORNO fuera de muestra (menor es mejor).

## Ablacion M0/M1/M2 (un aporte a la vez)

| modelo | descripcion | K | dist | covs | bloques | n_oos | log-loss OOS/obs |
| :-- | :-- | --: | :-- | :-- | --: | --: | --: |
| M0 | GARCH un solo regimen (piso) | 1 | t | - | 8 | 1461 | 2.2664 |
| M1 | HMM K=2 P constante | 2 | t | - | 8 | 1461 | 2.2724 |
| M2 | +TVTP tecnico ['sma_gap', 'bb_width_z'] | 2 | t | ['sma_gap', 'bb_width_z'] | 8 | 1461 | 2.2708 |

## Diebold-Mariano entre peldanos consecutivos (perdida predictiva OOS)

| A vs B | DM stat | p-value | dif. media |
| :-- | --: | --: | --: |
| M1 vs M0 | 1.296 | 0.195 | 0.00605 |
| M2 vs M1 | -0.795 | 0.427 | -0.00165 |

> DM<0 => el primer modelo (A) tiene MENOR perdida (mejor). HAC Newey-West + correccion de muestra pequena Harvey-Leybourne-Newbold.

## Criterio de aceptacion: Diebold-Mariano titular V1 vs V0

- **DM(M2 titular V1 vs V0) = 0.209, p = 0.835**: el titular es no mejor que V0, no significativo (p>=0.05).
  - V0 = K=2 Normal P-constante, re-corrido sobre la MISMA muestra y malla de bloques que V1 (perdidas emparejadas por fecha). dif. media de perdida = 0.00101.

## Pesaran-Timmermann (precision direccional del titular)

- PT stat = -0.129, p = 0.551, aciertos = 0.498 vs esperado bajo independencia 0.499. H1 unilateral: precision direccional mayor que bajo independencia

> Nota honesta: la senal direccional del retorno usa un proxy debil (dominancia del regimen calmo); la senal direccional fuerte es tarea de la capa de noticias (V2). Si el test degenera por signo constante, se reporta NaN, no se maquilla.

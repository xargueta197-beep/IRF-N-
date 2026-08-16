# Validacion V1 - IRF-N (SPY) -- CORRIDA PROVISIONAL (--quick, NO cumple R6)

generado: 2026-07-13T01:57:10.415117+00:00  |  asof: 2026-07-10  |  n_obs: 3400 (desde 2013-01-02)

Alcance V1: TVTP + seleccion de K + innovaciones t de Student. SIN NOTICIAS (surprise/Hawkes son V2/V3).

> **Bloqueante macro documentado (R4):** hy_oas_z EXCLUIDA: sin ALFRED_API_KEY y HY OAS (BAMLH0A0HYM2) limitado a ventana rodante de 3 anios en FRED/ALFRED, insuficiente para el walk-forward (reports/data_audit.md). Titular corre tecnico-solo; jamas se rellena con FRED revisado (R4). Retomar cuando haya fuente vintage con cobertura.

Titular tecnico-solo: covariables del TVTP = `['sma_gap', 'bb_width_z']`.

## 1. Seleccion de K: tabla BIC completa (no se esconden los perdedores)

Multistart R6 = 6 arranques por celda. BIC in-sample; la separacion fuera de muestra se dirime en la ablacion walk-forward (fase 2).

| K | dist | loglik | k_params | BIC | AIC | conv/arranques | nu |
| --: | :-- | --: | --: | --: | --: | :-- | :-- |
| 1 | normal | -4201.6 | 5 | 8443.9 | 8413.3 | 6/6 | - |
| 2 | normal | -4049.1 | 12 | 8195.8 **<-** | 8122.3 | 4/6 | - |
| 3 | normal | -4027.8 | 21 | 8226.3 | 8097.5 | 1/6 | - |
| 4 | normal | -4017.5 | 32 | 8295.1 | 8098.9 | 1/6 | - |
| 1 | t | -4083.4 | 6 | 8215.6 | 8178.8 | 6/6 | 5.9 |
| 2 | t | -4048.0 | 14 | 8209.8 | 8123.9 | 1/6 | 15.2, 3.9 |
| 3 | t | -4025.5 | 24 | 8246.1 | 8098.9 | 1/6 | 3.9, 1629.7, 12.4 |
| 4 | t | -4013.2 | 36 | 8319.2 | 8098.5 | 1/6 | 250.1, 7180.2, 37.7, 3.8 |

**Ganador por BIC: K=2, dist=normal** (BIC menor = mejor).

## 2. Test de numero de regimenes: bootstrap parametrico K vs K-1

Por que NO el LR con chi2: bajo H0 (K-1) los parametros del regimen extra no estan identificados (Davies; parametros de molestia solo bajo la alternativa), asi que 2*(llK - llK-1) NO es chi2. Hansen (1992) exacto es prohibitivo por costo (malla multidimensional x optimizacion por punto). Se usa bootstrap parametrico: se simula la distribucion nula del LR desde el modelo K-1 ajustado. Decision y costo documentados en validation/tests_stat.py.

| K vs K-1 | LR_obs | p-value (boot) | replicas ok/tot | LR boot q50 | q95 |
| :-- | --: | --: | :-- | --: | --: |
| 2 vs 1 | 305.03 | 0.167 | 5/5 | 6.58 | 15.71 |
| 3 vs 2 | 42.70 | 0.167 | 5/5 | 11.98 | 14.74 |
| 4 vs 3 | 20.62 | 0.167 | 5/5 | 9.52 | 13.70 |

> p pequeno => hay evidencia de que K regimenes mejora sobre K-1. Se reporta la escalera completa; ningun K se esconde.

## 3. Contraste con la apuesta a priori del director

> Apuesta: K=3 le gana a K=4 fuera de muestra porque el 4o estado casi nunca tiene suficientes observaciones para estimar su GARCH. **El resultado manda.**

- BIC(K=3,normal)=8226.3 vs BIC(K=4,normal)=8295.1: K=3 le gana a K=4 en BIC (coherente con la apuesta).
- La prueba definitiva es fuera de muestra (ablacion walk-forward, fase 2), no el BIC in-sample. Ver seccion de DM cuando la fase 2 corra.

## 4. Modelo titular V1 y betas del logit de transicion (R7: por MLE con SE)

Titular: K=2, dist=normal, covariables `['sma_gap', 'bb_width_z']`, lambda L1 (elegido por CV DENTRO de la muestra, jamas mirando test) = 32.000. loglik=-4053.1, convergencia 1/6, hessiano_ok=False.

### Interpretacion de los signos (generada de los numeros reales)

Referencia del logit: el regimen de mayor varianza ('alta volatilidad') tiene beta=0 por identificacion. Un beta POSITIVO sobre la columna de un regimen calmo sube las probabilidades de ESE regimen frente a risk-off, es decir BAJA la probabilidad de transitar a risk-off (y viceversa).

- desde 'baja volatilidad', hacia 'baja volatilidad', covariable `sma_gap`: beta=0.001 (SE=0.018, z=0.03, no significativo). Efecto sobre prob. de risk-off: la baja; coherente con el prior.
- desde 'baja volatilidad', hacia 'baja volatilidad', covariable `bb_width_z`: beta=0.000 (SE=0.008, z=0.04, no significativo). Efecto sobre prob. de risk-off: la baja; CONTRARIO al prior (+).
- desde 'alta volatilidad', hacia 'baja volatilidad', covariable `sma_gap`: beta=0.000 (SE no fiable). Efecto sobre prob. de risk-off: la baja; coherente con el prior.
- desde 'alta volatilidad', hacia 'baja volatilidad', covariable `bb_width_z`: beta=0.000 (SE no fiable). Efecto sobre prob. de risk-off: la baja; CONTRARIO al prior (+).

Tabla del CV de lambda (no se esconden los lambdas perdedores):

| lambda | val loglik/obs | train loglik/obs | conv |
| --: | --: | --: | :-- |
| 0.00 | -1.2123 | -1.1874 | 1 |
| 0.50 | -1.2165 | -1.1886 | 1 |
| 2.00 | -1.2172 | -1.1888 | 1 |
| 8.00 | -1.2019 | -1.1903 | 1 |
| 32.00 | -1.1960 | -1.1932 | 1 |

## 5. Pendiente de la fase 2 (ablacion walk-forward + Diebold-Mariano vs V0)

Correr `python scripts/run_v1.py ablation`. Completara: log-loss OOS de M0/M1/M2, DM(titular V1 vs V0), Pesaran-Timmermann direccional, y el veredicto fuera de muestra. Hasta entonces la seleccion de K de arriba es IN-SAMPLE (BIC + bootstrap).


---

# Fase 2: walk-forward fuera de muestra

generado: 2026-07-16T01:21:44.772270+00:00. K=2, dist=normal, covariables `['sma_gap', 'bb_width_z']`. Bloques de test: 19 (>=6, R2/R8), test=6 meses, 20 arranques/bloque (R6). Metrica primaria: log-loss predictiva del RETORNO fuera de muestra (menor es mejor).

## Ablacion M0/M1/M2 (un aporte a la vez)

| modelo | descripcion | K | dist | covs | bloques | n_oos | log-loss OOS/obs |
| :-- | :-- | --: | :-- | :-- | --: | --: | --: |
| M0 | GARCH un solo regimen (piso) | 1 | normal | - | 19 | 2386 | 1.2940 |
| M1 | HMM K=2 P constante | 2 | normal | - | 19 | 2386 | 1.2629 |
| M2 | +TVTP tecnico ['sma_gap', 'bb_width_z'] | 2 | normal | ['sma_gap', 'bb_width_z'] | 19 | 2386 | 1.2566 |

## Diebold-Mariano entre peldanos consecutivos (perdida predictiva OOS)

| A vs B | DM stat | p-value | dif. media |
| :-- | --: | --: | --: |
| M1 vs M0 | -3.146 | 0.002 | -0.03110 |
| M2 vs M1 | -1.144 | 0.253 | -0.00631 |

> DM<0 => el primer modelo (A) tiene MENOR perdida (mejor). HAC Newey-West + correccion de muestra pequena Harvey-Leybourne-Newbold.

## Criterio de aceptacion: Diebold-Mariano titular V1 vs V0

- **DM(M2 titular V1 vs V0) = -1.144, p = 0.253**: el titular es MEJOR que V0, no significativo (p>=0.05).
  - V0 = K=2 Normal P-constante, re-corrido sobre la MISMA muestra y malla de bloques que V1 (perdidas emparejadas por fecha). dif. media de perdida = -0.00631.

## Pesaran-Timmermann (precision direccional del titular)

- PT stat = 1.663, p = 0.048, aciertos = 0.553 vs esperado bajo independencia 0.542. H1 unilateral: precision direccional mayor que bajo independencia

> Nota honesta: la senal direccional del retorno usa un proxy debil (dominancia del regimen calmo); la senal direccional fuerte es tarea de la capa de noticias (V2). Si el test degenera por signo constante, se reporta NaN, no se maquilla.


---

# Fase 2: walk-forward fuera de muestra

generado: 2026-07-30T02:06:55.612061+00:00. K=2, dist=normal, covariables `['sma_gap', 'bb_width_z']`. Bloques de test: 19 (>=6, R2/R8), test=6 meses, 20 arranques/bloque (R6). Metrica primaria: log-loss predictiva del RETORNO fuera de muestra (menor es mejor).

## Ablacion M0/M1/M2 (un aporte a la vez)

| modelo | descripcion | K | dist | covs | bloques | n_oos | log-loss OOS/obs |
| :-- | :-- | --: | :-- | :-- | --: | --: | --: |
| M0 | GARCH un solo regimen (piso) | 1 | normal | - | 19 | 2386 | 1.2940 |
| M1 | HMM K=2 P constante | 2 | normal | - | 19 | 2386 | 1.2629 |
| M2 | +TVTP tecnico ['sma_gap', 'bb_width_z'] | 2 | normal | ['sma_gap', 'bb_width_z'] | 19 | 2386 | 1.2566 |

## Diebold-Mariano entre peldanos consecutivos (perdida predictiva OOS)

| A vs B | DM stat | p-value | dif. media |
| :-- | --: | --: | --: |
| M1 vs M0 | -3.146 | 0.002 | -0.03110 |
| M2 vs M1 | -1.144 | 0.253 | -0.00631 |

> DM<0 => el primer modelo (A) tiene MENOR perdida (mejor). HAC Newey-West + correccion de muestra pequena Harvey-Leybourne-Newbold.

## Criterio de aceptacion: Diebold-Mariano titular V1 vs V0

- **DM(M2 titular V1 vs V0) = -1.144, p = 0.253**: el titular es MEJOR que V0, no significativo (p>=0.05).
  - V0 = K=2 Normal P-constante, re-corrido sobre la MISMA muestra y malla de bloques que V1 (perdidas emparejadas por fecha). dif. media de perdida = -0.00631.

## Pesaran-Timmermann (precision direccional del titular)

- PT stat = 1.663, p = 0.048, aciertos = 0.553 vs esperado bajo independencia 0.542. H1 unilateral: precision direccional mayor que bajo independencia

> Nota honesta: la senal direccional del retorno usa un proxy debil (dominancia del regimen calmo); la senal direccional fuerte es tarea de la capa de noticias (V2). Si el test degenera por signo constante, se reporta NaN, no se maquilla.

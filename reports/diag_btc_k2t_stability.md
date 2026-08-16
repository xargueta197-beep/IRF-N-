# Diagnostico de estabilidad K=2 t — BTC (@diagnostic_only)

generado: 2026-08-15T20:24:50.015784+00:00  |  seed: 42  |  muestra: 2987 obs (2018-05-13..2026-07-16)

**Esta corrida NO decide el titular, NO publica, NO toca `_load_titular_K_dist` ni `run_v3`.** Responde una sola pregunta: el optimo de K=2 t que gano el test de Hansen (1/20 en el kselect R6) es real o un artefacto del multistart?

## Los tres candidatos con 100 arranques (comparacion en igualdad)

| modelo | BIC (100 arr.) | log-lik | estabilidad (100 arr.) | ref. R6 (20 arr.) |
| :-- | --: | --: | :-- | :-- |
| K=1 t | 14559.11 | -7255.55 | 95/100 (95%) | BIC 14559.11, 18/20 |
| K=2 t | 14567.90 | -7227.94 | 1/100 (1%) | BIC 14569.94, 1/20 |
| K=2 normal | 14665.57 | -7284.77 | 44/100 (44%) | BIC 14665.57, 10/20 |

> BIC menor = mejor. La columna de estabilidad es la fraccion de arranques que alcanzo el optimo global (tolerancia de estimate.py): baja = superficie multimodal, el optimo no es de fiar.

## Hansen (bootstrap LR) K=2 vs K=1 (dist t) con el optimo estabilizado

- Ajuste observado de K=2 con 100 arranques (no 1/20): log-lik_alt=-7227.94, log-lik_null=-7255.55.
- **LR_obs = 55.23, p = 0.020** (49/49 replicas ok).
- Referencia R6 (optimo 1/20): LR_obs=53.18, p=0.02.

## Lectura (descriptiva; la decision es del director)

- Estabilidad de K=2 t: SIGUE INESTABLE (1/100).
- BIC(K=2 t) - BIC(K=1 t) = 8.79 (K=1 t sigue mejor por BIC).
- Hansen con el optimo estabilizado: p=0.020 (sigue significativo al 5%).

# Validacion V0 - IRF-N (SPY)

run_id: `f5e37a1b0d02`  |  generado: 2026-08-15 20:21 UTC

Especificacion: MS-GJR-GARCH K=2, matriz de transicion CONSTANTE, innovaciones Normales. Sin TVTP, sin noticias (V1+).

## Veredicto honesto

- El ξ PREDICHO out-of-sample, ¿le gana a la climatologia? **PARCIAL**.
  - Log-loss: modelo=0.1790 vs baseline=0.1765 (NO gana).
  - Brier:    modelo=0.0792 vs baseline=0.0818 (gana).
  - ECE (error de calibracion esperado): 0.0690.
  - Observaciones out-of-sample evaluadas: 2387.

> Baseline = predecir siempre las frecuencias marginales de regimen (climatologia). Es el minimo que un predictor condicional debe batir. El objetivo realizado es un proxy (argmax ξ_{t|t}); y_t = argmax ξ_{t|t} es un PROXY del estado latente, no la verdad. La calibracion mide consistencia predicho(t|t-1) vs. concluido(t|t).

## Auditoria point-in-time

- Invarianza de prefijo: **VERDE** (max_abs_diff = 0.00e+00, atol = 1e-10).
- Re-estimacion por bloque distinta (R2): **VERDE**.

## Bloques de walk-forward

Total: **19** bloques (minimo exigido: 6; R2, R8). Cada bloque re-estima desde cero sobre su ventana de entrenamiento y arrastra el ultimo ξ_{t|t} del entrenamiento al test (no reinicia).

| bloque | train desde | test | n_test | loglik/obs test | kappa (persistencia) |
| --: | :-- | :-- | --: | --: | :-- |
| 0 | 2013-01-03 | 2017-01-03..2017-07-03 | 125 | -0.6285 | 0.899, 0.540 |
| 1 | 2013-07-03 | 2017-07-03..2018-01-03 | 127 | -0.4817 | 0.914, 0.942 |
| 2 | 2014-01-03 | 2018-01-03..2018-07-03 | 125 | -1.3151 | 0.925, 0.324 |
| 3 | 2014-07-03 | 2018-07-03..2019-01-03 | 126 | -1.2591 | 0.925, 0.856 |
| 4 | 2015-01-03 | 2019-01-03..2019-07-03 | 125 | -1.0952 | 0.938, 0.524 |
| 5 | 2015-07-03 | 2019-07-03..2020-01-03 | 127 | -0.9833 | 0.941, 0.621 |
| 6 | 2016-01-03 | 2020-01-03..2020-07-03 | 126 | -2.1227 | 0.924, 0.303 |
| 7 | 2016-07-03 | 2020-07-03..2021-01-03 | 126 | -1.3446 | 0.932, 0.997 |
| 8 | 2017-01-03 | 2021-01-03..2021-07-03 | 126 | -1.2097 | 0.936, 0.999 |
| 9 | 2017-07-03 | 2021-07-03..2022-01-03 | 126 | -1.0769 | 0.932, 0.996 |
| 10 | 2018-01-03 | 2022-01-03..2022-07-03 | 125 | -1.9188 | 0.923, 0.995 |
| 11 | 2018-07-03 | 2022-07-03..2023-01-03 | 126 | -1.8738 | 0.941, 1.000 |
| 12 | 2019-01-03 | 2023-01-03..2023-07-03 | 124 | -1.3419 | 0.986, 0.986 |
| 13 | 2019-07-03 | 2023-07-03..2024-01-03 | 127 | -1.1011 | 0.952, 0.999 |
| 14 | 2020-01-03 | 2024-01-03..2024-07-03 | 125 | -0.9907 | 0.955, 1.000 |
| 15 | 2020-07-03 | 2024-07-03..2025-01-03 | 127 | -1.3454 | 0.973, 1.000 |
| 16 | 2021-01-03 | 2025-01-03..2025-07-03 | 123 | -1.6059 | 0.970, 0.907 |
| 17 | 2021-07-03 | 2025-07-03..2026-01-03 | 127 | -0.9684 | 0.964, 0.605 |
| 18 | 2022-01-03 | 2026-01-03..2026-07-03 | 124 | -1.2367 | 0.968, 0.886 |

- Loglik media por observacion: train=-1.2014, test=-1.2579. La brecha train-test mide sobreajuste.

## Limitaciones conocidas de V0

- K=2 con P constante y Normal: un regimen puede colapsar a un estado casi integrado (kappa->1) que absorbe outliers en vez de ser un regimen persistente de alta vol. No es un bug (recovery lo prueba sobre datos limpios); lo ataca V1 con t de Student, TVTP y seleccion de K.
- Sin covariables ni capa de noticias: la matriz de transicion no reacciona a condiciones de mercado. Es V0 por diseno.

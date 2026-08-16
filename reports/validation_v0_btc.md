# Validacion V0 - IRF-N (BTC)

run_id: `16d4190d17e2`  |  generado: 2026-07-16 20:02 UTC

Especificacion: MS-GJR-GARCH K=2, matriz de transicion CONSTANTE, innovaciones Normales. Sin TVTP, sin noticias (V1+).

## Veredicto honesto

- El ξ PREDICHO out-of-sample, ¿le gana a la climatologia? **NO**.
  - Log-loss: modelo=0.4501 vs baseline=0.4240 (NO gana).
  - Brier:    modelo=0.2762 vs baseline=0.2561 (NO gana).
  - ECE (error de calibracion esperado): 0.1174.
  - Observaciones out-of-sample evaluadas: 1645.

> Baseline = predecir siempre las frecuencias marginales de regimen (climatologia). Es el minimo que un predictor condicional debe batir. El objetivo realizado es un proxy (argmax ξ_{t|t}); y_t = argmax ξ_{t|t} es un PROXY del estado latente, no la verdad. La calibracion mide consistencia predicho(t|t-1) vs. concluido(t|t).

## Auditoria point-in-time

- Invarianza de prefijo: **VERDE** (max_abs_diff = 0.00e+00, atol = 1e-10).
- Re-estimacion por bloque distinta (R2): **VERDE**.

## Bloques de walk-forward

Total: **9** bloques (minimo exigido: 6; R2, R8). Cada bloque re-estima desde cero sobre su ventana de entrenamiento y arrastra el ultimo ξ_{t|t} del entrenamiento al test (no reinicia).

| bloque | train desde | test | n_test | loglik/obs test | kappa (persistencia) |
| --: | :-- | :-- | --: | --: | :-- |
| 0 | 2017-08-18 | 2021-08-18..2022-02-18 | 184 | -2.6412 | 0.958, 0.931 |
| 1 | 2018-02-18 | 2022-02-18..2022-08-18 | 181 | -2.6565 | 0.959, 0.906 |
| 2 | 2018-08-18 | 2022-08-18..2023-02-18 | 184 | -2.3042 | 0.951, 0.944 |
| 3 | 2019-02-18 | 2023-02-18..2023-08-18 | 181 | -2.1436 | 0.968, 0.953 |
| 4 | 2019-08-18 | 2023-08-18..2024-02-18 | 184 | -2.1452 | 0.957, 0.967 |
| 5 | 2020-02-18 | 2024-02-18..2024-08-18 | 182 | -2.4597 | 0.958, 0.999 |
| 6 | 2020-08-18 | 2024-08-18..2025-02-18 | 184 | -2.2958 | 0.961, 0.963 |
| 7 | 2021-02-18 | 2025-02-18..2025-08-18 | 181 | -2.1389 | 0.959, 0.969 |
| 8 | 2021-08-18 | 2025-08-18..2026-02-18 | 184 | -2.2453 | 0.977, 0.998 |

- Loglik media por observacion: train=-2.5400, test=-2.3367. La brecha train-test mide sobreajuste.

## Limitaciones conocidas de V0

- K=2 con P constante y Normal: un regimen puede colapsar a un estado casi integrado (kappa->1) que absorbe outliers en vez de ser un regimen persistente de alta vol. No es un bug (recovery lo prueba sobre datos limpios); lo ataca V1 con t de Student, TVTP y seleccion de K.
- Sin covariables ni capa de noticias: la matriz de transicion no reacciona a condiciones de mercado. Es V0 por diseno.

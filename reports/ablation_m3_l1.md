# Ablacion M0-M3 CON L1 -- test justo de la macro (M3)

generado: 2026-07-30T02:53:11.576194+00:00  |  K=2, dist=normal  |  n_starts=20, L1 grid=[0.0, 0.5, 2.0, 8.0, 32.0]

muestra: 3151 obs, 2013-12-27..2026-07-10, 17 bloques


## Log-loss OOS por peldano

| modelo | covs | n_oos | log-loss OOS/obs | lambdas L1 por bloque |
| :-- | :-- | --: | --: | :-- |
| M0 | - | 2135 | 1.3713 | - |
| M1 | - | 2135 | 1.3321 | - |
| M2 | ['sma_gap', 'bb_width_z'] | 2135 | 1.3369 | [2.0, 8.0, 32.0, 32.0, 32.0, 0.0, 8.0, 2.0, 8.0, 8.0, 8.0, 2.0, 8.0, 8.0, 8.0, 2.0, 32.0] |
| M3 | ['sma_gap', 'bb_width_z', 'slope_2s10y', 'hy_oas_z'] | 2135 | 1.3416 | [0.0, 32.0, 32.0, 32.0, 32.0, 8.0, 8.0, 8.0, 32.0, 8.0, 32.0, 8.0, 8.0, 32.0, 0.5, 8.0, 8.0] |

## Diebold-Mariano entre peldanos (DM<0 => A mejor)

| A vs B | DM stat | p-value | dif. media |
| :-- | --: | --: | --: |
| M1 vs M0 | -3.401 | 0.001 | -0.03927 |
| M2 vs M1 | 2.247 | 0.025 | 0.00488 |
| M3 vs M2 | 1.038 | 0.299 | 0.00468 |

## Veredicto M3 (macro) CON L1

**M3 no aporta** (DM=1.038, p=0.299): incluso con L1 la macro no mejora a M2.
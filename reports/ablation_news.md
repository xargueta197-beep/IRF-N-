# Ablacion de la capa de noticias: M3 vs M4 (V2) y M4 vs M5 (V3)

generado: 2026-08-19T23:50:33.700729+00:00  |  run_id: `9d30960c94cc`  |  K=2, dist=normal

> Compromiso pre-registrado (V3, guia 8.1): la pregunta titular es M5 vs M4 (el flujo de noticias aporta POR ENCIMA de la sorpresa calendarizada?). Si M5 no mejora la log-loss OOS de M4 de forma distinguible (DM p < 0.10), el indicador se publica sin lambda_N_z en el logit.

> Compromiso pre-registrado del diagnostico M2+H (esta sesion, ANTES de correrlo): M2+H = M2 + lambda_N_z responde una pregunta mas debil que M5 vs M4 (aporte sobre el TVTP tecnico, no sobre la sorpresa). Si M2+H no mejora la log-loss OOS de M2 de forma distinguible (DM p < 0.10), lambda_N_z se publica INACTIVA como covariable y el titular queda tecnico. Este diagnostico NO sustituye a M5 vs M4, que sigue bloqueada aguas arriba.

## Veredicto de la comparacion pre-registrada M4 vs M5

**M4 y M5 NO SE PUDIERON CORRER.** No es un resultado negativo de la capa de noticias (ningun DM se perdio): es la ausencia del peldano previo de la escalera. El bloqueante VINCULANTE aguas arriba es M4 (sorpresa); M3 (macro) NO bloquea -- su estado real se reporta abajo:

- M3 (macro): NO es el bloqueante: ALFRED_API_KEY esta configurada y la ablacion M3 con L1 ya se corrio (M3 vs M2 DM=1.038, p=0.299 -> la macro NO aporta; reports/ablation_m3_l1.md). El bloqueante real de M5 vs M4 es M4 (sorpresa): sin consenso historico. M5 depende de M4 por diseno de la escalera.
- M4 (sorpresa): solo 0 evento(s) valido(s) de consenso acumulados (se necesitan >= 30); sin SI_t no hay capa de sorpresa. Mismo bloqueante que V2 (reports/data_audit.md).
- M5 = M4 + lambda_N_z depende de M4 por el diseno acumulativo de la escalera (guia 8.1: una variable a la vez).

## Diagnostico pre-declarado M2+H (esta sesion)

**M2+H NO corrio.** Motivo: cobertura de lambda_N_z insuficiente para el walk-forward pre-registrado: muestra alineada de 1.7 anios y se requieren >= 7.0 (train 4a + 6 bloques de 6m). El backfill de GDELT sigue creciendo hacia atras; re-correr cuando alcance. NO se encoge la malla de bloques (R8).

El diagnostico queda pre-declarado con su compromiso (arriba) y se corre en cuanto la cobertura del backfill alcance.

Recordatorio: M2+H responde una pregunta MAS DEBIL que M5 vs M4 (aporte sobre el TVTP tecnico, no sobre la sorpresa calendarizada). No sustituye la comparacion pre-registrada; es el minimo honesto para decidir si lambda_N_z entra al titular mientras M4 siga bloqueada. Decision de diseno para revision del director (docstring de scripts/run_v3.py).

## Ablacion corrida esta sesion

Peldanos corridos: `['M0', 'M1', 'M2']` (M0..M5 declarados en validation/ablation.full_ladder_specs; M2+H declarado en scripts/run_v3.py).

| modelo | descripcion | covs | bloques | n_oos | log-loss OOS/obs |
| :-- | :-- | :-- | --: | --: | --: |
| M0 | GARCH un solo regimen (piso) | - | 19 | 2386 | 1.2940 |
| M1 | HMM K=2 P constante | - | 19 | 2386 | 1.2583 |
| M2 | +TVTP tecnico ['sma_gap', 'bb_width_z'] | ['sma_gap', 'bb_width_z'] | 19 | 2386 | 1.2672 |

## Diebold-Mariano entre peldanos consecutivos (perdida OOS)

| A vs B | DM stat | p-value | dif. media |
| :-- | --: | --: | --: |
| M1 vs M0 | -3.210 | 0.001 | -0.03579 |
| M2 vs M1 | 1.617 | 0.106 | 0.00892 |

> DM<0 => el primer modelo (A) tiene MENOR perdida (mejor). HAC Newey-West + correccion Harvey-Leybourne-Newbold.

## w_i estimados (V2, sin cambios: capa de sorpresa bloqueada)

| indicador | w_i | SE | t | distinguible de cero | n eventos |
| :-- | --: | --: | --: | :-- | --: |
| CPI | n/a | n/a | n/a | False | 0 |
| NFP | n/a | n/a | n/a | False | 0 |
| FOMC | n/a | n/a | n/a | False | 0 |
| PMI | n/a | n/a | n/a | False | 0 |
| RETAIL_SALES | n/a | n/a | n/a | False | 0 |
| UNEMPLOYMENT | n/a | n/a | n/a | False | 0 |

> Bloqueante de la capa de sorpresa: solo 0 evento(s) valido(s) de consenso acumulados (se necesitan >= 30); sin SI_t no hay capa de sorpresa. Mismo bloqueante que V2 (reports/data_audit.md).

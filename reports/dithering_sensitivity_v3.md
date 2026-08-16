# Chequeo de sensibilidad del dithering (Hawkes V3, SPY)

**Diagnostico** (`@diagnostic_only`) -- no cambia metodologia, no publica en `artifacts/`. Responde el aviso #5 de `reports/auditoria_pre_corrida.md`.

- Activo: **SPY**  |  corpus: **95085 eventos** puntuados, cobertura 2023-11-18 -> 2026-08-11 (240 dias, 758 fantasma).
- Multistart: `n_starts=30`, **semilla del multistart FIJA en 42** (se varia SOLO la semilla del dithering, para aislar el efecto del ruido inyectado del efecto de que arranques se probaron).
- Semilla base (fija las SE de referencia): **42**.

## Parametros por semilla de dithering

| dither seed | mu_N | alpha | beta | n | E[hijos] | KS stat | KS p | arranques |
| --: | --: | --: | --: | --: | --: | --: | --: | :-- |
| 42 | 103.0426 | 329.6631 | 283.9588 | 0.7388 | 3.83 | 0.0289 | 0.0000 | 30/30 |
| 1 | 102.3523 | 322.2459 | 276.9147 | 0.7406 | 3.85 | 0.0303 | 0.0000 | 30/30 |
| 7 | 102.6768 | 325.0267 | 279.6145 | 0.7398 | 3.84 | 0.0308 | 0.0000 | 30/30 |
| 123 | 102.6769 | 328.2332 | 282.3733 | 0.7398 | 3.84 | 0.0298 | 0.0000 | 30/30 |
| 2024 | 102.3261 | 324.5556 | 278.8743 | 0.7406 | 3.86 | 0.0293 | 0.0000 | 30/30 |

SE marginal del ajuste base (semilla 42): mu_N=1.1612, alpha=3.5543, beta=3.2037. SE del branching ratio n (delta) = 0.0039, IC95 [0.7311, 0.7465].

## Desviacion frente a la base, en SE MARGINAL de la base

El criterio literal del auditor (cada parametro `|param_seed - param_base| / SE_base < 1`) es un primer filtro, pero **subestima** en un kernel exponencial: `alpha` y `beta` estan co-identificados (cresta de la verosimilitud) y su SE marginal ignora esa correlacion. Ver la seccion siguiente.

| dither seed | mu_N (SE) | alpha (SE) | beta (SE) | d_alpha | d_beta | d_alpha/d_beta |
| --: | --: | --: | --: | --: | --: | --: |
| 1 | 0.594 | 2.087 | 2.199 | -7.417 | -7.044 | 1.053 |
| 7 | 0.315 | 1.304 | 1.356 | -4.636 | -4.344 | 1.067 |
| 123 | 0.315 | 0.402 | 0.495 | -1.430 | -1.586 | 0.902 |
| 2024 | 0.617 | 1.437 | 1.587 | -5.107 | -5.085 | 1.005 |

Desviacion marginal maxima (alpha/beta) = **2.199 SE**. Pero `d_alpha/d_beta ~ 1` en TODAS las semillas: alpha y beta se mueven **juntos** (la excitacion se re-parametriza sobre la cresta), no de forma independiente.

## Lo que se PUBLICA: branching ratio n y cascada

`n = alpha * E[s] / beta` es invariante al deslizamiento sobre la cresta, y es la cantidad que viaja al artefacto y a la pantalla 3 (no `alpha`/`beta` por separado).

- Rango de `n` entre las 5 semillas: **0.0018** (0.7388 a 0.7406).
- Eso es **0.46 SE** de `n` (SE = 0.0039).
- Todas las `n` caen **dentro del IC95 base** [0.7311, 0.7465]: **True**.
- `mu_N` (piso): desviacion maxima = **0.617 SE** (< 1).
- Cascada esperada E[hijos]: 3.83 a 3.86 (estable).

## Veredicto: PASA (con matiz)

**Las cantidades que se publican -- `n`, la cascada esperada y `mu_N` -- son robustas a la semilla del dithering, dentro de su propia incertidumbre.** El `n` se mueve menos de una fraccion de su SE y no sale de su IC95; `mu_N` se mueve < 1 SE. El unico movimiento > 1 SE marginal es el de `alpha` y `beta`, y es un **co-movimiento ~1:1 sobre la cresta de la verosimilitud** (estan co-identificados en el kernel exponencial): su SE marginal, calculada ignorando esa correlacion, exagera la aparente sensibilidad. El de-empate desliza el par (alpha, beta) por esa cresta sin tocar el cociente que define la excitacion.

**Conclusion honesta:** el MLE del Hawkes NO esta dominado por el ruido inyectado del dithering en lo que se reporta. La decision del director de usar dithering con semilla fija queda respaldada. Esto NO revierte los avisos #4/#6: `n` sigue siendo una **cota superior cualitativa** bajo un kernel exponencial que el KS rechaza (stat ~0.03, p=0) -- la mala especificacion del kernel, no el de-empate, es la limitacion vigente. Los dos chequeos mas estrictos que se sugirieron (variar tambien la semilla del multistart, y promediar `n` por imputacion multiple) se ejecutaron: ver las secciones (a) y (b) abajo.

## Extension (a): el optimo del MLE es global (barrido de la semilla del multistart)

Chequeo **ortogonal** al del dithering: se FIJA la realizacion del de-empate (semilla de dithering 42) y se VARIA la semilla del multistart (`n_starts=30`, R6). Si el optimo es unimodal, tandas distintas de arranques aleatorios deben caer en el MISMO optimo. Esto confirma que el `n` reportado es el maximo global, no un arranque afortunado.

| multistart seed | log-verosimilitud | n | arranques en el optimo |
| --: | --: | --: | :-- |
| 42 | 508265.859035 | 0.738832 | 30/30 |
| 1 | 508265.859035 | 0.738831 | 30/30 |
| 7 | 508265.859035 | 0.738832 | 30/30 |
| 123 | 508265.859035 | 0.738832 | 30/30 |
| 2024 | 508265.859035 | 0.738832 | 30/30 |

Rango de la log-verosimilitud entre semillas = **1.38e-07**; rango de `n` = **1.58e-06**. Optimo global confirmado (mismo optimo a precision numerica): **True**. La verosimilitud es unimodal para estos datos; el multistart de R6 no es el eslabon debil.

## Extension (b): agrupacion de `n` por imputacion multiple (regla de Rubin)

Cada semilla de dithering es **una imputacion** de los timestamps sub-bin no observados (supuesto de llegada uniforme intra-bin). En vez de fijar una semilla arbitraria, se agrupan las `m` imputaciones con la regla de Rubin, que separa la incertidumbre en dos fuentes: la **de cada ajuste** (hessiano) y la **entre imputaciones** (el dithering).

| cantidad | valor |
| :-- | --: |
| m (imputaciones) | 5 |
| `n` agrupado (Q&#772;) | 0.73991 |
| W&#772; (varianza DENTRO, media de SE_i^2) | 1.555e-05 |
| B (varianza ENTRE imputaciones) | 5.501e-07 |
| T = W&#772; + (1+1/m)&middot;B (varianza TOTAL) | 1.621e-05 |
| SE de una sola imputacion (&asymp; el publicado) | 0.00394 |
| **SE agrupado (Rubin)** | **0.00403** |
| r = incremento relativo de varianza por el dithering | 0.0424 |
| IC95 agrupado | [0.7320, 0.7478] |

El dithering **añade** aproximadamente **2.1%** al SE del branching ratio (r = 0.0424): el SE agrupado 0.00403 apenas supera al de una sola imputacion 0.00394. Es una fuente de incertidumbre **real pero de segundo orden**, dominada por la del ajuste y muy por debajo de la limitacion vigente (el KS rechaza el kernel exponencial).

**Decision del director (R3):** el `n` agrupado y el SE de Rubin son un **diagnostico**. Adoptarlos como el `n`/IC **publicado** (en vez del de la semilla fija) cambia COMO se reporta la incertidumbre del indicador y por tanto es decision del director, no de Claude. Con la evidencia actual el aporte es minusculo, asi que no hay urgencia; tendria mas sentido revisitarlo si el kernel power-law (Opcion C) resuelve el rechazo del KS y el dithering deja de estar tapado por esa limitacion.


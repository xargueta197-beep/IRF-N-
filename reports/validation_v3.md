# Validacion V3 (SPY) — Hawkes y branching ratio

generado: 2026-08-15T06:53:51.771035+00:00  |  run_id: `dbdc61daa50c`  |  **corrida --quick (provisional, no cumple R6)**

## Criterios de aceptacion de V3 (guia, checklist)

| criterio | estado |
| :-- | :-- |
| test_hawkes_recovery pasa | ver seccion Tests (suite de pytest) |
| KS de re-escalamiento reportado (pase o falle) | reportado: stat=0.0289, p=0.0000 — RECHAZA Exp(1); se documenta, kernel power-law candidato a futuro |
| n = alpha*E[s]/beta < 1 verificado | n = 0.7388 (estacionario) |
| chequeo de timestamps en pantalla 6 | VERDE (vacuo: 0 titulares emparejados; el calendario de consenso sigue vacio) |
| ablacion M4 vs M5 | bloqueada aguas arriba; diagnostico M2+H bloqueado por cobertura (reports/ablation_news.md) |

## MLE de Hawkes sobre el corpus real de titulares

Corpus: 95,085 titulares GDELT, 2023-11-18 a 2026-08-11 (240 dias; 758 faltantes; 10 censurados por el cap de 250/consulta del API — la censura sesga lambda_N A LA BAJA en los picos y esta contada).

| parametro | estimado | SE |
| :-- | --: | --: |
| mu_N (titulares/dia) | 103.0426 | 1.1612 |
| alpha | 329.6631 | 3.5543 |
| beta (1/dia) | 283.9588 | 3.2037 |

E[s] = 0.6364 (media de la relevancia FinBERT). **Branching ratio n = alpha*E[s]/beta = 0.7388** (IC95 metodo delta [0.7311, 0.7465]; correccion por marcas: NO alpha/beta = 1.1610). Cascada esperada E[hijos] = 1/(1-n) = 3.829.

### Correccion de la ventana de observacion (Parte A, 2026-08-15)

El Hawkes se ajusta sobre el TIEMPO OBSERVADO, no sobre el span calendario (decision del director que REVIERTE la del 2026-08-14). El compensador Lambda(T)=mu_N*T integraba mu_N sobre los dias fantasma (rango sin titulares capturados), sesgando mu_N a la baja e inflando n hacia la frontera. Se comprime el reloj excindiendo esos dias. Trazabilidad antes vs despues:

| ventana | T (dias) | mu_N | n = alpha*E[s]/beta |
| :-- | --: | --: | --: |
| ANTES (span calendario, sesgado) | 998.0 | 0.0783 | 0.9994 |
| DESPUES (tiempo observado) | 241.0 | 103.0426 | 0.7388 |

Multistart (R6): 8/8 arranques convergieron al mismo optimo (tolerancia 1e-4 en log-verosimilitud). Pocos arranques en el optimo indicarian un problema de identificacion; no es el caso.

### Sensibilidad al dithering de de-empate (aviso #5, 2026-08-15)

El seendate de GDELT esta cuantizado a 15 min (~83% de titulares empatados); sin dithering U(0,15min) el MLE continuo degenera. Como el ruido se inyecta en la mayoria de los datos, se verifico que el optimo lo fijan los datos y no la realizacion del dithering: se re-ajusto con **5 semillas de dithering** (42/1/7/123/2024), con la semilla del multistart FIJA en 42 y `n_starts=30` (R6), para aislar el efecto del de-empate.

| cantidad | rango entre semillas | en unidades de su SE | veredicto |
| :-- | :-- | :-- | :-- |
| n (branching ratio, PUBLICADO) | 0.7388-0.7406 (0.0018) | 0.46 SE de n; dentro del IC95 [0.731, 0.747] | robusto |
| mu_N (piso, PUBLICADO) | 102.33-103.04 | < 0.62 SE | robusto |
| E[hijos] (cascada, PUBLICADO) | 3.83-3.86 | -- | robusto |
| alpha, beta (marginales) | ~2 SE marginales | co-movidos ~1:1 (ratio 0.90-1.07) | cresta de la verosimilitud |

**Veredicto: PASA con matiz.** Lo que se publica (`n`, cascada, `mu_N`) es estable al dithering dentro de su propia incertidumbre. `alpha` y `beta` se mueven ~2 SE marginales, pero **juntos** (co-identificados en el kernel exponencial): el de-empate desliza el par sobre la cresta de la verosimilitud sin tocar `n = alpha*E[s]/beta`. El MLE no lee ruido inyectado como senal en el indicador reportado. No revierte el rechazo del KS (abajo): `n` sigue siendo cota superior cualitativa bajo un kernel mal especificado. Diagnostico reproducible: `scripts/dithering_sensitivity_v3.py` (`@diagnostic_only`) -> `reports/dithering_sensitivity_v3.md`.

Dos extensiones ejecutadas en el mismo diagnostico: (a) **optimo global** -- barriendo la semilla del multistart con el dithering fijo, las 5 semillas caen en el mismo optimo (log-verosimilitud rango 1.4e-07, `n` rango 1.6e-06), asi que el MLE es unimodal y R6 no es el eslabon debil; (b) **imputacion multiple (Rubin)** sobre las semillas de dithering -- `n` agrupado 0.7399, SE 0.00394 -> 0.00403 (+2.1%; r=0.042, el dithering añade ~4% a la varianza), IC95 [0.732, 0.748]. El aporte del dithering a la incertidumbre es real pero de segundo orden; adoptar el `n`/IC agrupado en la ruta publicada seria decision del director (R3).

### Bondad de ajuste (teorema de re-escalamiento temporal)

KS sobre interarribos re-escalados vs Exp(1): stat=0.0289, p=0.000000 (n=95084). **SE RECHAZA.** La respuesta NO es forzar el modelo (guia 6.6): se reporta tal cual y el kernel power-law queda como candidato para una version futura. Mientras tanto, lambda_N y n se leen como aproximaciones bajo kernel exponencial, con este caveat impreso tambien en pantalla 3.

## Auditoria de timestamps de titulares (Trampa 3)

titular vs evento: pase VACUO declarado (0 titulares emparejados con releases; el calendario de consenso sigue vacio — mismo bloqueante de V2). El chequeo se vuelve sustantivo en cuanto ambos feeds acumulen datos.

resolucion del feed: 95,085 titulares; 1.48% con timestamp de medianoche exacta (VERDE: el feed publica horas reales).

## Decisiones de diseno para revision del director

1. (mu_N, alpha, beta) del Hawkes estimados UNA VEZ sobre todo el corpus, no por bloque de walk-forward — analogo a delta en V2 (parametros del proceso puntual, estimados sin mirar retornos; los beta_ij del logit si se re-estiman por bloque, R2). Detalle en el docstring de scripts/run_v3.py.
2. Diagnostico M2+H pre-declarado como decision intermedia mientras M5 vs M4 este bloqueada aguas arriba; compromiso pre-registrado en reports/ablation_news.md.
3. s_i = 1 - P(neutral) de FinBERT como relevancia (Trampa 4: jamas se pregunta la direccion; restriccion documentada en features/relevance.py, con la limitacion medida de que el score solo es significativo dentro del dominio financiero que filtra la consulta GDELT).

## Pendientes que esta version NO resuelve

- Backfill de GDELT hasta headlines.start_date (2017): en curso, reanudable (scripts/capture_headlines.py); la cobertura actual viaja en el artefacto.
- M5 vs M4: requiere ALFRED_API_KEY (gratis, minutos) y consenso historico (Trading Economics de pago o acumulacion hacia adelante).
- Invarianza de prefijo del titular: VERDE en esta corrida (pantalla 6).
- Re-correr sin --quick para cumplir R6 (multistart completo) antes de dar por cerrada la version.

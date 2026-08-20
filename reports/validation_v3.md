# Validacion V3 (SPY) — Hawkes y branching ratio

generado: 2026-08-20T03:43:11.906597+00:00  |  run_id: `75f650b1d59d`

## Criterios de aceptacion de V3 (guia, checklist)

| criterio | estado |
| :-- | :-- |
| test_hawkes_recovery pasa | ver seccion Tests (suite de pytest) |
| KS de re-escalamiento reportado (pase o falle) | reportado: stat=0.0306, p=0.0000 — RECHAZA Exp(1); se documenta, kernel power-law candidato a futuro |
| n = alpha*E[s]/beta < 1 verificado | n = 0.7385 (estacionario) |
| chequeo de timestamps en pantalla 6 | VERDE (vacuo: 0 titulares emparejados; el calendario de consenso sigue vacio) |
| ablacion M4 vs M5 | bloqueada aguas arriba; diagnostico M2+H bloqueado por cobertura (reports/ablation_news.md) |

## MLE de Hawkes sobre el corpus real de titulares

Corpus: 95,234 titulares GDELT, 2023-11-18 a 2026-08-16 (241 dias; 762 faltantes; 10 censurados por el cap de 250/consulta del API — la censura sesga lambda_N A LA BAJA en los picos y esta contada).

| parametro | estimado | SE |
| :-- | --: | --: |
| mu_N (titulares/dia) | 102.9185 | 1.1576 |
| alpha | 329.4885 | 3.5525 |
| beta (1/dia) | 283.8530 | 3.2035 |

E[s] = 0.6362 (media de la relevancia FinBERT). **Branching ratio n = alpha*E[s]/beta = 0.7385** (IC95 metodo delta [0.7308, 0.7462]; correccion por marcas: NO alpha/beta = 1.1608). Cascada esperada E[hijos] = 1/(1-n) = 3.824.

### Correccion de la ventana de observacion (Parte A, 2026-08-15)

El Hawkes se ajusta sobre el TIEMPO OBSERVADO, no sobre el span calendario (decision del director que REVIERTE la del 2026-08-14). El compensador Lambda(T)=mu_N*T integraba mu_N sobre los dias fantasma (rango sin titulares capturados), sesgando mu_N a la baja e inflando n hacia la frontera. Se comprime el reloj excindiendo esos dias. Trazabilidad antes vs despues:

| ventana | T (dias) | mu_N | n = alpha*E[s]/beta |
| :-- | --: | --: | --: |
| ANTES (span calendario, sesgado) | 1003.0 | 0.0807 | 0.9992 |
| DESPUES (tiempo observado) | 242.0 | 102.9185 | 0.7385 |

Multistart (R6): 28/30 arranques convergieron al mismo optimo (tolerancia 1e-4 en log-verosimilitud). Pocos arranques en el optimo indicarian un problema de identificacion; no es el caso.

### Bondad de ajuste (teorema de re-escalamiento temporal)

KS sobre interarribos re-escalados vs Exp(1): stat=0.0306, p=0.000000 (n=95233). **SE RECHAZA.** La respuesta NO es forzar el modelo (guia 6.6): se reporta tal cual y el kernel power-law queda como candidato para una version futura. Mientras tanto, lambda_N y n se leen como aproximaciones bajo kernel exponencial, con este caveat impreso tambien en pantalla 3.

## Auditoria de timestamps de titulares (Trampa 3)

titular vs evento: pase VACUO declarado (0 titulares emparejados con releases; el calendario de consenso sigue vacio — mismo bloqueante de V2). El chequeo se vuelve sustantivo en cuanto ambos feeds acumulen datos.

resolucion del feed: 95,234 titulares; 1.48% con timestamp de medianoche exacta (VERDE: el feed publica horas reales).

## Decisiones de diseno para revision del director

1. (mu_N, alpha, beta) del Hawkes estimados UNA VEZ sobre todo el corpus, no por bloque de walk-forward — analogo a delta en V2 (parametros del proceso puntual, estimados sin mirar retornos; los beta_ij del logit si se re-estiman por bloque, R2). Detalle en el docstring de scripts/run_v3.py.
2. Diagnostico M2+H pre-declarado como decision intermedia mientras M5 vs M4 este bloqueada aguas arriba; compromiso pre-registrado en reports/ablation_news.md.
3. s_i = 1 - P(neutral) de FinBERT como relevancia (Trampa 4: jamas se pregunta la direccion; restriccion documentada en features/relevance.py, con la limitacion medida de que el score solo es significativo dentro del dominio financiero que filtra la consulta GDELT).

## Pendientes que esta version NO resuelve

- Backfill de GDELT hasta headlines.start_date (2017): en curso, reanudable (scripts/capture_headlines.py); la cobertura actual viaja en el artefacto.
- M5 vs M4: requiere ALFRED_API_KEY (gratis, minutos) y consenso historico (Trading Economics de pago o acumulacion hacia adelante).
- Invarianza de prefijo del titular: VERDE en esta corrida (pantalla 6).

# Condiciones de reapertura de capas bloqueadas por datos

Las capas **M4/V2** (sorpresa/consenso) y **M2+H** (Hawkes como covariable en el
walk-forward) estan cerradas por **restriccion de datos**, no por bug ni por
decision de diseno (ver `README.md` "Estado de modulos" y los avisos del
artefacto). Este documento fija la condicion EXACTA y OBSERVABLE bajo la cual
cada una se reabre, para que la reapertura sea automatica y verificable, nunca
"cuando alguien se acuerde" (plan `reports/plan_mejoras_avisos_2026-08-18.md`,
Franja E).

La condicion se chequea con:

    python scripts/check_reopen_status.py

que lee SOLO la cobertura ya publicada en `artifacts/latest/irfn.json` y la
compara contra umbrales derivados de `config/` (no numeros magicos). No calcula
nada del modelo ni re-corre nada.

## M4/V2 — capa de sorpresa (consenso historico)

- **Umbral:** `config` `v2.delta_mle.min_events_total` (hoy = 30) releases con
  consenso acumulados en total. Es el mismo minimo que exige el fit conjunto de
  `delta` antes de intentarse.
- **Estado hoy:** 0/30.
- **Fuente de datos:** `scripts/capture_consensus.py` acumula hacia adelante
  desde la Sesion 0 (sin exitos aun: el demo `guest:guest` da 410). Alternativa:
  pagar Trading Economics point-in-time (decision del director).
- **Al reabrir:** re-correr `scripts/run_v2.py` (ablacion M4 real).

## M2+H — lambda_N_z como covariable en el walk-forward pre-registrado

- **Umbral:** el corpus GDELT debe cubrir el span del walk-forward
  pre-registrado = `walkforward.train_years` (4a) + `walkforward.n_blocks` (6) x
  `walkforward.test_months` (6m) = **7 anios ~= 2557 dias**. Con menos, la malla
  de 6 bloques NO cabe.
- **Estado hoy:** 240/2557 dias (~9.4 %).
- **PROHIBIDO (R8):** encoger la malla de bloques para "alcanzar" la cobertura.
  Eso convierte una limitacion honesta en una mentira (aviso #3 del plan).
- **Fuente de datos:** `scripts/capture_headlines.py` (backfill hacia atras,
  lento por el rate limit de GDELT sobre IP compartida). Ver `RESUME_GDELT.md`.
- **Al reabrir:** re-correr la ablacion M2+H (M5 vs M4).

## Nota

Ninguna de estas reaperturas se arregla con codigo: son blockers del mundo. El
corpus de 240 dias BASTA para AJUSTAR/inspeccionar el Hawkes (>= 200 eventos) y
por eso la capa Hawkes se publica ACTIVA standalone; lo que NO alcanza es la
ablacion M5-vs-M4 en walk-forward (esa necesita los ~7 anios de arriba).

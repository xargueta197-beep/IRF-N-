# Estado del proyecto IRF-N — cierre de sesion 2026-08-19

Documento de proceso y estado. Punto unico para saber que hay hecho, que esta
vivo y que falta, sin leer todo el historial.

---

## 1. Estado en produccion (verificado en vivo)

| Componente | Estado |
| :-- | :-- |
| **SPY (linea principal)** | Publicado: M1 (K=2, matriz constante, `covariates=[]`, `tvtp=false`), `run_id=75f650b1d59d`, asof **2026-08-19**. Con las DOS bandas de `n` + supresion F2.c del regimen degenerado. |
| **BTC** | Publicado: K=1 Student-t, `run_id=d7dca6e40eeb`, asof **2026-08-21** (refrescado 2026-08-21, antes `ece4ad3df66f`/2026-08-16). Display "sin estructura de regimenes" (K=1 correcto por BIC). |
| **Panel publico** | VIVO: https://xargueta197-beep.github.io/IRF-N-/ (GitHub Pages via Actions; repo publico). |
| **Git** | Todo en `origin/master`, arbol limpio, 0 sin pushear. |
| **Tests** | 157 passed (`-m "not slow"`). |

## 2. Conclusion cientifica (lo que quedo PROBADO, R8)

- **Los regimenes de volatilidad importan:** M1 > M0 (DM p=0.001).
- **Ninguna covariable REZAGADA mejora la prediccion OOS**, con test formal en cada eje:
  - Tecnicas (M2): PEOR que M1 (DM p=0.025).
  - Macro (M3): no aporta (DM p=0.299).
  - Noticias/sorpresa (M4/M5): CERRADAS por datos (sin consenso gratis, GDELT 240/2557d) +
    caso bias-variance/potencia. `reports/cierre_m4_m5_2026-08-19.md`.
  - Cripto-nativa BTC (GARCH-X): volumen NEGATIVO (empeora OOS), sentimiento/FNG NEGATIVO
    (sobreajuste), **funding PROMETEDOR pero no cruza** (in-sample t=3.41, OOS DM+1.77 p=0.078,
    subapoderado 5 bloques). `reports/garchx_btc_resultados_2026-08-19.md`.
- **La capa Hawkes** se publica como indicador de fragilidad standalone (n, cascada, KS, dos
  bandas de sensibilidad), NO como covariable del logit (M5 cerrado).

## 3. Lo que se hizo en esta sesion (2026-08-19)

- **Franja 1** (avisos): banda OOS en Historico, paridad panel (`value=null`->"no anualizable"),
  reapertura observable (`scripts/check_reopen_status.py` + `docs/reopen_conditions.md`).
- **Franja 2** (decisiones del director): kernel = mantener exponencial (power-law falla criterio
  pre-registrado); regimen degenerado = documentar; `n` = **dos bandas separadas** (IC del MLE vs
  banda de ventana), implementadas y publicadas. Auditoria metodologica: la banda (b) span-calendario
  es estimador SESGADO, etiquetado como tal. `reports/nota_decisiones_director_franja2_2026-08-19.md`.
- **Franja 3**: M4/M5 cerrados definitivamente + panel desplegado publico.
- **BTC**: confirmado K=1 correcto por BIC (colas gordas, no regimenes); display arreglado.
- **Capa cripto-nativa GARCH-X** (linea nueva): `models/garchx.py` + `scripts/run_garchx_btc.py`
  + `prices.load_volume` + capturadores FNG/funding; 3 drivers probados, pre-registrados.

## 4. Lo que FALTA (nada bloqueante, por impacto)

> **Actualizacion 2026-08-20/21:** #1 (F6) y #2 (BTC) RESUELTOS. Solo queda #3 (funding, opcional).

1. ~~**F6 — `validation.json` stale**~~ **RESUELTO (2026-08-20).** `validation_v4.md` reescrito
   sobre `75f650b1d59d`; Tests 2/3/5/6 recomputados sobre el run vivo, Tests 1/4/7 (seleccion de K
   y ablacion) etiquetados como estructurales; panel re-exportado con `stale=false` y chequeo espejo
   OK. Nota: con datos frescos el Test 3 (PT) paso de p=0.086 a p=0.032 (cruza el 5% unilateral pero
   FRAGIL, reportado tal cual, no señal robusta). Test 5 re-corrido: NO SUPERA.
2. ~~**BTC 3 dias por detras de SPY**~~ **RESUELTO (2026-08-21).** BTC refrescado (cache a .stale +
   re-run + promote): publicado `d7dca6e40eeb`, asof 2026-08-21. Modelo sin cambios de fondo.
3. **Revisit de funding** *(investigacion opcional; PENDIENTE por decision del director 2026-08-20).*
   El unico lead prometedor. Requiere
   PRE-REGISTRAR un test mas potente (mas historia de perps, o datos de 8h, o 3a train
   pre-especificado) ANTES de correrlo. NO re-correr con train mas corto post-hoc = p-hacking.

## 5. Donde vive todo

- Modelo/artefactos: `artifacts/latest/` (SPY), `artifacts/btc/latest/` (BTC). Runbook de
  re-publicacion: `artifacts/README.md`.
- Panel: `panel/` (Next static export). Deploy: `.github/workflows/deploy-panel.yml`.
- Decisiones y resultados: `reports/` (cierre_m4_m5, nota_decisiones_franja2, diseno_capa_cripto_btc,
  garchx_btc_resultados, plan_mejoras_avisos, este documento).
- Reapertura de lineas cerradas: `scripts/check_reopen_status.py`, `docs/reopen_conditions.md`.

# Plan de desarrollo IRF-N — mejora del índice, salud del proyecto y cierre de huecos

**Fecha:** 2026-08-16
**Origen:** reconciliación de la auditoría profunda de hoy con `auditoria_mejoras.md` (producto)
y `auditoria_sistema_2026-08-16.md` (matemática).
**Principio rector:** este plan NO reabre nada cerrado (§0) y separa por gobernanza lo que
Claude Code puede ejecutar solo (código, sin metodología) de lo que exige OK del director
(palancas de modelo) — CLAUDE.md §3 + R8. Antes de tocar cualquier palanca de modelo se
consultan los dos documentos de referencia absoluta (CLAUDE.md raíz).

---

## ESTADO DE EJECUCIÓN (2026-08-16)

- **Fase 0 (reconciliación doc):** HECHA (CLAUDE.md §ESTADO actualizado: BTC R6+atómico,
  power-law comparado, hawkes_powerlaw en producción, 149 tests).
- **Fase 1 (sprint de honestidad A2+A3+A5):** CÓDIGO HECHO Y VERIFICADO — 149 passed, 0 failed.
  - A2 maxdd (F3): IC apagado en `bootstrap.py`. HECHO.
  - A2 régimen degenerado (F4): IC apagado en `publish.py` + config + 4 orquestadores. HECHO.
  - A3 (D del KS): warning enriquecido en `run_v3.py`. HECHO.
  - A5 (F6): guardarraíl + estampado `stale` en `export_panel_data.py` + tipo del frontend +
    panel re-exportado. HECHO. (Ninguna página renderiza hoy `validation.json`.)
  - **PENDIENTE de decisión del usuario:** re-publicar el artefacto para que A2/A3 lleguen al
    `latest/` vivo (commit por R5 → `run_v3 --no-capture` ~37 min R6 → `promote.py` atómico →
    re-exportar). El código y los tests ya están verdes; falta solo la corrida.
- **Fase 2 (decisiones del director A1/A4/A6):** NO iniciada (metodología; requiere OK).
- **Fase 3 (limbos de datos):** NO iniciada.

---

## 0. Congelado — NO reabrir (guardarraíl anti-re-trabajo)

Estos ejes están cerrados con evidencia. Reabrir sin motivo nuevo es desperdicio:

- Covariables de transición (M2 vs M1, M3 vs M2 p=0.299) — cerrado en negativo, con L1.
- Eje económico / regla de trading — walk-forward económico NO supera buy-and-hold (R8).
- Test 3 direccional — sin señal al 5% (índice de volatilidad, no de dirección).
- V2 (sorpresa) — congelada; sin fuente gratuita point-in-time (4 fuentes descartadas).
- Sensibilidad del dithering (aviso #5) — resuelto, `n` robusto dentro de su SE.
- BTC R6 + tubería atómica — hecho hoy (multistart 30, publicación atómica).
- Comparación de kernels Hawkes — power-law gana AIC (ΔAIC +180) pero ninguno pasa KS
  (confirma que el KS está sobre-apoderado a n=95k, F1). Falta SOLO decidir con el resultado.

---

## Ruta en 4 fases (secuenciada por riesgo: menor riesgo / mayor honestidad primero)

```
Fase 0  Reconciliación documental        [Claude, 15 min, riesgo cero]
Fase 1  Sprint de honestidad (A2+A5+A3)  [Claude, código puro, sin metodología]
Fase 2  Nota de decisiones al director   [Claude prepara; director decide A1/A4/A6]
Fase 3  Cierre de limbos de datos        [director decide sí/no; Claude ejecuta]
```

La Fase 1 no depende de la Fase 2. La Fase 2 no se implementa hasta tener OK explícito.
La Fase 3 son 4 decisiones sí/no independientes que pueden resolverse en cualquier orden.

---

## FASE 0 — Reconciliación documental (riesgo cero, ~15 min)

**Motivo:** los docs van por detrás de lo hecho hoy → la próxima sesión re-hace trabajo cerrado.

| # | Acción | Archivo |
|---|--------|---------|
| 0.1 | Marcar BTC como R6 + atómico (recs 10-11 → hechas) | `reports/auditoria_mejoras.md` |
| 0.2 | Marcar power-law como comparado; queda solo decidir (rec 3) | `reports/auditoria_mejoras.md` |
| 0.3 | Registrar `hawkes_powerlaw.py` + su test en producción | CLAUDE.md §ESTADO |
| 0.4 | Corregir conteo de tests: 140 → **141 passed** | CLAUDE.md §ESTADO |

**Criterio de aceptación:** doc ↔ realidad sin discrepancias (las 4 del §4 de la auditoría).

---

## FASE 1 — Sprint de honestidad (código puro, SIN metodología)

Cierra los 3 huecos que hoy **publican precisión falsa**. No cambia ningún número del modelo:
solo deja de afirmar intervalos que no son válidos y sincroniza el panel público. Es lo que
más mejora el índice sin arriesgar nada.

### A2 · Apagar dos intervalos de confianza inválidos (emitir NaN/None)

- **maxdd (F3):** el bootstrap estacionario NO es válido sobre un funcional de valor extremo
  (el máximo drawdown). Verificado en artefacto: `value=-0.314` pegado a `ci_low=-0.333`,
  IC asimétrico. → devolver `(punto, None, None)` para `maxdd`, igual que ya se hace con
  pocas observaciones.
- **Régimen degenerado (F4):** Sharpe −5.56 con IC [−8.53, −2.65] sobre ~100 días sueltos =
  precisión espuria. El banner ya advierte, pero el JSON publica el IC igual. → suprimir el IC
  (None) cuando el régimen es el absorbe-outliers / n_obs por debajo de un umbral de confianza.
- **Dónde:** `src/irfn/validation/bootstrap.py` (`stationary_bootstrap_ci` /
  `bootstrap_regime_stats`) + consumo en `src/irfn/outputs/publish.py` (`conditional_stats`).
  El esquema `MetricWithCI` ya admite `ci_low/ci_high = null` (V2) → sin romper contrato.
- **Test:** ampliar `test_bootstrap` / `test_app_components` para exigir `None` en maxdd y en
  el régimen degenerado; la app ya sabe renderizar celdas sin IC.

### A3 · Reportar el tamaño de efecto D del KS (no solo p=0)

- El KS del re-escalamiento reporta `p=0` a n=95k; el número honesto es **D=0.029** (efecto
  pequeño). Cambio de **redacción/campo**, no de cálculo.
- **Dónde:** `src/irfn/models/hawkes_mle.py` (ya calcula el stat KS) → asegurar que `D` viaja
  al artefacto; `src/irfn/outputs/schema.py`+`publish.py` (sección `news`, params Hawkes);
  aviso en `app/pages/3_Noticias.py`.

### A5 · Coherencia del panel público (F6) — SIGUE ROTO

- Verificado: `panel/public/data/irfn.json` es el run vigente (asof 2026-08-14) pero
  `panel/public/data/validation.json` sigue en `generated_at: 2026-07-14`. La página pública
  "Validación" describe un modelo más viejo que la de "hoy".
- **Acción:** (a) re-ejecutar `scripts/export_panel_data.py` para regenerar los 3 JSON del run
  vigente; (b) añadir a `export_panel_data.py` un **chequeo espejo de `contract.py`** que
  FALLE (exit≠0) si `irfn.json`, `validation.json` e `history.json` no comparten `run_id`.
  Así el panel no puede volver a divergir en silencio.

**Verificación de fin de Fase 1 (obligatoria):**
1. `pytest -m "not slow"` → 141+ passed (los nuevos tests incluidos), 0 failed.
2. Re-publicar/re-exportar y correr `python -m irfn.outputs.contract artifacts/latest --repo-root .`
   → PROMOVIBLE, y el nuevo chequeo espejo del panel → verde (3 JSON mismo run_id).
3. Levantar app privada (8501) y panel (3000) y confirmar en navegador: maxdd y régimen
   degenerado sin IC, KS con D, "Validación" pública coherente con "hoy".

**Gobernanza:** nada de esto toca metodología → Claude lo ejecuta sin gate. (Es la respuesta
directa al encargo: "apagar 3 IC que fingen precisión y re-exportar el panel".)

---

## FASE 2 — Nota de decisiones al director (metodología; NO implementar sin OK)

Estas son las **palancas matemáticas reales** para mejorar la estructura del modelo. Ninguna
se toca sin OK del director (R8: criterio pre-registrado ANTES de correr) y consulta previa de
los dos Google Docs. Claude prepara UNA nota con: problema, corrección cerrada, costo de
cómputo, impacto en artefacto/contrato y criterio de aceptación pre-registrado. Decide Xavier.

### A1 · Hawkes sobre soporte observado (F2) — el único número publicado hoy sesgado

- **Problema:** `n=0.739` está inflado porque el compensador integra `μ_N` sobre **758 días
  fantasma** (fuera de las ventanas realmente observadas del corpus de 240 días).
- **Corrección cerrada:** no integrar `μ_N` fuera de las ventanas observadas (ajustar
  `Λ(T)` al tiempo observado). `n = α·E[s]/β` no cambia de fórmula; cambia el tiempo base del
  compensador y, por tanto, el MLE.
- **Costo:** re-correr walk-forward (R2) + multistart (R6) para SPY (y BTC).
- **Impacto:** cambia `branching_ratio`, `expected_cascade`, `mu_N` publicados. Debe pasar el
  contrato atómico. Es el cambio de mayor impacto en honestidad de la capa Hawkes.
- **Contexto que ya tenemos:** con el power-law ya comparado, esta conversación es informada
  (kernel + soporte se deciden juntos).

### A4 · Publicar M1 en vez de M2 (F5) — argumento bias-variance

- **Problema:** ninguna covariable de transición aporta OOS; M2 tiene el mismo sesgo que M1 con
  más varianza. Formalmente conviene publicar M1.
- **Impacto en contrato:** hoy `contract.py` R6 exige `V1+ ⇒ tvtp=true` (con K≥2). Publicar M1
  relaja esa regla → cambio de contrato, no solo de run. Requiere redacción cuidadosa para no
  reabrir el guardarraíl anti-regresión.

### A6 · Régimen degenerado (E[D]≈1 día) — es el óptimo global, no un bug

- **Diagnóstico previo (cerrado):** absorbe-outliers estructural, óptimo global (17/50
  arranques), ni Student-t ni K=3 lo corrigen (`reports/diag_degenerate_regime.md`).
- **Palanca propuesta:** probar piso de persistencia / componente de salto (jump) / mezcla de
  colas, **con criterio pre-registrado**: E[D] > 5 días **sin perder log-loss**. Si no cumple,
  se mantiene el actual con caveat (ya lo hace).
- **Dónde (si se aprueba):** parametrización en `src/irfn/models/params.py` / `hamilton.py` /
  `estimate.py`.

**Entregable de la Fase 2:** `reports/nota_decisiones_director_2026-08-16.md` con las 3
palancas en una sola nota, cada una con su criterio de aceptación pre-registrado. **Sin código
en `src/` hasta el OK.**

---

## FASE 3 — Cierre de limbos de datos (decisión sí/no explícita cada uno)

Hoy son "deuda silenciosa". El objetivo es que cada uno quede con un sí/no explícito, no en
limbo. Claude ejecuta la rama elegida.

| Limbo | Estado | Decisión requerida |
|-------|--------|--------------------|
| **GDELT M5** | 240/998 días; único hueco `2026-01-13` | Backfill a ~7 años desde IP no compartida **o** cerrar M5 formalmente como *bloqueado-por-datos* |
| **V2 consenso** | Sin fuente gratuita point-in-time | Pagar Trading Economics **o** declararla cerrada (deja de ser deuda) |
| **Panel público** | Construido y verificado, no desplegado | Desplegar (credenciales Vercel) **o** dejar local documentado |
| **Caché de precios** | `run_v3` usó caché rancia (5 semanas), corregida a mano | Unificar el loader con **política de frescura** (higiene de ingeniería, bajo riesgo) |

Nota: la caché de precios es la única de las 4 que es pura ingeniería (sin decisión de
producto) → puede ejecutarse junto con la Fase 1 si se prioriza.

---

## Resumen de una línea

El sistema técnico está verde y blindado. Este plan (1) sincroniza los docs, (2) **apaga 3
intervalos de confianza que fingen precisión y re-exporta el panel** —puro código, ejecutable
ya—, (3) lleva al director las 3 palancas de modelo (Hawkes-soporte, M1-vs-M2, régimen
degenerado) en una sola nota pre-registrada, y (4) cierra con un sí/no los 3 huecos de datos y
la caché de precios. Nada reabre lo ya cerrado.

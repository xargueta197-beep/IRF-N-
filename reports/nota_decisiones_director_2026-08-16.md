# Nota de decisiones para el director — palancas de modelo (Fase 2)

**Fecha:** 2026-08-16
**Para:** Xavier (director de proyecto)
**De:** Claude Code (implementación)
**Regla:** ninguna de estas palancas se toca sin tu OK explícito (CLAUDE.md §3, R8) y sin
consultar antes los dos documentos de referencia absoluta (Reglas Absolutas + Guía de
Implementación). Esta nota **propone y cuantifica**; no implementa nada.

> Contexto: la Fase 1 (sprint de honestidad, A2+A3+A5) ya está hecha, verificada y publicada
> (run `58077981ed78`). Esta nota cubre solo las **3 palancas matemáticas** que sí requieren tu
> decisión. Al redactarla se corrigió además un emisor stale (ver §A1): el número Hawkes NO
> está inflado como decía el aviso viejo.

---

## A1 · Kernel del Hawkes (el KS rechaza el exponencial)

**ACLARACIÓN IMPORTANTE — el audit estaba desactualizado en un punto.** El audit listaba A1
como "n=0.739 inflado porque el compensador integra μ_N sobre 758 días fantasma". **Eso ya no
es cierto:** el código ajusta el Hawkes sobre el **tiempo observado** (`times_obs`/`T_obs`,
días fantasma excindidos) desde la Parte A — decisión tuya del **2026-08-15** que revierte la
del 2026-08-14. El `n=0.7388` publicado ES la estimación sobre soporte observado (μ_N=103, no
sesgada a la baja). El span-fit (n=0.9994) es **solo diagnóstico, no se publica**. El aviso del
artefacto que aún decía "span completo / n inflado / cota superior" era un **emisor stale** y se
corrigió hoy como parte de la honestidad (no es metodología). **Conclusión: A1-como-sesgo está
CERRADO.**

**La decisión real que queda (kernel):** el KS de re-escalamiento **rechaza** el kernel
exponencial (D=0.0289, p≈2e-69, n≈95k). El tamaño de efecto D es **pequeño** (el rechazo lo
domina el n enorme), pero el kernel exponencial tiene un solo timescale y no captura el
clustering multiescala de titulares. Ya se comparó el power-law (`_compare_kernels.log`,
`hawkes_powerlaw.py`): **gana en AIC (ΔAIC ≈ +180) pero tampoco pasa el KS.**

- **Opción 1 (recomendada): mantener el exponencial con el caveat honesto del KS.** Es lo que
  hay hoy: n como estimación exponencial sobre la ventana observada, con el aviso KS (D pequeño,
  rechazo n-driven). Cero cómputo, cero riesgo. El power-law no resuelve el KS, así que cambiar
  de kernel no compra honestidad, solo AIC.
- **Opción 2: publicar el power-law como indicador principal.** Mejor AIC, cola pesada. Costo:
  O(n²) con numba (~10⁹ ops/eval), re-correr walk-forward (R2) + multistart (R6); cambia
  `branching_ratio`/`expected_cascade`/`attribution` publicados; el KS **sigue** rechazando.
- **Criterio de aceptación pre-registrado (si eliges Opción 2):** adoptar power-law solo si
  (a) ΔAIC > 0 (ya cumplido, +180) **y** (b) su D del KS < D del exponencial en ≥20% **y**
  (c) n_powerlaw < 1 (estacionario). Declararlo ANTES de re-correr.

**Recomendación:** Opción 1. El KS sobre-apoderado a n=95k (F1) rechaza cualquier kernel; el
valor honesto ya está reportado. Reabrir el kernel solo si quieres el power-law por AIC, no por
el KS.

---

## A4 · Publicar M1 en vez de M2 (argumento bias-variance, F5)

**Estado.** Ninguna covariable de transición aporta OOS (cerrado en negativo, con L1: M2 vs M1
no distinguible, M3 vs M2 p=0.299). Hoy se publica **M2** (TVTP técnico). Formalmente, M2 tiene
el mismo sesgo que M1 con **más varianza** → M1 domina en bias-variance.

- **Corrección propuesta:** publicar M1 (K=2, matriz constante o TVTP sin covariables técnicas
  informativas) como indicador principal.
- **Impacto en contrato:** `contract.py` R6 exige hoy `V1+ ⇒ tvtp=true` (con K≥2). Publicar M1
  relajaría esa regla → **cambio de contrato**, no solo de run. Requiere redacción cuidada para
  no reabrir el guardarraíl anti-regresión (que existe justo para impedir downgrades silenciosos).
- **Costo:** bajo (M1 ya se estima en la escalera; re-publicar con M1 como principal). Re-correr
  R6 del titular M1.
- **Criterio de aceptación pre-registrado:** publicar M1 si su log-loss OOS no es peor que la de
  M2 más allá de 1 SE (M1 ya ≤ M2 en la ablación) **y** el contrato se ajusta explícitamente
  (V1+ ⇒ tvtp puede ser false cuando ninguna covariable aporta).
- **Riesgo/tensión:** es la decisión más delicada de gobernanza — toca el contrato que blinda
  contra downgrades. Necesita tu OK explícito sobre CÓMO relajar la regla sin abrir la puerta a
  regresiones V0.

**Recomendación:** favorable en lo estadístico (M1 es el modelo honesto), pero solo con una
enmienda explícita y acotada del contrato. Decides tú.

---

## A6 · Régimen degenerado (E[D]≈1 día) — reparametrización

**Estado.** El régimen "alta volatilidad" es un **absorbe-outliers** (E[D]=1.0 d): óptimo
global, no un bug (17/50 arranques; ni Student-t ni K=3 lo corrigen —
`reports/diag_degenerate_regime.md`). La Fase 1 ya **apagó su IC** (F4) y el banner ya avisa; no
se esconde. Lo que queda es si quieres intentar que persista.

- **Palanca propuesta:** probar (a) piso de persistencia (acotar p_kk por abajo), (b) componente
  de salto (jump), o (c) mezcla de colas — para que el segundo régimen capture volatilidad
  persistente en vez de outliers sueltos.
- **Costo:** medio-alto (cambia la parametrización en `models/params.py`/`hamilton.py`/
  `estimate.py`; re-verificar `test_hamilton_recovery`, `test_label_ordering`; re-correr R6 +
  walk-forward).
- **Criterio de aceptación pre-registrado:** adoptar la reparametrización solo si **E[D] > 5
  días SIN perder log-loss OOS** (empeorar la densidad predictiva para "arreglar" la estética
  del régimen sería un mal trade). Si ninguna variante lo logra, se mantiene el actual con caveat.
- **Riesgo:** es metodología estructural (R5 identificación); alto cuidado.

**Recomendación:** baja prioridad. El régimen degenerado ya se reporta con honestidad (IC
apagado + banner). Solo vale la pena si un piso/jump mejora la densidad predictiva, no por
cosmética.

---

## Resumen de decisiones que te pido

| Palanca | Estado real | Qué decides | Recom. |
|---|---|---|---|
| **A1 kernel** | Sesgo ya cerrado (soporte observado + aviso stale corregido hoy) | ¿Power-law por AIC, o exponencial + caveat KS? | Exponencial + caveat |
| **A4 M1 vs M2** | M2 publicado; M1 domina en bias-variance | ¿Publicar M1 + enmendar contrato V1+⇒tvtp? | Favorable con enmienda explícita |
| **A6 régimen degenerado** | IC apagado + banner (honesto) | ¿Piso/jump/mezcla con criterio E[D]>5d sin perder log-loss? | Baja prioridad |

Ninguna se implementa hasta tu OK y la consulta de los dos documentos de referencia.

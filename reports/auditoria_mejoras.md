# Auditoría de mejoras del índice IRF-N — recomendaciones priorizadas

Fecha: 2026-08-16. Base: artefacto vigente `run_id=7773faae4863` (V3, promovido) +
diagnósticos de esta sesión (`diag_degenerate_regime.md`, `validation_v4.md`,
ablaciones M1..M5). La remediación de **publicación** ya está cerrada (ver
`artifacts/README.md`); esto es sobre la **calidad del índice en sí**.

Cada recomendación lleva: evidencia, qué decisión implica y si toca metodología
(en cuyo caso la decide el director, no se aplica en silencio — regla del proyecto).

---

## P1 — Modelo (mayor palanca sobre los números publicados)

### 1. Atacar el régimen "absorbe-outliers" de forma explícita
- **Evidencia (Fase 7):** el segundo régimen dura ~1 día (E[D]≈1, ~3% de días) y es
  el **óptimo global**, no un artefacto del multistart (K=2 Normal, 50 arranques:
  17/50 convergen al mismo punto). **Ni Student-t ni K=3** lo corrigen limpiamente.
- **Recomendación (metodología, decisión del director):** en vez de "vivir con él",
  probar una de estas reparametrizaciones y re-correr walk-forward (R2) + multistart
  (R6): (a) **piso de persistencia** sobre `p_kk` (p.ej. p_kk ≥ 0.8) en la
  parametrización; (b) **componente explícito de saltos** (jump) que absorba los
  outliers fuera de la cadena de regímenes; (c) **mezcla de colas** (2 componentes de
  varianza por régimen). Criterio de éxito pre-registrado: E[D] del régimen de alta
  vol > 5 días **sin** perder log-loss OOS frente a la climatología.
- **Impacto:** hoy el retorno condicional de alta vol (−139%/año) solo es publicable
  con un caveat (banner). Un régimen de alta vol persistente lo volvería interpretable.

### 2. Considerar publicar M1 (regímenes, P constante) en vez de M2
- **Evidencia:** la ablación es tajante — **M1 > M0** (los regímenes aportan, DM
  p≈0.001) pero **ninguna covariable de transición** (técnica M2 ni macro M3) mejora
  OOS (M2 vs M1 no significativo con L1). El artefacto publicado es **M2** (con
  `sma_gap`, `bb_width_z`), que añade complejidad sin ganancia demostrada.
- **Recomendación (decisión del director):** evaluar publicar **M1** como el índice
  "de producción" (el modelo más simple que captura el único aporte robusto), y dejar
  M2/TVTP como línea de investigación. Nota de contrato: si se adopta M1, la regla
  "V1+ ⇒ tvtp=true" de `contract.py` debe relajarse a propósito.

### 3. Hawkes: probar el kernel power-law
- **Evidencia:** el KS de re-escalamiento **rechaza** el kernel exponencial (p=0), así
  que `n=0.739` es una **cota superior cualitativa**, no una estimación creíble. Ya
  existe `src/irfn/models/hawkes_powerlaw.py` y `scripts/compare_hawkes_kernels.py`.
- **Recomendación:** ajustar el power-law y comparar KS/AIC contra el exponencial. Si
  el power-law no es rechazado, `n` pasa a ser un número reportable, no solo un aviso.

---

## P2 — Datos (desbloquear las capas hoy inactivas)

### 4. Corpus GDELT: 240/998 días — decidir el destino de la capa de noticias
- **Evidencia:** el Hawkes se publica como indicador de fragilidad standalone, pero la
  **ablación M5-vs-M4** (¿el flujo de titulares mejora la predicción de régimen?) NO
  es corrible: necesita ~7 años (train 4a + 6×6m) y hay 240 días. Cuello de botella =
  rate limit de GDELT sobre IP compartida (CGNAT), no CPU.
- **Recomendación (decisión del director):** o (a) backfill desde una IP no
  compartida / repartido en el tiempo hasta ~7 años para correr M5, o (b) **cerrar
  formalmente M5 como bloqueado-por-datos** y quedarse con el branching ratio como
  indicador cualitativo (lo que hay). No dejarlo en limbo.

### 5. Capa de sorpresa (V2): resolver la fuente de consenso o cerrarla
- **Evidencia:** V2 congelada; no hay fuente **gratuita** de consenso histórico
  point-in-time (Forex Factory, Investing, Econoday, FXStreet descartadas con
  evidencia). `capture_consensus.py` acumula hacia adelante pero muy lento.
- **Recomendación (decisión del director):** decidir explícitamente **pagar Trading
  Economics** (única vía point-in-time real) o declarar V2 cerrada. Mantenerla
  "congelada indefinidamente" es deuda silenciosa.

### 6. Unificar la ingesta de precios (caché fresca)
- **Evidencia (hallada esta sesión):** `run_pipeline` usa `load_returns` (baja fresco)
  pero `run_v3` usa `load_close` con **caché en disco** que se quedó en 2026-07-10
  (~5 semanas rancio). Tuve que borrar la caché a mano para que el V4 usara datos a
  2026-08-14.
- **Recomendación (ingeniería, sin metodología):** un único loader con política de
  frescura explícita — p.ej. `load_close(..., max_age_days=3)` que re-baja si la
  caché está vieja, o un flag `--refresh-prices`. Elimina el modo de fallo "publiqué
  con datos de hace 5 semanas sin darme cuenta".

---

## P3 — Validación y producto

### 7. Reducir la "cola" del walk-forward (hueco de 44 días)
- **Evidencia:** el histórico OOS termina 2026-07-01 mientras `asof`=2026-08-14; el
  último tramo (~6 semanas) no forma un bloque de test completo. Hoy se explica con un
  banner, pero es una limitación real del producto.
- **Recomendación (metodología menor):** permitir un **último bloque expandible** o un
  bloque de test más corto al final, para que la serie publicada llegue más cerca de
  `asof`. Cuidar no introducir look-ahead (el test `prefix_invariance` lo protege).

### 8. Enmarcar el producto como índice de VOLATILIDAD/fragilidad, no direccional
- **Evidencia:** Pesaran-Timmermann (Test 3) — sin señal direccional al 5% (hit rate
  55% vs 54% esperado). El modelo distingue regímenes de **volatilidad**, no de
  dirección; la regla económica **no bate** comprar-y-mantener (documentado, negativo).
- **Recomendación:** posicionar el índice explícitamente como medidor de régimen de
  vol + fragilidad informativa, no como señal de retorno. Evita la tentación de leer el
  retorno condicional como predicción.

### 9. Desplegar el panel público (Next.js) — pendiente de V4
- **Evidencia:** el panel estático está construido y verificado, pero **no desplegado**
  (falta decisión + credenciales de Vercel). Ahora que `latest/` es un V3 coherente,
  el panel consumiría datos correctos.
- **Recomendación (decisión + credenciales del director):** desplegar, o exponer al
  menos la app Streamlit tras la remediación.

---

## P4 — Rigor / higiene

### 10. Cerrar R6 en TODAS las líneas
- **Evidencia:** SPY V4 ya es R6 (multistart 30, wf 20). Pero **BTC** y varios tests
  (BIC/Hansen/económico) siguen `--quick`.
- **Recomendación:** re-correr la línea BTC y los tests pendientes sin `--quick` para
  que todo lo publicado cumpla R6 (como ya lo hace SPY).

### 11. Promover BTC a la nueva tubería atómica
- **Evidencia:** la remediación se aplicó y verificó sobre SPY. BTC (`artifacts/btc/`)
  usa los mismos orquestadores (ya retargeteados) pero no se ha corrido/promovido con
  el nuevo flujo.
- **Recomendación:** correr `run_v3 --asset BTC` (auto-contenido ahora) y promover con
  `scripts/promote.py artifacts/btc/runs/<id> --slug btc`.

---

## Resumen ejecutivo
El hallazgo transversal de toda la validación se mantiene: **el único aporte robusto
del modelo es que los regímenes de volatilidad importan (M1>M0); ninguna covariable de
transición añade valor OOS.** Las mejoras de mayor impacto son (1) arreglar el régimen
degenerado y (2) decidir si el índice de producción debe ser M1 en vez de M2 — ambas
decisiones de metodología del director. Todo lo demás (GDELT, consenso, power-law,
despliegue) desbloquea capas que hoy están honestamente inactivas.

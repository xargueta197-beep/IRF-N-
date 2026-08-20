# Plan de mejora de los avisos del artefacto (SPY / BTC)

**Fecha:** 2026-08-18
**Estado:** PROPUESTA PARA EL DIRECTOR — nada de esto se ha implementado.
**Autor:** Claude Code (bajo CLAUDE.md R1–R9; los dos documentos de referencia consultados antes de redactar: Reglas Absolutas y Guía de Implementación §6.6, §10).
**Insumo:** los avisos de corrida de SPY V3/M1 (`run_id=3b4f1e39b59c`) y BTC V3 K=1-t, más los hallazgos de `audits/AUDIT_MATH_v1.md` (2026-08-18).

---

## 0. Principio rector (no negociable)

**Los avisos NO son bugs a silenciar. Son divulgación honesta (R8).** El proyecto emite estos mensajes *a propósito*: reportan blockers de datos reales, limitaciones conocidas del modelo, y decisiones de metodología ya tomadas. La Guía (Parte 10) lo dice sin adornos: *"La capa de noticias puede no aportar nada... El diseño está hecho para que el test lo revele en vez de esconderlo"*.

Por eso "mejorar los avisos" significa exactamente **tres cosas, y ninguna es hacerlos desaparecer con código**:

- **(M) Mitigar** una limitación real, cuando hay un fix *sancionado por la Guía* (p. ej. power-law para el KS, §6.6). Requiere OK del director: es metodología.
- **(C) Clarificar** un aviso que es correcto pero se lee peor de lo necesario. Bajo riesgo, sin tocar números del modelo.
- **(E) Esperar** — mantener corriendo el pipeline de datos y fijar el umbral de reapertura. Los blockers de datos NO se arreglan con código; borrarlos sería falsificar (viola R8 y la honestidad del proyecto).

**Anti-patrón prohibido:** encoger la malla de bloques del walk-forward, rellenar con FRED revisado (R4), o quitar un caveat para que el panel "se vea más limpio". Cualquiera de esos convierte una limitación honesta en una mentira.

---

## 1. Triage de todos los avisos

| # | Aviso | Categoría | ¿Defecto? | Acción | Quién decide |
| :-: | :-- | :-: | :-- | :-- | :-- |
| 1 | M3 macro no aporta (DM p=0.299) | E/estado | No | Ninguna: es un resultado OOS negativo documentado (R8). Ya cerrado. | — |
| 2 | M5 vs M4 infranqueable por M4 (sin consenso) | **E — datos** | No | Mantener `capture_consensus.py`; fijar umbral de reapertura | Director (pagar TE o esperar) |
| 3 | M2+H bloqueado: λ_N_z cubre 1.7 a, se necesitan ≥7 | **E — datos** | No | Seguir backfill GDELT hacia atrás; NO encoger malla (R8) | Automático + revisión |
| 4 | 10 días censurados por el cap de GDELT | E/estado | No | Ya contado en `missing_days`; opcional re-capturar picos densos | Bajo, opcional |
| 5 | Corpus 240/998 d; ajuste sobre tiempo observado | E/estado | No | Correcto (decisión director 2026-08-15). Caveat vigente. | — |
| 6 | Dithering intra-bin (83% empates) | E/estado | No | Correcto (decisión director 2026-08-14). | — |
| 7 | **KS rechaza el kernel exponencial** (D=0.0289) | **M — modelo** | Limitación real, no bug | Decidir exp+caveat vs power-law (ya implementado) | **Director** |
| 8 | V2/sorpresa inactiva (0 consenso, se necesitan ≥30) | **E — datos** | No | Igual que #2 | Director |
| 9 | **Matriz de transición casi rango-1** (dif ≤0.05) | **M — modelo** | Rasgo de datos, NO bug | Documentar; opcional K=3 / piso persistencia | **Director** |
| 10 | OOS hasta 2026-07-01 vs asof 2026-08-14 (44 d) | **C — UI** | No | Clarificar copy (ya explicado, mejorable) | Bajo |
| 11 | Régimen "alta vol" absorbe-outliers (~1 d), no anualizable | **M/C** | Limitación K=2, ya avisada | Copy ya hecho (F2.c); opcional reparametrizar | **Director** (si reparametriza) |
| 12 | **n varía ~1300× según ventana; IC no capta esa sensibilidad** | **M — modelo** | Limitación de reporte | Reportar banda de sensibilidad (Rubin ya corrido) | **Director** |

---

## 2. Categoría E — Blockers de datos (avisos 2, 3, 8; relacionados 4)

**No hay fix de código. La Guía (Parte 10) es explícita: _"El cuello de botella real es el consenso, no las matemáticas. Sin consenso histórico no hay índice de sorpresa. Empieza a guardarlo hoy."_**

Escalera de dependencia (del propio artefacto): **M5 depende de M4 (sorpresa) depende de consenso histórico.** Y el walk-forward pre-registrado de M2+H depende de la cobertura de λ_N_z (GDELT).

Estado real de datos hoy:
- **Consenso (M4/V2):** 0 eventos válidos acumulados; se necesitan ≥30. `capture_consensus.py` acumula hacia adelante desde Sesión 0. Alternativa: pagar Trading Economics point-in-time. **Decisión pendiente del director** (ya en la memoria del proyecto, sin cambios).
- **GDELT (λ_N_z):** **242 días** capturados; el walk-forward pre-registrado necesita **~2555 días (7 años)** = train 4a + 6 bloques de 6m. Faltan ~6.3 años de backfill hacia atrás.

**Propuesta (sin metodología, solo operación):**
1. Dejar el aviso tal cual (es honesto). NO tocar el código para ocultarlo.
2. **Fijar umbrales de reapertura explícitos y observables**, para que la reapertura sea automática y no "cuando alguien se acuerde":
   - M4/V2: reabrir cuando `n_with_consensus ≥ 30` en `≥1` indicador (ya emitido en `news_layer_params.coverage`).
   - M2+H: reabrir cuando `load_headlines().attrs['coverage']['n_days'] ≥ 2555` (o el mínimo real que exija la malla de 6 bloques sin encogerla).
3. Mantener vivos los dos capturadores. Nota de red: la IP es compartida y GDELT limita por tasa ([[project_gdelt_network]]); el backfill es lento por diseño, no por bug.
4. **Prohibido:** encoger la malla de bloques para "alcanzar" la cobertura (R8). El aviso #3 ya lo dice.

**Riesgo honesto:** el backfill de 6 años a ~1 día/varias-peticiones sobre IP compartida puede tardar meses, y el consenso gratuito no existe (auditoría de datos §7-9 lo cerró). Es posible que M4/M5 **nunca** se puedan correr sin pagar TE. Eso es un hecho del mundo, no una tarea de ingeniería — el plan lo nombra, no lo esconde.

---

## 3. Categoría M — Limitaciones de modelo con fix sancionado (avisos 7, 9, 12; relacionado 11)

**Todo lo de esta sección es METODOLOGÍA → requiere OK explícito del director antes de tocar `src/` (R3 del reparto de roles: "no cambies metodología por tu cuenta").** Yo preparo el análisis; el director decide.

### M-1 · Aviso #7 — KS rechaza el kernel exponencial

Conexión con la auditoría (**MATH-D7, S5**): el rechazo lo **domina el tamaño muestral**, no un desajuste grande de forma. D_obs=0.0289 vs D_crít(5%)=0.0044 → ratio 6.6; y el p es tipo Lilliefors (parámetros estimados de la misma muestra), así que 2.0e-69 no es literalmente interpretable. El desajuste de forma es *pequeño*.

**Dato clave: el power-law YA está implementado y comparado** (`src/irfn/models/hawkes_powerlaw.py`, `_compare_kernels.log`):

| Kernel | n | KS p | AIC | Veredicto |
| :-- | :-: | :-: | :-: | :-- |
| Exponencial (publicado) | 0.695 | 3.4e-17 | −110 228.8 | falla KS |
| Power-law | 0.836 | 6.7e-13 | **−110 408.4** (mejor por 180) | **también falla KS** |

Lo que esto significa para el director (la disyuntiva real, sin adorno):
- **Opción A (statu quo):** exponencial + caveat honesto. Coherente con la Guía §6.6 ("si el KS falla, repórtalo... considerar power-law en V3+") y con MATH-D7 (el efecto es pequeño). **Recomendada** salvo que el director quiera el mejor AIC.
- **Opción B (adoptar power-law):** mejor AIC, pero **mueve el número titular n de 0.69 a 0.84** (más cerca de la frontera reflexiva) y **sigue sin pasar el KS**. Cambiar el titular por un kernel que tampoco pasa el test, y que empuja n hacia "criticidad", es una decisión de riesgo comunicacional que solo el director puede tomar.
- **Opción C (ninguno es el titular):** ya ejecutada parcialmente — el panel **sacó n de los KPIs** (ver aviso #12). Si n no es titular, la elección de kernel pesa menos y A basta.

**Propuesta:** mantener A (exponencial + caveat + MATH-D7 cuantificado en el warning), y dejar el power-law como diagnóstico comparativo ya disponible. Reabrir solo si el director prioriza AIC sobre estabilidad del titular.

### M-2 · Aviso #9 — Matriz de transición casi rango-1

Conexión con la auditoría (**A.2 punto 4, resultado de primera página + MATH-C3, S5**): en datos sintéticos con P bien condicionada **el estimador SÍ recupera rango 2** (sv_ratio 0.859 vs 0.88). Por tanto el rango-1 en SPY real **es un rasgo de los datos, no un defecto del estimador**. Es la misma raíz del régimen degenerado (A6): un estado "alta vol" absorbe-outliers de ~1 día. κ≈1−1e-16 en 2/19 bloques es su firma numérica (MATH-C3), no un bug de identificación.

Opciones (todas metodología, del director):
- **Statu quo:** documentar que P≈rango-1 y el segundo régimen es absorbe-outliers, con banner data-driven (ya existe). El más honesto con "los regímenes son latentes, no reales" (Guía Parte 10).
- **Reparametrizar** el régimen degenerado: piso de persistencia, componente de salto, o mezcla de colas. La auditoría de mejoras (`reports/auditoria_mejoras.md` #1) y `diag_degenerate_regime.md` ya lo estudiaron: es óptimo global, ni Student-t ni K=3 lo corrigieron. Alta probabilidad de no aportar.
- **K=3:** el BIC ya eligió K=2 sobre K=3 (`v1_kselect.json`: BIC K=2=8195.8 < K=3=8220.5). Cambiarlo contradiría la selección de modelo ya validada.

**Propuesta:** statu quo + documentación. Reparametrizar solo si el director lo prioriza como línea nueva (no lo recomiendo: evidencia previa en contra).

### M-3 · Aviso #12 — Sensibilidad de n a la ventana; el IC no la capta

Conexión con la auditoría (**MATH-D7 + nota del artefacto**): el IC95 [0.731, 0.747] captura el ruido del MLE *dentro de una elección de ventana*, no la sensibilidad *a* esa elección (span calendario n=0.9994 vs tiempo observado n=0.7388). El panel **ya sacó n de los KPIs** por esto — decisión correcta y ya tomada.

El trabajo de sensibilidad al dithering **ya se corrió** (Rubin/imputación múltiple, `dithering_sensitivity_v3.md`): el dithering añade ~4% a la varianza de n (segundo orden). Lo que falta es la sensibilidad a la **elección de ventana** (calendario vs observado), que es de primer orden y actualmente se comunica en prosa, no como banda.

**Propuesta (reporte, requiere OK del director porque cambia CÓMO se publica la incertidumbre, R3/R7):** publicar n con **dos incertidumbres separadas y etiquetadas**: (a) IC del MLE dentro de la ventana observada (lo actual), y (b) rango entre elecciones de ventana (observada vs calendario), como cota de sensibilidad estructural. Nunca colapsarlas en un solo IC que finge precisión. Esto formaliza en el artefacto lo que el aviso #12 ya dice en palabras.

---

## 4. Categoría C — Presentación (aviso 10; relacionado 11)

Bajo riesgo, sin tocar números del modelo (R9: la interfaz no calcula). Puedo prepararlo yo y el director solo revisa el copy.

- **Aviso #10 (OOS hasta 07-01 vs asof 08-14):** el texto actual ya explica que el último tramo no forma un bloque de test completo. Mejora propuesta: añadir en la pantalla Histórico una banda visual "tramo aún no evaluable (OOS incompleto)" entre `history_end` y `asof`, para que el hueco de 44 días se *vea* como intencional y no como dato faltante. Solo UI.
- **Aviso #11 (no anualizable):** ya resuelto en el trabajo F2.c (supresión del punto anualizado + n_obs visible). Verificar que el copy del panel público (Next) refleje lo mismo que la app Streamlit.

---

## 5. Secuencia propuesta (por dependencia, no por severidad)

Ninguna de estas tareas desbloquea a otra (son independientes); el orden es por valor/coste:

1. **C (UI) — inmediato, sin director:** clarificar el hueco OOS (#10) y verificar paridad de copy "no anualizable" en el panel público (#11). Riesgo nulo, R9 respetado.
2. **E (datos) — operación continua:** fijar los dos umbrales de reapertura observables (#2, #3, #8) y confirmar que los capturadores siguen vivos. No es código de modelo.
3. **M-3 (reporte de n) — requiere director:** si aprueba, publicar la banda de sensibilidad de ventana junto al IC del MLE (#12).
4. **M-1 (kernel) — requiere director:** decisión exp vs power-law (#7). El material comparativo ya existe; es una llamada de criterio, no de ingeniería.
5. **M-2 (régimen degenerado) — requiere director, baja prioridad:** documentar (#9); reparametrizar solo si lo prioriza (evidencia previa en contra).

---

## 6. Qué puedo hacer yo vs qué decide el director

| Puedo ejecutar yo (sin metodología) | Solo el director aprueba (metodología) |
| :-- | :-- |
| Clarificar copy del hueco OOS (#10) — solo UI | Adoptar power-law como titular (#7) |
| Paridad de copy "no anualizable" (#11) — solo UI | Reparametrizar el régimen degenerado (#9, #11) |
| Fijar umbrales de reapertura observables (#2,#3,#8) | Cambiar CÓMO se publica el IC de n (#12) |
| Mantener/verificar los capturadores de datos | Pagar Trading Economics para desbloquear M4/M5 |
| Documentar todo lo anterior (R8) | Cualquier cambio a `src/` de modelo/estimador |

---

## 7. Lo que este plan NO propone (para que quede escrito)

- **No** silenciar ningún aviso de datos con código (viola R8).
- **No** encoger la malla de bloques del walk-forward (aviso #3 lo prohíbe explícitamente).
- **No** rellenar macro con FRED revisado (R4).
- **No** cambiar K=2→K=3 (contradice el BIC ya validado).
- **No** inventar un IC de n que finja captar la sensibilidad de ventana (el problema exacto que el aviso #12 denuncia).

---

*Referencias: `audits/AUDIT_MATH_v1.md` (MATH-D7, MATH-C3, A.2), Guía de Implementación §6.6 y Parte 10, `reports/auditoria_mejoras.md`, `reports/diag_degenerate_regime.md`, `reports/dithering_sensitivity_v3.md`, `_compare_kernels.log`, `v1_kselect.json`.*

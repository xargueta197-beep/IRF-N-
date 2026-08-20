# AUDITORÍA MATEMÁTICA ADVERSARIAL — IRF-N V3 (M1)

**Informe:** AUDIT_MATH_v1
**Ejecutor:** Claude Code, bajo contrato CLAUDE.md (R1–R9), modo SOLO LECTURA sobre `src/`.
**Director:** Xavier Argueta — única autoridad para aprobar cambios.

---

## Encabezado de congelamiento (Regla Cero)

| Campo | Valor |
| :-- | :-- |
| **Commit auditado (HEAD)** | `e01e0f8dac8831932c41fa2b19629421edb313e3` |
| **Tag ancla inmutable** | `audit-math-v1-base` |
| **Base previa registrada** | `04b7d5a` (último commit del proyecto antes del snapshot de auditoría) |
| **run_id de `artifacts/latest/`** | `3b4f1e39b59c` (M1: K=2, matriz de transición CONSTANTE, `tvtp=false`, `covariates=[]`, capa Hawkes standalone activa) |
| **asof** | 2026-08-14 |
| **version** | V3 |
| **Tests en verde (marcador rápido, commit auditado)** | **155 passed, 12 deselected (slow)** (`pytest -m "not slow"`, 160 s) |
| **Tests en verde (suite completa, incl. slow)** | **167 passed, 0 failed** (`pytest tests/`, 53 min; incluye los 2 slow de recuperación Hawkes) |

El commit `e01e0f8` congela el trabajo de sesión F2.c que ya estaba en el árbol al iniciar (supresión de puntos anualizados en regímenes no anualizables + pantallas 7/8 de la app). **La auditoría no modificó nada de `src/`, `tests/`, `artifacts/` ni `panel/`.** Todo lo escrito vive en `audits/` (este informe) y `audits/probes/` (scripts de prueba desechables que importan producción sin tocarla).

**Fuera de alcance (§11):** rediseño de arquitectura, cambios de kernel, despliegue del panel, optimización. Los seis ejes cerrados se auditaron por *corrección matemática* del número que sostuvo el cierre; ninguno se reactivó como línea de trabajo.

---

## Resumen ejecutivo (jerarquía obligatoria)

**No se encontró ningún hallazgo S0 (estructural) ni S1 (sesgo confirmado que contamine el artefacto).** La matemática del núcleo publicado —filtro de Hamilton, verosimilitud MS-GJR-GARCH, verosimilitud y branching ratio del Hawkes marcado, y la recuperación de parámetros sobre verdad conocida— **está confirmadamente sana**, con pruebas numéricas ejecutables que lo demuestran (§ Fase A, B, C, D, F).

El resultado más importante del encargo (Fase A.2, punto 4): **el estimador MS-GARCH SÍ recupera una matriz de transición de rango 2 en datos sintéticos bien condicionados** (razón de valores singulares estimada 0.859 vs verdadera 0.88). Por tanto, el régimen degenerado que se observa en datos reales (A6) **no es un defecto del estimador**: M1 —el único aporte OOS robusto del proyecto— se apoya en un cimiento que aquí queda verificado, no refutado.

Los cinco hallazgos son **S2 (inferencia) y S5 (presentación/reproducibilidad)**. **Ninguno contamina el artefacto publicado.** El más sustantivo (MATH-E2) muestra un sesgo anti-conservador de segundo orden en el p-valor del test de número de regímenes, que **no cambia la decisión K=2** (el estadístico observado supera a la distribución nula por un factor de ~17).

---

## Protocolo de evidencia

Todos los scripts están en `audits/probes/`; sus salidas crudas en `audits/probes/out/*.json` y `*_run.log`. Cada uno importa el código de producción y lo ejerce sin modificarlo.

| Fase | Script | Veredicto |
| :-- | :-- | :-- |
| A — recuperación sobre verdad conocida | `probe_a1_hawkes.py`, `probe_a2_msgarch.py` | PASA (Hawkes + MSGARCH); A.3/A.4 NO APLICAN |
| B — doble implementación | `probe_b_crosscheck.py` | PASA (0 fallos, 14 checks) |
| C — invariantes de probabilidad | `invariants.py` | PASA (0 fallos, 18 aserciones) |
| D — Hawkes línea por línea | `probe_d_hawkes.py` | PASA con 1 matiz de presentación (KS Lilliefors) |
| E — inferencia y comparación | `probe_e_inference.py` | 1 hallazgo S2 (asimetría de multistart) + confirmaciones |
| F — walk-forward y fugas | `probe_f_walkforward.py` | PASA (reconstrucción independiente reproduce el bloque) |
| G — capa económica y Kelly | (lectura de `economic.py`) | Kelly NO EXISTE; regla auditada, sin hallazgo |
| H — reconciliación código↔artefacto↔panel | `probe_h_reconcile.py` | PASA con 1 gap de reproducibilidad S5 |

---

## FASE A — Recuperación de parámetros sobre verdad conocida (el núcleo)

> Nota de especificación: el encargo fija μ=0.5, α=0.3, β=1.2 "⇒ n=0.25", que es la cuenta **sin** marcas (α/β). El canon del proyecto y la física del modelo marcado es **n = α·E[s]/β**. Para que la *verdad* sea n=0.25 exacto con las marcas empíricas de FinBERT (E[s]=0.6364, del corpus real `headline_rug.parquet`), se fijó α = n·β/E[s] manteniendo μ y β. El objetivo del criterio (verdad n=0.25) se respeta.

### Tabla de cierre de Fase A

| Estimador | Escenario | Verdad | Recuperado | Criterio | Veredicto |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Hawkes (A.1) | n=0.25, lejos de frontera, N≈20 188 | n=0.25 | n̂=0.2627; μ,α,β todos dentro de ±3 SE | ±3 SE y \|n̂−n\|≤0.03 | **PASA** |
| Hawkes (A.1) | n=0.85, cerca de frontera, N≈18 266 | n=0.85 | n̂=0.8443; todos dentro de ±3 SE | ídem | **PASA** |
| Hawkes (A.1) | **soporte censurado, escala producción** (μ=100, β=30, span 998 d, 240 observados) | n=0.74 | ingenuo T=span: n̂=**0.997** (inflado); producción `compress_to_observed_time`: n̂=**0.730** | recuperar n al censurar | **PASA — el fix del 2026-08-15 está completo** |
| Hawkes (A.1) | 50 réplicas, n=0.25 | — | sesgo n = **+0.32 %**, RMSE 0.0085, cobertura IC95 **0.94** | sesgo <5 % | **PASA** |
| Hawkes (A.1) | 50 réplicas, n=0.85 | — | sesgo n = **−0.10 %**, RMSE 0.0081, cobertura IC95 **0.96** | sesgo <5 % | **PASA** |
| MS-GARCH (A.2) | K=2, P conocida, T=5000, R6×20 | ver script | μ,v,α,γ,β **todos dentro de ±3 SE en ambos regímenes**; P recuperada (entradas dentro de ±3 SE); clasificación hit 0.823, AUC 0.866 | recuperar P + GARCH + clasificar | **PASA** |
| MS-GARCH (A.2) | **rango de P (prueba decisiva)** | rango 2, sv_ratio 0.88 | sv_ratio estimado **0.859** (rango 2 recuperado) | recuperar rango 2 en sintético | **PASA** |
| MS-GARCH (A.2) | adversarial P casi absorbente (p₁₁=0.999) | ocupación régimen 2 = 2.1 % | estimador ubica el estado raro (AUC 0.884), no lo colapsa | comportamiento del régimen degenerado | **PASA** |

**A.3 (particle filter / jump-diffusion) y A.4 (diferenciación fraccionaria): NO APLICAN.** No existe ningún filtro de partículas ni procedimiento ARFIMA/fracdiff en `src/` (grep exhaustivo sin resultados). El pipeline no usa esos objetos. Documentado como no-verificado-por-inexistencia, no como hueco.

**Interpretación del resultado de rango (encargo A.2.4, primera página):** en sintético con P bien condicionada el estimador recupera rango 2; en datos reales colapsa a rango ~1. Según la lógica del propio encargo, esto sitúa la causa **en los datos / la identificación en real, no en el estimador**. El régimen degenerado (A6) es un óptimo global de la muestra real, no un artefacto numérico — consistente con `reports/diag_degenerate_regime.md`.

---

## FASE B — Doble implementación (cross-check independiente)

Todas las referencias son reimplementaciones lentas y obviamente correctas, escritas de cero en el probe. Concordancia con producción:

| Objeto | Referencia independiente | Tolerancia exigida | Error observado | Veredicto |
| :-- | :-- | :-- | :-- | :-- |
| log-verosim. Hawkes | doble suma O(n²) explícita, N=3000 | 1e-10 | **1.2e-15** (rel) | PASA |
| log-verosim. Hawkes escala producción | ídem, β=30/d, N=2000 | 1e-10 | **3.3e-15** | PASA |
| Compensador Λ(T) | cuadratura adaptativa `scipy.quad` (≠ fórmula analítica) | 1e-8 | **1.4e-16** | PASA |
| Identidad re-escalamiento Λ(tᵢ) | fórmula directa O(n²) | 1e-10 | 1.1e-13 | PASA |
| Filtro Hamilton (constante) | bucle ingenuo, densidades `scipy.stats`, normalización explícita | 1e-10 | **0.0** (loglik), 6.7e-16 (ξ) | PASA |
| Filtro Hamilton (t de Student) | ídem | 1e-10 | 0.0 / 1.3e-15 | PASA |
| Filtro Hamilton (TVTP) | ídem con `transition_matrices` | 1e-10 | 0.0 / 5.0e-16 | PASA |
| log-verosim. MS-GARCH | `statsmodels.MarkovRegression`, caso comparable (varianza constante por régimen) | 1e-6 rel | **3.9e-11** | PASA |
| BIC (K=1 t, K=2 normal, K=2 t) | conteo manual enumerado de k | exacto | k coincide (6/12/14) y BIC bit a bit | PASA |
| Pesos fracdiff | — | — | **NO APLICA** (no existe fracdiff) | — |

**Consecuencia:** la recursión O(n) del Hawkes coincide con la doble suma O(n²) a 1e-15. No hay error estructural en la verosimilitud; todo lo que depende de `n` (la fragilidad diaria publicada) descansa sobre una verosimilitud verificada.

---

## FASE C — Invariantes de probabilidad (aserciones ejecutables)

`audits/probes/invariants.py` — 18 aserciones sobre el artefacto vigente y corridas sintéticas. **0 fallos.**

**C.1 Cadena de Markov.** En 2000 θ aleatorios (K∈{1,2,3}, normal/t, TVTP con 0–2 covariables): filas de P (y de cada Pₜ bajo TVTP) suman 1 con desvío máximo <1e-12; todas las entradas en [0,1]. `transition_matrix_today` publicada suma 1 por filas y no tiene 0 ni 1 exactos. **ξ₀ = distribución estacionaria de P** (`hamilton.py:103`), función solo de parámetros → prefix-safe (verificado: la loglik del filtro coincide con la reimplementación arrancada en la estacionaria a <1e-8). Sensibilidad de ξ₀ sobre T=2000: ΔLL(uniforme)=+0.61, ΔLL(casi-degenerada)=−19.4 en unidades absolutas — efecto acotado al transitorio inicial, no estructural. **R3:** el `.shift(1)` de `hawkes_feature` es un desfase real en el índice temporal (verificado alineando la serie), no solo en el nombre de columna.

**C.2 Filtradas y predictivas del artefacto.** ξ_{t|t} y ξ_{t|t−1} ∈ [0,1], suman 1 (desvío <1e-9), sin NaN. **Orientación de la transpuesta verificada SOBRE LOS DATOS PUBLICADOS:** dentro de cada bloque del walk-forward, `xi_pred[t]` reproduce `Pᵦ' · xi_filt[t−1]` con error máximo <1e-9, mientras que la versión **sin** transponer (control) da error de orden ~1 → la orientación no es ambigua, es la correcta. **R1:** `publish()` lanza `LookAheadViolation` ante una clave `xi_smoothed` **anidada** (no solo de primer nivel).

**C.3 Estabilidad / restricciones GARCH.** En 2000 θ aleatorios: ω>0, α,γ,β≥0 y estacionariedad κ=α+γ/2+β<1 se cumplen **por construcción** (0 violaciones) — impuestas en la parametrización (`params.py`), no corregidas post-hoc (R5). κ por bloque del walk-forward: máximo 1.0000000 en 2/19 bloques (persistencia GARCH muy alta, típica; no viola la cota estricta pero queda pegada a ella — ver hallazgo MATH-C3 abajo).

**C.4 Calibración de las probabilidades OOS publicadas** (descomposición de Brier, Murphy 1973):
- vs proxy del proyecto (argmax ξ_{t|t}): Brier 0.0354, **resolution 0.0873 > reliability 0.0063** → el modelo resuelve mucho y está bien calibrado contra su propio proxy.
- vs proxy observable externo (quintil superior de \|r\|): Brier 0.2302, resolution 0.0030 — bajo, esperable: ξ es probabilidad de *régimen de volatilidad*, no de un día extremo puntual. Se reporta como diagnóstico, no como métrica de fallo.

**R1 — rastreo de flujo del smoother (chequeo de flujo, no de nombre):** `kim_smoother` existe, está decorado `@diagnostic_only`, y su único llamador (`run_pipeline._write_diagnostic_smoother`) escribe en `data/interim/` (fuera de `artifacts/`); la app lo dibuja solo bajo `DEV_MODE` desde ese directorio interino. `FORBIDDEN_KEYS = {xi_smoothed, smoothed, kim_smoother, xi_tT}` bloquea cualquier ruta hacia el payload. **Ninguna serie publicada proviene del suavizador.**

---

## FASE D — Hawkes línea por línea

- **D.1 Unidades.** Tiempos en DÍAS desde `origin` ⇒ μ_N y β en 1/día, n adimensional. Rastreado de la ingesta al reporte. El re-ajuste sobre el corpus real (30 arranques, semilla 42) reproduce μ_N≈102.9, α≈329.5, β≈283.9, n≈0.7385 (publicado: 103.04 / 329.66 / 283.96 / 0.73883) — coincide salvo un desvío de ~5e-4 relativo atribuible a que el corpus creció de 240 a 245 snapshots tras la publicación (snapshot `2026-08-16.json` posterior al run). **No es un bug de unidades ni de fórmula** (ver "no verificado").
- **D.2 Fórmula de n.** Para el kernel implementado φ(u,m)=α·m·exp(−β·u), la integral analítica es α·m/β por evento ⇒ **n = α·E[s]/β**. Verificación aritmética del n publicado desde (α, β, mean_mark): coincide a 1e-12. La cascada 1/(1−n) también.
- **D.3 Censura activa en serialización.** `run_v3.py:572` llama `expected_cascade_reported(n, ci_high, trigger=0.95)`; el artefacto publica `expected_cascade_bounded=True` coherente con ci_high=0.7465<0.95. La regla no está solo definida: está en el path.
- **D.4 Efecto de borde t₀.** Sesgo cuantificado descartando la historia previa (supuesto λ(t₀)=μ del estimador): kernel lento adversarial (β=1.2/d) Δn=−0.0072; escala producción (β=284/d) Δn=−0.0004. Despreciable al β real.
- **D.5 Empates de timestamp.** Corpus real: **83.2 % de eventos empatados** antes del dithering (confirma el diagnóstico que motivó el dithering intra-bin U(0,15min), decisión del director 2026-08-14). Tras el dithering: 95 234/95 234 timestamps únicos, orden restaurado.
- **D.6 IC de n cerca de la frontera.** IC delta publicado [0.7311, 0.7465] vs **verosimilitud perfilada** [0.7328, 0.7448] sobre el corpus real. El delta es **ligeramente más ancho** (conservador) que el perfilado — a n≈0.74 la aproximación normal es fiable, como afirma el docstring. Sin hallazgo: el IC publicado no sub-cubre.
- **D.7 KS.** Ver hallazgo MATH-D7 (matiz de presentación, ya documentado por el proyecto como A3).

---

## FASE E, F, G, H — resumen

- **E.1** BIC del `v1_kselect.json`: las 8 filas reverificadas (k_free = conteo manual, BIC = −2·loglik+k·ln n_obs) — todo exacto. Ganador K=2 normal.
- **E.3 Diebold-Mariano M2 vs M1:** recomputo **independiente** desde los dos `history.parquet` persistidos (M1=`3b4f1e39b59c`, M2=`7c44a7fac16d`), con Newey-West + HLN propios. Con la regla de rezagos del proyecto (lags=20): DM=1.617, **p=0.106** — reproduce el reportado. Sensibilidad al rezago: p∈[0.099, 0.124] para lags∈{0,5,20,40}. La perdida es −loglik predictiva un-paso-adelante (h=1): **la premisa "h>1 con ventanas solapadas" del encargo NO aplica a este pipeline** (no hay objetivo de vol a 20 días en el DM; ese objetivo es de otra capa, ver Fase F). El estadístico está bien construido; la decisión de cerrar M2 se apoya en un p correctamente calculado.
- **E.4** Cobertura del IC95 de `sharpe_ci` (bootstrap estacionario) sobre AR(1): 0.90/1.00 (aceptable para bloque estacionario con dependencia). Supresión A2 (maxdd sin IC; régimen degenerado sin IC) **activa y verificada en el artefacto publicado**.
- **F.4 Reconstrucción independiente del bloque 5** del walk-forward (estimador MS-GJR-GARCH escrito de cero, sin reutilizar producción, 20 arranques): loglik train/obs mía −1.028081 vs artefacto −1.028048 (Δ=3.3e-5); loglik test/obs mía −0.968269 vs −0.968256 (Δ=1.3e-5). **El número OOS publicado se reproduce desde una implementación independiente.** PIT del tramo OOS: KS=0.075, p=0.455 (uniformidad no rechazada), autocorr lag-1 −0.011 (dentro de ±0.177).
- **F.3 Vintages (R4):** `data/alfred.py` solicita la ventana realtime correcta (`realtime_start` = fecha de publicación real) y reconstruye la serie point-in-time con margen de días hábiles; no usa la serie corriente revisada. (Verificación de código; la ablación macro está cerrada por datos, fuera de alcance reactivar.)
- **F.5 Pre-registro:** `docs/preregistro_regla_trading.md` y `economic.py` entran ambos en el repositorio en el mismo commit base `77a27e9`; el doc dice "NADA DE ESTO SE HA CORRIDO" y su fecha de diseño (2026-07-18) precede a la implementación (mtime 2026-07-18 07:03, 13 min después del doc). No hay evidencia de que el criterio cambiara tras ver resultados.
- **G Kelly:** **no existe capa de Kelly.** La capa económica (`economic.py`) es una regla de des-riesgo binaria pre-registrada (w=1 o w=0 según P(alta vol)>0.5), con `shift(1)` explícito (R3), costo por turnover y éxito por IC bootstrap del Sharpe. No hay sizing continuo, ni apalancamiento, ni fracciones cripto/acciones. Los puntos G.1–G.4 del encargo (Kelly fraccional, encogimiento de p̂, cotas de leverage) **no tienen objeto que auditar**. La regla implementada es honesta con su pre-registro.
- **H Reconciliación:** run_id idéntico en irfn/manifest/panel; `validation.json` del panel `stale=false` validando `3b4f1e39b59c`; asof sincronizado; el panel se etiqueta M1 (tvtp=false, covariates=[], K=2). **5 de 6 números publicados reconcilian exacto** desde disco (E[D], entropía, entropy_max, n, cascada). El 6º (vol_ann de `conditional_stats`) ver hallazgo MATH-H1. **R9:** no hay suavizado ni lógica de modelo en `panel/` (grep de rolling/smooth/window sin resultados); solo argmax y ln(K) recomputados de datos ya publicados. **Desfase documental del encargo (H.5) YA CERRADO:** la sección ESTADO ACTUAL de `CLAUDE.md` sí tiene la entrada de la migración a M1 (commit `04b7d5a`, "registra en ESTADO ACTUAL la sesion de migracion a M1 (commit 6bb8ae1)").

---

## HALLAZGOS

### MATH-E2 — Asimetría de multistart en el bootstrap-LR del número de regímenes

| Campo | Contenido |
| :-- | :-- |
| **ID** | MATH-E2 |
| **Ubicación** | `src/irfn/validation/tests_stat.py:161-186` (`bootstrap_lr_test`); config `v1.ktest.boot_n_starts=6` vs `bic_n_starts=20` |
| **Fórmula implementada** | LR_obs = 2(ℓ_alt − ℓ_null) con **20** arranques por modelo; réplicas nulas LR_b = 2(ℓ₁−ℓ₀) con **6** arranques por modelo; p = (1+#{LR_b≥LR_obs})/(B+1), B=49 |
| **Fórmula canónica** | El bootstrap paramétrico de razón de verosimilitud (Davidson & MacKinnon 2004; McLachlan 1987 para mezclas) exige que el estadístico observado y las réplicas se computen con **el mismo procedimiento de optimización**, para que LR_b y LR_obs sean idénticamente distribuidos bajo H₀. |
| **Discrepancia** | El presupuesto de multistart es asimétrico: 20 arranques hallan mejor el óptimo del alternativo que 6. En una prueba de 4 réplicas simuladas del nulo K=1 ajustado a la muestra del kselect (SPY, N=3400), LR con 20 arranques superó al de 6 en promedio **+0.99** (réplicas: +0.005, +1.999, +1.958, +0.000). Dirección del sesgo: las réplicas **subestiman** el LR nulo ⇒ la cola nula queda corta ⇒ **p sesgado a la baja (anti-conservador)**. |
| **Prueba numérica** | `audits/probes/probe_e_inference.py` → `e_results.json.E2_asimetria_multistart` |
| **Severidad + impacto** | **S2 — Inferencia inválida.** **CONTAMINA ARTEFACTO PUBLICADO: NO.** Atenuante decisivo: LR_obs = **305.0** vs cuantil 95 de la nula = **18.4** (y máximo nulo ~ decenas). La evidencia de K=2 es abrumadora; el sesgo de ~1 unidad LR es irrelevante frente a una separación de ~17×. Además, con B=49 el **p mínimo alcanzable es 1/50 = 0.02**, que es exactamente el p reportado: la evidencia real es "p ≤ 0.02" (LR_obs superó a las 49 réplicas), no un p finamente estimado. La decisión K=2 es robusta; lo que **no** es literalmente interpretable es el "0.02" como magnitud de evidencia fina. |

### MATH-D7 — KS de bondad de ajuste: Lilliefors + sobre-poder muestral

| Campo | Contenido |
| :-- | :-- |
| **ID** | MATH-D7 |
| **Ubicación** | `src/irfn/models/hawkes_mle.py:468` (`stats.kstest(inter, "expon")`) |
| **Fórmula implementada** | KS de una muestra de los interarribos re-escalados τ contra Exp(1) con parámetros del **mismo ajuste MLE**, N≈95 084; D=0.0289, p_nominal=1.98e-69 |
| **Fórmula canónica** | El KS de una muestra supone parámetros **conocidos a priori**. Con parámetros estimados de la misma muestra, la distribución nula de D es la de **Lilliefors** (más concentrada), y el p nominal de `kstest` no es exacto. Adicionalmente, el poder del KS crece con √N: a N grande, un D diminuto rechaza. |
| **Discrepancia** | (a) El p=1.98e-69 no es literalmente interpretable (Lilliefors). (b) El rechazo lo **domina N**: D crítico al 5 % ≈ 1.3581/√N = **0.0044**, y D_obs/D_crit = **6.6** — el desajuste de forma es pequeño (2.9 puntos porcentuales de desviación máxima en la CDF) pero N lo vuelve significativo. |
| **Prueba numérica** | `audits/probes/probe_d_hawkes.py` → `d_results.json.D7_ks` |
| **Severidad + impacto** | **S5 — Presentación.** **CONTAMINA ARTEFACTO: NO.** El proyecto **ya documenta esto** (A3, 2026-08-16): el warning del artefacto enriquece D/p/n y encuadra D como efecto pequeño; `reports/validation_v3.md` lo describe. Esta auditoría **cuantifica** el argumento (D_crit=0.0044, ratio 6.6) que el proyecto afirmaba cualitativamente, y **añade** el matiz Lilliefors (p no exacto), que el proyecto no menciona. La conclusión de fondo (el kernel exponencial no ajusta perfectamente; el power-law gana AIC pero tampoco pasa KS, F1) es correcta y no depende del p literal. |

### MATH-H1 — `conditional_stats` no es reconciliable desde el artefacto en disco

| Campo | Contenido |
| :-- | :-- |
| **ID** | MATH-H1 |
| **Ubicación** | `scripts/run_v3.py:541-550` (frame de `today_run` in-sample) → `src/irfn/outputs/publish.py:243` (`conditional_stats`) |
| **Fórmula implementada** | Las estadísticas condicionales por régimen (mean_ann, vol_ann, sharpe, maxdd) se calculan sobre el **frame in-sample** de `today_run` (filtro de todo el histórico con el único ajuste "de hoy"), cuyo `argmax_idx` asigna cada día a un régimen. |
| **Fórmula canónica** | Reconciliación de auditoría (§H.1): todo número publicado debe recalcularse desde los artefactos en disco. |
| **Discrepancia** | El `history.parquet` publicado es el frame **OOS del walk-forward** (2386 filas, asignación de régimen distinta). El frame in-sample que alimenta `conditional_stats` **no se persiste**. Recalcular vol_ann del régimen 0 desde `history.parquet` da 0.1735 vs 0.1593 publicado (17 % de diferencia) — no por un error de cálculo, sino porque **son dos frames distintos**. Un tercero no puede reproducir `conditional_stats` desde disco. |
| **Prueba numérica** | `audits/probes/probe_h_reconcile.py` → `h_results.json.H1_vol_ann_regimen0` (los otros 5 números reconcilian exacto) |
| **Severidad + impacto** | **S5 — Presentación / reproducibilidad.** **CONTAMINA ARTEFACTO: NO** (los valores publicados son descriptivos correctos sobre su frame). El gap es de *auditabilidad*: el insumo de `conditional_stats` (la serie de `argmax` in-sample con el modelo publicado) no viaja al artefacto. Propuesta (no ejecutada): persistir ese frame, o documentar en el contrato que `conditional_stats` es in-sample-full mientras `history.parquet` es OOS. |

### MATH-C3 — κ pegado a 1 en 2/19 bloques del walk-forward (observación, no defecto)

| Campo | Contenido |
| :-- | :-- |
| **ID** | MATH-C3 |
| **Ubicación** | `artifacts/latest/walkforward.json`, `kappa` por bloque |
| **Fórmula implementada** | κ_k = α_k + γ_k/2 + β_k = sigmoid(b_k) ∈ (0,1) estricto por construcción (`params.py:193`) |
| **Fórmula canónica** | Estacionariedad GARCH: κ < 1. |
| **Discrepancia** | La cota se respeta **estrictamente** (κ = sigmoid nunca alcanza 1), pero en 2 de 19 bloques (y 7 de 19 con κ≥0.999) el régimen de alta varianza queda con κ = 1 − 1.1e-16 (indistinguible de 1 en doble precisión). Es el régimen degenerado absorbe-outliers (E[D]≈1 día): persistencia GARCH máxima pegada a IGARCH. |
| **Prueba numérica** | `audits/probes/invariants.py` → `c_results.json.C3_kappa_bloques_publicados` |
| **Severidad + impacto** | **S5 — Presentación.** **CONTAMINA ARTEFACTO: NO.** No viola la restricción (la parametrización la garantiza); es una **manifestación numérica del régimen degenerado ya documentado** (A6), no un problema de identificación nuevo. Se registra para que quien lea κ≈1 en el artefacto sepa que es esperado, no un error. |

### MATH-H4 — El panel recomputa argmax y ln(K) en vez de leerlos (observación menor R9)

| Campo | Contenido |
| :-- | :-- |
| **ID** | MATH-H4 |
| **Ubicación** | `panel/components/RegimeHistory.tsx:31,44`; `RegimeCard.tsx:43` |
| **Discrepancia** | El panel calcula `entropyMax = Math.log(K)` y `argmax = xi.indexOf(Math.max(...xi))` localmente, en vez de leer `regime.entropy_max` y `regime.argmax` del artefacto (que sí los publica). |
| **Prueba numérica** | grep sobre `panel/` (sin rolling/smooth); lectura de las tres líneas. |
| **Severidad + impacto** | **S5 — Presentación.** **CONTAMINA ARTEFACTO: NO.** No es look-ahead ni lógica de modelo (son funciones deterministas de datos ya publicados, y coinciden con el artefacto). Es una desviación menor de R9 ("la interfaz no calcula"): dos cómputos duplicados que podrían derivar en silencio si el artefacto cambiara de convención. No hay suavizado cosmético (lo que R9 realmente prohíbe). Observación, no exige acción. |

---

## §10 — Anti-complacencia (obligatorio)

### 10.1 Lista de lo NO verificado

1. **PIT de densidad predictiva OOS global (C.4).** `walkforward.json` no persiste θ por bloque; la densidad predictiva completa no es reconstruible sin re-correr el walk-forward (cómputo pesado, fuera del presupuesto). **Mitigación parcial:** el PIT del bloque 5 reconstruido (F.4b) sí se verificó (KS p=0.455). Queda sin verificar el PIT sobre los 19 bloques juntos.
2. **Test de Hansen (1992) exacto.** No se implementó el procedimiento de bandas con malla de parámetros nuisance; se auditó la implementación **real** (bootstrap-LR, MATH-E2). Verificar Hansen exacto exigiría días de CPU (la propia justificación del proyecto). El conflicto "BIC vs Hansen" del encargo no aplica: el proyecto nunca implementó Hansen, usa bootstrap-LR, y BIC (K=2) y bootstrap-LR (K=2, p≤0.02) **coinciden** en K=2.
3. **Multiplicidad / Deflated Sharpe Ratio (E.5).** Se contó que el proyecto probó ≥6 especificaciones (M0–M5) documentadas más variantes; el número crudo de configuraciones contra el mismo conjunto OOS supera 20 a lo largo del proyecto. **No se calculó el DSR** porque el resultado económico ya es negativo (Test 5 NO SUPERA, `success=false`): deflactar un Sharpe que ya no excluye 0 no cambia el veredicto. Se documenta el riesgo de multiplicidad; no se cuantificó por no ser accionable sobre un resultado nulo.
4. **Recuperación del particle filter y fracdiff (A.3, A.4).** No existen en `src/`; no verificados por inexistencia, no por omisión.
5. **Reproducción bit-exacta del fit Hawkes publicado (D.1).** El refit da un desvío de 5e-4 relativo, atribuido a que el corpus creció de 240 a 245 snapshots tras la publicación (`2026-08-16.json` posterior al run `3b4f1e39b59c`). No se re-corrió sobre el corpus congelado exacto del run (requeriría revertir el estado de `data/raw/headlines/`, prohibido por la Regla Cero). La fórmula y las unidades sí quedaron verificadas; lo no verificado es la igualdad numérica exacta con el corpus histórico.

### 10.2 Autofalsación de los tres hallazgos de mayor severidad

- **MATH-E2:** *¿Qué lo refutaría?* Que LR con 6 y con 20 arranques coincidieran (delta≈0), o que LR_obs estuviera cerca del cuantil nulo (donde 1 unidad de LR sí movería el p). *Intento:* medí ambos y el delta medio es +0.99 (existe) pero LR_obs=305 ≫ q95=18.4 (irrelevante para la decisión). **El hallazgo sobrevive como S2, pero su impacto queda acotado a "no contamina": confirmado.**
- **MATH-D7:** *¿Qué lo refutaría?* Que D_obs/D_crit fuera ~1 (rechazo genuino de forma, no sobre-poder). *Intento:* D_crit=0.0044, ratio 6.6 → el rechazo es de sobre-poder, no de forma grande. **Sobrevive como S5.** *Segundo intento de refutación:* que el proyecto NO lo documentara (sería S2 oculto) — pero A3 sí lo documenta, así que baja a S5 confirmado.
- **MATH-H1:** *¿Qué lo refutaría?* Que `conditional_stats` se calculara sobre `history.parquet` (entonces reconciliaría). *Intento:* rastreé el flujo hasta `today_run.frame` (in-sample) en `run_v3.py:542`; el frame OOS es otro objeto. **Sobrevive como S5.** El valor publicado no es falso; lo no reconciliable es su insumo.

### 10.3 Hallazgos negativos (lo confirmadamente sano, con prueba)

Esto es lo que acota el daño: **el núcleo matemático está limpio.**

- Verosimilitud Hawkes O(n) = doble suma O(n²) a **1e-15** (B1).
- Compensador = cuadratura independiente a **1e-16** (B2).
- Filtro de Hamilton (constante, t, TVTP) = bucle ingenuo a **1e-10 o mejor** (B3); la loglik sale del denominador (verificado, sin segunda pasada).
- MS-GARCH = statsmodels a **3.9e-11** (B4).
- Conteo de BIC exacto en 3 especificaciones (B5).
- **Recuperación de parámetros limpia** en Hawkes (sesgo <0.4 %, cobertura 0.94–0.96) y MS-GARCH (todos dentro de ±3 SE, rango 2 recuperado, clasificación AUC 0.87) (A1, A2).
- **El fix de censura del 2026-08-15 funciona**: producción devuelve n=0.730 donde el ingenuo infla a 0.997 (A1).
- Orientación de la transpuesta correcta **sobre datos publicados** (C2).
- Restricciones GARCH y suma-1 de P por construcción, 0 violaciones en 2000 θ (C1, C3).
- R1 (smoother) bloqueado en el path, incluso anidado; ninguna serie publicada viene del suavizador (C2).
- Branching ratio, cascada, E[D], entropía reconcilian **exacto** desde el artefacto (H1).
- **Reconstrucción independiente del bloque 5 del walk-forward reproduce el número OOS a 3e-5** (F4).
- DM M2-vs-M1 reproducido independiente (p=0.106): la decisión de cerrar M2 se apoya en un p bien calculado (E3).
- run_id/asof/panel sincronizados; panel sin lógica de modelo ni suavizado (H2, R9).

### 10.4 Cuota de honestidad

Tras las ocho fases **no aparece ningún S0 ni ningún S1.** La auditoría no fabricó hallazgos menores para llenar el informe: los cinco reportados son reales y verificados, pero su severidad honesta es S2 (uno) y S5 (cuatro), y **ninguno contamina el artefacto publicado**. El sistema matemático de IRF-N, en su núcleo, está sano. Inflar esto a un hallazgo crítico sería corromper la auditoría; el resultado legítimo es: *el cimiento aguanta.*

### 10.5 Orden de reparación propuesto (solo propuesto — decisión del director)

Ordenado por **dependencia causal**, no por severidad:

1. **MATH-E2 (si se quiere un p fino de K):** igualar `boot_n_starts` a `bic_n_starts` (o subir B) — barato, elimina el sesgo anti-conservador. *No urgente:* la decisión K=2 no cambia. Se hace **primero** solo si el director quiere reportar un p-valor de número de regímenes con magnitud interpretable.
2. **MATH-H1 (auditabilidad):** persistir el frame in-sample que alimenta `conditional_stats`, o documentar explícitamente en el contrato que es in-sample-full (distinto de `history.parquet` OOS). Cambia *cómo se audita*, no ningún número.
3. **MATH-D7 (presentación):** añadir el matiz Lilliefors al warning del KS (el proyecto ya reporta D/D_crit cualitativamente; solo falta decir "p no exacto por parámetros estimados"). Cosmético.
4. **MATH-C3 y MATH-H4:** documentación (κ≈1 esperado en el régimen degenerado; nota R9 sobre argmax/ln K recomputados). Sin acción de código necesaria.

Ninguna reparación cambia el diagnóstico de otra; no hay dependencias causales entre los hallazgos (todos son independientes y de bajo impacto). El orden es por *facilidad y valor de auditabilidad*, no por desbloqueo.

---

*Fin de AUDIT_MATH_v1. Scripts de prueba: `audits/probes/`. Salidas crudas: `audits/probes/out/`. La auditoría no modificó `src/`, `tests/`, `artifacts/` ni `panel/`.*

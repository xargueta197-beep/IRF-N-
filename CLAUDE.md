# CLAUDE.md — Proyecto IRF-N

Este archivo se lee al inicio de **cada** sesión. Si algo en el código contradice este archivo, el código está mal.

## 1. Qué es este proyecto

**IRF-N — Índice de Régimen Filtrado con Noticias.**

Un indicador diario, publicable y auditable, que estima:

- la **probabilidad de estar en cada régimen de mercado** (ξ_{t|t}),
- la **confianza** de esa estimación (entropía),
- la **persistencia esperada** del régimen (duración),
- la **atribución** del movimiento entre precio y noticias,

usando **exclusivamente** información disponible en la fecha de publicación.

Modelo: **Markov-switching GJR-GARCH** (especificación Haas, Mittnik & Paolella 2004) con **probabilidades de transición variables en el tiempo (TVTP)** moduladas por covariables, entre ellas un **índice de presión de noticias** construido como proceso puntual autoexcitante (Hawkes).

## 2. Qué NO es

- **No** es una señal de trading en vivo. No hay ejecución, no hay broker, no hay dinero real.
- **No** es un proyecto de NLP. El sentimiento de titulares es una covariable, no el producto.
- **No** es un dashboard. Si el panel se ve espectacular y la auditoría PIT falla, el proyecto fracasó.
- **No** es una promesa de rendimiento.

## 3. Reparto de roles

- **Xavier** es el director de proyecto. Decide metodología, valida criterios de aceptación, aprueba avances de versión.
- **Claude Code** implementa. Escribe código, tests y documentación técnica.
- **Claude Code no cambia metodología por su cuenta.** Si crees que una decisión de diseño es incorrecta, **detente y dilo antes de implementar**. No la "arregles" silenciosamente.

# LAS 9 REGLAS INNEGOCIABLES

Violación de cualquiera de estas = el código se revierte, no se discute.

### R1 — Jamás se publica el smoother

xi_smoothed / ξ_{t|T} puede existir en código de diagnóstico, marcado @diagnostic_only. **Está prohibido en cualquier ruta que llegue a artifacts/ o a app/.** outputs/publish.py debe lanzar LookAheadViolation si detecta cualquiera de estas claves en el payload: xi_smoothed, smoothed, kim_smoother, xi_tT.

Motivo: el smoother usa el futuro. Produce gráficas hermosas y modelos que fallan en producción.

### R2 — Re-estimación por bloque en walk-forward

**Prohibido estimar una sola vez sobre todo el histórico.** Cada bloque de walk-forward re-estima los parámetros desde cero, usando solo datos de entrenamiento de ese bloque. Mínimo **6 bloques de test**.

### R3 — Toda covariable entra rezagada: x_{t-1}

Todo feature lleva un .shift(1) **explícito y comentado**. En el filtro de Hamilton, la matriz de transición del paso *t* se evalúa en x[t-1], nunca en x[t].

### R4 — Datos macro desde ALFRED (vintages), nunca desde FRED revisado

El M2 de enero que se ve hoy en FRED no es el que se publicó en enero. Entrenar con la serie revisada le da al modelo información que nadie tenía en ese momento.

### R5 — Restricción de identificación estructural

Varianza incondicional **estrictamente creciente**: v₁ < v₂ < ... < v_K, impuesta **en la parametrización**, no corregida post-hoc:

```
v_1 = exp(a_1)
v_k = v_{k-1} + exp(a_k)   # k = 2..K
```

Motivo: sin esto hay label switching y los regímenes se intercambian entre bloques.

### R6 — Multistart obligatorio

20–50 arranques aleatorios en cada estimación. Se conserva el de mayor log-verosimilitud. **Semilla fija y registrada en el artefacto.** Reportar cuántos arranques convergieron al mismo óptimo.

### R7 — Cero pesos asignados a mano

Todo peso se **estima**:

- w_i (impacto de cada indicador macro) → por regresión de impacto, con error estándar.
- β_ij (logit de transición) → por máxima verosimilitud, con error estándar.
- δ (decaimiento del índice de sorpresa) → maximizando la verosimilitud del modelo completo.
- α, β, μ_N (Hawkes) → por MLE.

Si encuentras un número mágico en el código que no viene de config/ ni de una estimación, es un bug.

### R8 — Ninguna versión avanza sin walk-forward documentado

Incluso si el resultado es **negativo**. El reporte se escribe igual. Diseñar el test antes de correrlo es lo que separa investigación de narrativa.

### R9 — La interfaz no calcula nada

app/ **solo lee** de artifacts/. Cero lógica de modelo. Cero suavizado cosmético en la capa de presentación (un rolling(7).mean() "para que se vea mejor" es look-ahead disfrazado de diseño).

# ESPECIFICACIÓN TÉCNICA CANÓNICA

## Capa de observación (Haas et al. 2004 — sin path dependence)

Para cada régimen k, en paralelo, con el mismo retorno observado:

```
ε_{k,t}  = r_t − μ_k
σ²_{k,t} = ω_k + α_k·ε²_{k,t-1} + γ_k·ε²_{k,t-1}·1{ε_{k,t-1} < 0} + β_k·σ²_{k,t-1}
r_t | S_t = k  ~  N(μ_k, σ²_{k,t})      # o t de Student en V1+
```

**No usar colapso de Gray/Klaassen sin discutirlo primero.** Haas es exacto y O(T·K).

## Parametrización sin restricciones (el optimizador trabaja en ℝⁿ)

```
v_1 = exp(a_1);  v_k = v_{k-1} + exp(a_k)      # R5: orden estructural
p_k = sigmoid(b_k)                              # persistencia ∈ (0,1)
(α_k, γ_k, β_k) = p_k * softmax(c_k1, c_k2, c_k3)
ω_k = v_k * (1 − p_k)                           # ω NO es libre
μ_k ∈ ℝ                                         # libre
β_{i,K} = 0 ∀i                                  # identificación del logit multinomial
```

## TVTP

```
p_ij(t) = exp(β'_ij · x_{t-1}) / Σ_m exp(β'_im · x_{t-1})
```

Covariables (todas rezagadas): sma_gap, bb_width_z, slope_2s10y, hy_oas_z, surprise_index, lambda_N_z.

**Arrancar con 2 covariables, no 5.** Penalización L1 sobre β_ij. El λ de la penalización se elige por CV **dentro del bloque de entrenamiento**, jamás mirando el test.

## Filtro de Hamilton (el núcleo)

```
Predicción:    ξ_{t|t-1}(j) = Σ_i p_ij(x_{t-1}) · ξ_{t-1|t-1}(i)
Actualización: ξ_{t|t}(j)   = ξ_{t|t-1}(j)·f(r_t|S_t=j) / Σ_m ξ_{t|t-1}(m)·f(r_t|S_t=m)
Log-verosim.:  log L = Σ_t log[ Σ_m ξ_{t|t-1}(m)·f(r_t|S_t=m) ]
```

**La log-verosimilitud sale del denominador de la actualización.** Si escribes una función compute_loglik separada que recorre los datos otra vez, algo está mal.

**Inicialización en walk-forward:** al empezar un bloque de test, se **arrastra** el último ξ_{t|t} del entrenamiento. **No** se reinicia a la distribución estacionaria.

## Hawkes con marcas

```
λ_N(t) = μ_N + Σ_{t_i < t} α·s_i·exp(−β·(t − t_i))

Recursión:    A_1 = 0;  A_i = exp(−β·Δt_i)·(A_{i-1} + s_{i-1});  λ(t_i) = μ_N + α·A_i
Compensador:  Λ(T) = μ_N·T + (α/β)·Σ_i s_i·(1 − exp(−β·(T − t_i)))
log L = Σ_i log λ(t_i) − Λ(T)
```

**Branching ratio con marcas:** n = α · E[s] / β — **NO** es α/β. Estacionariedad: n < 1.

**Bondad de ajuste:** teorema de re-escalamiento temporal → τ_i = Λ(t_i) debe ser Poisson unitario → KS test sobre interarribos contra Exp(1).

## Índice de sorpresa

```
z_i  = (actual_i − consenso_i) / σ_i          # σ_i específico de ESE indicador
w_i  ← estimado por:  |r_ventana| = a + w_i·|z_i| + u
SI_t = Σ_{i: t_i ≤ t} w_i · z_i · exp(−δ·(t − t_i))     # δ estimado por MLE
```

**SI_t y λ_N(t) entran como covariables SEPARADAS en el logit.** Prohibido colapsarlas en un solo "news score" con pesos inventados.

# TESTS OBLIGATORIOS

Estos corren en cada commit. Si fallan, el trabajo no está terminado.

| Test | Qué verifica |
| :-- | :-- |
| test_prefix_invariance | **El más importante.** Correr el pipeline sobre data[:t] produce exactamente el mismo ξ_{t\|t} que correrlo sobre data[:T]. Si difiere, hay información futura filtrándose. |
| test_hamilton_recovery | Simular MS-GJR-GARCH con parámetros conocidos (N=5000) → estimar → los parámetros verdaderos caen dentro del IC 95%. |
| test_garch_vs_arch | Con K=1, la implementación reproduce a la librería arch con tolerancia numérica. |
| test_no_smoother_in_outputs | Publicar un payload con xi_smoothed lanza LookAheadViolation. |
| test_label_ordering | v₁ < v₂ < ... < v_K tras cada estimación de cada bloque. El régimen "risk-off" no cambia de índice entre bloques. |
| test_hawkes_recovery | Simular Hawkes (Ogata thinning) con (μ, α, β) conocidos → recuperar. KS de re-escalamiento pasa. |

# STACK Y CONVENCIONES

- Python 3.11, venv aislado, pyproject.toml.
- pandas + pyarrow (parquet). scipy.optimize (L-BFGS-B). numpy.
- arch **solo** para el test de sanidad K=1. No se usa en producción.
- numba **solo si el profiling lo justifica**. Perfilar antes de optimizar.
- Interfaz: **Streamlit** + Plotly. Migra a Next.js solo en V4, consumiendo el mismo irfn.json.
- Config: pydantic-settings + YAML en config/. **Cero constantes mágicas fuera de config/.**
- Reproducibilidad: run_id = hash de (config + commit git + semilla). Va dentro de cada artefacto.
- notebooks/ es **solo exploración**. src/ nunca importa de notebooks/.
- Código en inglés, comentarios y documentación en español.
- Nada de emojis en código, commits ni salidas.

# CONTRATO DE SALIDA

artifacts/latest/irfn.json — definido en src/irfn/outputs/schema.py con pydantic. Si el esquema cambia, la interfaz rompe **a propósito**. Campos obligatorios:

run_id, generated_at, git_commit, config_hash, asof, version, model{K, spec, tvtp, covariates, news_layer, estimation}, regime{labels, xi_filtered, entropy, entropy_max, confidence, expected_duration_days, argmax, xi_momentum_5d}, transition_matrix_today, news{surprise_index, lambda_N, lambda_N_z, branching_ratio, expected_cascade, attribution}, conditional_stats, warnings, validation_ref, disclaimer.

# ESTADO ACTUAL DEL PROYECTO

Claude Code: actualiza esta sección al final de cada sesión.

- **Sesión 2026-08-15 (chequeo de sensibilidad del dithering — aviso #5, SPY):**
  - **Aviso #5 de la auditoría pre-corrida RESUELTO para SPY.** El `seendate` de GDELT está
    cuantizado a 15 min (~83% de titulares empatados) y sin dithering U(0,15min) el MLE del
    Hawkes degenera. El auditor pidió confirmar que el óptimo lo fijan los datos y no la
    realización del ruido. Nuevo diagnóstico `scripts/dithering_sensitivity_v3.py`
    (`@diagnostic_only`, NO publica, NO toca `artifacts/`): replica la preparación de datos
    de `run_v3` para SPY (corpus 95,085 titulares, cache de relevancia) y re-ajusta el Hawkes
    variando **solo la semilla del dithering** (5 semillas: 42/1/7/123/2024), con la semilla
    del **multistart FIJA en 42** y `n_starts=30` (R6) — así cualquier variación del óptimo es
    atribuible al de-empate, no a qué arranques se probaron.
  - **Resultado (PASA con matiz):** las cantidades **publicadas** son robustas dentro de su
    propia incertidumbre — `n` (branching ratio) se mueve 0.0018 entre semillas = **0.46 SE de
    `n`**, todas dentro del IC95 [0.731, 0.747]; `mu_N` < 0.62 SE; cascada 3.83-3.86. `alpha` y
    `beta` sí se mueven ~2 SE marginales, PERO **co-movidos ~1:1** (ratio d_alpha/d_beta
    0.90-1.07): es deslizamiento sobre la **cresta de la verosimilitud** (co-identificación del
    kernel exponencial), que `n = alpha·E[s]/beta` absorbe. Conclusión honesta: el MLE NO lee
    ruido del dithering como señal en lo que se reporta; la decisión del director (dithering con
    semilla fija) queda respaldada. NO revierte los avisos #4/#6: `n` sigue siendo cota superior
    cualitativa bajo un kernel que el KS rechaza (stat 0.029, p=0). El matiz clave documentado:
    el criterio literal del auditor (cada parámetro < 1 SE marginal) es demasiado estricto para
    un kernel exponencial con `alpha`/`beta` co-identificados; lo correcto es juzgar la cantidad
    publicada (`n`) contra su propia SE.
  - **Extensiones (a) y (b) EJECUTADAS a petición** (mismo diagnóstico, `@diagnostic_only`):
    (a) **óptimo global** — barriendo la semilla del multistart con el dithering fijo, las 5
    semillas caen en el MISMO óptimo (log-verosimilitud rango 1.4e-07, `n` rango 1.6e-06, 30/30
    arranques): la verosimilitud es unimodal y el multistart de R6 no es el eslabón débil.
    (b) **imputación múltiple (Rubin)** sobre las semillas de dithering — `n` agrupado 0.7399
    (≈ el de semilla fija 0.7388), SE 0.00394 → **0.00403** (+2.1%; r=0.042, el dithering añade
    ~4% a la varianza), IC95 [0.732, 0.748]. El aporte del dithering a la incertidumbre es real
    pero de **segundo orden**. Adoptar el `n`/IC agrupado en la ruta publicada (en vez del de
    semilla fija) es **decisión del director (R3)** porque cambia CÓMO se reporta la
    incertidumbre; con el aporte actual no hay urgencia (más sentido revisitarlo si el kernel
    power-law resuelve el rechazo del KS).
  - **Documentado en:** `reports/dithering_sensitivity_v3.md` (tabla completa + secciones (a) y
    (b)), `reports/auditoria_pre_corrida.md` (aviso #5 marcado RESUELTO + extensiones),
    `reports/validation_v3.md` (subsección "Sensibilidad al dithering" + extensiones).
    Sin cambios en `src/` → sin regresión de tests.

- **Sesión 2026-07-30 (R6 vía paralelización + veredicto macro + corpus GDELT):**
  - **Paralelización del walk-forward (solo velocidad, resultado IDÉNTICO al serial).**
    `validation/walkforward.py::walk_forward` acepta `n_jobs`: reparte los bloques
    (independientes por R2) en un `ProcessPoolExecutor`; cuerpo del bloque extraído a
    `_compute_block` (pura, module-level). BLAS a 1 hilo/proceso. Determinismo
    verificado bit a bit (serial vs paralelo, max |diff|=0.0 en theta/loglik/serie OOS).
    `n_jobs` NO entra en el checkpoint ni en config_hash → serial y paralelo comparten
    checkpoints. Propagado a `ablation.run_ablation`, `run_v2.py --jobs`, `run_v1.py
    ablation --jobs` (`-1` = núcleos-1). Estimador (`estimate.py`/`hamilton.py`) NO
    tocado → tests numéricos intactos (76 passed). Multistart dentro de `fit()` NO
    paralelizado (necesitaría OK del director; titular/kselect siguen a 1 núcleo).
  - **Bug REAL del bloqueo de M3 encontrado y corregido:** no era falta de datos sino un
    gate estricto en `run_v2.py::try_macro_covariates` (`if X_macro.isna().any().any()`).
    El único NaN eran 249 filas de warm-up del z-score de `hy_oas_z` (0 huecos internos;
    `slope_2s10y` 100%). Fix aprobado por el director: el gate permite warm-up INICIAL y
    rechaza solo NaN INTERNOS (R4); la ablación se alinea al tramo común sin NaN (dropna).
  - **Ablación M3 EJECUTADA con R6.** SIN L1 (`run_v2 --jobs -1`, `l1_grid=[0.0]`):
    M2 y M3 salían PEOR que M1 — pero era ARTEFACTO de no regularizar. CON L1 (test justo,
    `scripts/run_m3_l1.py --jobs -1`, grid [0,0.5,2,8,32], 2135 obs OOS, 17 bloques):
    M0=1.3713, M1=1.3321, M2=1.3369, M3=1.3416. DM: M1vsM0 −3.40 p=0.001 (regímenes
    aportan); **M3vsM2 +1.04 p=0.299 → macro NO aporta** (ya no degrada con L1). Eje de
    covariables de transición CERRADO en negativo, justo. `reports/ablation_m3_l1.md`.
  - **Tests de V4 que quedaron R6:** Test 1 (K=2 por BIC y Hansen p=0.02, corrida
    completa 2026-07-22 `v1_kselect.json`); Test 3 (direccional p=0.048, marginal, antes
    0.082); Test 4 (M1 vs M0 DM=−3.15 p=0.002 robusto; M2 vs M1 con L1 p=0.253);
    Test 7 (ablación, cerrado). **Aún `--quick`: Tests 5 (económico) y 6 (Sharpe)** —
    re-correr sus scripts sobre `walkforward_v1.json`. `reports/validation_v4.md`
    reconciliado (resumen, Tests 1/3/4/7, Limitaciones). Hallazgo transversal reforzado
    y ahora R6: **el único aporte robusto es M1>M0 (regímenes); ninguna covariable de
    transición (técnica ni macro) añade valor OOS.**
  - **REGRESIÓN del artefacto publicado (integridad):** `artifacts/latest/irfn.json` y el
    panel están en **V0** (no V3 como decían los docs) desde el 2026-07-16/18 — corridas
    V0 del walk-forward republicaron `latest`. El V3 real (`02db03d3d6d3`) está a salvo en
    `artifacts/runs/` (solo irfn.json+audit.json, sin history.parquet). Decisión: arreglar
    la publicación UNA vez al cerrar V4, no antes (menos churn). NO restaurado.
  - **Corpus GDELT (M5) en construcción.** Decisión final del usuario: acotar a ~238-244
    días (descartó el corpus completo 2017→hoy tras entender que la ablación M5-vs-M4 en
    walk-forward necesita ~7 años: train_years=4+6×6m). Estado al cierre: ~97 días en disco
    (~40%). Alcanza para AJUSTAR/inspeccionar Hawkes (≥200 eventos), NO para la ablación
    completa. Cuello de botella = rate limit de GDELT sobre IP compartida (CGNAT), NO CPU.
    La laptop suspende cada 3h y mata los procesos; se desactivó el standby (powercfg) y se
    relanzó. **Reanudar con `irfn/RESUME_GDELT.md`** (`capture_headlines.py --max-days 180`
    + `scripts/gdelt_watchdog.py` = puntos de control con validación cada 5% + avisos móvil).
  - **Archivos nuevos:** `scripts/run_m3_l1.py` (ablación M3 con L1, no publica),
    `scripts/gdelt_watchdog.py` (checkpoints 5% + integridad), `irfn/RESUME_GDELT.md`.
    pytest fast: 76 passed, 0 failed.

- **Sesión 2026-07-18 (diagnóstico + bug macro):**
  - **Bug de M3 CORREGIDO.** `data/alfred.py::point_in_time_series` deduplicaba el índice ANTES de aplicar el margen de días hábiles; `+ BusinessDay(n)` colapsa publicaciones de fin de semana sobre el mismo día hábil (vie/sáb/dom + 1 hábil = lunes), reintroduciendo duplicados que hacían reventar el `reindex` de `features/macro.py` con `cannot reindex on an axis with duplicate labels`. Confirmado con vintages reales: DGS10/DGS2 traen 12.482/8.890 realtime_start en fin de semana; BAMLH0A0HYM2 producía 11 duplicados. Fix: dedup DESPUÉS del margen (el offset preserva el orden de publicación, `keep="last"` sigue siendo "gana el último estado publicado"). Test de regresión nuevo `test_point_in_time_weekend_collision_no_duplicates` en `tests/test_features.py`. Verificado end-to-end: `macro_features` construye `slope_2s10y` (cobertura 100%) y `hy_oas_z` sobre el calendario real de SPY. **M3 queda desbloqueado** (la ablación macro real aún no se ha corrido). Nota: cobertura no-NaN de `hy_oas_z` = 16,6% — es la ventana rodante de 3 años de FRED para series ICE BofA (documentada en el módulo y en data_audit), no parte del bug.
  - pytest `-m "not slow"`: **65 passed, 0 failed** (64 previos + regresión nueva).
  - **Test 3 (Pesaran-Timmermann) de V4 EJECUTADO.** El bloqueo (`r_pred_mean` no persistido) ya no existía: el walk-forward regenerado en la sesión BTC (`run_id=16d4190d17e2`) sí persiste la columna (`walkforward.py:262`). Resultado SPY: hit rate 55,0% vs 54,1% esperado, PT=1,39, p=0,082 — **sin señal direccional al 5%**, como anticipaba la nota metodológica (regímenes de volatilidad, no de dirección). BTC: test degenerado (signo predicho 100% positivo, coherente con K=1). `reports/validation_v4.md` actualizado (resumen ejecutivo, sección Test 3, limitaciones) y `panel/public/data/validation.json` re-exportado. Sigue siendo `--quick` (n_starts=8): provisional hasta R6.
  - **401 de Trading Economics: NO había key.** La variable existe VACÍA en `.env`; la cadena vacía se colaba como key (`c=` en la URL → 401 en vez del 410 del demo). Fix en `capture_consensus.py::_api_key` (`or "guest:guest"`). El bloqueante de fondo (sin fuente de consenso) no cambia.
  - **Decisiones del director (2026-07-18):** (1) auditar Econoday y FXStreet antes de pagar Trading Economics — al ir a hacerlo se encontró que la auditoría YA existía (`data_audit.md` §8-9, del 2026-07-15, otra parte no documentada de esa sesión): ambas descartadas, no quedan alternativas; (2) **V2 CONGELADA por decisión explícita del director**: la capa de sorpresa queda inactiva por falta de datos, `capture_consensus.py` sigue acumulando hacia adelante, se reabre solo si algún día se paga Trading Economics — este pendiente queda CERRADO, no es deuda activa; (3) **integrar el histórico Wayback de BAMLH0A0HYM2** (`data/raw/recovered/`, 1996-2025, QA'd contra ALFRED) en la capa macro — decisión de metodología aprobada, defensa R4 en `data_audit.md` §3 (serie no revisada); (4) SÍ habrá regla de trading: simple, pre-registrada, presentada al director ANTES de correr el walk-forward económico (R8).
  - Backfill GDELT lanzado en background (tanda de 365 días, reanudable, newest-first desde 2026-07-17).
  - **Prefijo Wayback de BAMLH0A0HYM2 INTEGRADO** (decisión del director aprobada): `src/irfn/data/recovered.py` (empalme como prefijo estricto en formato vintages, lag medido de ALFRED = 0d, en el solape manda ALFRED), flag `macro.use_recovered_prefix: true` en `config/base.yaml` + `config.py`, cableado en `run_v2.py::try_macro_covariates`. Cobertura `hy_oas_z` sobre calendario SPY: 16,6% → **98,6%**. `tests/test_recovered_prefix.py` nuevo (4 tests). `data_audit.md` §3 actualizado. pytest: **69 passed, 0 failed**.
  - **V2 congelada documentada** también en `config/news.yaml` (bloque de `surprise_start_date`).
  - **Walk-forward ECONÓMICO ejecutado (Test 5 de V4, parte económica).** Regla pre-registrada en `docs/preregistro_regla_trading.md` (congelada y aprobada por el director ANTES de correr, R8): des-riesgo binario — efectivo cuando P(alta vol, ξ filtrada de t−1) > 0.5, costo 2 pb/cambio, éxito = IC95 bootstrap del Sharpe de la diferencia vs buy-and-hold excluye 0 por arriba. Implementación: `src/irfn/validation/economic.py` (pura, con `shift(1)` explícito R3), config `v4.economic` en `base.yaml`/`config.py`, `scripts/run_economic_v4.py`, `tests/test_economic.py` (5 tests, incluido no-look-ahead). **Resultado: NO SUPERA** (diff −0.03, IC95 [−0.70, 0.53]; a 0 pb +0.03, a 5 pb −0.12 — veredicto insensible al costo), confirmando la hipótesis nula declarada. Informativo: drawdown 31.4% vs 33.7%, 132 cambios en 9.5 años. BTC no aplica (K=1, regla vacua). `artifacts/latest/validation.json` existe ahora (+ copia en el run); `validation_v4.md` actualizado (resumen, Test 5, limitaciones — también la limitación de Test 7: M3 desbloqueado pendiente de re-correr, M4 congelado, M5 esperando GDELT); panel re-exportado. **El eje económico queda CERRADO con resultado negativo documentado.** pytest final de la sesión: **74 passed, 0 failed**.
  - Diagnóstico completo del proyecto realizado (10 pendientes priorizados, ver conversación); pendientes activos tras esta sesión: 401 de Trading Economics (la URL capturada muestra `c=&` — la key no llega a la petición, por investigar), backfill GDELT (~1 día capturado de ~9 años), decisión del director sobre regla de trading, `r_pred_mean` sin persistir, R6 sin cumplir en ninguna corrida, V4 formal para BTC, despliegue del panel.

- **[Sesión 2026-07-15/16 — línea BTC, documentada retroactivamente: la sesión original no actualizó esta sección]:**
  - **Segunda línea de activo: BTC (BTCUSDT, Binance klines diarios desde 2017-08-17) corrida V0→V3 en paralelo a SPY**, sin tocar los artefactos de SPY. Artefactos en `artifacts/btc/`, reportes `validation_v0_btc.md`, `validation_v1_btc.md`, `ablation_news_btc.md`, `validation_v3_btc.md` y la nota comparativa `reports/spy_vs_btc.md`. Todo `--quick` (NO cumple R6). Se implementó la descarga real de Binance en `src/irfn/data/prices.py` (antes `NotImplementedError`).
  - Hallazgos BTC: V0 pierde contra climatología en ambas métricas (más claro que SPY); **BIC elige K=1 con dist=t para BTC** (divergencia real frente a K=2 de SPY); ambos regímenes persistentes en los 9 bloques. V2/V3 inactivas por los mismos bloqueantes de datos compartidos que SPY.
  - **Hallazgo colateral:** `ALFRED_API_KEY` YA estaba configurada en `.env` (contradecía lo documentado abajo); al probar M3 con la key apareció el bug de `macro.py` (corregido en la sesión 2026-07-18, ver arriba).
  - Query GDELT: narrowing de 7 a 6 términos (se quitó "stock market", razones metodológicas en `config/news.yaml`). Backfill GDELT sigue casi en cero (1 día capturado, 115 titulares puntuados < 200 `min_events_fit`).
  - BTC no tiene validación V4 formal ni tests dedicados en `tests/`.

- **Versión actual: V4 (validación estadística formal + panel público estático) COMPLETA en su parte de codigo, PENDIENTE DE DESPLIEGUE.**
  - `reports/validation_v4.md`: los 7 tests de validación (K, calibración, Pesaran-Timmermann, Diebold-Mariano, walk-forward, bootstrap del Sharpe, ablación de noticias) sobre `run_id=02db03d3d6d3` (V3) / `run_id=c1a8e85ca408` (walk-forward y BIC consumidos, **`--quick`, NO cumple R6**). Hallazgo central: el modelo aporta en densidad predictiva (vence a M0 de un régimen, DM p=0.001) y su valor se concentra en los **6/19 bloques de estrés** del walk-forward (COVID, 2020-21, SVB); no vence a la climatología de régimen en log-loss, y el Sharpe condicional es indistinguible del de comprar-y-mantener SPY. Test 3 (direccional) no ejecutable: `r_pred_mean` no se persiste en `history.parquet`. No existe walk-forward económico (`validation.json`): el proyecto no define una regla de trading.
  - `src/irfn/validation/bootstrap.py`: `optimal_block_length` (Politis-White 2004, selección automática), `stationary_bootstrap`, `sharpe_ci` (con `includes_zero`). `src/irfn/validation/tests_stat.py`: `calibration_metrics` (con el assert que impide pasar `xi_filtered` donde va `xi_predicted`), `brier_score_multiclass`, `log_loss`, `reliability_diagram_data`; PT y DM ampliados con `interpretation`/`statistic`/`better` sin romper `ablation.py`. `tests/test_bootstrap.py` nuevo (cobertura del IC verificada por simulación).
  - `panel/`: panel público **estático** (Next.js 14 App Router, `output: "export"`, Tailwind, Recharts, fuentes Instrument Serif + DM Sans). Tres páginas: `/` (estado de hoy), `/historico` (serie de ξ filtrada + franja de entropía), `/metodologia` (renderiza `docs/metodologia.md`). Lee exclusivamente de `panel/public/data/*.json`, nunca de `artifacts/` en runtime (R9). `npx next build` verificado: compila y exporta sin errores (`panel/out/`). Colores de régimen ajustados de la paleta de marca para ser distinguibles en escala de grises (verificado con luminancia relativa sRGB y luma PIL: grises ~85/122/38 sobre 255, separación mínima 37 puntos).
  - `scripts/export_panel_data.py`: `artifacts/latest/irfn.json` → `panel/public/data/irfn.json`; `artifacts/latest/history.parquet` → `history.json` (**nota:** el encargo pedía `regime_history.parquet`, que no existe; se usa el artefacto real `history.parquet` del walk-forward OOS, documentado en el docstring del script); `reports/validation_v4.md` (resumen ejecutivo) → `validation.json`.
  - `docs/metodologia.md`: documento para audiencia no técnica, con la analogía del médico para filtrada-vs-suavizada y los resultados de validación (incluidos los negativos) en lenguaje llano.
  - **Panel público: NO DESPLEGADO.** No hay URL de Vercel: desplegar a un hosting compartido es una acción de infraestructura visible a terceros y requiere confirmación explícita + credenciales (cuenta/token de Vercel) que esta sesión no tiene. El panel corre localmente (`cd panel && npm install && npm run build`, sirve `panel/out/` con cualquier servidor estático) y quedó verificado en navegador real, incluyendo el estado `confidence === "el modelo no distingue"` (probado con datos sintéticos, revertido antes de cerrar la sesión).
  - **Pendientes que V4 no resuelve:** desplegar el panel (decisión + credenciales del director); persistir `r_pred_mean` en el walk-forward para desbloquear el Test 3 (Pesaran-Timmermann); definir (o descartar explícitamente) una regla de trading para tener un walk-forward económico real; re-correr BIC/walk-forward/Hansen sin `--quick` (R6); M3/M4/M5 de la ablación siguen bloqueados por los mismos datos de siempre (ALFRED, consenso histórico, corpus de titulares).
  - pytest completo (`-m "not slow"`): 64 passed, 0 failed — sin regresiones; no se tocó ningún archivo de `src/` en esta sesión (`bootstrap.py`/`tests_stat.py` se ampliaron en la sesión anterior, de validación).

- **[Contexto histórico V3] V3 (capa de Hawkes / indicador de fragilidad) implementada de punta a punta y VERIFICADA.** `artifacts/latest/irfn.json` tiene `version: "V3"` (`run_id=02db03d3d6d3`, corrida `--quick --no-capture`). El **indicador de Hawkes se publica INACTIVO** por un bloqueante de DATOS honesto, no por un bug: `data/raw/headlines/` está vacío (el backfill de GDELT — `scripts/capture_headlines.py`, 1 pet./5 s, reanudable — aún no ha corrido en esta máquina). Toda la maquinaria corre y se documenta (R8). El código Hawkes es reutilizable tal cual en la Fase 5 del pipeline cuantitativo (API autocontenida, solo numpy+scipy).
- **Implementado y verificado esta sesión (V3):**
  - `src/irfn/models/hawkes_mle.py`: MLE recursivo O(n) (logsumexp, sin doble loop), simulación por thinning de Ogata, branching ratio CON MARCAS `n = alpha*E[s]/beta` (no alpha/beta), cascada esperada `1/(1-n)`, KS de re-escalamiento temporal, SE por hessiano numérico, multistart R6. **Sin dependencias del resto de irfn.**
  - `tests/test_hawkes_recovery.py`: **pasa** (antes `skipped`). Fast: verosimilitud O(n) == referencia O(n²) bit a bit, identidad del re-escalamiento, branching con E[s], `lambda_on_grid`. Slow: recuperación de (μ,α,β) dentro del IC 95% + KS no rechaza Exp(1) sobre datos simulados + control negativo (KS rechaza modelo malo).
  - `src/irfn/data/headlines.py`: ingesta GDELT con TIMESTAMPS AUDITADOS (Trampa 3: `hora_titular` vs `hora_evento`, masa negativa = rojo; resolución del feed), snapshots inmutables, censura del cap documentada (`cap_hit`), `timestamp_audit`.
  - `src/irfn/features/relevance.py`: `s_i = 1 - P(neutral)` con FinBERT local (Trampa 4: SOLO relevancia, jamás dirección; documentado y sin fallback heurístico por R7).
  - `src/irfn/features/hawkes_features.py`: `lambda_N` diaria (causal, NaN fuera de cobertura) y covariable `lambda_N_z` con `shift(1)` explícito (R3), SEPARADA de `surprise_index` (guía 6.7).
  - `outputs/schema.py`/`publish.py`: contrato V3 — `model.hawkes_layer_params` (μ_N/α/β con SE, branching ratio, cascada, KS, cobertura, `reflexive_threshold`, blocker); `news.branching_ratio`/`expected_cascade`/`attribution`.
  - `validation/ablation.py`: `full_ladder_specs` extendido a M0..M5 (M5 = M4 + `lambda_N_z`).
  - `app/pages/3_Noticias.py`: pantalla 3 completa — `lambda_N(t)` con rug de titulares, branching ratio con barra de criticidad y alerta "modo reflexivo" al acercarse a n=1, cascada esperada, tabla de parámetros con KS, atribución de 3 vías. `app/pages/6_Auditoria_PIT.py`: sección 6 (chequeo de timestamps, Trampa 3) — VERDE vacuo hoy (0 titulares emparejados: consenso vacío).
  - `scripts/run_v3.py`: orquestador end-to-end (exit 0, PIT=VERDE). Comparación pre-registrada M5 vs M4 (bloqueada aguas arriba por M3/M4) + diagnóstico pre-declarado M2+H (bloqueado hoy por cobertura de `lambda_N_z`). Reports `reports/ablation_news.md` y `reports/validation_v3.md` escritos (R8).
  - **pytest: 61 passed (fast), 0 failed; los 2 slow de Hawkes pasan.** `test_hawkes_recovery` ya no se salta.
  - **Pendientes que V3 NO resuelve (documentados en el artefacto y en validation_v3.md):** correr el backfill de GDELT para activar el indicador; `ALFRED_API_KEY` + consenso histórico para M5 vs M4; re-correr sin `--quick` para cerrar R6.

- **[Contexto histórico V2] Versión anterior: V2 implementada de punta a punta (plomería completa), pero la capa de sorpresa está PUBLICADA INACTIVA** — no por un bug, sino porque hoy no hay ni un solo día de consenso histórico real (ver hallazgo de esta sesión abajo). `artifacts/latest/irfn.json` tiene `version: "V2"`, `model.news_layer_params.active: false`, `model.news_layer_params.blocker` documentado. V0 cerrado y validado (ver más abajo). **V1 fase 1 (kselect) corrida en modo `--quick` (PROVISIONAL, no cumple R6)**; **V1 fase 2 (ablación walk-forward + DM vs V0) todavía NO se ha corrido con multistart completo** — pendiente, anterior a esta sesión.
- **Hallazgo crítico de esta sesión (auditoría de datos, no metodología):** se investigó Forex Factory como posible fuente alternativa de consenso histórico gratuito. **Descartada con evidencia directa** (`reports/data_audit.md` sección 7): sus términos de servicio (`forexfactory.com/notices`, leídos en vivo) prohíben explícitamente tanto el acceso automatizado fuera de su interfaz como la redistribución de su histórico compilado ("FEED"); además está detrás de un reto anti-bot de Cloudflare activo. Mismo desenlace que Investing.com (sección 4), con evidencia más fuerte. **Conclusión: sigue sin haber ninguna fuente gratuita y honesta de consenso histórico.** La única vía real es pagar Trading Economics (`point-in-time`) o acumular hacia adelante con `scripts/capture_consensus.py` (ya corriendo desde Sesión 0, cero éxitos hasta hoy porque el demo `guest:guest` da 410 Gone).
- **Bloqueantes de datos ACTIVOS ahora mismo (ambos documentados como warnings en el artefacto, ninguno oculto):**
  1. `TRADING_ECONOMICS_API_KEY` vacía → 0 eventos de consenso capturados nunca → `news_layer_params.active=false`, `delta=null`, `surprise_start_date=null` en `config/news.yaml`.
  2. `ALFRED_API_KEY` vacía → M3 (macro: `slope_2s10y`, `hy_oas_z`) sigue bloqueado, igual que en V1 → M4 (que depende de M3 por diseño de la escalera) tampoco puede correr aunque hubiera datos de sorpresa.
  - Ninguno de los dos es un bug ni una decisión de diseño: son huecos de datos reales, documentados con la misma disciplina que V1 ya usaba para `hy_oas_z` (R8: reportar aunque el resultado sea "no se pudo correr").
- **Implementado esta sesión (V2 — plomería, ablación, interfaz, reporte):**
  - `src/irfn/data/calendar.py`: parser point-in-time de los snapshots diarios de `capture_consensus.py` (consenso = de la captura MÁS TEMPRANA que lo vio, nunca reescrito por una captura posterior) + fetcher real (no verificado en vivo, sin key pagada) contra el endpoint `economic_calendar/point-in-time` para backfill futuro.
  - `config/news.yaml`: `surprise_start_date` (null hoy, documentado por qué). `config/base.yaml` + `config.py`: sección `v2` (ladder de covariables M3, bootstrap, delta_mle).
  - `validation/bootstrap.py`: bootstrap ESTACIONARIO (Politis-Romano) implementado — estaba vacío, era un stub.
  - `outputs/schema.py` / `outputs/publish.py`: contrato roto A PROPÓSITO (V2) — `model.news_layer_params` (w_i con SE por indicador, delta, cobertura, bloqueante) y `conditional_stats.*.*` ahora son `{value, ci_low, ci_high}` (IC por bootstrap; `ci_low/ci_high=null` con pocas observaciones, nunca un intervalo inventado).
  - `validation/ablation.py`: `full_ladder_specs` declara M0..M4 (M3=+macro, M4=+surprise_index, acumulativo) SIEMPRE, aunque no corran (R8: diseño escrito antes de correrlo).
  - `audit/pit.py`: `consensus_vintage_ledger` (fecha_evento vs fecha en que el proyecto vio el consenso) y `expanding_window_check` (prefix-invariance de `sigma_i`, análogo a la sección 1 pero para la capa de sorpresa) — ambos en VERDE hoy de forma **vacua** (0 eventos, lo dicen explícitamente, no fingen una verificación sustantiva que no ocurrió).
  - `scripts/run_v2.py`: orquestador real, corrido de punta a punta (`run_id=f9d881066858`, `--quick`). **Decisión de diseño para que el director la revise:** `delta` (decaimiento de SI_t) se estima UNA VEZ por MLE de muestra completa (como `z_window`/`bb_window`: hiperparámetro fijo de ingeniería de features), no re-estimado por bloque de walk-forward; los `beta_ij` del logit (los parámetros de verdad sujetos a R2) SÍ se re-estiman desde cero en cada bloque. Documentado en el docstring de `fit_delta_mle`.
  - `tests/test_prefix_invariance_news.py`: análogo de `test_prefix_invariance_tvtp.py` con `surprise_index` como covariable del pipeline completo (Hamilton+TVTP). Pasa.
  - `app/pages/3_Noticias.py` (reescrita), `4_Retornos_condicionales.py` (IC por celda), `6_Auditoria_PIT.py` (secciones 4-5 nuevas) — probadas en navegador real (`streamlit run`), sin regresiones en 1/2/5.
- **pytest: 52 passed, 1 skipped (`test_hawkes_recovery`, Hawkes es V3), 0 failed** — suite completa (`-m "not slow"`: 45 passed, 1 skipped, 38s; `-m "slow"`: 7 passed, 66m47s). `test_prefix_invariance_news` nuevo, pasa (incluye la capa de noticias en el test PIT más importante del repo). `test_delta_in_mle` (recupera delta vía `surprise_spec`) pasa. Fixture de `test_no_smoother_in_outputs.py` actualizada al contrato V2 (`news_layer_params` + `MetricWithCI`).
- **Siguiente entregable:** (1) decidir con el director si se paga Trading Economics o se investiga Econoday/FXStreet (únicas alternativas no verificadas que quedan); (2) conseguir `ALFRED_API_KEY` (gratis, minutos) para desbloquear M3 — no resuelve M4 pero permite correr la ablación macro real que V1 ya dejó pendiente; (3) correr la fase 2 de V1 (ablación completa, no `--quick`) que quedó pendiente de antes de esta sesión; (4) una vez haya `>= sigma_min_obs` sorpresas por indicador, re-correr `scripts/run_v2.py` sin `--quick` para el veredicto real de M4.
- **Deuda menor (heredada, sin tocar esta sesión):** `use_container_width=True` deprecado por Streamlit; migrar a `width='stretch'`.

# CUANDO TENGAS DUDA

1. Si una decisión afecta la metodología → **pregunta, no decidas.**
2. Si un test falla y "es solo tolerancia numérica" → **no relajes la tolerancia sin explicar por qué.**
3. Si el resultado sale peor de lo esperado → **repórtalo tal cual.** Un resultado negativo documentado vale más que uno positivo maquillado.
4. Si te encuentras escribiendo xi_smoothed fuera de una función @diagnostic_only → **detente.**

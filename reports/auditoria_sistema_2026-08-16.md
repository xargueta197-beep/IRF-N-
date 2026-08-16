# Auditoría de sistema IRF-N — justificada matemáticamente

Fecha: 2026-08-16. Base: artefacto vigente `run_id=7773faae4863` (V3, promovido
atómicamente, asof 2026-08-14), la app Streamlit privada (7 pantallas) y el panel
público estático (`panel/public/data/`). Motivación: los avisos de artefacto que
aparecen en ambas pantallas. Objetivo: no "quitar los avisos", sino clasificarlos
por **severidad matemática** (¿sesgan un número publicado, o son una limitación
honesta?) y proponer la corrección **correcta**, no la cosmética.

Esta auditoría es complementaria a `auditoria_mejoras.md` (recomendaciones de
producto). Aquí el criterio es exclusivamente matemático-estadístico y añade seis
hallazgos que la auditoría de mejoras no cubre (F1–F6).

---

## 1. Taxonomía de los avisos de pantalla por severidad matemática

Los siete avisos del artefacto (`irfn.json::warnings`) más los banners data-driven
de la app (`render_header`, `render_freshness_gap`, `degenerate_regimes`) no son
todos iguales. Los separo en tres clases según su efecto sobre los **números que se
publican** (ξ_{t|t}, n, cascada, retornos condicionales).

### Clase I — SESGAN un número publicado (prioridad real)

| Aviso en pantalla | Cantidad publicada afectada | Raíz matemática |
| :-- | :-- | :-- |
| "corpus 240/998 días … μ_N sesgada a la baja, n inflado … leer n como COTA SUPERIOR" | `branching_ratio` n=0.7388, `expected_cascade` 3.83, `lambda_N` | **Verosimilitud mal especificada** por el compensador sobre días fantasma. Ver F2. |
| Banner "absorbe-outliers … retorno condicional NO es esperado" | `conditional_stats.alta volatilidad.*` (mean_ann −139%, sharpe −5.56 **y su IC**) | Estadísticos sobre ~100 días sueltos sin muestra efectiva. Ver F4. |
| (Sin aviso hoy) IC de `maxdd` | `conditional_stats.*.maxdd.ci_*` | **Bootstrap no válido** para un funcional extremo/no suave. Ver F3. |

### Clase II — LIMITACIÓN honesta, no sesga (correcto reportarlas, no urgente)

| Aviso | Por qué NO sesga |
| :-- | :-- |
| "KS rechaza el kernel exponencial (p=0)" | El ajuste es imperfecto pero el n publicado ya se marca como cota. **Además el aviso sobreestima el problema: ver F1.** |
| "capa de noticias V2 inactiva (0 consenso, ≥30)" | La capa simplemente no entra al modelo; `active:false`. Bloqueo de datos, no error de cálculo. |
| "M3/M5 bloqueado … macro NO aporta (DM p=0.299)" | Resultado negativo documentado (R8). El modelo publicado (M2) no usa esas capas. |
| "lambda_N_z inactiva en el logit (1.7 a de 7 requeridos)" | La covariable no entra; el walk-forward no se encoge (R8). Honesto. |
| Banner régimen degenerado (E[D]≈1 d) | Es el **óptimo global** de la especificación K=2/Normal elegida por BIC, no un bug (diag Fase 7). Estructural. |

### Clase III — Informacional / segundo orden (cosmético)

- "hueco de frescura" (serie OOS termina 44 d antes de asof): correcto por diseño del
  walk-forward por bloques; el banner lo explica.
- "dithering U(0,15min), semilla fija": ya se demostró (dithering_sensitivity_v3) que
  aporta ~4% de varianza a n, de segundo orden; n robusto dentro de su SE.
- "10 días censurados por el cap de GDELT": censura contada, no oculta; subestima
  λ_N en picos puntuales, no sesga el ajuste global.

**Conclusión de la sección:** de los siete avisos, **uno** (corpus 240/998) toca de
verdad un número publicado; el resto son limitaciones honestas bien reportadas. Y
hay **dos sesgos que NINGÚN aviso cubre hoy** (IC de maxdd, IC del régimen
degenerado). El foco matemático debe ir a la Clase I, no a "silenciar" la Clase II.

---

## 2. Hallazgos matemáticos nuevos (F1–F6)

### F1 — El KS del Hawkes está sobre-apoderado; el aviso sobreestima el desajuste
- **Hecho:** KS stat = **0.0289**, p = 1.98e-69, sobre **n = 95 085** eventos.
- **Matemática:** la potencia del KS crece con √n. El error estándar del estadístico
  bajo H0 es ~1/√n ≈ 0.0032, así que un tamaño de efecto **D = 0.029** son ~9 desv.
  estándar → p≈0 **por construcción**, no porque el exponencial esté "muy mal". A
  n=95k, KS rechaza cualquier kernel paramétrico razonable, **incluido el power-law**.
- **Consecuencia 1 (pantalla):** el aviso "KS RECHAZA el kernel" es literalmente
  cierto pero comunica más alarma de la que la evidencia soporta. Lo honesto es
  reportar el **tamaño de efecto** (D=0.029, un desajuste pequeño) además del p-valor.
- **Consecuencia 2 (recomendación previa):** "ajustar power-law para que n sea
  reportable" (auditoria_mejoras rec 3) **no se sostiene tal cual**: el power-law
  también será rechazado a este n. El criterio correcto para elegir kernel no es el
  p-valor del KS sino **AIC / tamaño de efecto del KS / QQ de re-escalamiento**.
  Reformular el objetivo: "¿el power-law reduce D y mejora AIC?", no "¿pasa el KS?".

### F2 — El sesgo del branching ratio es verosimilitud mal especificada, con corrección exacta
- **Hecho:** el Hawkes se ajusta con origen 2023-11-18 y T=998 días, pero solo 240
  días tienen titulares (758 "fantasma"). El compensador publicado es
  Λ(T)=μ_N·T + (α/β)Σ s_i(1−e^{−β(T−t_i)}).
- **Matemática:** Λ(T)=μ_N·T integra la tasa de fondo sobre 758 días con **cero
  eventos observados no porque no ocurrieran, sino porque no se capturaron**. El MLE
  empuja μ_N hacia abajo para no "predecir" eventos en días fantasma, y compensa
  subiendo α; como n=α·E[s]/β, **n queda inflado**. Es exactamente lo que el aviso
  dice — pero lo trata como algo a "leer con cuidado" cuando tiene **corrección
  cerrada**.
- **Corrección correcta (no cosmética):** ajustar sobre el **soporte observado**. Si
  O = ∪ ventanas observadas, el compensador correcto es
  Λ_O = Σ_{ventanas} [ μ_N·|ventana| + (α/β) Σ_{t_i en ventana} s_i(1−e^{−βΔ}) ],
  es decir, **no integrar μ_N sobre los días fantasma**. Es el tratamiento estándar de
  un Hawkes con observación censurada por huecos. Elimina el sesgo de μ_N en lugar de
  etiquetarlo. **Decisión de metodología del director** (cambia cómo se estima), pero
  es la respuesta matemáticamente correcta al aviso de Clase I.

### F3 — El intervalo de confianza de `maxdd` no es válido (bootstrap sobre funcional extremo)
- **Hecho (artefacto):** `conditional_stats.SPY.baja volatilidad.maxdd` =
  value **−0.314**, ci_low **−0.333**, ci_high **−0.102**. El punto estimado está
  pegado al borde inferior del IC y el intervalo es fuertemente asimétrico.
- **Matemática:** el bootstrap estacionario (Politis-White) da IC válidos para
  funcionales **Hadamard-diferenciables** (media, vol, Sharpe → tienen TLC). El
  **máximo drawdown** = −min_t (retorno acumulado_t − máx previo) es un funcional de
  **valor extremo**, no suave, sin TLC estándar: su distribución bootstrap no
  converge a la del estimador. La firma es justo la observada: punto estimado en un
  extremo del intervalo, asimetría severa. **El IC publicado no significa lo que
  parece.**
- **Corrección:** o (a) **no publicar IC para maxdd** (solo el punto, como ya se hace
  con celdas de pocas obs), o (b) usar subsampling / un método de valor extremo. La
  media, vol y Sharpe conservan su IC bootstrap válido. Ingeniería, no metodología de
  modelo (afecta reporte de incertidumbre → confirmar con el director, R3).

### F4 — Los estadísticos condicionales del régimen degenerado no son estimables
- **Hecho:** régimen "alta volatilidad" con E[D]=1.0 día, ~3% de 3425 obs ≈ **100 días
  sueltos, no contiguos**. Se publica mean_ann=−139%, **sharpe=−5.56 con IC
  [−8.53,−2.65]** y maxdd.
- **Matemática:** anualizar una media desde ~100 retornos diarios dispersos multiplica
  por √252 el error; la muestra efectiva para un Sharpe con estructura temporal es
  ~0 (no hay tramos contiguos que bootstrapear en bloque). El **banner** ya prohíbe
  leer el −139% como retorno esperado — correcto — **pero el IC se calcula y se
  publica igual**, dándole una precisión espuria a un número no estimable.
- **Corrección:** cuando E[D]<`DEGENERATE_DURATION_DAYS`, **emitir NaN en los IC** de
  ese régimen (el esquema ya soporta `ci_low/ci_high=null`, se usa para la capa de
  sorpresa con pocas obs). El punto estimado se muestra con el caveat; el IC no finge
  precisión. Cierra el hueco de que el banner advierte pero el JSON no.

### F5 — Publicar M2 en vez de M1 es un costo de varianza sin reducción de sesgo
- **Hecho:** la ablación es tajante — M1>M0 (regímenes aportan, DM p≈0.001) pero
  **ninguna covariable de transición mejora OOS** (M2 vs M1 no significativo con L1;
  M3 vs M2 p=0.299). El artefacto publicado es **M2** (`sma_gap`, `bb_width_z`).
- **Matemática (bias-variance):** bajo la propia evidencia de la ablación, los β del
  logit TVTP no reducen el sesgo del ξ predictivo (no mejoran densidad OOS). Pero sí
  **añaden parámetros → aumentan la varianza de estimación** de ξ_{t|t} en cada
  bloque (más grados de libertad, SE más anchos, la superficie de verosimilitud es
  más plana). En MSE de la probabilidad filtrada, **M1 domina débilmente a M2**: mismo
  sesgo, menor varianza. Refuerza la rec 2 de auditoria_mejoras con un argumento
  formal, no solo "parsimonia". **Decisión del director** (relaja el contrato
  "V1+ ⇒ tvtp=true").

### F6 — Incoherencia de procedencia en el panel PÚBLICO (no protegida)
- **Hecho:** `panel/public/data/irfn.json` es el run vigente (`7773faae4863`, asof
  2026-08-14), pero `panel/public/data/validation.json` tiene
  `generated_at: 2026-07-14` y describe corridas hasta 2026-07-22. La página pública
  "Validación" describe un modelo **más viejo** que la página pública "hoy".
- **Matemática/ingeniería:** `export_panel_data.py` no se re-ejecutó tras la promoción
  V4. La app **privada** tiene guardarrail de coherencia (`render_header` avisa si
  irfn/audit/manifest no comparten run_id), pero el **panel público NO** tiene ese
  chequeo → puede mostrar procedencia mezclada en silencio, que es exactamente la
  regresión que se remedió del lado privado.
- **Corrección:** (a) re-exportar el panel tras cada promoción; (b) añadir a
  `export_panel_data.py` un chequeo espejo de `contract.py` que **falle** si el
  `run_id`/`asof` de las tres salidas JSON no son coherentes. Ingeniería pura.

---

## 3. Recomendaciones priorizadas (con justificación matemática)

| # | Acción | Clase / palanca | Metodología del director |
| :-- | :-- | :-- | :-: |
| A1 | **Ajustar el Hawkes sobre el soporte observado** (Λ restringido, F2) → elimina el sesgo de μ_N/n en lugar de etiquetarlo | Clase I — corrige el único número publicado que hoy está sesgado | **Sí** |
| A2 | **NaN en IC de maxdd** (F3) y **NaN en IC del régimen degenerado** (F4) | Clase I — quita precisión espuria de dos IC inválidos | Confirmar (R3) |
| A3 | **Reportar D del KS, no solo p** (F1); reformular el objetivo del power-law como AIC/tamaño de efecto, no "pasar KS" | Clase II — comunica el desajuste con su magnitud real | No (reporte) |
| A4 | **Evaluar publicar M1** (F5): mismo sesgo, menor varianza que M2 bajo la ablación | Modelo — MSE del ξ publicado | **Sí** |
| A5 | **Coherencia de procedencia en el panel público** (F6): re-exportar + chequeo que falle | Integridad — cierra el hueco que la remediación dejó del lado público | No |
| A6 | Régimen degenerado: probar piso de persistencia / jump / mezcla de colas (ya en auditoria_mejoras rec 1) con criterio pre-registrado E[D]>5 d sin perder log-loss OOS | Modelo — estructural | **Sí** |

**Orden de ejecución sugerido:** A2 y A5 son ingeniería de bajo riesgo y cierran
huecos de honestidad **hoy** (IC inválidos + panel público). A1 y A4 son las palancas
matemáticas reales sobre los números publicados, pero requieren decisión del director
y re-correr walk-forward (R2) + multistart (R6). A3 es un cambio de redacción del
aviso. A6 es la línea de investigación de mayor alcance.

---

## 4. Lo que NO hay que hacer (trampas)

1. **No "quitar" los avisos de Clase II.** Son la evidencia de que el proyecto reporta
   sus límites (R8). Silenciarlos sería el fracaso que CLAUDE.md §2.3 describe.
2. **No perseguir "que pase el KS" con el power-law** (F1): a n=95k es inalcanzable y
   no es el criterio correcto.
3. **No leer el −139%/año del régimen degenerado como predicción** (F4): ni con IC ni
   sin él. El banner es correcto; falta solo apagar el IC.
4. **No re-parametrizar el régimen degenerado en silencio** (A6): el diag Fase 7
   muestra que es el óptimo global; cualquier cambio es decisión de metodología y
   exige re-validar.

---

## Resumen ejecutivo

El sistema es **honesto**: siete avisos, y solo uno (corpus 240/998, F2) sesga de
verdad un número publicado. La contribución de esta auditoría son tres sesgos/errores
de incertidumbre que **ningún aviso cubría** (F2 tiene corrección cerrada, F3/F4 son
IC inválidos que se están publicando con precisión espuria) más una incoherencia de
procedencia en el panel público (F6) y un argumento formal para simplificar a M1
(F5). Las correcciones de bajo riesgo e inmediatas son **A2** (apagar dos IC no
válidos) y **A5** (coherencia del panel público). Las de mayor palanca matemática
—**A1** (Hawkes sobre soporte observado) y **A4** (M1 vs M2)— son decisiones de
metodología del director y exigen re-correr walk-forward y multistart.

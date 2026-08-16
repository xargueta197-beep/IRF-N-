# Auditoria pre-corrida — veredictos y recomendaciones sobre los avisos

fecha: 2026-08-14 (correcciones aplicadas 2026-08-15)  |  run auditada: `190ff64f0511` (V3, SPY)  |  auditor: Claude Code
alcance: revisar cada AVISO emitido antes de la corrida, dar veredicto (valido / obsoleto / requiere accion)
y una recomendacion accionable, SIN cambiar metodologia (CLAUDE.md, reparto de roles: se senala, no se decide).

> ## Estado de las correcciones (2026-08-15)
> - **Aviso #1 (emisor obsoleto): CORREGIDO en codigo.** `scripts/run_v3.py` tenia un string
>   hardcodeado (`macro_blocker`, lineas 339-342) que afirmaba "ALFRED_API_KEY sin configurar"
>   sin comprobarlo nunca. Reemplazado por `_macro_status()`, que en tiempo de ejecucion (1)
>   comprueba la key con el mismo mecanismo que `data/alfred.py` (dotenv + entorno) y (2) lee el
>   DM de M3 vs M2 del archivo persistido `reports/ablation_m3_l1.json` (NO se hardcodea el numero).
>   Corregidos ademas: el warning del artefacto, el `ctx` del reporte, la intro de `ablation_news.md`
>   y el docstring del modulo. Suite: **110 passed, 0 failed**. Salida real ahora:
>   "M3 (macro): NO es el bloqueante: ALFRED_API_KEY esta configurada y la ablacion M3 con L1 ya se
>   corrio (M3 vs M2 DM=1.038, p=0.299 -> la macro NO aporta)...".
> - **Matematica del indicador (Hawkes): VERIFICADA, sin bug.** Ver seccion nueva "Verificacion
>   matematica" al final. Bajo la regla de no-invenciones NO se fabrico ningun arreglo: el codigo
>   reproduce el spec canonico a precision de maquina. `n=0.9994` es un estimado REAL (artefacto de
>   dias fantasma), no un error numerico.
> - **BANDERA A (n/cascada): RESUELTA (Partes A+B, aprobadas por el director 2026-08-15).**
>   - **Parte A (raiz):** el Hawkes ahora se ajusta sobre el TIEMPO OBSERVADO (241 d), no el span
>     calendario (998 d): `hf.compress_to_observed_time()` excinde los 758 dias fantasma. Resultado
>     REAL sobre el corpus: **n = 0.9994 -> 0.7388** (mu_N 0.0783 -> 103.04), cascada **1673.6 -> 3.83**
>     (finita, interpretable). Revierte la decision del 2026-08-14 (con OK explicito del director).
>   - **Parte B (reporte honesto):** IC del branching ratio por metodo delta (**IC95 [0.731, 0.747]**),
>     mostrado junto al valor en artefacto y pantalla 3; regla de censura (IC superior >= 0.95 ->
>     "cascada no acotada", config `news.hawkes.cascade_ci_trigger`). Aqui el IC superior (0.747) esta
>     lejos del disparador, asi que la cascada se publica como 3.83.
>   - Re-corrida unica `run_v3 --quick --no-capture` hecha: `run_id=dbdc61daa50c`, PIT=VERDE. Artefacto
>     y reportes en disco YA reflejan emisor corregido + n corregido + IC + censura. Suite: 117 passed.
>   - Trazabilidad antes/despues: en el log de la corrida y en `validation_v3.md` (tabla dedicada).
> - **Aviso #5 (sensibilidad del dithering): RESUELTO (SPY, 2026-08-15).** Se re-ajusto el Hawkes
>   variando SOLO la semilla del dithering (5 semillas: 42/1/7/123/2024), con la semilla del
>   multistart FIJA en 42 y `n_starts=30` (R6), para aislar el ruido de de-empate. Resultado:
>   **la cantidad publicada es robusta** -- `n` se mueve 0.0018 en total (0.46 SE de `n`, todas
>   dentro del IC95 [0.731, 0.747]); `mu_N` < 0.62 SE; cascada 3.83-3.86. `alpha` y `beta` si se
>   mueven ~2 SE marginales, pero **co-movidos ~1:1** (ratio d_alpha/d_beta 0.90-1.07): es
>   deslizamiento sobre la cresta de la verosimilitud (co-identificacion del kernel exponencial),
>   que `n = alpha*E[s]/beta` absorbe. Veredicto: **PASA con matiz** -- el MLE no lee ruido del
>   dithering como senal en lo que se reporta. Detalle en `reports/dithering_sensitivity_v3.md`;
>   diagnostico reproducible en `scripts/dithering_sensitivity_v3.py` (`@diagnostic_only`, no publica).
>   **Extensiones (ejecutadas a peticion, mismo diagnostico):** (a) barrido de la semilla del
>   MULTISTART con el dithering fijo -> las 5 semillas caen en el MISMO optimo (log-verosimilitud
>   rango 1.4e-07, n rango 1.6e-06, 30/30 arranques): el optimo del MLE es GLOBAL, R6 no es el
>   eslabon debil. (b) agrupacion de `n` por imputacion multiple (Rubin) sobre las semillas de
>   dithering -> `n` agrupado 0.7399, SE sube de 0.00394 a 0.00403 (+2.1%; r=0.042, el dithering
>   añade ~4% a la varianza), IC95 [0.732, 0.748]: fuente real pero de SEGUNDO ORDEN. Adoptar el
>   `n`/IC agrupado en ruta publicada seria decision del director (R3, cambia como se reporta la
>   incertidumbre); con el aporte actual no hay urgencia.
> - **Pendiente (decisiones del director, no tocadas):** kernel power-law (Opcion C, el KS sigue
>   rechazando el exponencial -- es mala especificacion del kernel, no la ventana); Opcion E (n como
>   diagnostico) -- el director pidio evaluarla despues de ver n corregido.

> Convencion de veredictos:
> - **OBSOLETO** = el aviso ya no es cierto; corregir el mensaje antes de reportarlo al director.
> - **VALIDO (aceptar)** = honesto y correcto; la accion es documentar y seguir.
> - **VALIDO (accion)** = cierto, pero pide una accion concreta antes o durante la proxima corrida.
> - **BANDERA** = hallazgo del auditor no incluido en los avisos; requiere decision del director.

---

## Resumen ejecutivo

| # | Aviso | Veredicto | Accion prioritaria |
| :-- | :-- | :-- | :-- |
| 1 | M3 bloqueado por `ALFRED_API_KEY` sin configurar | **OBSOLETO** | Corregir el banner: la key existe y autentica. M3 no esta bloqueado por datos. |
| 2 | M2+H bloqueado: cobertura `lambda_N_z` 1.6a < 7.0a | VALIDO (aceptar) | No encoger la malla (R8). Seguir backfill. Verificar el 1.6a vs 240 dias. |
| 3 | 10 dias censurados por el cap de GDELT | VALIDO (aceptar) | Aceptable y contado. Opcional: re-capturar esos 10 dias en ventanas finas. |
| 4 | Corpus 240/998; `mu_N` sesgado bajo, `n` inflado | VALIDO (accion) | Ver BANDERA A: `n=0.9994` esta PEGADO a la frontera; no publicar cascada como punto. |
| 5 | `seendate` cuantizado 15 min; dithering aplicado | VALIDO (accion) -> **RESUELTO** | Chequeo hecho (5 semillas, SPY): `n`/cascada/`mu_N` robustos dentro de su SE; solo alpha/beta co-mueven sobre la cresta. `reports/dithering_sensitivity_v3.md`. |
| 6 | KS rechaza el kernel exponencial | VALIDO (aceptar) | Coherente con #4: leer `n`/cascada como ordinal, no cardinal. Power-law a futuro. |
| 7 | Capa V2 (sorpresa) inactiva: 0 eventos (>=30) | VALIDO (aceptar) | V2 congelada por el director. Dejar inactiva; el capturador acumula hacia adelante. |

**Lo mas importante para el director, en una linea:** ninguno de los avisos invalida la corrida — pero (a) el aviso #1 es un mensaje muerto que hay que apagar para no reportar un bloqueante inexistente, y (b) el numero `n=0.9994` y la "cascada esperada = 1673" NO deben viajar como cantidades: son un estimador en la frontera de estacionariedad, consecuencia directa del sesgo del aviso #4.

---

## 1. M3 (macro) bloqueado por `ALFRED_API_KEY` sin configurar — **OBSOLETO**

**Evidencia directa (esta auditoria):**
- `.env` tiene `ALFRED_API_KEY` NO vacia, formato correcto (32 hex).
- Sonda en vivo: `GET api.stlouisfed.org/fred/series?series_id=DGS10` -> **HTTP 200** con vintages reales (`realtime_start=2026-08-14`). La key autentica.
- Historia del repo: el bloqueo de M3 NUNCA fue la key — fueron dos bugs de codigo ya corregidos: dedup antes del margen en `data/alfred.py` (2026-07-18) y el gate `isna().any().any()` en `run_v2.py` (2026-07-30). Con ambos corregidos, **la ablacion M3 con L1 YA se corrio**: M3 vs M2 DM=+1.04, p=0.299 -> macro no aporta (negativo, documentado en `reports/ablation_m3_l1.md`).

**Conclusion:** el aviso mezcla dos cosas y ambas estan mal hoy: (i) la key existe; (ii) M3 no esta "sin correr", esta CERRADO en negativo. El texto "mismo bloqueante desde V1" es un arrastre de plantilla.

**Recomendacion:**
1. Apagar/renovar la cadena del banner. La logica que lo emite trata `ALFRED_API_KEY` vacia como bloqueante; hoy no lo esta, asi que o el chequeo mira otra variable (p.ej. una copia en el entorno del proceso, no en `.env`) o el string esta hardcodeado. Localizar el emisor (probable `scripts/run_v3.py` / `ablation_news.md` writer) y hacer que refleje el estado real: "M3 corrido, macro no aporta (p=0.299)".
2. Aclarar el encadenamiento: M4/M5 NO dependen de M3 por falta de la key macro; dependen del **consenso historico** (aviso #7), que si sigue bloqueado. No atribuir a ALFRED un bloqueo que es de Trading Economics.

---

## 2. M2+H bloqueado: cobertura de `lambda_N_z` (1.6a) < 7.0a requeridos — **VALIDO (aceptar)**

El walk-forward pre-registrado exige train 4a + 6 bloques de 6m = 7.0a. La cobertura alineada no llega. Esto es exactamente lo que R8 pide reportar: el test se disena antes de correrlo y, si no hay datos, se dice "no se pudo correr", no se afloja el diseno.

**Recomendacion:**
1. **No encoger la malla de bloques** para forzar un veredicto (R8). Correcto tal como esta.
2. **Consistencia a verificar:** el aviso dice "muestra alineada de 1.6 anios" pero el corpus son 240 dias (~0.66a) dentro de un span de 998 (~2.73a). Confirmar que el 1.6a es la superposicion real `lambda_N_z` x retornos (no un forward-fill que infle la cobertura aparente). Si el 1.6a viene de rellenar dias fantasma, esta contaminado por el mismo sesgo del #4 y no cuenta como cobertura efectiva.
3. Mantener el diagnostico M2+H **pre-declarado** con su compromiso (DM p<0.10) y correrlo cuando el backfill alcance. Sigue siendo la pregunta debil (aporte sobre el TVTP tecnico), no sustituto de M5 vs M4.

---

## 3. 10 dias censurados por el cap del API de GDELT — **VALIDO (aceptar)**

Censura contada, no oculta: `lambda_N` queda subestimada en esos picos y esta marcada (`cap_hit`). Es el manejo honesto.

**Recomendacion:** aceptar. Opcional y de baja prioridad: re-capturar esos 10 dias densos con la tecnica ya probada para `2026-01-13` (`capture_headlines.py --since D --until D` en ventanas de 3h, sin nada mas compitiendo por la IP). Reduce la subestimacion en los picos, que es justo donde `lambda_N` importa. No es bloqueante.

---

## 4. Corpus 240/998; `mu_N` sesgado a la baja, `n` inflado — **VALIDO (accion)** + **BANDERA A**

El Hawkes se ajusta sobre todo el span (origen 2023-11-18) con `Lambda(T)=mu_N*T` integrando 758 dias sin titulares capturados (fantasma, no vacios). Decision del director: ajustar sobre el span completo y documentar. Correcto como decision.

**BANDERA A (hallazgo del auditor, NO estaba en los avisos):**
El resultado de ese sesgo no es "un poco de sesgo": es un estimador **pegado a la frontera de estacionariedad**.
- `n = 0.9994` (a 6 diezmilesimas de 1).
- `E[hijos] = 1/(1-n) = 1673.6`.

Un `n` a esa distancia de 1 no es "casi critico y estacionario"; es numericamente indistinguible del limite y su transformada `1/(1-n)` es **inestable** (un cambio de 0.0004 en `n` mueve la cascada de 1673 a infinito). Leerlo como "1673 hijos esperados" es reportar ruido como cantidad. El propio KS que rechaza (#6) confirma que el kernel no captura la estructura, asi que ese `n` no es interpretable como excitacion real.

**Recomendacion:**
1. Publicar `n` y `expected_cascade` **como cota superior cualitativa, no como punto**: en el artefacto y en pantalla 3, `n ~ 1 (frontera; cota superior de excitacion bajo kernel exponencial mal ajustado)`, y NO imprimir "cascada = 1673.6" como si fuera un pronostico. El caveat ya existe en texto; falta que el NUMERO no se muestre con 4 decimales de falsa precision.
2. **Contencion (tranquilizador para el director):** este sesgo NO contamina ningun resultado validado. `lambda_N_z` no entra en ningun logit aceptado — M2+H y M5 estan bloqueados (avisos #2 y #7). El Hawkes hoy es diagnostico/inspeccion, no covariable publicada. El dano del sesgo esta confinado a una pantalla informativa.
3. Diagnostico barato que respalda la decision del director sin cambiarla: reportar tambien el `n` ajustado SOLO sobre el sub-span cubierto (como sensibilidad `@diagnostic_only`, sin publicar), para cuantificar cuanto del `0.9994` es sesgo de dias fantasma. Si al quitar los fantasmas `n` cae claramente por debajo de 1, confirma que el `0.9994` es artefacto del compensador y refuerza leerlo como cota superior. Presentar como diagnostico, no como cambio de metodologia (pregunta al director antes de que entre a cualquier ruta publicada).

---

## 5. `seendate` cuantizado a 15 min; dithering U(0,15min) aplicado — **VALIDO (accion)**

~83% de titulares empatados; sin dithering el MLE degenera. Supuesto: llegada uniforme dentro del bin. Semilla fija. Decision del director. Metodologicamente defendible.

**Recomendacion:** un unico chequeo de robustez antes de fiarse del MLE, dado que el 83% de los datos recibe ruido inyectado: **re-ajustar con 2-3 semillas alternativas** y confirmar que `(mu_N, alpha, beta)` se mueven dentro de sus SE. Si el optimo salta con la semilla, el estimador esta dominado por el dithering y no por los datos — hay que saberlo antes de leer nada del Hawkes. Es barato (el fit es O(n) y ya esta escrito) y honesto. No cambia metodologia; solo mide la sensibilidad de una decision ya tomada.

---

## 6. KS de re-escalamiento rechaza el kernel exponencial — **VALIDO (aceptar)**

stat=0.0848, p=0.000000, n=95084. Se reporta tal cual, no se fuerza el modelo (guia 6.6). Es el manejo correcto: el ajuste es imperfecto y se dice.

**Recomendacion:** aceptar y encadenar con #4 — mientras el KS rechace, `n`, `lambda_N` y la cascada son aproximaciones bajo un kernel que sabemos mal especificado, luego **ordinales, no cardinales**. Dejar el kernel power-law como candidato de version futura (guia 6.6). No es deuda que bloquee V3; es una limitacion declarada.

> Nota estadistica menor: con n=95084 el KS rechaza casi cualquier desviacion practica. El p=0.000000 confirma "el exponencial no es el generador", pero la magnitud relevante es el **stat=0.085**, no el p. Reportar el stat como la medida de cuan lejos esta el ajuste evita sobre-dramatizar un p que la muestra gigante vuelve inevitable.

---

## 7. Capa V2 (sorpresa) inactiva: 0 eventos de consenso (>=30) — **VALIDO (aceptar)**

Sin consenso historico no hay `z_i`, sin `z_i` no hay `SI_t`, sin `SI_t` no hay capa de sorpresa. Mismo bloqueante que V2, ya CONGELADA por decision explicita del director (2026-07-18): se reabre solo si se paga Trading Economics; `capture_consensus.py` acumula hacia adelante.

**Recomendacion:** aceptar; no es deuda activa, es un pendiente cerrado por decision. Confirmar que el capturador hacia adelante sigue vivo (para que el dia que se decida pagar, el prefijo exista). No atribuir este bloqueo a ALFRED (ver #1): son fuentes distintas.

---

## Banderas del auditor (no estaban en los avisos)

- **BANDERA A** (detallada en #4): `n=0.9994` / cascada 1673 son frontera, no cantidades. No publicar con falsa precision.
- **BANDERA B — coherencia del banner:** el aviso #1 demuestra que el generador de avisos puede emitir bloqueantes muertos. Recomiendo una revision del emisor de warnings para que lea estado real (key -> sonda; M3 -> ultimo resultado de ablacion) en vez de plantillas heredadas. Un banner que miente en 1 de 7 lineas erosiona la confianza en los otros 6, que si son correctos.
- **BANDERA C — R6 en el Hawkes:** multistart 15/30 al mismo optimo cumple R6 (20-50 arranques), pero conviene registrar la semilla en el artefacto (R6 lo exige explicitamente). Verificar que `run_id=190ff64f0511` la lleva.

---

## Veredicto global

La corrida es **publicable como V3 con el Hawkes INACTIVO**, que es su estado declarado. Los 7 avisos son en su mayoria honestos y correctos; el trabajo de esta auditoria deja tres acciones concretas antes de reportar al director:

1. **Apagar el aviso #1** (ALFRED no bloquea nada; corregir el mensaje).
2. **Degradar `n`/cascada de numero a cota cualitativa** (#4/BANDERA A).
3. **Un chequeo de sensibilidad del dithering** (#5) barato que blinda el MLE. **HECHO (2026-08-15, SPY):**
   la cantidad publicada (`n`, cascada, `mu_N`) es robusta a la semilla del dithering dentro de su
   propia SE; PASA con matiz (`reports/dithering_sensitivity_v3.md`).

Ninguna toca metodologia; las tres son de honestidad de reporte, que es exactamente lo que las 9 reglas priorizan sobre cualquier grafica bonita.

---

## Verificacion matematica del indicador (Hawkes) — 2026-08-15

A peticion explicita del director ("rigor total en las matematicas del indicador, sin invenciones"),
se auditaron `src/irfn/models/hawkes_mle.py` y `src/irfn/features/hawkes_features.py` contra el spec
canonico de CLAUDE.md, y se verificaron EMPIRICAMENTE (no solo por inspeccion) contra referencias de
fuerza bruta O(n^2) e integracion numerica. Semilla fija, 200 eventos sinteticos.

| Cantidad | Implementacion | Referencia independiente | max error |
| :-- | :-- | :-- | --: |
| `A_i` (estado de decaimiento) | recursion O(n) en log-dominio | suma directa O(n^2) | 2.7e-14 |
| `lambda(t_i)` | `mu + alpha*A_i` | `mu + alpha*A_brute` | 3.2e-14 |
| `Lambda(T)` (compensador) | formula analitica | cuadratura fina (4e5 nodos) | 3.7e-6 (err. de cuadratura) |
| `log L` | `sum log lambda - Lambda(T)` | definicion directa | 8.5e-14 |
| `tau_i = Lambda(t_i)` (re-escalamiento) | identidad `mu*t + (a/b)(cumS - A)` | compensador evaluado en cada `t_i` | 4.3e-14 |
| `n = alpha*E[s]/beta` | con marcas | verificado != `alpha/beta` | exacto |

Ademas: recuperacion por MLE sobre datos simulados (Ogata thinning, parametros conocidos) recupera el
branching ratio identificado, y el KS de re-escalamiento NO rechaza sobre datos bien especificados
(control positivo). Todo coincide con el spec.

**Veredicto matematico: el indicador es CORRECTO.** No hay bug que arreglar; bajo la regla de
no-invenciones no se fabrica un arreglo inexistente. En particular:

- El branching ratio esta bien: `n = alpha*E[s]/beta`, NO `alpha/beta` (la trampa de las marcas).
- El compensador integra sobre TODA la ventana `[0, T]`, incluidos los 758 dias sin titulares
  capturados. Ahi nace, matematicamente, el sesgo del aviso #4: `Lambda(T)=mu_N*T` reparte la masa
  del compensador sobre dias fantasma, empujando `mu_N` a la baja y `alpha/beta` (y por tanto `n`)
  al alza. **Es una consecuencia CORRECTA de la ecuacion aplicada a datos con huecos, no un defecto
  del codigo.** El `n=0.9994` es el MLE honesto de esa situacion.
- `stationary = (n < 1.0)`: con `n=0.9994` el modelo pasa como estacionario y se publica. Aqui vive
  BANDERA A: matematicamente `1/(1-n)` es correcto pero numericamente inestable a esa distancia de 1.
  El manejo (mostrarlo como cota superior + alerta "modo reflexivo" en pantalla 3 cuando `n` se acerca
  a `reflexive_threshold`) es una decision del director, no un arreglo de codigo.

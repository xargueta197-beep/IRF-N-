# IRF-N — Índice de Régimen Filtrado con Noticias

Un indicador diario, publicable y auditable, que estima la **probabilidad de
estar en cada régimen de mercado**, la **confianza** de esa estimación, la
**persistencia esperada** del régimen, y la **atribución** del movimiento
entre precio y noticias — usando exclusivamente información disponible en la
fecha de publicación.

Modelo: Markov-switching GJR-GARCH (Haas, Mittnik & Paolella 2004) con
probabilidades de transición variables en el tiempo (TVTP), moduladas por un
índice de presión de noticias construido como proceso puntual autoexcitante
(Hawkes).

No es una señal de trading en vivo. No es una promesa de rendimiento. Ver
`CLAUDE.md` para la especificación técnica completa y las 9 reglas
innegociables que gobiernan este repo.

## Estado actual

**V4 — validación estadística formal.** Modelo completo y vivo: MS-GJR-GARCH
K=2 con filtro de Hamilton, TVTP y capa Hawkes. Indicador publicado
atómicamente en `artifacts/latest/` (SPY `run_id=75f650b1d59d`, asof 2026-08-19, R6). El detalle
por sesión vive en "ESTADO ACTUAL DEL PROYECTO" (`CLAUDE.md`); la validación
formal en `reports/validation_v4.md`.

### Estado de módulos (cierre documental, 2026-08-16)

Escalera de covariables M0→M5 (ver `reports/validation_v4.md`, Test 7). El único
aporte robusto OOS son los **regímenes de volatilidad** (M1>M0); ninguna
covariable de transición añade valor OOS distinguible.

| Módulo | Qué es | Estado | Motivo |
| :-- | :-- | :-- | :-- |
| **M1** (regímenes) | HMM K=2, P constante | **MODELO DE PRODUCCIÓN — PUBLICADO** (SPY `75f650b1d59d`, asof 2026-08-19; BTC `d7dca6e40eeb`, K=1, asof 2026-08-21) | Único aporte robusto (DM vs M0 p=0.001). Mismo sesgo que M2 con menor varianza (bias-variance). El contrato ya NO exige tvtp para V1+ (`outputs/contract.py`). |
| **M2** (TVTP técnico) | + sma_gap, bb_width_z en el logit | Reemplazado por M1 (era `7c44a7fac16d`) | DM M2 vs M1 p=0.106: no aporta OOS distinguible. Válido bajo el contrato, pero A4 publica M1. |
| **M3** (macro) | + slope_2s10y, hy_oas_z | **CERRADO POR INEFICIENCIA OOS** | Con L1 canónica, DM M3 vs M2 p=0.299: no aporta (`reports/ablation_m3_l1.md`). |
| **V2 / M4** (sorpresa) | Índice de sorpresa SI_t (consenso point-in-time) | **CERRADO DEFINITIVAMENTE** (decisión del director 2026-08-19, `reports/cierre_m4_m5_2026-08-19.md`) | No se paga Trading Economics. Base: (a) el eje de covariables de transición ya salió negativo con test formal (M2 **peor** que M1, DM p=0.025; M3 no aporta, p=0.299); (b) bias-variance/potencia en contra; (c) sin fuente gratuita de consenso (4 descartadas, `reports/data_audit.md`). Reabre solo si `check_reopen_status.py` cumple el umbral gratis. |
| **M5** (GDELT/Hawkes como covariable) | λ_N(t) del Hawkes en el logit de transición | **CERRADO DEFINITIVAMENTE** (decisión del director 2026-08-19, `reports/cierre_m4_m5_2026-08-19.md`) | No se persigue el backfill de ~6 años (corpus ~240/2557 días; cuello = rate limit IP compartida). Mismo caso estadístico que M4. La capa Hawkes SÍ se publica como **indicador standalone** (n, cascada, KS + dos bandas de sensibilidad), no como covariable `lambda_N_z` del logit. Reabre solo si el corpus alcanza el umbral gratis. |

**Nota de transición (A4) — MIGRACIÓN COMPLETADA 2026-08-16.** El contrato permite M1
como modelo de producción válido (bias-variance) desde el commit `cd0e45b`. La
migración del artefacto publicado se ejecutó el 2026-08-16 (autorización explícita del
director): `artifacts/latest/` es ahora **M1** (`run_id=3b4f1e39b59c`, K=2, matriz de
transición constante, `covariates=[]`, `tvtp=false`), reemplazando al M2 saliente
(`7c44a7fac16d`). La migración se hizo con `run_v3 --publish-m1` + `promote.py`
(atómico), re-validación formal en `reports/validation_v4.md` y panel re-exportado
(`stale=false`). Histórico de la decisión: contrato primero (`cd0e45b`), migración del
artefacto después (`3b4f1e39b59c`). **Actualización:** el artefacto SPY publicado hoy es
`75f650b1d59d` (mismo modelo M1, datos frescos asof 2026-08-19; re-validación F6 2026-08-20
con `stale=false`); BTC refrescado a `d7dca6e40eeb` (K=1, asof 2026-08-21).

**A1 (kernel Hawkes).** El sesgo de "días fantasma" está **cerrado**: el Hawkes se
ajusta sobre tiempo observado (decisión del director 2026-08-15; `run_v3.py`,
μ_N corregido, n=0.7388). El único eje abierto es kernel exponencial vs power-law:
el KS rechaza el exponencial (D=0.029, pequeño, dominado por n≈95k), el power-law
gana AIC pero tampoco pasa el KS → se mantiene el exponencial con el caveat honesto
del KS. Ver `reports/nota_decisiones_director_2026-08-16.md`.

**A6 (régimen degenerado).** El segundo régimen es un absorbe-outliers (E[D]≈1 día):
es el **óptimo global**, no un bug. Decisión informada del director: se conserva la
estimación puntual **sin parches de persistencia artificiales** (piso/jump/mezcla);
se reporta con honestidad (IC condicional suprimido + banner en la app). NO es una
omisión: es una decisión documentada de no forzar el modelo.

## Instalación

Requiere Python 3.11.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install -e ".[dev]"
cp .env.example .env            # y rellenar las API keys que se tengan
```

## Cómo correr

```bash
pytest                                    # corre los tests (6 obligatorios; hoy en rojo/skip)
python scripts/run_pipeline.py --dry-run  # imprime el plan de ejecución sin ejecutar
python scripts/run_pipeline.py            # corre el pipeline completo (cuando exista el modelo)
streamlit run app/Home.py                 # interfaz de investigación (lee artifacts/, R9)
```

Captura diaria del calendario macro con consenso (arranca desde Sesión 0,
ver §"Cron de captura de consenso" más abajo):

```bash
python scripts/capture_consensus.py
```

## Estructura

```
irfn/
├── CLAUDE.md            # las 9 reglas + spec técnica canónica. Autoridad máxima del repo.
├── config/               # YAML + pydantic. Cero constantes mágicas fuera de aquí.
├── data/                 # raw (inmutable) / vintages (ALFRED) / interim / processed
├── src/irfn/
│   ├── data/             # ingesta: precios, ALFRED, calendario, titulares
│   ├── features/         # point-in-time, todo rezagado (R3)
│   ├── models/           # hamilton.py, msgarch.py, tvtp.py, hawkes_mle.py — el núcleo
│   ├── validation/       # walk-forward, calibración, tests estadísticos, ablación
│   ├── outputs/          # schema.py (contrato) + publish.py (guardián anti-smoother)
│   └── audit/            # test de invarianza de prefijo, detector de label switching
├── app/                  # Streamlit. Solo lee artifacts/ (R9). Cero lógica de modelo.
├── artifacts/            # irfn.json + parquet, por run_id
├── reports/              # validation_vN.md, ablation_news.md, data_audit.md
├── tests/                # 6 tests obligatorios, corren en cada commit
├── scripts/               # run_pipeline.py, make_artifacts.py, capture_consensus.py
└── notebooks/            # solo exploración, nunca importado por src/
```

## Las 9 reglas innegociables

Especificación completa en `CLAUDE.md`. Violarlas revierte el código, no se discute.

| # | Regla | Consecuencia de violarla |
| :-: | :-- | :-- |
| R1 | Jamás se publica el smoother (ξ_{t\|T}). Solo se publica ξ_{t\|t}. | Es *el* look-ahead. Gráficas hermosas, modelos que fallan en producción. |
| R2 | Re-estimación por bloque en walk-forward. Prohibido estimar una vez sobre todo el histórico. | El pecado original del look-ahead de MSGARCH. |
| R3 | Toda covariable entra rezagada: x_{t-1}, con `.shift(1)` explícito y comentado. | Usar el dato de hoy para predecir el régimen de hoy es circular. |
| R4 | Datos macro desde ALFRED (vintages), nunca desde FRED revisado. | El dato revisado le da al modelo información que nadie tenía en su momento. |
| R5 | Varianza incondicional estrictamente creciente (v₁ < v₂ < ... < v_K), impuesta en la parametrización. | Sin esto hay label switching entre bloques. |
| R6 | Multistart obligatorio: 20–50 arranques aleatorios, semilla fija y registrada. | La superficie de verosimilitud tiene múltiples máximos locales. |
| R7 | Cero pesos asignados a mano. Todo peso (w_i, β_ij, δ, Hawkes) se estima por MLE/regresión. | Es la crítica que se le hace a los modelos con pesos inventados. No se repite aquí. |
| R8 | Ninguna versión avanza sin walk-forward corrido y documentado, incluso si el resultado es negativo. | Diseñar el test antes de correrlo separa investigación de narrativa. |
| R9 | La interfaz no calcula nada. `app/` solo lee de `artifacts/`. | Un `rolling(7).mean()` "para que se vea mejor" es look-ahead disfrazado de diseño. |

## Documentos de referencia

La metodología completa (spec matemática, roadmap por versiones, diseño de
validación) vive en los dos documentos de dirección de proyecto referenciados
en el `CLAUDE.md` raíz de este directorio de trabajo. Este repo (`irfn/`) es
la implementación; esos documentos son la autoridad.

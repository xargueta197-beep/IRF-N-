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

**Sesión 0 — andamiaje.** No existe modelo todavía: ni filtro de Hamilton, ni
MS-GJR-GARCH, ni TVTP, ni Hawkes. Esta sesión entrega estructura de repo,
contrato de salida, guardián anti-look-ahead, tests (en rojo/skip) y
auditoría de fuentes de datos. Ver `reports/data_audit.md` y la sección
"ESTADO ACTUAL DEL PROYECTO" en `CLAUDE.md`.

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

# GATES metodológicos — antes de que el módulo `sentiment` toque el modelo

Este módulo es **infraestructura de datos / exploración**. Producir un DataFrame
limpio y consolidado **no** lo convierte en una señal válida. Antes de que
cualquier feature derivada de aquí entre al pipeline (logit TVTP, relevancia s_i,
Hawkes), TODOS estos gates deben estar en verde. Cada uno mapea a una regla del
proyecto (ver CLAUDE.md). Un gate rojo no se "salta": se documenta (R8).

| # | Gate | Regla | Por qué / riesgo si se ignora | Estado hoy |
| :- | :-- | :-- | :-- | :-- |
| G1 | **Profundidad temporal ≥ ~7 años** de historia alineada por fuente | R2/R8 | La ablación M5-vs-M4 en walk-forward necesita `train 4a + 6×6m`. Con 240 días NO es evaluable: sin esto no hay veredicto, solo narrativa. | 🔴 240 d |
| G2 | **Timestamp point-in-time honesto** | R3 | Cada API tiene su semántica de fecha; si alguna re-fecha o rellena hacia atrás, inyecta look-ahead. Debe auditarse (test tipo prefix-invariance sobre el stream de eventos). | 🔴 sin auditar |
| G3 | **De-dup cross-fuente principiada** (no solo por `url`) | — (integridad estadística) | La misma noticia con URLs distintas (Finnhub/GDELT/APITube) se cuenta varias veces → **infla el branching ratio del Hawkes** (ya vimos n≈0.9994; esto lo empuja a criticalidad espuria). Dedup semántico (título+tiempo) validado ANTES de construir intensidad. | 🔴 solo por url |
| G4 | **Conmensurabilidad del sentiment** | R7 | El `overall.score` de APITube y el ratio de votos de CryptoPanic **no son la misma variable**. Colapsarlos en una columna es error de categoría sin una calibración por-fuente ESTIMADA (con SE), o manteniéndolos separados. | 🔴 columna única |
| G5 | **Estacionariedad de composición** | R2 | Si la mezcla de fuentes cambia en el tiempo (qué API pagabas ese mes), la intensidad se mueve por razones ajenas al mercado → covariable de confusión. Composición fija y documentada sobre toda la ventana de estimación. | 🔴 mezcla variable |
| G6 | **Relevancia, no dirección** | Trampa 4 | El modelo usa presión de noticias (tiempos + relevancia s_i), NO la dirección del sentimiento. `sentiment_score` no entra como señal direccional; a lo sumo informa relevancia. | ⚠️ por diseño, vigilar |
| G7 | **Ablación documentada** (pase lo que pase) | R8 | Aun con G1–G6 verdes, el resultado de M5-vs-M4 (positivo o negativo) se reporta igual. Diseñar el test antes de correrlo. | 🔴 pendiente de G1 |

## Consecuencia operativa

Mientras cualquiera de G1–G5 esté en rojo, **el consolidado de este módulo NO se
conecta al modelo**. Sirve para: (a) inspeccionar cobertura y calidad de fuentes,
(b) comparar resolución temporal (Finnhub/APITube al segundo vs GDELT a 15 min),
(c) acumular historia hacia adelante para algún día cerrar G1.

## Nota sobre certeza matemática

Añadir *más fuentes* no cierra G1 (profundidad), que es la restricción que ata el
método. La palanca de certeza más alta hoy NO es más ingesta, sino:
1. **Kernel power-law para el Hawkes** (el KS rechaza el exponencial, p=0.0000):
   corrige la *especificación*, no el volumen de datos.
2. **Cerrar R6** en el resto de V4 (BIC, Hansen, walk-forward económico siguen
   `--quick`): vuelve definitivos los veredictos existentes.

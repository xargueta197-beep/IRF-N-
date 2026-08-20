# Diseno (pre-registro) — capa cripto-nativa para BTC: sentimiento + liquidez

**Fecha:** 2026-08-19
**Estado:** PROPUESTA / PRE-REGISTRO. Nada implementado. Requiere OK del director (R3).
**Motivacion (director):** BTC deberia estar afectado por sentimiento de mercado y por
cambios de liquidez; el modelo actual no lo captura.
**Referencias consultadas:** Guia de Implementacion (capa de noticias, indice de sorpresa)
+ Reglas Absolutas (R2, R3, R4, R6, R7, R8) via CLAUDE.md. Esta es una direccion NUEVA
(GARCH-X) que la Guia no cubre; por eso va como pre-registro para tu aprobacion.

---

## 1. El hueco (verificado en el artefacto)

BTC titular vive: `K=1`, `covariates=[]`, `news_layer=[]`, `tvtp=false`. Es un MS-GJR-GARCH
de UN regimen (Student-t) sobre **solo el precio** + capa Hawkes standalone alimentada con
titulares **US-macro** (mismo corpus que SPY, NO cripto-nativo). **No hay ninguna variable
de sentimiento cripto ni de liquidez** (funding, netflows, on-chain, order-book) en `src/`.

Conclusion honesta: BTC no refleja sentimiento/liquidez **porque esas variables no estan en
el modelo**, no porque el modelo las descarte. La intuicion del director (BTC es
sentiment/liquidity-driven) es empiricamente razonable; el modelo simplemente no fue
construido para eso.

## 2. Restriccion matematica: con K=1 la variable NO puede entrar por el logit

La maquinaria M4/M5 (noticias/sorpresa en el logit de transicion, TVTP) **exige K>=2**: un
logit que mueva probabilidad ENTRE regimenes. BTC (K=1) no tiene ese logit. Por tanto una
covariable de sentimiento/liquidez **no puede modular transiciones** en BTC.

La via correcta es que el driver exogeno entre en la **ecuacion de la varianza** — un
**GARCH-X**:

```
sigma^2_{t} = omega + alpha*eps^2_{t-1} + gamma*eps^2_{t-1}*1{eps_{t-1}<0}
              + beta*sigma^2_{t-1} + Sum_j theta_j * g(x_{j,t-1})
```

- `x_{j,t-1}`: driver exogeno REZAGADO (R3, `.shift(1)` explicito). Nunca `x_t`.
- `g(.)`: transformacion que garantiza `sigma^2 > 0` (p.ej. driver estandarizado pasado por
  softplus, o restriccion `theta_j >= 0` con drivers no negativos). A definir en la parametrizacion.
- `theta_j`: ESTIMADO por MLE con su SE (R7, cero pesos a mano). Penalizacion L1 sobre
  `theta_j` con lambda por CV DENTRO del bloque de entrenamiento (anti-sobreajuste, como el TVTP).

**Hipotesis testeable adicional:** re-correr la seleccion de K **con** los terminos GARCH-X.
Si al anadir el driver el BIC pasa a preferir K>=2, entonces el sentimiento/liquidez es lo que
"crea" la estructura de regimenes que hoy no aparece. Resultado limpio en cualquier direccion.

## 3. Variables candidatas + factibilidad de datos (priorizadas por: gratis, historico, sin look-ahead)

| Driver | Tipo | Fuente | Costo | Backfill historico | Look-ahead (R4) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **Volumen/turnover BTC** | liquidez | Binance klines (col [5]) | GRATIS, **ya se descarga** | completo (2017->hoy) | limpio (dato de cierre del dia) |
| **Funding rate perp** | liquidez/posicionamiento | Binance/Bybit API | GRATIS | ~2019->hoy | limpio (publicado por intervalo) |
| **Fear & Greed Index** | sentimiento cripto | alternative.me | GRATIS | 2018->hoy, diario | limpio (publicado diario, no revisado) |
| Netflows de exchange | liquidez | Glassnode/CryptoQuant | PAGO | si | revisiones -> cuidado R4 |
| CryptoPanic (sentiment) | sentimiento | CryptoPanic | PAGO (paywall) | limitado | cuidado |
| Order-book depth/spread | liquidez | snapshots de exchange | infra propia | dificil (real-time) | RIESGO look-ahead |

**Recomendacion de arranque (minima friccion, maximo aprendizaje):** empezar con las **3
gratis y limpias**: (a) **volumen** (ya en mano — el loader hoy tira la col [5], re-fetchear
con volumen es trivial), (b) **funding rate**, (c) **Fear & Greed**. Order-book y netflows
quedan para una fase 2 solo si las 3 baratas muestran senal.

## 4. Protocolo de test PRE-REGISTRADO (congelar ANTES de correr, R8)

- **Baseline:** el titular BTC actual (K=1 Student-t GARCH, sin exogenas).
- **Candidato:** K=1 Student-t **GARCH-X** con el/los driver(s), `x_{t-1}` (R3).
- **Walk-forward:** re-estimacion por bloque (R2), >= 6 bloques de test, multistart R6.
- **Metrica primaria:** densidad predictiva OOS (log-loss) del candidato vs baseline;
  **Diebold-Mariano**. **Criterio de aceptacion:** adoptar GARCH-X solo si DM favorece al
  candidato a **p < 0.05** Y `theta_j` es significativo (SE por Hessiano). Si no, **se reporta
  el negativo** (R8) y se cierra, como se hizo con M2/M3.
- **Chequeo secundario (K):** ¿el BIC prefiere K>=2 con los drivers dentro? Se reporta.
- **Disciplina de datos (R4):** cada driver entra POINT-IN-TIME / sin revision. Funding y
  Fear&Greed como se publicaron; nada de series revisadas retroactivamente.
- **PIT (R1/R9):** el `test_prefix_invariance` debe pasar con la exogena en el pipeline
  (correr sobre data[:t] == data[:T] en ξ_{t|t}). Sin esto no avanza.

## 5. Caveats honestos (no maquillar)

1. **La escalera de SPY advierte:** anadir covariables **empeoro** OOS (M2 peor que M1,
   DM p=0.025; M3 no aporta). Cripto **podria** ser distinto (mas sentiment/liquidity-driven),
   pero **el burden of proof esta en los datos**, no en la hipotesis. Prior esceptico pero abierto.
2. **Sobreajuste:** 3 drivers x parametros nuevos sobre una sola serie -> L1 + CV interna
   obligatorias.
3. **Modelo NUEVO:** GARCH-X no existe en `src/`. Requiere: parametrizacion con positividad de
   varianza, recuperacion sobre verdad conocida (analogo a `test_hamilton_recovery`), y
   `test_garch_vs_arch` extendido. No es "un peldano mas".
4. **Order-book/liquidez intradia:** dificil de conseguir historico SIN look-ahead; por eso NO
   esta en el arranque.

## 6. Primer paso recomendado (barato y decisivo)

**Un solo driver, cero datos nuevos: el VOLUMEN de BTC** (ya en los klines de Binance).
GARCH-X con `log(volumen)_{t-1}` estandarizado como regresor de varianza. Si mejora la densidad
OOS -> expandir a funding + Fear&Greed. Si no -> senal negativa fuerte y **barata** de que la
liquidez (al menos via volumen) no ayuda a la densidad predictiva de BTC, documentada (R8).

## 7. Qué decides tu (director)

- ¿Abro esta linea (GARCH-X cripto-nativo para BTC)? Es metodologia nueva -> tu OK (R3).
- ¿Empezamos por el paso 6 (solo volumen, cero datos nuevos) o quieres las 3 gratis de una?
- ¿Presupuesto para fuentes de pago (Glassnode/CryptoPanic) mas adelante, o solo lo gratis?

Nada se implementa hasta tu OK y la consulta de los dos documentos de referencia.

# Resultados — capa cripto-nativa BTC (GARCH-X): volumen, sentimiento, funding

**Fecha:** 2026-08-19
**Pre-registro:** `reports/diseno_capa_cripto_btc_2026-08-19.md` (director 2026-08-19).
**Modelo:** K=1 GJR-GARCH-t + un driver exogeno REZAGADO (R3) y ESTANDARIZADO en la
ecuacion de varianza (`src/irfn/models/garchx.py`). Baseline = mismo modelo con theta=0.
**Estimador validado:** recovery test PASA (theta dentro de IC95, significativo cuando
theta!=0, NO significativo en control theta=0; leve atenuacion => test CONSERVADOR).
Sanity theta=0: logL bit-identico al baseline.
**Criterio de aceptacion pre-registrado:** adoptar un driver solo si (a) theta significativo
in-sample Y (b) **DM OOS > 0 con p < 0.05** en walk-forward (R2). El OOS decide (R8).

---

## Resultados por driver

| Driver | Clase | Transform | In-sample (theta, t, LR p) | OOS (DM, p, dif) | Veredicto |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **Volumen** | liquidez | log(vol) | theta=-0.003, t=-0.11, p=0.91 | DM **-3.01**, p=0.003 | **NEGATIVO — empeora** |
| **FNG** | sentimiento | \|FNG-50\| | theta=+0.064, t=+1.67, p=0.060 | DM +0.08, p=0.94 | **NEGATIVO — no aguanta** |
| **Funding** | apalancamiento | \|funding\| | theta=+0.386, **t=+3.41**, **p<1e-6** | DM **+1.77**, p=0.078, dif +0.0053 | **NO PASA (borderline)** |

### Volumen (liquidez) — NEGATIVO
In-sample nulo (logL bit-identico al baseline) y OOS **significativamente peor** (DM=-3.01,
p=0.003). Coherente con la teoria (Mixture of Distributions Hypothesis): volumen y volatilidad
se mueven JUNTOS el mismo dia; el volumen REZAGADO es redundante con el retorno^2 que el GARCH
ya usa. Anadirlo solo suma varianza de estimacion -> degrada OOS (misma leccion que M2 en SPY).

### FNG / Fear & Greed (sentimiento) — NEGATIVO
In-sample borderline (theta>0 con el signo correcto, p=0.060) pero **OOS dead null** (DM=+0.08,
p=0.94, dif~0). El "casi-significativo" in-sample era **sobreajuste**; no sobrevive fuera de
muestra. Justo lo que el walk-forward pre-registrado existe para atrapar.

### Funding (apalancamiento) — EL MAS PROMETEDOR, pero NO cruza el umbral
- **In-sample fuerte y significativo:** theta=+0.386 (t=+3.41), LR=30.4 (p<1e-6). El signo es
  coherente: |funding| alto = apalancamiento saturado -> riesgo de liquidacion -> mas varianza.
- **OOS: signo correcto y sugestivo, pero no concluyente:** DM=+1.77, **p=0.078 > 0.05**. La
  diferencia media es positiva (+0.0053).
- **Limitacion honesta (R2):** solo **5 bloques** de walk-forward (n_oos=910), por debajo del
  minimo de 6 -- los perpetuos de Binance existen solo desde 2019, la serie es corta. El test
  esta SUBAPODERADO: p=0.078 con 5 bloques podria cruzar con mas historia.

**Por el criterio pre-registrado (DM>0 Y p<0.05), funding NO se adopta.** Pero es el UNICO
driver con senal positiva consistente (in-sample significativo + OOS con signo correcto),
fallando solo el umbral estricto en un test corto. Es el lead a revisitar.

---

## Conclusion

**Ninguno de los 3 drivers cripto-nativos cumple el criterio de aceptacion.** Se documenta el
negativo (R8), como se hizo con M2/M3/M4/M5:

- **Volumen (liquidez):** cerrado, empeora OOS. Redundante con la dinamica GARCH.
- **Sentimiento (FNG):** cerrado, in-sample era sobreajuste.
- **Funding (apalancamiento):** NO adoptado por el criterio, PERO el unico prometedor
  (in-sample p<1e-6, OOS DM +1.77 con signo correcto). Falla solo por potencia (5 bloques,
  perps desde 2019). **Recomendacion: revisitar cuando la historia de perps sea mas larga**, o
  con un test de mayor frecuencia/potencia. NO se publica hoy en el artefacto (no cumple el bar).

**Leccion transversal (consistente con SPY):** el GJR-GARCH-t de un regimen es un baseline
fuerte; los drivers exogenos rezagados en la varianza en general anaden varianza sin beneficio
OOS. La unica excepcion con base teorica (funding -> liquidaciones) queda **abierta pero no
establecida** estadisticamente. La intuicion del director (BTC es sentiment/liquidity-driven)
NO se confirma en la densidad predictiva a horizonte diario con estos proxies rezagados --
posiblemente porque esos efectos son CONTEMPORANEOS (intradia), no predictivos a 1 dia.

**Codigo conservado** (`models/garchx.py`, `scripts/run_garchx_btc.py`, `data/prices.load_volume`,
capturadores de FNG/funding): reutilizable para el revisit de funding o mas drivers.

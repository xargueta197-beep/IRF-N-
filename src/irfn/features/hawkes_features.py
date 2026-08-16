"""Features derivados del proceso de Hawkes sobre titulares: lambda_N(t) diaria
y su covariable lambda_N_z para el logit de transicion.

--------------------------------------------------------------------------------
INTEGRACION (guia 6.7): NO COLAPSAR SI_t Y lambda_N
--------------------------------------------------------------------------------
z(lambda_N(t-1)) entra como covariable SEPARADA del logit, junto a (no mezclada
con) SI_{t-1}. PROHIBIDO promediarlas en un solo "news score" con pesos
inventados: entran por separado y la verosimilitud (los beta_ij del logit, con
error estandar) decide cuanto pesa cada una. Ese es el punto entero de la capa.

--------------------------------------------------------------------------------
R3 (rezago): el shift(1) vive AQUI y SOLO aqui
--------------------------------------------------------------------------------
`hawkes_feature` devuelve lambda_N_z YA rezagada: el valor fechado en t es el
z-score de la intensidad conocida al cierre del dia calendario t-1 (UTC). El
punto de evaluacion de lambda_N para el dia d es la MEDIANOCHE UTC que cierra d
(d+1 00:00 UTC, con eventos t_i <= u): todo titular contado ahi precede
estrictamente a la sesion bursatil del dia d+1 (que abre 14:30 UTC), asi que
tras el shift(1) la covariable en t solo contiene titulares anteriores a la
sesion de t. Registrado en FEATURE_REGISTRY para el lag_ledger.

--------------------------------------------------------------------------------
Cobertura honesta: NaN fuera de la ventana de datos, no "calma" fabricada
--------------------------------------------------------------------------------
Fuera de [primer dia capturado, ultimo dia capturado], lambda_N NO se evalua:
el modelo diria mu_N (calma basal) en fechas donde simplemente no hay datos, y
eso seria fabricar senal de la ausencia de datos. Se devuelve NaN y el llamador
alinea la muestra (igual que el warm-up de las ventanas tecnicas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from irfn.features.technical import rolling_zscore
from irfn.models.hawkes_mle import lambda_on_grid

# Registro para el lag_ledger (R3). El shift aplicado es EXACTAMENTE el del
# codigo de hawkes_feature; si alguien cambia uno sin el otro, la auditoria lo delata.
FEATURE_REGISTRY: list[dict] = [
    {
        "feature": "lambda_N_z",
        "role": "covariable",
        "applied_shift": 1,
        "note": (
            "z-score movil de lambda_N (intensidad de Hawkes sobre titulares) al cierre "
            "del dia calendario UTC; shift(1) explicito en features/hawkes_features.py. "
            "Covariable SEPARADA de surprise_index (guia 6.7: prohibido colapsarlas)."
        ),
    },
]


def headline_event_times(
    scored: pd.DataFrame, origin: pd.Timestamp | None = None
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    """Convierte titulares puntuados (hora_titular UTC + columna s) al formato
    del MLE: tiempos en DIAS (float) desde `origin` y marcas s_i, ordenados.

    origin por defecto = medianoche UTC del dia del primer titular, de modo que
    times[0] >= 0 (contrato de fit_hawkes_mle: el reloj arranca al inicio de la
    ventana de observacion).
    """
    if scored.empty:
        raise ValueError("no hay titulares puntuados para construir eventos de Hawkes.")
    df = scored.dropna(subset=["hora_titular", "s"]).sort_values("hora_titular")
    ts = pd.DatetimeIndex(df["hora_titular"])
    if origin is None:
        origin = ts[0].normalize()
    times = (ts - origin).total_seconds().to_numpy() / 86400.0
    marks = df["s"].to_numpy(dtype=float)
    return times, marks, origin


def compress_to_observed_time(
    times: np.ndarray, origin: pd.Timestamp, missing_days: list[str]
) -> tuple[np.ndarray, float]:
    """Re-mapea los tiempos de evento del reloj CALENDARIO al reloj TIEMPO-OBSERVADO,
    excindiendo los dias fantasma (dias del rango SIN titulares capturados).

    Motivo (auditoria 2026-08-14/15, decision del director de revertir el ajuste
    sobre span completo): la verosimilitud de Hawkes tiene el compensador
    Lambda(T) = mu_N*T + ..., que integra mu_N sobre TODA la ventana [0, T]. Si el
    corpus cubre 240 dias dentro de un span calendario de ~998, ajustar con
    T = span calendario reparte la masa basal sobre 758 dias que NO se observaron
    (no genuinamente vacios): mu_N se sesga A LA BAJA y, por construccion, el
    branching ratio n = alpha*E[s]/beta se INFLA hacia la frontera n->1 (artefacto,
    no criticidad real). La correccion es ajustar sobre el TIEMPO OBSERVADO: se
    comprime el eje de tiempo quitando los dias fantasma, de modo que
    T_observado = numero de dias cubiertos.

    Por que comprimir NO distorsiona la dinamica: el kernel es exponencial con
    beta ~ decenas por dia (excitacion muerta en minutos) y ningun evento cruza un
    hueco (los eventos solo caen en dias cubiertos), asi que ningun interarribo que
    importe cambia: solo desaparece el termino espurio mu_N*(dias vacios). La
    compresion es monotona (preserva el orden de los eventos) -> no reordena marcas.

    Parametros
    ----------
    times : tiempos de evento en DIAS desde `origin` (reloj calendario, >= 0).
    origin : medianoche UTC del primer dia (mismo `origin` de headline_event_times).
    missing_days : fechas ISO (YYYY-MM-DD) de los dias del rango SIN titulares
        (coverage["missing_days"] de load_headlines).

    Devuelve
    --------
    (times_obs, T_observado) : tiempos en el reloj comprimido y la duracion
        observada total en dias. times_obs conserva el orden de `times`.
    """
    t = np.asarray(times, dtype=float)
    if len(t) == 0:
        return t, 0.0
    origin_ts = pd.Timestamp(origin)
    origin_day = origin_ts.normalize()
    # Offsets enteros (en dias desde origin) de los dias fantasma, ordenados.
    if missing_days:
        miss = pd.to_datetime(list(missing_days))
        miss_day = miss.tz_localize(origin_day.tz) if (miss.tz is None and origin_day.tz is not None) else miss
        miss_offsets = np.sort(((miss_day.normalize() - origin_day).days).to_numpy())
    else:
        miss_offsets = np.zeros(0, dtype=int)
    # Reloj observado = t menos la MEDIDA de dias fantasma en [0, t]. Definirlo como
    # una medida (no como "dias enteros antes") lo hace MONOTONO por construccion
    # (pendiente en [0,1]): asi un evento que el dithering empujo cruzando la
    # medianoche a un dia fantasma se ancla al fin del ultimo dia cubierto en vez de
    # romper el orden. Para eventos en dias cubiertos (el caso normal) coincide con
    # (day_offset - dias_fantasma_antes) + frac.
    day_offset = np.floor(t).astype(int)
    frac = t - day_offset
    missing_before = np.searchsorted(miss_offsets, day_offset, side="left")
    in_phantom = np.isin(day_offset, miss_offsets)          # el dia del evento es fantasma?
    phantom_measure = missing_before + np.where(in_phantom, frac, 0.0)
    times_obs = t - phantom_measure
    # Duracion observada: dias del rango menos los fantasma. El +1 pasa de "ultimo
    # offset" a "conteo de dias"; el max con el ultimo evento cubre eventos cuyo
    # seendate rebasa la medianoche nominal del ultimo dia (frontera, no dato).
    span_days = int(np.max(day_offset)) + 1
    n_missing_in_span = int(np.searchsorted(miss_offsets, span_days, side="left"))
    T_obs = float(span_days - n_missing_in_span)
    T_obs = max(T_obs, float(times_obs[-1]))
    return times_obs, T_obs


# Ancho de la malla temporal del feed de GDELT: el seendate se cuantiza a
# ventanas de 15 minutos (verificado empiricamente: minutos en {0,15,30,45},
# gap modal 900 s entre instantes distintos).
GDELT_BIN_DAYS = 15.0 / 1440.0


def dither_quantized_times(
    times: np.ndarray, marks: np.ndarray, *, seed: int, bin_days: float = GDELT_BIN_DAYS
) -> tuple[np.ndarray, np.ndarray]:
    """De-empata tiempos cuantizados sumando ruido U(0, bin_days) a cada evento.

    Por que: el seendate de GDELT es el PISO del instante real a la malla de 15
    min, asi que ~83% de los titulares comparten timestamp exacto (hasta 52 en un
    mismo instante). Un Hawkes de tiempo continuo con kernel exponencial NO es
    identificable sobre empates: el MLE manda beta->inf para explicar el pico
    instantaneo y el ajuste degenera (alpha,beta enormes, n=0, KS rechaza). El
    dithering restaura la identificabilidad bajo el supuesto de LLEGADA UNIFORME
    dentro del bin observado: t_real = t_grid + U(0, Delta).

    Propiedades:
      - Causal / sin look-ahead: el desplazamiento queda DENTRO del bin ya
        observado (nunca lo rebasa; bins de ancho `bin_days` no se solapan), asi
        que no reordena eventos de bins distintos ni usa informacion futura.
      - Reproducible (R6): determinista dado `seed` (np.random.default_rng).
      - Reordena solo DENTRO de un bin, donde los eventos empatados no tienen
        orden real; se devuelve (times, marks) re-ordenados de forma ascendente
        y alineados, como exige la recursion de Hawkes.

    Decision de metodologia aprobada por el director (2026-08-14); el supuesto de
    uniformidad intra-bin se documenta en el artefacto y en validation_v3.md.
    """
    if len(times) != len(marks):
        raise ValueError("times y marks deben tener la misma longitud.")
    if len(times) == 0:
        return times, marks
    rng = np.random.default_rng(seed)
    dithered = times + rng.uniform(0.0, bin_days, size=len(times))
    order = np.argsort(dithered, kind="stable")
    return dithered[order], marks[order]


def lambda_daily(
    times_days: np.ndarray,
    marks: np.ndarray,
    params: dict[str, float],
    target_index: pd.DatetimeIndex,
    *,
    origin: pd.Timestamp,
    coverage_first: pd.Timestamp,
    coverage_last: pd.Timestamp,
) -> pd.Series:
    """lambda_N al cierre del dia calendario UTC, para cada fecha de `target_index`.

    Punto de evaluacion del dia d: u = (d+1) 00:00 UTC, con eventos t_i <= u
    (los titulares del propio d ya son conocidos a su cierre). Fechas fuera de
    la ventana de cobertura [coverage_first, coverage_last] devuelven NaN (ver
    docstring del modulo: no se fabrica calma donde no hay datos).
    """
    target = pd.DatetimeIndex(target_index)
    naive = target.tz_localize("UTC") if target.tz is None else target.tz_convert("UTC")
    eval_points = naive.normalize() + pd.Timedelta(days=1)
    u = ((eval_points - origin).total_seconds() / 86400.0).to_numpy()

    lam = lambda_on_grid(times_days, marks, params, u, include_equal=True)

    dates = naive.normalize()
    inside = (dates >= coverage_first.normalize()) & (dates <= coverage_last.normalize())
    out = pd.Series(np.where(inside, lam, np.nan), index=target, name="lambda_N")
    return out


def hawkes_feature(lambda_raw: pd.Series, *, z_window: int) -> pd.Series:
    """Covariable lambda_N_z YA rezagada (R3) para el logit de transicion.

    z-score MOVIL causal (misma receta que bb_width_z / hy_oas_z: la ventana
    termina en t, jamas usa futuro; features/technical.rolling_zscore) y luego
    .shift(1) EXPLICITO: el valor fechado en t es el z de la intensidad conocida
    al cierre de t-1, que es el x_{t-1} que el logit usa para predecir la
    transicion HACIA t.
    """
    z = rolling_zscore(lambda_raw, z_window)
    # ----------------------------------------------------------------------
    # R3: shift(1) EXPLICITO. Sin esta linea los titulares del propio dia t
    # entrarian a predecir la transicion hacia t: look-ahead intradia.
    # ----------------------------------------------------------------------
    return z.shift(1).rename("lambda_N_z")

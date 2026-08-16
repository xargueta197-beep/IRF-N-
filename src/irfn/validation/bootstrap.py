"""Bootstrap estacionario (Politis-Romano 1994) para intervalos de confianza de
las estadisticas condicionales por regimen (pantalla 4).

Por que estacionario y no bloques de tamano fijo: los retornos diarios dentro de
un regimen tienen autocorrelacion de corto plazo (clustering de volatilidad, aun
DENTRO de un estado). Bloques de tamano fijo introducen un artefacto en los
bordes (siempre se corta en el mismo punto relativo); el bootstrap estacionario
de Politis-Romano usa bloques de longitud GEOMETRICA aleatoria (media
`block_len`), lo que produce una serie remuestreada que es ella misma
estacionaria y evita ese artefacto. Referencia: Politis & Romano (1994), "The
Stationary Bootstrap", JASA.

Uso (outputs/publish.conditional_stats): por cada (activo, regimen), se
remuestrea la serie de retornos diarios asignados a ese regimen `n_boot` veces,
se recalcula el estadistico en cada replica, y el IC es el percentil empirico
[(1-ci_level)/2, 1-(1-ci_level)/2] de las replicas. Con menos de `min_obs`
observaciones en el regimen, NO se intenta: se devuelve (punto, None, None) y la
pantalla 4 muestra la celda de IC vacia en vez de inventar un intervalo sobre
dos datos.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# --------------------------------------------------------------------------- #
# Estadisticos sobre retornos DIARIOS en escala decimal (mismos que publish.py
# usa inline; centralizados aqui para que el punto y el IC usen EXACTAMENTE la
# misma formula).
# --------------------------------------------------------------------------- #
TRADING_DAYS = 252


def ann_mean(r: np.ndarray) -> float:
    return float(np.mean(r) * TRADING_DAYS)


def ann_vol(r: np.ndarray) -> float:
    if len(r) < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(r: np.ndarray) -> float:
    v = ann_vol(r)
    if not np.isfinite(v) or v <= 0:
        return 0.0
    return float(ann_mean(r) / v)


def max_drawdown(r: np.ndarray) -> float:
    equity = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


STATISTICS: dict[str, Callable[[np.ndarray], float]] = {
    "mean_ann": ann_mean,
    "vol_ann": ann_vol,
    "sharpe": sharpe_ratio,
    "maxdd": max_drawdown,
}


# --------------------------------------------------------------------------- #
# Bootstrap estacionario
# --------------------------------------------------------------------------- #
def _stationary_resample(x: np.ndarray, block_len: float, rng: np.random.Generator) -> np.ndarray:
    """Una replica de bootstrap estacionario: longitud n, bloques geometricos de
    media `block_len` con arranque uniforme y wraparound circular (Politis-Romano)."""
    n = len(x)
    p = 1.0 / max(block_len, 1.0)
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(0, n)
    continue_block = rng.random(n) >= p   # True = seguir el bloque; False = reiniciar
    for t in range(1, n):
        if continue_block[t]:
            idx[t] = (idx[t - 1] + 1) % n
        else:
            idx[t] = rng.integers(0, n)
    return x[idx]


def stationary_bootstrap_ci(
    x: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int,
    block_len: float,
    ci_level: float,
    seed: int,
    min_obs: int = 2,
) -> tuple[float, float | None, float | None]:
    """Punto + IC (percentil empirico) del `statistic` sobre `x` vía bootstrap
    estacionario. El PUNTO se calcula con apenas 2 observaciones finitas (es el
    mismo estadistico descriptivo de siempre); el IC solo se intenta con al
    menos `min_obs` (config v2.bootstrap.min_obs) -- con menos, (punto, None,
    None): la celda de IC queda vacia en vez de inventar un intervalo sobre un
    punado de datos. Tambien (punto, None, None) si demasiadas replicas
    degeneran (estadistico no finito, p.ej. Sharpe con vol=0 en una replica corta).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan"), None, None

    point = float(statistic(x))
    if n < min_obs:
        return point, None, None

    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        reps[b] = statistic(_stationary_resample(x, block_len, rng))
    reps = reps[np.isfinite(reps)]

    if len(reps) < max(10, n_boot // 2):
        return point, None, None

    alpha = 1.0 - ci_level
    lo = float(np.quantile(reps, alpha / 2.0))
    hi = float(np.quantile(reps, 1.0 - alpha / 2.0))
    return point, lo, hi


def bootstrap_regime_stats(
    r: np.ndarray,
    *,
    n_boot: int,
    block_len: float,
    ci_level: float,
    seed: int,
    min_obs: int,
) -> dict[str, tuple[float, float | None, float | None]]:
    """Los 4 estadisticos condicionales (mean_ann, vol_ann, sharpe, maxdd) con su
    IC, sobre la MISMA serie de retornos diarios `r` (decimal) de un regimen."""
    return {
        name: stationary_bootstrap_ci(
            r, fn, n_boot=n_boot, block_len=block_len, ci_level=ci_level,
            seed=seed, min_obs=min_obs,
        )
        for name, fn in STATISTICS.items()
    }


# --------------------------------------------------------------------------- #
# V4 -- Seleccion AUTOMATICA de longitud de bloque (Politis-White 2004) + API
# de alto nivel para el reporte de validacion (BLOQUE 1 del encargo V4).
#
# La API de arriba (stationary_bootstrap_ci / bootstrap_regime_stats) recibe
# `block_len` desde config (v2.bootstrap.block_len) y la consume publish.py para
# los IC de la pantalla 4. Las funciones de abajo NO reciben la longitud: la
# ELIGEN de los datos con Politis-White, y devuelven un IC al 95%. Son las que
# usa el reporte V4 (Test 6, "el Sharpe es real?").
# --------------------------------------------------------------------------- #
def _flat_top_lambda(t: np.ndarray) -> np.ndarray:
    """Ventana de lag "flat-top" trapezoidal de Politis (1995), la que usan
    Politis-White (2004):

        lambda(t) = 1                 si |t| <= 1/2
                  = 2 (1 - |t|)       si 1/2 < |t| <= 1
                  = 0                 si |t| > 1
    """
    a = np.abs(t)
    return np.where(a <= 0.5, 1.0, np.where(a <= 1.0, 2.0 * (1.0 - a), 0.0))


def optimal_block_length(series: np.ndarray) -> int:
    """Longitud optima de bloque para el bootstrap ESTACIONARIO, por el metodo
    automatico de Politis & White (2004), "Automatic Block-Length Selection for
    the Dependent Bootstrap" (Econometric Reviews 23(1)), con la correccion de
    Patton, Politis & White (2009). La longitud NO se elige a ojo ni se hardcodea:
    sale de la estructura de autocovarianza de la propia serie.

    Formula exacta (bootstrap estacionario, SB):

        b_opt = ( 2 * Ghat^2 / Dhat_SB )^(1/3) * n^(1/3)

    donde, con R(k) la autocovarianza muestral al rezago k y lambda(.) la ventana
    flat-top de `_flat_top_lambda`:

        ghat(0) = sum_{k=-M}^{M} lambda(k/M) * R(k)          (densidad espectral en 0)
        Ghat    = sum_{k=-M}^{M} lambda(k/M) * |k| * R(k)
        Dhat_SB = 2 * ghat(0)^2

    de modo que b_opt = ( Ghat / ghat(0) )^(2/3) * n^(1/3), recortado a
    [1, min(3*sqrt(n), n/3)].

    El ancho de banda M se elige de forma adaptativa (Politis 2003): M = 2*mhat,
    donde mhat es el menor rezago tal que las siguientes `kn` autocorrelaciones
    son todas insignificantes al umbral c*sqrt(log10(n)/n) (c=2, kn=max(5,
    ceil(log10(n)))). Si ninguna racha es insignificante, mhat = mayor rezago
    significativo. Serie sin dependencia (ruido blanco) => M pequeno => b_opt ~ 1,
    que es justo el bootstrap iid: correcto.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return 1
    x = x - x.mean()

    kn = max(5, int(np.ceil(np.log10(n))))
    mmax = min(int(np.ceil(np.sqrt(n))) + kn, n - 1)
    b_max = float(np.ceil(min(3.0 * np.sqrt(n), n / 3.0)))

    # Autocovarianzas 0..mmax (con divisor n, sesgadas: la convencion de P&W).
    acv = np.empty(mmax + 1)
    acv[0] = float(x @ x) / n
    if acv[0] <= 0:
        return 1
    for k in range(1, mmax + 1):
        acv[k] = float(x[k:] @ x[:-k]) / n
    rho = acv[1:] / acv[0]                                   # rezagos 1..mmax

    crit = 2.0 * np.sqrt(np.log10(n) / n)
    insig = np.abs(rho) < crit
    mhat = None
    for m in range(0, mmax - kn + 1):
        if insig[m:m + kn].all():
            mhat = m
            break
    if mhat is None:
        sig = np.where(~insig)[0]
        mhat = int(sig[-1] + 1) if sig.size else 1
    M = max(1, min(2 * mhat, mmax))

    kk = np.arange(-M, M + 1)
    acv_sym = acv[np.abs(kk)]                                # R(k) = R(-k)
    w = _flat_top_lambda(kk / M)
    g0 = float(np.sum(w * acv_sym))                          # ghat(0)
    G = float(np.sum(w * np.abs(kk) * acv_sym))              # Ghat
    if g0 <= 0 or not np.isfinite(G):
        return 1
    b = (2.0 * G ** 2 / (2.0 * g0 ** 2)) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    if not np.isfinite(b):
        return 1
    return int(round(min(max(b, 1.0), b_max)))


def _stationary_resample_fast(
    x: np.ndarray, block_len: float, rng: np.random.Generator
) -> np.ndarray:
    """Una replica de bootstrap estacionario, VECTORIZADA. Concatena bloques de
    longitud GEOMETRICA (media `block_len`, p=1/block_len) con arranque uniforme y
    wraparound circular, hasta cubrir n observaciones. Equivalente en distribucion
    a `_stationary_resample` pero sin el bucle python por observacion (importa
    porque el test de cobertura corre miles de replicas)."""
    n = len(x)
    p = 1.0 / max(block_len, 1.0)
    # Longitudes geometricas (>=1, media 1/p); se sobre-generan y se recortan.
    m = int(np.ceil(n * p)) + 8
    lens = rng.geometric(p, size=m)
    while lens.sum() < n:
        lens = np.concatenate([lens, rng.geometric(p, size=m)])
    ends = np.cumsum(lens)
    k = int(np.searchsorted(ends, n, side="left")) + 1      # bloques necesarios
    lens = lens[:k]
    starts = rng.integers(0, n, size=k)
    total = int(lens.sum())
    prefix = np.cumsum(lens) - lens                          # inicio de cada bloque en el plano
    offset = np.arange(total) - np.repeat(prefix, lens)      # 0..L-1 dentro del bloque
    idx = (np.repeat(starts, lens) + offset) % n
    return x[idx[:n]]


def stationary_bootstrap(
    series: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Estimacion puntual + IC 95% de `statistic_fn` sobre `series`, por bootstrap
    estacionario (Politis-Romano 1994) con longitud de bloque elegida AUTOMATICA-
    mente por Politis-White (`optimal_block_length`). Cada replica usa longitud
    geometrica aleatoria de media = la optima.

    Devuelve (point_estimate, ci_lower_95, ci_upper_95). Con <2 observaciones
    finitas devuelve (punto, nan, nan): no se inventa un IC.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        pt = float(statistic_fn(x)) if n else float("nan")
        return pt, float("nan"), float("nan")

    point = float(statistic_fn(x))
    block = float(optimal_block_length(x))
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        reps[i] = statistic_fn(_stationary_resample_fast(x, block, rng))
    reps = reps[np.isfinite(reps)]
    if reps.size < max(10, n_boot // 2):
        return point, float("nan"), float("nan")
    lo = float(np.quantile(reps, 0.025))
    hi = float(np.quantile(reps, 0.975))
    return point, lo, hi


def sharpe_ci(returns: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    """IC 95% del Sharpe (anualizado) por bootstrap estacionario con longitud de
    bloque automatica (Politis-White). El bloque geometrico respeta la
    autocorrelacion de corto plazo de los retornos: ignorarla (bootstrap iid)
    estrecha el IC de forma enganosa.

    Devuelve
        {"sharpe", "ci_lower", "ci_upper", "block_length", "includes_zero"}.

    `includes_zero` (el IC cubre el 0) se calcula AQUI y se expone: es el campo
    con que el reporte decide si la seccion Test 6 abre en mayusculas. No se
    recalcula en el reporte. Si el IC no se pudo formar (pocos datos), se marca
    includes_zero=True de forma conservadora (sin evidencia de Sharpe != 0).
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    block = int(optimal_block_length(x)) if x.size >= 2 else 1
    point, lo, hi = stationary_bootstrap(x, sharpe_ratio, n_boot=n_boot, seed=seed)
    if np.isfinite(lo) and np.isfinite(hi):
        includes_zero = bool(lo <= 0.0 <= hi)
    else:
        includes_zero = True
    return {
        "sharpe": point,
        "ci_lower": lo,
        "ci_upper": hi,
        "block_length": block,
        "includes_zero": includes_zero,
    }

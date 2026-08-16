"""Proceso de Hawkes exponencial con marcas: verosimilitud O(n), MLE, simulacion
por thinning de Ogata y re-escalamiento temporal (bondad de ajuste).

Modelo (CLAUDE.md, guia 6.6):

    lambda_N(t) = mu_N + sum_{t_i < t} alpha * s_i * exp(-beta * (t - t_i))

    Recursion:    A_1 = 0;  A_i = exp(-beta*(t_i - t_{i-1})) * (A_{i-1} + s_{i-1})
                  lambda(t_i) = mu_N + alpha * A_i
    Compensador:  Lambda(T) = mu_N*T + (alpha/beta) * sum_i s_i * (1 - exp(-beta*(T - t_i)))
    log L = sum_i log lambda(t_i) - Lambda(T)

BRANCHING RATIO CON MARCAS -- esto se equivoca seguido (guia 6.6):

    n = alpha * E[s] / beta        <-- NO es alpha/beta cuando hay marcas s_i
    Cascada esperada: E[hijos] = 1 / (1 - n)
    Estacionariedad: n < 1. Si sale n >= 1 el modelo es explosivo y hay un bug;
    NO se publica (el orquestador lo trata como bloqueante, no como numero).

--------------------------------------------------------------------------------
API DELIBERADAMENTE AUTOCONTENIDA (solo numpy + scipy)
--------------------------------------------------------------------------------
Este modulo NO importa nada del resto de irfn: se reutiliza tal cual en la Fase 5
del pipeline cuantitativo (cascadas de liquidacion en cripto: funding extremo ->
liquidacion -> mas liquidaciones es el MISMO formalismo, con la marca s_i como
tamano de la liquidacion en vez de relevancia del titular). Las unidades de
tiempo son las del llamador (aqui: dias); mu_N y beta quedan en 1/esa unidad.

--------------------------------------------------------------------------------
Por que la recursion se evalua en dominio log y vectorizada
--------------------------------------------------------------------------------
A_i = exp(-beta*t_i) * sum_{j<i} s_j*exp(beta*t_j) es matematicamente identica a
la recursion del spec, pero sum s_j*exp(beta*t_j) desborda para beta*t_j grande.
En dominio log: log A_i = logsumexp_{j<i}(log s_j + beta*t_j) - beta*t_i, y
np.logaddexp.accumulate lo computa en O(n) SIN loop de Python. Esto mantiene la
verosimilitud exacta del spec (no una aproximacion), estable para muestras de
cientos de miles de eventos, y sin necesitar numba (que la guia solo permite si
el profiling lo justifica -- aqui no hace falta).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

__all__ = [
    "HawkesFit",
    "hawkes_loglik",
    "lambda_at_events",
    "lambda_on_grid",
    "compensator",
    "branching_ratio",
    "branching_ratio_ci",
    "expected_cascade",
    "expected_cascade_reported",
    "fit_hawkes_mle",
    "simulate_hawkes_ogata_thinning",
    "time_rescaling",
    "rescaling_ks",
]

# Cota inferior de las intensidades evaluadas dentro del log de la verosimilitud:
# solo evita log(0) exacto por subdesbordamiento en puntos extremos del
# optimizador; no acota los parametros (que viven en log-espacio, ya positivos).
_LOG_FLOOR = 1e-300


def _validate(times: np.ndarray, marks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=float)
    s = np.asarray(marks, dtype=float)
    if t.ndim != 1 or s.shape != t.shape:
        raise ValueError("times y marks deben ser vectores 1-D de la misma longitud.")
    if len(t) and np.any(np.diff(t) < 0):
        raise ValueError("times debe venir ordenado de forma no decreciente.")
    if np.any(s < 0) or not np.all(np.isfinite(s)):
        raise ValueError("marks deben ser finitas y >= 0 (relevancia s_i in [0,1]).")
    if len(t) and not np.all(np.isfinite(t)):
        raise ValueError("times contiene valores no finitos.")
    return t, s


def _decay_state_at_events(times: np.ndarray, marks: np.ndarray, beta: float) -> np.ndarray:
    """A_i = sum_{j < i} s_j * exp(-beta*(t_i - t_j)), para cada evento i.

    Identica a la recursion A_i = exp(-beta*dt)*(A_{i-1} + s_{i-1}) del spec,
    evaluada en dominio log (ver docstring del modulo). A_1 = 0.
    """
    n = len(times)
    if n == 0:
        return np.zeros(0)
    with np.errstate(divide="ignore"):  # log(0) = -inf es el neutro correcto
        log_terms = np.log(marks) + beta * times
    # logsumexp acumulado de los terminos j <= i; para A_i se usa el prefijo j <= i-1.
    log_cum = np.logaddexp.accumulate(log_terms)
    A = np.zeros(n)
    A[1:] = np.exp(log_cum[:-1] - beta * times[1:])
    return A


def lambda_at_events(times: np.ndarray, marks: np.ndarray, mu: float, alpha: float, beta: float) -> np.ndarray:
    """lambda(t_i) = mu + alpha * A_i en cada tiempo de evento (limite por la izquierda:
    el propio evento i no se auto-excita)."""
    t, s = _validate(times, marks)
    return mu + alpha * _decay_state_at_events(t, s, beta)


def compensator(times: np.ndarray, marks: np.ndarray, mu: float, alpha: float, beta: float, T: float) -> float:
    """Lambda(T) = mu*T + (alpha/beta) * sum_i s_i * (1 - exp(-beta*(T - t_i)))."""
    t, s = _validate(times, marks)
    if len(t) and t[-1] > T:
        raise ValueError("T debe ser >= al ultimo tiempo de evento.")
    tail = float(np.sum(s * (1.0 - np.exp(-beta * (T - t))))) if len(t) else 0.0
    return mu * T + (alpha / beta) * tail


def hawkes_loglik(
    times: np.ndarray, marks: np.ndarray, mu: float, alpha: float, beta: float, T: float
) -> float:
    """log L = sum_i log lambda(t_i) - Lambda(T) sobre la ventana [0, T]."""
    t, s = _validate(times, marks)
    lam = mu + alpha * _decay_state_at_events(t, s, beta)
    return float(np.sum(np.log(np.maximum(lam, _LOG_FLOOR)))) - compensator(t, s, mu, alpha, beta, T)


def branching_ratio(alpha: float, beta: float, marks: np.ndarray) -> float:
    """n = alpha * E[s] / beta. Con marcas NO es alpha/beta: cada hijo llega con
    intensidad alpha*s_i, asi que el numero medio de hijos por evento es
    alpha*E[s]/beta. E[s] se estima con la media muestral de las marcas."""
    s = np.asarray(marks, dtype=float)
    if len(s) == 0:
        return float("nan")
    return float(alpha * s.mean() / beta)


def expected_cascade(n: float) -> float:
    """E[descendencia total por evento exogeno] = 1/(1-n) para n < 1; inf si n >= 1
    (regimen explosivo: no es un numero publicable, es un bug o un bloqueante)."""
    if not np.isfinite(n) or n >= 1.0:
        return float("inf")
    return float(1.0 / (1.0 - n))


def branching_ratio_ci(
    alpha: float,
    beta: float,
    marks: np.ndarray,
    cov_alpha_beta: np.ndarray | None,
    ci_level: float = 0.95,
) -> dict:
    """IC del branching ratio n = alpha*E[s]/beta por METODO DELTA.

    n es una funcion no lineal de (alpha, beta, m) con m = E[s]. Con la matriz de
    covarianza Cov(alpha, beta) del hessiano del MLE (parametros naturales) y la
    varianza muestral de la media de marcas:

        Var(n) ~= g' Cov(alpha,beta) g  +  (dn/dm)^2 * Var(m)
        g       = (dn/dalpha, dn/dbeta) = (m/beta, -alpha*m/beta^2)
        dn/dm   = alpha/beta,   Var(m) = Var(s)/N

    Se devuelve un IC de Wald simetrico: n +- z * sqrt(Var(n)). NOTA HONESTA: cerca
    de la frontera (n ~ 1) el Wald simetrico es DEBIL -- por eso la cascada 1/(1-n)
    se CENSURA via expected_cascade_reported cuando el IC superior alcanza el
    disparador. Lejos de la frontera (el caso tras corregir la ventana de
    observacion) la aproximacion normal es fiable. cov_alpha_beta=None (hessiano no
    definido positivo) -> SE/IC en NaN: jamas se inventa un intervalo."""
    s = np.asarray(marks, dtype=float)
    N = len(s)
    nan = {"n": float("nan"), "se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    if N == 0 or not np.isfinite(alpha) or not np.isfinite(beta) or beta <= 0:
        return nan
    m = float(s.mean())
    n = alpha * m / beta
    if cov_alpha_beta is None:
        return {"n": float(n), "se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    g = np.array([m / beta, -alpha * m / beta**2])
    var_params = float(g @ np.asarray(cov_alpha_beta, dtype=float) @ g)
    var_m = float(s.var(ddof=1) / N) if N > 1 else 0.0
    var_n = var_params + (alpha / beta) ** 2 * var_m
    if not np.isfinite(var_n) or var_n < 0:
        return {"n": float(n), "se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    se = float(np.sqrt(var_n))
    z = float(stats.norm.ppf(0.5 + ci_level / 2.0))
    return {"n": float(n), "se": se, "ci_low": float(n - z * se), "ci_high": float(n + z * se)}


def expected_cascade_reported(
    n: float, ci_high: float, trigger: float
) -> tuple[float | None, bool]:
    """Cascada esperada REPORTABLE con la regla de censura (decision del director
    2026-08-15). Devuelve (valor, acotada):

      - Si n no es finito, n >= 1, o el IC SUPERIOR de n alcanza `trigger`
        (p.ej. 0.95), la cascada 1/(1-n) diverge o pierde sentido cerca de la
        frontera -> (None, False): el llamador muestra la etiqueta "no acotada".
      - En otro caso -> (1/(1-n), True): el valor puntual es interpretable.

    `trigger` se pasa desde config (news.hawkes.cascade_ci_trigger); este modulo
    NO lee config -- se mantiene autocontenido (solo numpy+scipy)."""
    if not np.isfinite(n) or n >= 1.0:
        return None, False
    if np.isfinite(ci_high) and ci_high >= trigger:
        return None, False
    return float(1.0 / (1.0 - n)), True


# --------------------------------------------------------------------------- #
# MLE con multistart (R6) y errores estandar por hessiano numerico
# --------------------------------------------------------------------------- #
@dataclass
class HawkesFit:
    """Resultado del MLE. params/se/ci_* estan en parametros NATURALES
    (mu, alpha, beta), no en el log-espacio del optimizador."""

    params: dict[str, float]
    se: dict[str, float]
    ci_low: dict[str, float]
    ci_high: dict[str, float]
    loglik: float
    n_events: int
    T: float
    mean_mark: float
    branching_ratio: float
    expected_cascade: float
    stationary: bool
    n_starts: int
    n_converged: int
    seed: int
    hessian_ok: bool
    starts_at_best: int = field(default=0)  # arranques que llegaron al mismo optimo (R6)
    # IC del branching ratio por metodo delta (Parte B, 2026-08-15). se/ci en NaN
    # si el hessiano no es definido positivo (nunca se inventa un intervalo).
    branching_ratio_se: float = field(default=float("nan"))
    branching_ratio_ci_low: float = field(default=float("nan"))
    branching_ratio_ci_high: float = field(default=float("nan"))


def _neg_loglik_log_space(theta: np.ndarray, t: np.ndarray, s: np.ndarray, T: float) -> float:
    mu, alpha, beta = np.exp(theta)
    if not np.all(np.isfinite([mu, alpha, beta])):
        return 1e30
    ll = hawkes_loglik(t, s, mu, alpha, beta, T)
    return -ll if np.isfinite(ll) else 1e30


def _numerical_hessian(f, x: np.ndarray, rel_step: float = 1e-4) -> np.ndarray:
    """Hessiano por diferencias centrales; paso relativo por coordenada."""
    k = len(x)
    h = rel_step * np.maximum(np.abs(x), 1e-3)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h[i]
            ej = np.zeros(k); ej[j] = h[j]
            fpp = f(x + ei + ej)
            fpm = f(x + ei - ej)
            fmp = f(x - ei + ej)
            fmm = f(x - ei - ej)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j])
    return H


def fit_hawkes_mle(
    times: np.ndarray,
    marks: np.ndarray,
    *,
    T: float | None = None,
    n_starts: int = 20,
    seed: int = 0,
    ci_level: float = 0.95,
) -> HawkesFit:
    """MLE de (mu_N, alpha, beta) por L-BFGS-B multistart en log-espacio.

    Parametros
    ----------
    times : tiempos de evento ordenados, medidos desde el inicio de la ventana
        de observacion (t=0). Unidades del llamador (el proyecto usa dias).
    marks : s_i >= 0 (relevancia del titular en [0,1]; la API acepta cualquier
        marca no negativa para reuso en Fase 5).
    T : fin de la ventana de observacion. Si None, se usa times[-1] (sesga mu_N
        levemente al alza al ignorar el silencio final; pasar T explicito si se
        conoce -- el orquestador siempre lo conoce).
    n_starts : arranques aleatorios (multistart estilo R6: se conserva el de
        mayor log L y se reporta cuantos convergieron al mismo optimo).
    seed : semilla del muestreo de arranques; va en el resultado (reproducible).
    ci_level : nivel del intervalo de confianza (Wald, hessiano numerico).

    Los errores estandar se calculan invirtiendo el hessiano numerico de la
    -log-verosimilitud en los parametros NATURALES en el optimo. Si el hessiano
    no es definido positivo se reporta hessian_ok=False y SE/IC en NaN -- jamas
    se inventa un intervalo.
    """
    t, s = _validate(times, marks)
    if len(t) < 10:
        raise ValueError(f"muy pocos eventos ({len(t)}) para un MLE de Hawkes honesto.")
    if T is None:
        T = float(t[-1])
    T = float(T)
    if T <= 0:
        raise ValueError("T debe ser > 0.")
    # El MLE trabaja con el reloj en 0: si el llamador paso tiempos absolutos
    # es su responsabilidad restarles el origen; aqui solo se valida.
    if t[0] < 0:
        raise ValueError("times debe medirse desde el inicio de la ventana (t >= 0).")

    rng = np.random.default_rng(seed)
    rate = len(t) / T                      # tasa media observada: escala natural de mu
    mean_mark = float(s.mean()) if len(s) else float("nan")
    dt = np.diff(t)
    dt_med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 1.0

    best_res, best_ll = None, -np.inf
    final_lls: list[float] = []
    n_converged = 0
    for _ in range(n_starts):
        # Arranques con significado fisico: mu una fraccion de la tasa observada,
        # n implicito uniforme en (0.05, 0.9), beta alrededor de 1/interarribo.
        mu0 = rate * rng.uniform(0.1, 1.0)
        beta0 = np.exp(rng.uniform(np.log(0.1 / dt_med), np.log(10.0 / dt_med)))
        n0 = rng.uniform(0.05, 0.9)
        alpha0 = n0 * beta0 / max(mean_mark, 1e-6)
        theta0 = np.log([mu0, alpha0, beta0])
        res = optimize.minimize(
            _neg_loglik_log_space, theta0, args=(t, s, T), method="L-BFGS-B",
        )
        if res.success:
            n_converged += 1
        final_lls.append(-res.fun)
        if -res.fun > best_ll:
            best_ll, best_res = -res.fun, res

    if best_res is None:
        raise RuntimeError("ningun arranque del MLE de Hawkes produjo un resultado.")

    mu_hat, alpha_hat, beta_hat = np.exp(best_res.x)
    params = {"mu": float(mu_hat), "alpha": float(alpha_hat), "beta": float(beta_hat)}
    # cuantos arranques llegaron (a tolerancia) al mismo optimo -- si son pocos,
    # hay un problema de identificacion y hay que decirlo (R6).
    starts_at_best = int(np.sum(np.abs(np.asarray(final_lls) - best_ll) < 1e-4))

    # SE en parametros naturales (hessiano de -logL en el optimo).
    x_nat = np.array([mu_hat, alpha_hat, beta_hat])

    def _nll_nat(x: np.ndarray) -> float:
        if np.any(x <= 0):
            return 1e30
        ll = hawkes_loglik(t, s, x[0], x[1], x[2], T)
        return -ll if np.isfinite(ll) else 1e30

    names = ("mu", "alpha", "beta")
    se = {k: float("nan") for k in names}
    cov_full: np.ndarray | None = None
    hessian_ok = False
    try:
        H = _numerical_hessian(_nll_nat, x_nat)
        cov = np.linalg.inv(H)
        diag = np.diag(cov)
        if np.all(np.isfinite(diag)) and np.all(diag > 0):
            hessian_ok = True
            cov_full = cov
            se = {k: float(np.sqrt(d)) for k, d in zip(names, diag)}
    except np.linalg.LinAlgError:
        hessian_ok = False

    zq = float(stats.norm.ppf(0.5 + ci_level / 2.0))
    ci_low = {k: (params[k] - zq * se[k]) if np.isfinite(se[k]) else float("nan") for k in names}
    ci_high = {k: (params[k] + zq * se[k]) if np.isfinite(se[k]) else float("nan") for k in names}

    n_hat = branching_ratio(alpha_hat, beta_hat, s)
    # IC del branching ratio por metodo delta (usa el bloque Cov(alpha,beta) del
    # hessiano; alpha,beta son los indices 1,2 en el orden (mu,alpha,beta)).
    cov_ab = cov_full[1:3, 1:3] if cov_full is not None else None
    br = branching_ratio_ci(alpha_hat, beta_hat, s, cov_ab, ci_level=ci_level)
    return HawkesFit(
        params=params, se=se, ci_low=ci_low, ci_high=ci_high,
        loglik=float(best_ll), n_events=int(len(t)), T=T, mean_mark=mean_mark,
        branching_ratio=n_hat, expected_cascade=expected_cascade(n_hat),
        stationary=bool(n_hat < 1.0),
        n_starts=n_starts, n_converged=n_converged, seed=seed,
        hessian_ok=hessian_ok, starts_at_best=starts_at_best,
        branching_ratio_se=br["se"],
        branching_ratio_ci_low=br["ci_low"],
        branching_ratio_ci_high=br["ci_high"],
    )


# --------------------------------------------------------------------------- #
# Simulacion por thinning de Ogata (para test de recuperacion)
# --------------------------------------------------------------------------- #
def simulate_hawkes_ogata_thinning(
    T: float,
    mu: float,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
    mark_sampler=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simula un Hawkes exponencial con marcas en [0, T] por thinning de Ogata.

    mark_sampler(rng) -> marca del evento aceptado; por defecto U(0,1), que es
    la distribucion nominal de la relevancia s_i. La estacionariedad requiere
    alpha*E[s]/beta < 1; se valida con la media del sampler por defecto y se
    deja al llamador la responsabilidad si pasa un sampler propio.

    El thinning es exacto: entre eventos la intensidad solo DECAE, asi que
    lambda_bar = lambda(t actual) es una cota superior valida hasta el proximo
    candidato; el candidato se acepta con probabilidad lambda(t_cand)/lambda_bar.
    """
    if T <= 0 or mu <= 0 or alpha < 0 or beta <= 0:
        raise ValueError("parametros invalidos para la simulacion (T, mu, beta > 0; alpha >= 0).")
    if mark_sampler is None:
        if alpha * 0.5 / beta >= 1.0:
            raise ValueError("alpha*E[s]/beta >= 1 con marcas U(0,1): proceso explosivo.")
        mark_sampler = lambda g: float(g.uniform(0.0, 1.0))  # noqa: E731

    times: list[float] = []
    marks: list[float] = []
    t = 0.0
    D = 0.0                                   # sum s_j exp(-beta*(t - t_j))
    while True:
        lam_bar = mu + alpha * D              # cota superior: D solo decae
        w = rng.exponential(1.0 / lam_bar)
        t_cand = t + w
        if t_cand > T:
            break
        D *= np.exp(-beta * (t_cand - t))
        t = t_cand
        lam_t = mu + alpha * D
        if rng.uniform() * lam_bar <= lam_t:  # aceptar candidato
            s_new = float(mark_sampler(rng))
            times.append(t)
            marks.append(s_new)
            D += s_new
    return np.asarray(times), np.asarray(marks)


# --------------------------------------------------------------------------- #
# Bondad de ajuste: teorema de re-escalamiento temporal
# --------------------------------------------------------------------------- #
def time_rescaling(times: np.ndarray, marks: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """tau_i = Lambda(t_i). Si el modelo es correcto, {tau_i} es un Poisson de
    tasa unitaria y los interarribos (tau_i - tau_{i-1}) son Exp(1).

    Identidad usada (evita el doble loop O(n^2)):
        Lambda(t_i) = mu*t_i + (alpha/beta) * (sum_{j<i} s_j - A_i)
    donde A_i = sum_{j<i} s_j exp(-beta*(t_i - t_j)) es el MISMO estado de la
    recursion de la verosimilitud.
    """
    t, s = _validate(times, marks)
    mu, alpha, beta = params["mu"], params["alpha"], params["beta"]
    A = _decay_state_at_events(t, s, beta)
    cum_prev = np.concatenate([[0.0], np.cumsum(s)[:-1]])
    return mu * t + (alpha / beta) * (cum_prev - A)


def rescaling_ks(times: np.ndarray, marks: np.ndarray, params: dict[str, float]) -> dict:
    """KS de los interarribos re-escalados contra Exp(1).

    Si el KS FALLA, la respuesta NO es forzar el modelo: es REPORTARLO y
    considerar un kernel power-law en una version futura (guia 6.6). Este
    resultado viaja al artefacto y al reporte tal cual, pase o falle.
    """
    tau = time_rescaling(times, marks, params)
    inter = np.diff(tau)
    if len(inter) < 2:
        return {"ks_stat": float("nan"), "p_value": float("nan"), "n": int(len(inter)), "passed": None}
    ks_stat, p_value = stats.kstest(inter, "expon")   # Exp(1): scale=1 por defecto
    return {
        "ks_stat": float(ks_stat),
        "p_value": float(p_value),
        "n": int(len(inter)),
        "passed": bool(p_value > 0.05),
    }


# --------------------------------------------------------------------------- #
# Evaluacion de lambda_N sobre una rejilla (feature diaria)
# --------------------------------------------------------------------------- #
def lambda_on_grid(
    times: np.ndarray,
    marks: np.ndarray,
    params: dict[str, float],
    grid: np.ndarray,
    *,
    include_equal: bool = True,
) -> np.ndarray:
    """lambda_N evaluada en cada punto u de `grid` (mismas unidades que times).

    include_equal=True cuenta eventos con t_i <= u (la lectura "al cierre de u":
    los titulares del propio dia u ya son conocidos al cierre de u). El rezago
    R3 NO se aplica aqui: es responsabilidad de la capa de features (shift(1)
    explicito en features/hawkes_features.py), una sola vez y en un solo lugar.
    """
    t, s = _validate(times, marks)
    u = np.asarray(grid, dtype=float)
    mu, alpha, beta = params["mu"], params["alpha"], params["beta"]
    if len(t) == 0:
        return np.full(len(u), mu)
    with np.errstate(divide="ignore"):
        log_terms = np.log(s) + beta * t
    log_cum = np.logaddexp.accumulate(log_terms)
    side = "right" if include_equal else "left"
    idx = np.searchsorted(t, u, side=side)         # eventos con t_i <= u (o < u)
    out = np.full(len(u), float(mu))
    has = idx > 0
    out[has] = mu + alpha * np.exp(log_cum[idx[has] - 1] - beta * u[has])
    return out

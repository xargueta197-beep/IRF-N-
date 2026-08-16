"""Proceso de Hawkes con kernel POWER-LAW (Omori/ETAS) y marcas: MLE, simulacion
por thinning de Ogata y re-escalamiento temporal (bondad de ajuste).

Motivacion (guia 6.6): el kernel exponencial de hawkes_mle.py fue RECHAZADO por el
KS de re-escalamiento sobre el corpus real de titulares (p=0.0000) con un branching
ratio casi critico (n~0.9994). El kernel exponencial tiene un solo timescale y no
captura el clustering multiescala de las noticias (rafagas de segundos + ecos de
horas/dias). El power-law tiene cola pesada con un unico exponente de cola theta.

Modelo:

    phi(u) = (u + c)^{-(1+theta)}        c>0 (offset, evita la singularidad en 0),
                                         theta>0 (exponente de cola)
    lambda(t) = mu + alpha * sum_{t_i < t} s_i * (t - t_i + c)^{-(1+theta)}

    Compensador (CERRADO, no numerico):
      Integral_0^x phi(u) du = (c^{-theta} - (x+c)^{-theta}) / theta
      Lambda(T) = mu*T + (alpha/theta) * sum_i s_i * (c^{-theta} - (T - t_i + c)^{-theta})
    log L = sum_i log lambda(t_i) - Lambda(T)

BRANCHING RATIO CON MARCAS:
    Integral_0^inf phi(u) du = c^{-theta}/theta   (para theta>0)
    n = alpha * E[s] * c^{-theta} / theta         <-- masa total del kernel * E[s]
    Estacionariedad: n < 1 (igual que el exponencial; n>=1 = explosivo, no publicable).

--------------------------------------------------------------------------------
POR QUE numba (y no la recursion O(n) del exponencial)
--------------------------------------------------------------------------------
El kernel exponencial es SIN MEMORIA: A_i = exp(-beta*dt)*(A_{i-1}+s_{i-1}) da la
intensidad en O(n). El power-law NO es sin memoria: lambda(t_i) suma sobre TODO el
pasado, O(n^2). Para ~10^5 eventos eso son ~10^9 operaciones por evaluacion de la
verosimilitud; se compila con numba (prange) -- este es exactamente el caso en que
la guia permite numba: "solo si el profiling lo justifica". La verosimilitud sigue
siendo EXACTA (no una aproximacion tipo suma-de-exponenciales).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numba
import numpy as np
from scipy import optimize, stats

__all__ = [
    "PowerLawFit",
    "pl_lambda_at_events",
    "pl_compensator",
    "pl_loglik",
    "pl_branching_ratio",
    "fit_powerlaw_mle",
    "simulate_powerlaw_ogata_thinning",
    "pl_time_rescaling",
    "pl_rescaling_ks",
]

_LOG_FLOOR = 1e-300


# --------------------------------------------------------------------------- #
# Nucleos O(n^2) compilados (numba). Devuelven SOLO la suma sobre el pasado;
# mu/alpha se aplican fuera para mantener las funciones puras y reutilizables.
# --------------------------------------------------------------------------- #
@numba.njit(parallel=True, cache=True, fastmath=True)
def _pl_excess(times: np.ndarray, marks: np.ndarray, c: float, power: float) -> np.ndarray:
    """E_i = sum_{j<i} s_j * (t_i - t_j + c)^{-power}, para cada i. O(n^2)."""
    n = times.shape[0]
    out = np.zeros(n)
    for i in numba.prange(n):
        ti = times[i]
        acc = 0.0
        for j in range(i):
            acc += marks[j] * (ti - times[j] + c) ** (-power)
        out[i] = acc
    return out


def _validate(times: np.ndarray, marks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=float)
    s = np.asarray(marks, dtype=float)
    if t.ndim != 1 or s.shape != t.shape:
        raise ValueError("times y marks deben ser vectores 1-D de la misma longitud.")
    if len(t) and np.any(np.diff(t) < 0):
        raise ValueError("times debe venir ordenado de forma no decreciente.")
    if np.any(s < 0) or not np.all(np.isfinite(s)):
        raise ValueError("marks deben ser finitas y >= 0.")
    return t, s


def pl_lambda_at_events(
    times: np.ndarray, marks: np.ndarray, mu: float, alpha: float, c: float, theta: float
) -> np.ndarray:
    """lambda(t_i) = mu + alpha * sum_{j<i} s_j (t_i - t_j + c)^{-(1+theta)}."""
    t, s = _validate(times, marks)
    if len(t) == 0:
        return np.zeros(0)
    return mu + alpha * _pl_excess(t, s, float(c), 1.0 + float(theta))


def pl_compensator(
    times: np.ndarray, marks: np.ndarray, mu: float, alpha: float, c: float, theta: float, T: float
) -> float:
    """Lambda(T) = mu*T + (alpha/theta) * sum_i s_i (c^{-theta} - (T - t_i + c)^{-theta}).
    Forma cerrada, O(n) vectorizada."""
    t, s = _validate(times, marks)
    if len(t) and t[-1] > T:
        raise ValueError("T debe ser >= al ultimo tiempo de evento.")
    if len(t) == 0:
        return mu * T
    tail = float(np.sum(s * (c ** (-theta) - (T - t + c) ** (-theta)))) / theta
    return mu * T + alpha * tail


def pl_loglik(
    times: np.ndarray, marks: np.ndarray, mu: float, alpha: float, c: float, theta: float, T: float
) -> float:
    """log L = sum_i log lambda(t_i) - Lambda(T)."""
    t, s = _validate(times, marks)
    lam = mu + alpha * _pl_excess(t, s, float(c), 1.0 + float(theta))
    return float(np.sum(np.log(np.maximum(lam, _LOG_FLOOR)))) - pl_compensator(t, s, mu, alpha, c, theta, T)


def pl_branching_ratio(alpha: float, c: float, theta: float, marks: np.ndarray) -> float:
    """n = alpha * E[s] * c^{-theta} / theta (masa total del kernel por E[s])."""
    s = np.asarray(marks, dtype=float)
    if len(s) == 0 or theta <= 0:
        return float("nan")
    return float(alpha * s.mean() * c ** (-theta) / theta)


def expected_cascade(n: float) -> float:
    if not np.isfinite(n) or n >= 1.0:
        return float("inf")
    return float(1.0 / (1.0 - n))


# --------------------------------------------------------------------------- #
# MLE multistart (R6). Parametros naturales positivos -> se optimiza en log-espacio.
# --------------------------------------------------------------------------- #
@dataclass
class PowerLawFit:
    params: dict[str, float]           # mu, alpha, c, theta
    se: dict[str, float]
    loglik: float
    aic: float
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
    starts_at_best: int = field(default=0)


_PARAM_NAMES = ("mu", "alpha", "c", "theta")


def _nll_log_space(psi: np.ndarray, t: np.ndarray, s: np.ndarray, T: float) -> float:
    mu, alpha, c, theta = np.exp(psi)
    if not np.all(np.isfinite([mu, alpha, c, theta])):
        return 1e30
    ll = pl_loglik(t, s, mu, alpha, c, theta, T)
    return -ll if np.isfinite(ll) else 1e30


def _numerical_hessian(f, x: np.ndarray, rel_step: float = 1e-4) -> np.ndarray:
    k = len(x)
    h = rel_step * np.maximum(np.abs(x), 1e-3)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h[i]
            ej = np.zeros(k); ej[j] = h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4.0 * h[i] * h[j])
    return H


def fit_powerlaw_mle(
    times: np.ndarray,
    marks: np.ndarray,
    *,
    T: float | None = None,
    n_starts: int = 20,
    seed: int = 0,
) -> PowerLawFit:
    """MLE de (mu, alpha, c, theta) por L-BFGS-B multistart en log-espacio.

    c y theta se arrancan sobre rejillas fisicas: c alrededor del interarribo
    mediano (offset sub-escala), theta en (0.1, 2.0) (exponentes de cola tipicos
    de procesos autoexcitados). n implicito de cada arranque en (0.05, 0.9). SE por
    hessiano numerico; si no es definido positivo -> NaN (jamas se inventa un IC).
    """
    t, s = _validate(times, marks)
    if len(t) < 10:
        raise ValueError(f"muy pocos eventos ({len(t)}) para un MLE honesto.")
    if T is None:
        T = float(t[-1])
    T = float(T)
    if len(t) and t[0] < 0:
        raise ValueError("times debe medirse desde el inicio de la ventana (t >= 0).")

    rng = np.random.default_rng(seed)
    rate = len(t) / T
    mean_mark = float(s.mean())
    dt = np.diff(t)
    dt_med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 1.0

    best_res, best_ll = None, -np.inf
    final_lls: list[float] = []
    n_converged = 0
    for _ in range(n_starts):
        mu0 = rate * rng.uniform(0.1, 1.0)
        c0 = dt_med * np.exp(rng.uniform(np.log(0.05), np.log(5.0)))
        theta0 = np.exp(rng.uniform(np.log(0.1), np.log(2.0)))
        n0 = rng.uniform(0.05, 0.9)
        # invertir n = alpha*E[s]*c^{-theta}/theta  ->  alpha0
        alpha0 = n0 * theta0 * c0 ** theta0 / max(mean_mark, 1e-6)
        psi0 = np.log([mu0, alpha0, c0, theta0])
        res = optimize.minimize(_nll_log_space, psi0, args=(t, s, T), method="L-BFGS-B")
        if res.success:
            n_converged += 1
        final_lls.append(-res.fun)
        if -res.fun > best_ll:
            best_ll, best_res = -res.fun, res

    if best_res is None:
        raise RuntimeError("ningun arranque del MLE power-law produjo resultado.")

    mu_h, alpha_h, c_h, theta_h = np.exp(best_res.x)
    params = {"mu": float(mu_h), "alpha": float(alpha_h), "c": float(c_h), "theta": float(theta_h)}
    starts_at_best = int(np.sum(np.abs(np.asarray(final_lls) - best_ll) < 1e-4))

    def _nll_nat(x: np.ndarray) -> float:
        if np.any(x <= 0):
            return 1e30
        ll = pl_loglik(t, s, x[0], x[1], x[2], x[3], T)
        return -ll if np.isfinite(ll) else 1e30

    se = {k: float("nan") for k in _PARAM_NAMES}
    hessian_ok = False
    try:
        H = _numerical_hessian(_nll_nat, np.array([mu_h, alpha_h, c_h, theta_h]))
        cov = np.linalg.inv(H)
        diag = np.diag(cov)
        if np.all(np.isfinite(diag)) and np.all(diag > 0):
            hessian_ok = True
            se = {k: float(np.sqrt(d)) for k, d in zip(_PARAM_NAMES, diag)}
    except np.linalg.LinAlgError:
        hessian_ok = False

    n_hat = pl_branching_ratio(alpha_h, c_h, theta_h, s)
    k_params = 4
    aic = 2 * k_params - 2 * best_ll
    return PowerLawFit(
        params=params, se=se, loglik=float(best_ll), aic=float(aic),
        n_events=int(len(t)), T=T, mean_mark=mean_mark,
        branching_ratio=n_hat, expected_cascade=expected_cascade(n_hat),
        stationary=bool(np.isfinite(n_hat) and n_hat < 1.0),
        n_starts=n_starts, n_converged=n_converged, seed=seed,
        hessian_ok=hessian_ok, starts_at_best=starts_at_best,
    )


# --------------------------------------------------------------------------- #
# Simulacion por thinning de Ogata (para el test de recuperacion)
# --------------------------------------------------------------------------- #
def _intensity_at(t: float, times: np.ndarray, marks: np.ndarray, mu: float, alpha: float, c: float, power: float) -> float:
    if len(times) == 0:
        return mu
    return mu + alpha * float(np.sum(marks * (t - times + c) ** (-power)))


def simulate_powerlaw_ogata_thinning(
    T: float, mu: float, alpha: float, c: float, theta: float, rng: np.random.Generator, mark_sampler=None
) -> tuple[np.ndarray, np.ndarray]:
    """Simula un Hawkes power-law con marcas en [0, T] por thinning de Ogata. La
    intensidad DECAE entre eventos (kernel monotono decreciente), asi que lambda en
    el t actual es una cota superior valida hasta el proximo evento aceptado."""
    if T <= 0 or mu <= 0 or alpha < 0 or c <= 0 or theta <= 0:
        raise ValueError("parametros invalidos (T, mu, c, theta > 0; alpha >= 0).")
    if mark_sampler is None:
        mark_sampler = lambda g: float(g.uniform(0.0, 1.0))  # noqa: E731
    power = 1.0 + theta
    times: list[float] = []
    marks: list[float] = []
    ta = np.zeros(0)
    ma = np.zeros(0)
    t = 0.0
    while True:
        lam_bar = _intensity_at(t, ta, ma, mu, alpha, c, power)
        w = rng.exponential(1.0 / lam_bar)
        t_cand = t + w
        if t_cand > T:
            break
        lam_cand = _intensity_at(t_cand, ta, ma, mu, alpha, c, power)
        if rng.uniform() * lam_bar <= lam_cand:
            s_new = float(mark_sampler(rng))
            times.append(t_cand)
            marks.append(s_new)
            ta = np.asarray(times)
            ma = np.asarray(marks)
        t = t_cand
    return np.asarray(times), np.asarray(marks)


# --------------------------------------------------------------------------- #
# Bondad de ajuste: re-escalamiento temporal
# --------------------------------------------------------------------------- #
def pl_time_rescaling(times: np.ndarray, marks: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """tau_i = Lambda(t_i) = mu*t_i + (alpha/theta)*(c^{-theta}*cumsum_prev - R_i),
    con R_i = sum_{j<i} s_j (t_i - t_j + c)^{-theta}."""
    t, s = _validate(times, marks)
    mu, alpha, c, theta = params["mu"], params["alpha"], params["c"], params["theta"]
    R = _pl_excess(t, s, float(c), float(theta))
    cum_prev = np.concatenate([[0.0], np.cumsum(s)[:-1]])
    return mu * t + (alpha / theta) * (c ** (-theta) * cum_prev - R)


def pl_rescaling_ks(times: np.ndarray, marks: np.ndarray, params: dict[str, float]) -> dict:
    """KS de los interarribos re-escalados contra Exp(1). Se reporta pase o falle."""
    tau = pl_time_rescaling(times, marks, params)
    inter = np.diff(tau)
    if len(inter) < 2:
        return {"ks_stat": float("nan"), "p_value": float("nan"), "n": int(len(inter)), "passed": None}
    ks_stat, p_value = stats.kstest(inter, "expon")
    return {"ks_stat": float(ks_stat), "p_value": float(p_value), "n": int(len(inter)), "passed": bool(p_value > 0.05)}

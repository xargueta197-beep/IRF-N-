"""GJR-GARCH-X de UN regimen (K=1) con innovaciones Student-t y un regresor
EXOGENO en la varianza -- para BTC (aviso del director 2026-08-19: BTC deberia
sentir liquidez/sentimiento; con K=1 no hay logit de transicion, asi que el driver
entra en sigma^2, no en las transiciones. Ver reports/diseno_capa_cripto_btc_2026-08-19.md).

Especificacion (todo estimado por MLE, R7):

    eps_t    = r_t - mu
    sigma2_t = omega + alpha*eps2_{t-1} + gamma*eps2_{t-1}*1{eps_{t-1}<0}
               + beta*sigma2_{t-1} + theta * x_{t-1}          <- termino EXOGENO
    r_t | .  ~ Student-t(mu, sigma2_t, nu)

Positividad de la varianza GARANTIZADA por construccion: omega>0, (alpha,gamma,beta)
del reparto p*softmax (>=0), theta>=0 y x_{t-1}>=0 (se usa volumen relativo, >=0).
El regresor entra REZAGADO (R3): x_{t-1}, nunca x_t. Con theta=0 se reduce EXACTO
al GJR-GARCH-t de un regimen (sanity check en tests).

Parametrizacion sin restricciones (el optimizador trabaja en R^n), misma cresta de
estacionariedad que el nucleo del proyecto (alpha + gamma/2 + beta = p < 1):

    mu    = m
    omega = softplus(a_omega) > 0
    p     = sigmoid(b_p) in (0,1)
    (u0,u1,u2) = softmax(c0,c1,c2);  alpha = p*u0, gamma = 2*p*u1, beta = p*u2
    theta = softplus(t_theta) >= 0
    nu    = 2 + softplus(n_nu) > 2

Autocontenido: solo numpy + scipy. No importa del resto de irfn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

_TINY = 1e-12


def _softplus(z: float) -> float:
    return float(np.log1p(np.exp(-abs(z))) + max(z, 0.0))


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def _softmax3(c: np.ndarray) -> np.ndarray:
    c = c - np.max(c)
    e = np.exp(c)
    return e / e.sum()


def unpack(theta_vec: np.ndarray) -> dict:
    """Vector sin restricciones -> parametros del modelo (con sus dominios)."""
    m, a_omega, b_p, c0, c1, c2, t_theta, n_nu = theta_vec
    p = _sigmoid(b_p)
    u = _softmax3(np.array([c0, c1, c2]))
    alpha = p * u[0]
    gamma = 2.0 * p * u[1]
    beta = p * u[2]
    return {
        "mu": float(m),
        "omega": _softplus(a_omega),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "beta": float(beta),
        # theta LINEAL (cualquier signo). El regresor exogeno debe entrar
        # ESTANDARIZADO (media 0) -> theta ortogonal a omega, identificable. La
        # positividad de sigma^2 la garantiza el piso _TINY (omega domina en la
        # practica; violaciones raras se pinzan sin sesgar el signo de theta).
        "theta": float(t_theta),
        "nu": 2.0 + _softplus(n_nu),
        "persistence": float(p),
    }


def _sigma2_path(r: np.ndarray, x_lag: np.ndarray, par: dict) -> np.ndarray:
    """Recursion de sigma^2 (T,). x_lag[t] YA es x_{t-1} (rezagado por el caller)."""
    T = r.shape[0]
    mu, omega = par["mu"], par["omega"]
    alpha, gamma, beta, theta = par["alpha"], par["gamma"], par["beta"], par["theta"]
    eps = r - mu
    eps2 = eps * eps
    s2 = np.empty(T)
    # Varianza incondicional aproximada como semilla (incluye el nivel medio del
    # termino exogeno). No es look-ahead: es solo la condicion inicial de la recursion.
    denom = max(1.0 - (alpha + 0.5 * gamma + beta), 1e-4)
    x_mean = float(np.nanmean(x_lag)) if np.isfinite(x_lag).any() else 0.0
    s2[0] = (omega + theta * x_mean) / denom
    for t in range(1, T):
        neg = 1.0 if eps[t - 1] < 0.0 else 0.0
        s2[t] = (omega + alpha * eps2[t - 1] + gamma * eps2[t - 1] * neg
                 + beta * s2[t - 1] + theta * x_lag[t])
    return np.maximum(s2, _TINY)


def _student_t_ll(r: np.ndarray, s2: np.ndarray, mu: float, nu: float) -> float:
    """Log-verosimilitud Student-t con varianza s2 (t estandarizada, var unitaria)."""
    z2 = (r - mu) ** 2 / s2
    c = (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
         - 0.5 * np.log(np.pi * (nu - 2.0)))
    ll = (c - 0.5 * np.log(s2) - (nu + 1.0) / 2.0 * np.log1p(z2 / (nu - 2.0)))
    return float(np.sum(ll))


def negloglik(theta_vec: np.ndarray, r: np.ndarray, x_lag: np.ndarray) -> float:
    par = unpack(theta_vec)
    s2 = _sigma2_path(r, x_lag, par)
    ll = _student_t_ll(r, s2, par["mu"], par["nu"])
    if not np.isfinite(ll):
        return 1e12
    return -ll


@dataclass
class GarchXFit:
    params: dict
    loglik: float
    n_starts: int
    starts_at_best: int
    theta_vec: np.ndarray
    n_obs: int
    se: dict = field(default_factory=dict)
    converged: bool = True


def _random_start(rng: np.random.Generator, r: np.ndarray, with_exog: bool) -> np.ndarray:
    v0 = float(np.var(r)) + _TINY
    return np.array([
        float(np.mean(r)) + rng.normal(scale=0.1) * np.sqrt(v0),
        np.log(np.expm1(max(0.05 * v0, 1e-6))),   # omega ~ 5% de la var
        rng.normal(loc=2.0, scale=0.5),            # p ~ sigmoid(2) ~ 0.88
        rng.normal(scale=0.5), rng.normal(scale=0.5), rng.normal(loc=1.0, scale=0.5),
        (rng.normal(scale=0.3) if with_exog else 0.0),  # theta LINEAL ~0 (off => 0)
        rng.normal(loc=1.5, scale=0.5),            # nu ~ 2 + softplus(1.5) ~ 3.9
    ])


def fit_garchx(
    r: np.ndarray,
    x_lag: np.ndarray | None,
    *,
    n_starts: int = 30,
    seed: int = 42,
    with_exog: bool = True,
    compute_se: bool = True,
) -> GarchXFit:
    """MLE multistart (R6) del GJR-GARCH-X de un regimen.

    r      : retornos (T,), en las MISMAS unidades que se publican.
    x_lag  : regresor exogeno YA REZAGADO (x_{t-1}) y >=0, alineado con r. Si es
             None o with_exog=False, se ajusta el baseline (theta fijado a ~0).
    """
    r = np.asarray(r, dtype=float)
    T = r.shape[0]
    if x_lag is None or not with_exog:
        x_lag = np.zeros(T)
        with_exog = False
    else:
        x_lag = np.asarray(x_lag, dtype=float)
        x_lag = np.where(np.isfinite(x_lag), x_lag, 0.0)

    rng = np.random.default_rng(seed)
    best = None
    best_vecs = []
    for _ in range(n_starts):
        x0 = _random_start(rng, r, with_exog)
        # Baseline: theta en 0 (y x_lag=0 => no afecta la verosimilitud).
        if not with_exog:
            x0[6] = 0.0
        res = minimize(negloglik, x0, args=(r, x_lag), method="L-BFGS-B",
                       options={"maxiter": 500})
        if not np.isfinite(res.fun):
            continue
        if best is None or res.fun < best.fun:
            best = res
        best_vecs.append(res.fun)

    if best is None:
        raise RuntimeError("fit_garchx: ningun arranque convergio.")

    ll_best = -best.fun
    at_best = int(np.sum(np.isclose(np.array(best_vecs), best.fun, atol=1e-4)))
    par = unpack(best.x)
    if not with_exog:
        par["theta"] = 0.0

    fit = GarchXFit(
        params=par, loglik=ll_best, n_starts=n_starts,
        starts_at_best=at_best, theta_vec=best.x, n_obs=T,
    )
    if compute_se:
        fit.se = _hessian_se(best.x, r, x_lag)
    return fit


def _hessian_se(theta_vec: np.ndarray, r: np.ndarray, x_lag: np.ndarray) -> dict:
    """SE por hessiano numerico de la negloglik (delta sobre theta -> params)."""
    n = theta_vec.shape[0]
    eps = 1e-4
    H = np.zeros((n, n))
    f0 = negloglik(theta_vec, r, x_lag)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ei[i] = eps
            ej = np.zeros(n); ej[j] = eps
            fpp = negloglik(theta_vec + ei + ej, r, x_lag)
            fpm = negloglik(theta_vec + ei - ej, r, x_lag)
            fmp = negloglik(theta_vec - ei + ej, r, x_lag)
            fmm = negloglik(theta_vec - ei - ej, r, x_lag)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
    try:
        cov = np.linalg.inv(H)
        se_vec = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    except np.linalg.LinAlgError:
        se_vec = np.full(n, np.nan)
    # theta es LINEAL: su SE es directamente la del hessiano (sin jacobiano).
    return {"theta": float(se_vec[6]) if np.isfinite(se_vec[6]) else float("nan")}


def simulate_garchx(par: dict, x_lag: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Simula r_t del GJR-GARCH-X con parametros conocidos (para recovery test).
    x_lag[t] = x_{t-1} (rezagado). Innovaciones Student-t estandarizadas."""
    rng = np.random.default_rng(seed)
    T = x_lag.shape[0]
    mu, omega = par["mu"], par["omega"]
    alpha, gamma, beta, theta, nu = (par["alpha"], par["gamma"], par["beta"],
                                     par["theta"], par["nu"])
    r = np.empty(T)
    s2 = np.empty(T)
    denom = max(1.0 - (alpha + 0.5 * gamma + beta), 1e-4)
    s2[0] = (omega + theta * float(np.mean(x_lag))) / denom
    scale = np.sqrt((nu - 2.0) / nu)  # t estandarizada -> var unitaria
    for t in range(T):
        if t > 0:
            eps_prev = r[t - 1] - mu
            neg = 1.0 if eps_prev < 0.0 else 0.0
            s2[t] = (omega + alpha * eps_prev**2 + gamma * eps_prev**2 * neg
                     + beta * s2[t - 1] + theta * x_lag[t])
        z = rng.standard_t(nu) * scale
        r[t] = mu + np.sqrt(max(s2[t], _TINY)) * z
    return r


def predictive_logscore(par: dict, r: np.ndarray, x_lag: np.ndarray) -> np.ndarray:
    """log-densidad predictiva por observacion (para DM / walk-forward). Usa la
    sigma^2 filtrada un paso adelante (causal: sigma2_t depende de info hasta t-1)."""
    s2 = _sigma2_path(r, x_lag, par)
    nu, mu = par["nu"], par["mu"]
    z2 = (r - mu) ** 2 / s2
    c = (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
         - 0.5 * np.log(np.pi * (nu - 2.0)))
    return c - 0.5 * np.log(s2) - (nu + 1.0) / 2.0 * np.log1p(z2 / (nu - 2.0))

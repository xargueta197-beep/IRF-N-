"""Capa de observacion: especificacion Haas, Mittnik & Paolella (2004).

La idea central que resuelve el path dependence: cada regimen k mantiene su
PROPIA recursion GARCH, y las K recursiones corren en PARALELO, todas
alimentadas por el mismo retorno observado r_t. No hay colapso (Gray/Klaassen),
no hay dependencia de la trayectoria de estados, y la verosimilitud es exacta y
O(T*K).

Contraste con un MS-GARCH ingenuo: alli sigma^2_t dependeria del regimen en t-1,
que dependia del regimen en t-2, ... y la verosimilitud sumaria sobre K^T
trayectorias (intratable). Aqui sigma^2_{k,t} depende solo de sigma^2_{k,t-1}
del MISMO regimen k y del residuo observado; el estado latente entra despues, en
el filtro de Hamilton, no en la recursion de varianza.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from irfn.models._kernels import garch_recursion
from irfn.models.params import unpack


def conditional_variances(r: np.ndarray, params: dict[str, np.ndarray], K: int) -> np.ndarray:
    """Devuelve sigma2 de forma (T, K): la varianza condicional de cada regimen
    en cada fecha, corriendo las K recursiones GJR-GARCH en paralelo.

    Inicializacion sigma2_{k,0} = v_k (la varianza incondicional del regimen).
    --------------------------------------------------------------------------
    Decision documentada: arrancamos en el valor estacionario E[sigma^2_{k,t}]
    = v_k, que es funcion SOLO de los parametros, no de los datos. Esto es
    deliberado por dos razones:
      1. Point-in-time / prefix-invariance: sigma2_0 debe ser identico corriendo
         sobre data[:t] o sobre data[:T]. Un arranque data-dependiente (p.ej. la
         varianza muestral de un burn-in, o el backcast exponencial que usa la
         libreria `arch` por defecto) haria que sigma2_0 dependiera de cuantos
         datos hay -> look-ahead disfrazado de inicializacion, y test_prefix_
         invariance fallaria con razon.
      2. No introduce parametros de molestia (nada de un sigma2_0 libre).
    Costo conocido: esto desalinea con `arch` (que usa backcast) en los primeros
    pasos; se maneja en test_garch_vs_arch comparando sobre series largas donde
    el transitorio se lava.
    """
    r = np.ascontiguousarray(r, dtype=float)
    # El loop secuencial vive en _kernels.garch_recursion (jitteado si hay numba).
    return garch_recursion(
        r,
        np.ascontiguousarray(params["mu"], dtype=float),
        np.ascontiguousarray(params["omega"], dtype=float),
        np.ascontiguousarray(params["alpha"], dtype=float),
        np.ascontiguousarray(params["gamma"], dtype=float),
        np.ascontiguousarray(params["beta"], dtype=float),
        np.ascontiguousarray(params["v"], dtype=float),
    )


def normal_logpdf(r: np.ndarray, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """log N(r; mu, sigma2) evaluada por regimen. r es (T,), mu es (K,),
    sigma2 es (T, K). Devuelve (T, K). Se trabaja en log para estabilidad.
    """
    r = np.asarray(r, dtype=float)[:, None]  # (T, 1)
    return -0.5 * (np.log(2.0 * np.pi) + np.log(sigma2) + (r - mu) ** 2 / sigma2)


def student_t_logpdf(
    r: np.ndarray, mu: np.ndarray, sigma2: np.ndarray, nu: np.ndarray
) -> np.ndarray:
    """log t de Student ESCALADA A VARIANZA, evaluada por regimen (V1).

    El punto delicado: sigma2 es la VARIANZA condicional del regimen (asi lo
    produce la recursion GARCH y asi lo exige R5, donde v_k es la varianza
    incondicional). Una t_nu estandar tiene varianza nu/(nu-2), no 1, asi que el
    scale^2 de la densidad NO es sigma2 sino

        s2 = sigma2 * (nu - 2) / nu        (requiere nu > 2, garantizado por
                                            la parametrizacion nu = 2 + exp(g))

    Con ese escalado, Var(r | S_t=k) = sigma2_{k,t} exactamente, y la Normal es
    el limite nu -> inf (test_student_t_limit). r es (T,), mu y nu son (K,),
    sigma2 es (T, K). Devuelve (T, K).
    """
    r = np.asarray(r, dtype=float)[:, None]                 # (T, 1)
    nu = np.asarray(nu, dtype=float)[None, :]               # (1, K)
    s2 = sigma2 * (nu - 2.0) / nu                           # scale^2, (T, K)
    z2 = (r - mu) ** 2 / s2
    return (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * (np.log(np.pi * nu) + np.log(s2))
        - 0.5 * (nu + 1.0) * np.log1p(z2 / nu)
    )


def obs_logpdf(
    r: np.ndarray, params: dict[str, np.ndarray], sigma2: np.ndarray
) -> np.ndarray:
    """Log-densidad de observacion por regimen, despachada por especificacion:
    t de Student si params trae "nu" (dist="t"), Normal si no. Quien decide
    entre ambas es el BIC (V1), no una constante en el codigo."""
    if "nu" in params:
        return student_t_logpdf(r, params["mu"], sigma2, params["nu"])
    return normal_logpdf(r, params["mu"], sigma2)


def simulate(
    theta: np.ndarray,
    K: int,
    T: int,
    seed: int,
    burn_in: int = 500,
    dist: str = "normal",
    n_cov: int = 0,
    X_lagged: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simula un MS-GJR-GARCH (especificacion Haas). Devuelve (r, states), ambos
    de longitud T.

    En cada paso:
      1. Se actualizan las K varianzas condicionales usando el retorno observado
         r_{t-1} (las K recursiones corren siempre en paralelo, igual que en el
         filtro; esto es lo que hace exacta a la verosimilitud).
      2. Se transiciona el estado latente s_t ~ P_t[s_{t-1}, :].
      3. Se genera r_t = mu_{s_t} + sqrt(sigma2_{s_t,t}) * z_t con z_t de
         varianza UNITARIA: Normal estandar, o t_nu reescalada por
         sqrt((nu-2)/nu) si dist="t" (sigma2 sigue siendo la varianza).

    TVTP (n_cov > 0): X_lagged debe ser (T, n_cov) y seguir el MISMO contrato que
    el filtro: la fila t son las covariables x_{t-1} que gobiernan la transicion
    HACIA t (models/tvtp.py). Durante el burn-in (sin covariables) se transiciona
    con la matriz de interceptos P = softmax(d), la de condiciones medias.

    Se descartan `burn_in` observaciones iniciales para lavar el transitorio del
    arranque en la varianza incondicional.
    """
    rng = np.random.default_rng(seed)
    params = unpack(theta, K, n_cov=n_cov, dist=dist)
    mu = params["mu"]
    v = params["v"]
    omega = params["omega"]
    alpha = params["alpha"]
    gamma = params["gamma"]
    beta = params["beta"]
    P = params["P"]

    if n_cov > 0:
        if X_lagged is None or X_lagged.shape != (T, n_cov):
            raise ValueError(f"TVTP: X_lagged debe ser ({T}, {n_cov}).")
        from irfn.models.tvtp import transition_matrices

        P_path = transition_matrices(params["d"], params["beta_tvtp"], X_lagged)

    if dist == "t":
        nu = params["nu"]

        def draw_z(k: int) -> float:
            # t_nu tiene varianza nu/(nu-2); se reescala a varianza 1.
            return rng.standard_t(nu[k]) * np.sqrt((nu[k] - 2.0) / nu[k])
    else:
        def draw_z(k: int) -> float:
            return rng.standard_normal()

    n = T + burn_in
    r = np.empty(n)
    states = np.empty(n, dtype=int)
    sigma2 = np.empty((n, K))
    sigma2[0, :] = v

    # estado inicial desde la distribucion estacionaria de P (interceptos)
    from irfn.models.hamilton import stationary_distribution

    pi0 = stationary_distribution(P)
    states[0] = rng.choice(K, p=pi0)
    r[0] = mu[states[0]] + np.sqrt(sigma2[0, states[0]]) * draw_z(states[0])

    for t in range(1, n):
        eps_prev = r[t - 1] - mu
        eps2_prev = eps_prev * eps_prev
        neg = (eps_prev < 0.0).astype(float)
        sigma2[t, :] = omega + alpha * eps2_prev + gamma * eps2_prev * neg + beta * sigma2[t - 1, :]

        if n_cov > 0 and t >= burn_in:
            P_t = P_path[t - burn_in]      # transicion hacia t con x_{t-1}
        else:
            P_t = P
        states[t] = rng.choice(K, p=P_t[states[t - 1]])
        r[t] = mu[states[t]] + np.sqrt(sigma2[t, states[t]]) * draw_z(states[t])

    return r[burn_in:], states[burn_in:]

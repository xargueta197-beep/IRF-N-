"""Kernels numericos calientes del nucleo, aislados para poder acelerarlos.

Las dos recursiones secuenciales (varianza GARCH y filtro de Hamilton) son
loops en t que no se pueden vectorizar (cada paso depende del anterior). En
Python puro, con arrays diminutos (K=2..3), el overhead de dispatch de numpy por
paso domina: ~100 ms por evaluacion del filtro sobre T=5000, lo que vuelve el
multistart (R6) y el test de recuperacion (N=5000) intratables (~40 min).

CLAUDE.md autoriza numba EN la recursion del filtro "solo si el profiling lo
justifica"; se perfilo y lo justifica. Si numba esta disponible, se compilan con
@njit; si no, se degradan a Python puro sin cambiar el resultado (fallback
correcto, solo mas lento). El resto del codebase no sabe si estan jitteados.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover - fallback si numba no esta instalado
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def _wrap(func):
            return func

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _wrap


@njit(cache=True)
def garch_recursion(r, mu, omega, alpha, gamma, beta, v):
    """K recursiones GJR-GARCH independientes en paralelo (Haas et al. 2004).

    sigma2_{k,0} = v_k (varianza incondicional); para t>=1
    sigma2_{k,t} = omega_k + alpha_k eps2 + gamma_k eps2 1{eps<0} + beta_k sigma2_{k,t-1}
    con eps = r_{t-1} - mu_k. Devuelve sigma2 de forma (T, K).
    """
    T = r.shape[0]
    K = mu.shape[0]
    sigma2 = np.empty((T, K))
    for k in range(K):
        sigma2[0, k] = v[k]
    for t in range(1, T):
        rt = r[t - 1]
        for k in range(K):
            eps = rt - mu[k]
            eps2 = eps * eps
            neg = 1.0 if eps < 0.0 else 0.0
            sigma2[t, k] = omega[k] + alpha[k] * eps2 + gamma[k] * eps2 * neg + beta[k] * sigma2[t - 1, k]
    return sigma2


# error_model="numpy": en pasos extremos del optimizador la prediccion puede
# poner masa 0 en el regimen de mayor densidad y denom colapsar a 0. Con la
# semantica numpy eso produce nan/-inf (que estimate._neg_loglik trata como
# punto infactible -> 1e12), en vez de lanzar ZeroDivisionError y abortar el fit.
@njit(cache=True, error_model="numpy")
def filter_recursion(logf, P_T, pi0):
    """Recursion de Hamilton. logf es (T,K) log-densidades, P_T = P^T (K,K),
    pi0 la distribucion inicial (K,). Devuelve (xi_filt, xi_pred, loglik).

    Prediccion en espacio lineal (combinacion convexa, sin underflow). En la
    actualizacion se factoriza el maximo de logf por paso para evitar underflow
    de densidades; esa constante se reintegra a la loglik, que sale del
    denominador de la actualizacion.
    """
    T = logf.shape[0]
    K = logf.shape[1]
    xi_pred = np.empty((T, K))
    xi_filt = np.empty((T, K))
    loglik = 0.0

    for t in range(T):
        # 1. PREDICCION
        if t == 0:
            for j in range(K):
                xi_pred[t, j] = pi0[j]
        else:
            for j in range(K):
                s = 0.0
                for i in range(K):
                    s += P_T[j, i] * xi_filt[t - 1, i]
                xi_pred[t, j] = s

        # 2. maximo de las log-densidades del paso (estabilidad)
        m = logf[t, 0]
        for k in range(1, K):
            if logf[t, k] > m:
                m = logf[t, k]

        # 3. ACTUALIZACION; la verosimilitud sale del denominador
        denom = 0.0
        for k in range(K):
            num_k = xi_pred[t, k] * np.exp(logf[t, k] - m)
            xi_filt[t, k] = num_k
            denom += num_k
        for k in range(K):
            xi_filt[t, k] /= denom
        loglik += np.log(denom) + m

    return xi_filt, xi_pred, loglik


@njit(cache=True, error_model="numpy")
def filter_recursion_tvtp(logf, P_path, pi0):
    """Recursion de Hamilton con matriz de transicion VARIABLE (TVTP, V1).

    P_path es (T, K, K): P_path[t] es la matriz que gobierna la prediccion del
    paso t, YA evaluada en x_{t-1} por models/tvtp.py (el contrato de rezago
    vive alli y en hamilton.py; este kernel solo consume la trayectoria).
    Identica a filter_recursion salvo que la combinacion convexa de la
    prediccion usa P_path[t] en vez de una P fija:

        xi_{t|t-1}(j) = sum_i p_ij(x_{t-1}) * xi_{t-1|t-1}(i)

    Devuelve (xi_filt, xi_pred, loglik); la loglik sale del denominador de la
    actualizacion, igual que siempre.
    """
    T = logf.shape[0]
    K = logf.shape[1]
    xi_pred = np.empty((T, K))
    xi_filt = np.empty((T, K))
    loglik = 0.0

    for t in range(T):
        # 1. PREDICCION con la matriz del paso t (evaluada en x_{t-1})
        if t == 0:
            for j in range(K):
                xi_pred[t, j] = pi0[j]
        else:
            for j in range(K):
                s = 0.0
                for i in range(K):
                    s += P_path[t, i, j] * xi_filt[t - 1, i]
                xi_pred[t, j] = s

        # 2. maximo de las log-densidades del paso (estabilidad)
        m = logf[t, 0]
        for k in range(1, K):
            if logf[t, k] > m:
                m = logf[t, k]

        # 3. ACTUALIZACION; la verosimilitud sale del denominador
        denom = 0.0
        for k in range(K):
            num_k = xi_pred[t, k] * np.exp(logf[t, k] - m)
            xi_filt[t, k] = num_k
            denom += num_k
        for k in range(K):
            xi_filt[t, k] /= denom
        loglik += np.log(denom) + m

    return xi_filt, xi_pred, loglik

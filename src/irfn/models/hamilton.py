"""EL NUCLEO. Recursion de prediccion + actualizacion de Hamilton.

hamilton_filter() devuelve (xi_filtered, xi_predicted, loglik). Nunca smoother.
El smoother de Kim vive aqui pero decorado @diagnostic_only y publish.py lo
bloquea (R1).

Regla de oro de este archivo: la log-verosimilitud SALE del denominador de la
actualizacion bayesiana. No existe una funcion compute_loglik separada que
recorra los datos otra vez; si la hubiera, algo estaria mal.
"""

from __future__ import annotations

import functools

import numpy as np

from irfn.models._kernels import filter_recursion, filter_recursion_tvtp
from irfn.models.msgarch import conditional_variances, obs_logpdf
from irfn.models.params import unpack


def diagnostic_only(func):
    """Marca una funcion como herramienta de diagnostico interno. Lo que produce
    JAMAS puede llegar a artifacts/ ni a app/ (R1). El bloqueo efectivo lo hace
    outputs/publish.py via FORBIDDEN_KEYS; este decorador es la senal de intencion
    en el codigo.
    """
    func._diagnostic_only = True

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._diagnostic_only = True
    return wrapper


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Distribucion estacionaria pi de una cadena de Markov con matriz de
    transicion P (filas suman 1): resuelve pi = P^T pi, sum(pi) = 1.

    Se obtiene como el nullspace de (P^T - I) con la restriccion de normalizacion,
    resuelto por minimos cuadrados para robustez numerica (evita depender de que
    un autovalor sea exactamente 1).
    """
    P = np.asarray(P, dtype=float)
    K = P.shape[0]
    if K == 1:
        return np.array([1.0])

    A = np.vstack([P.T - np.eye(K), np.ones(K)])
    b = np.concatenate([np.zeros(K), [1.0]])
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    # limpiar ruido numerico: probabilidades no negativas que suman 1
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def hamilton_filter(
    r: np.ndarray,
    params: dict[str, np.ndarray] | np.ndarray,
    K: int,
    X_lagged: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Filtro de Hamilton para un MS-GJR-GARCH. Matriz de transicion constante
    (V0) o TVTP (V1) segun params traiga o no "beta_tvtp"; innovaciones Normales
    o t de Student segun traiga o no "nu" (msgarch.obs_logpdf despacha).

    Parametros
    ----------
    r : (T,) retornos observados.
    params : dict de parametros naturales (salida de unpack) o vector theta
        (solo para la especificacion V0: constante + Normal).
    K : numero de regimenes.
    X_lagged : (T, n_cov) o None. La fila t contiene x_{t-1}: covariables
        construidas SOLO con informacion hasta el cierre de t-1 (el rezago viene
        aplicado desde la capa de features y registrado en el lag_ledger, R3;
        ver el contrato en models/tvtp.py). Obligatoria si params trae
        "beta_tvtp"; prohibida si no.

    Devuelve
    --------
    xi_filtered  : (T, K) prob. filtrada ξ_{t|t}   -- LO UNICO que se publica.
    xi_predicted : (T, K) prob. predicha ξ_{t|t-1} -- para verosimilitud/calibracion.
    loglik       : escalar, log-verosimilitud del modelo.

    NO devuelve smoother. Ese es kim_smoother(), aislado y @diagnostic_only.
    """
    if isinstance(params, np.ndarray):
        params = unpack(params, K)

    r = np.asarray(r, dtype=float)
    P = params["P"]

    sigma2 = conditional_variances(r, params, K)                    # (T, K)
    logf = np.ascontiguousarray(obs_logpdf(r, params, sigma2))      # (T, K)

    # ξ_{0|-1} = distribucion estacionaria de P (solo params -> prefix-safe).
    # Con TVTP, P es la matriz de interceptos (covariables en su media de
    # entrenamiento): sigue siendo funcion SOLO de los parametros, nunca de los
    # datos, asi que la inicializacion no rompe la invarianza de prefijo.
    pi0 = stationary_distribution(P)

    tvtp = "beta_tvtp" in params
    if tvtp:
        if X_lagged is None:
            raise ValueError("params trae beta_tvtp: hamilton_filter requiere X_lagged (TVTP).")
        X_lagged = np.ascontiguousarray(X_lagged, dtype=float)
        if X_lagged.shape[0] != r.shape[0]:
            raise ValueError(
                f"X_lagged ({X_lagged.shape[0]} filas) no alinea con r ({r.shape[0]})."
            )
        from irfn.models.tvtp import transition_matrices

        # LA LINEA QUE SEPARA UN MODELO DE UNA ILUSION: la matriz de transicion
        # del paso t se evalua en x[t-1], NUNCA en x[t]. Aqui X_lagged[t] ES
        # x_{t-1} -- el rezago lo aplico la capa de features con su .shift(1)
        # explicito / lag de publicacion real (R3, lag_ledger) -- de modo que
        # P_path[t] = softmax(d + B·x_{t-1}) predice la transicion HACIA t sin
        # haber visto nada de t. Si alguien alimenta covariables sin rezagar,
        # el modelo "predice" el regimen de hoy usando el cierre de hoy: una
        # ilusion que la auditoria PIT delata (test_prefix_invariance_tvtp).
        P_path = np.ascontiguousarray(transition_matrices(params["d"], params["beta_tvtp"], X_lagged))
        xi_filt, xi_pred, loglik = filter_recursion_tvtp(logf, P_path, pi0)
    else:
        if X_lagged is not None:
            raise ValueError("params sin beta_tvtp: X_lagged no debe pasarse (evita falsas expectativas).")
        P_T = np.ascontiguousarray(P.T)                    # (K, K), para la prediccion
        xi_filt, xi_pred, loglik = filter_recursion(logf, P_T, pi0)

    return xi_filt, xi_pred, float(loglik)


def predictive_loglik_per_obs(
    r: np.ndarray,
    params: dict[str, np.ndarray],
    K: int,
    X_lagged: np.ndarray | None = None,
) -> np.ndarray:
    """Contribucion de CADA observacion a la log-verosimilitud del filtro:

        ll_t = log sum_m xi_{t|t-1}(m) f(r_t | S_t=m)

    Es el denominador de la actualizacion de Hamilton (la densidad predictiva
    un-paso-adelante), NO una segunda recorrida de la verosimilitud: sum_t ll_t
    reproduce exactamente la loglik que devuelve hamilton_filter. Se expone por
    observacion porque la validacion la necesita en rebanadas: la perdida
    predictiva OOS del Diebold-Mariano, el CV del lambda L1 y las metricas por
    bloque del walk-forward son sumas parciales de ESTA serie.
    """
    r = np.ascontiguousarray(r, dtype=float)
    sigma2 = conditional_variances(r, params, K)
    logf = obs_logpdf(r, params, sigma2)                       # (T, K)
    _, xi_pred, _ = hamilton_filter(r, params, K, X_lagged=X_lagged)
    a = np.log(np.clip(xi_pred, 1e-300, None)) + logf          # (T, K)
    m = a.max(axis=1, keepdims=True)
    return m[:, 0] + np.log(np.exp(a - m).sum(axis=1))


@diagnostic_only
def kim_smoother(
    xi_filtered: np.ndarray,
    xi_predicted: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """SMOOTHER DE KIM. USA EL FUTURO (ξ_{t|T}). JAMAS SE PUBLICA (R1).

    Solo diagnostico interno: producir graficas hermosas para ENTENDER, nunca
    para reportar. Cualquier ruta que lleve su salida a artifacts/ o a app/ es
    una violacion de R1 y outputs/publish.py debe lanzar LookAheadViolation antes
    de que eso ocurra.

    Recursion hacia atras de Kim (1994):
        ξ_{t|T}(j) = ξ_{t|t}(j) * Σ_k [ P_jk * ξ_{t+1|T}(k) / ξ_{t+1|t}(k) ]
    """
    T, K = xi_filtered.shape
    xi_smooth = np.empty((T, K))
    xi_smooth[-1] = xi_filtered[-1]

    for t in range(T - 2, -1, -1):
        ratio = xi_smooth[t + 1] / np.clip(xi_predicted[t + 1], 1e-300, None)  # (K,)
        xi_smooth[t] = xi_filtered[t] * (P @ ratio)
        xi_smooth[t] /= xi_smooth[t].sum()  # renormalizar por seguridad numerica

    return xi_smooth

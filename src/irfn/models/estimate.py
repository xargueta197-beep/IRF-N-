"""Estimacion por maxima verosimilitud directa sobre la recursion de Hamilton.

- L-BFGS-B sobre la parametrizacion sin restricciones de params.py.
- Multistart (R6): 20-50 arranques aleatorios, se conserva el de mayor loglik.
  Se reporta cuantos convergieron al MISMO optimo; si son pocos, hay un problema
  de identificacion y hay que decirlo, no esconderlo.
- Errores estandar por hessiano numerico invertido, propagados al espacio de
  parametros naturales por el metodo delta (la SE de a_1 NO es la SE de v_1).

V1: la misma maquinaria estima TVTP (betas del logit multinomial, R7: por MLE
con error estandar) e innovaciones t de Student. La penalizacion L1 sobre los
beta_ij entra como verosimilitud penalizada; su lambda se elige por CV temporal
DENTRO del bloque de entrenamiento (cv_l1_lambda), JAMAS mirando el test.

--------------------------------------------------------------------------------
Como entra la L1 en un optimizador suave (decision documentada)
--------------------------------------------------------------------------------
L-BFGS-B asume gradiente continuo y |x| no lo tiene en 0. Se usa el suavizado
estandar |x| ~ sqrt(x^2 + eps^2) - eps con eps diminuto (config
v1.tvtp.l1_smooth_eps): para |x| >> eps es |x| con error < eps; en 0 es
diferenciable. El sesgo que introduce es del orden de eps (1e-6), ordenes de
magnitud por debajo de la resolucion estadistica de los beta. La alternativa
(orthant-wise/proximal) exigiria otro optimizador para un beneficio nulo aqui.
La loglik que se REPORTA (y el BIC) es siempre la NO penalizada en el optimo.

Advertencia honesta sobre los SE con lambda > 0: se calculan con el hessiano de
la verosimilitud NO penalizada en el optimo penalizado. Son una aproximacion
condicional (no capturan la incertidumbre de la seleccion del lambda ni el sesgo
de contraccion). Si el CV elige lambda=0, la advertencia es vacua.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from irfn.models.hamilton import hamilton_filter, predictive_loglik_per_obs
from irfn.models.params import n_params, pack, unpack

logger = logging.getLogger("irfn.estimate")

# Orden fijo de los parametros naturales para el vector g(theta) del metodo delta.
_NATURAL_ORDER = ["mu", "v", "alpha", "gamma", "beta", "omega"]


@dataclass
class FitResult:
    K: int
    params: dict[str, np.ndarray]      # parametros naturales en el optimo
    theta: np.ndarray                  # vector sin restricciones en el optimo
    loglik: float                      # NO penalizada, aunque l1_lambda > 0
    se: dict[str, np.ndarray]          # errores estandar de los params naturales
    bic: float
    aic: float
    n_converged: int                   # arranques dentro de 1e-4 del mejor loglik
    n_starts: int
    seed: int
    hessian_ok: bool | None            # None si no se calcularon SE (compute_se=False)
    P: np.ndarray = field(default=None)
    se_P: np.ndarray = field(default=None)
    dist: str = "normal"               # "normal" | "t" (V1)
    n_cov: int = 0                     # covariables del TVTP (0 = P constante)
    l1_lambda: float = 0.0             # penalizacion L1 usada (elegida por CV)


def _smooth_l1(x: np.ndarray, eps: float) -> float:
    """Suavizado diferenciable de sum|x|: sum sqrt(x^2+eps^2) - eps (ver docstring)."""
    return float(np.sum(np.sqrt(x * x + eps * eps) - eps))


def _build_full_X(
    X_lagged: np.ndarray | None,
    params: dict,
    surprise_spec: dict | None,
) -> np.ndarray | None:
    """Ensambla la matriz de covariables COMPLETA del logit de transicion.

    Con capa de sorpresa (surprise_spec no None), la columna surprise_index es
    funcion del parametro LIBRE delta (unpack -> params["delta"]): se reconstruye
    en CADA evaluacion de la verosimilitud llamando surprise_spec["si_of_delta"]
    (un callable delta -> columna (T,) ya rezagada, rellenada y estandarizada por
    el llamador) y se APPENDEA como ULTIMA covariable. Asi delta se estima por MLE
    del modelo completo (R7): la verosimilitud depende de delta a traves de esta
    columna. Sin surprise_spec, X es la tecnica tal cual (V0/V1 exactos)."""
    if surprise_spec is None:
        return X_lagged
    si_col = np.asarray(surprise_spec["si_of_delta"](params["delta"]), dtype=float).reshape(-1, 1)
    if X_lagged is None:
        return si_col
    return np.column_stack([X_lagged, si_col])


def _neg_loglik(
    theta: np.ndarray,
    r: np.ndarray,
    K: int,
    X_lagged: np.ndarray | None = None,
    dist: str = "normal",
    n_cov: int = 0,
    l1_lambda: float = 0.0,
    l1_smooth_eps: float = 0.0,
    surprise_spec: dict | None = None,
    surprise: bool = False,
) -> float:
    """-loglik (+ penalizacion L1 sobre los beta del TVTP si l1_lambda > 0).

    Los interceptos d NO se penalizan: la penalizacion contrae el EFECTO de las
    covariables hacia el modelo de matriz constante (V0 anidado), no la matriz
    misma hacia la uniforme. Con surprise=True, theta trae delta y la columna
    surprise_index se reconstruye de delta en cada evaluacion (ver _build_full_X).
    """
    try:
        params = unpack(theta, K, n_cov=n_cov, dist=dist, surprise=surprise)
        X_full = _build_full_X(X_lagged, params, surprise_spec)
        _, _, ll = hamilton_filter(r, params, K, X_lagged=X_full)
    except (ValueError, FloatingPointError):
        return 1e12
    if not np.isfinite(ll):
        return 1e12
    pen = 0.0
    if l1_lambda > 0.0 and n_cov > 0:
        pen = l1_lambda * _smooth_l1(params["beta_tvtp"], l1_smooth_eps)
    return -ll + pen


def _random_start(
    r: np.ndarray,
    K: int,
    rng: np.random.Generator,
    n_cov: int = 0,
    dist: str = "normal",
    surprise: bool = False,
    delta_init: float = np.log(2.0) / 25.0,
) -> np.ndarray:
    """Genera un theta inicial razonable muestreando parametros naturales y
    empaquetandolos. Diagonal de P dominante para arrancar en regimenes
    persistentes (evita partir de una cadena que barre estados cada dia).

    V1: nu inicial moderada (colas claramente no Normales pero varianza comoda)
    y betas TVTP cerca de 0 con ruido pequeno -- el arranque natural es el
    modelo constante anidado; el ruido rompe la simetria entre arranques.
    Los rangos de muestreo son heuristicas del OPTIMIZADOR (como los de V0),
    no pesos del modelo: todo lo que se reporta se estima por MLE (R7).
    """
    s2 = float(np.var(r))
    sd = float(np.std(r))

    mu = rng.normal(scale=0.5 * sd, size=K)

    # v estrictamente creciente alrededor de la varianza muestral
    v = np.empty(K)
    v[0] = s2 * rng.uniform(0.2, 0.8)
    for k in range(1, K):
        v[k] = v[k - 1] + s2 * rng.uniform(0.5, 3.0)

    p = rng.uniform(0.85, 0.99, size=K)              # persistencia alta
    # reparto (u_a, u_g, u_b) por Dirichlet suave -> alpha, gamma, beta
    u = rng.dirichlet(np.ones(3), size=K)            # (K, 3)
    alpha = p * u[:, 0]
    gamma = 2.0 * p * u[:, 1]
    beta = p * u[:, 2]

    if K == 1:
        P = np.array([[1.0]])
    else:
        P = np.empty((K, K))
        for i in range(K):
            stay = rng.uniform(0.9, 0.98)
            off = (1.0 - stay) / (K - 1)
            P[i, :] = off
            P[i, i] = stay

    params = {"mu": mu, "v": v, "alpha": alpha, "gamma": gamma, "beta": beta, "P": P}
    if dist == "t":
        params["nu"] = rng.uniform(5.0, 15.0, size=K)
    if n_cov > 0:
        params["beta_tvtp"] = rng.normal(scale=0.1, size=(K, K - 1, n_cov))
    if surprise:
        # delta inicial cerca del arranque del proyecto (vida media ~25 dias),
        # perturbado en log para romper simetria entre arranques. Es heuristica
        # del OPTIMIZADOR (como los demas rangos), no un peso del modelo (R7).
        params["delta"] = float(delta_init * np.exp(rng.normal(scale=0.3)))
    return pack(params, K, n_cov=n_cov, dist=dist, surprise=surprise)


def _numerical_hessian(f, x: np.ndarray, rel_step: float = 1e-4) -> np.ndarray:
    """Hessiano por diferencias finitas centrales (segundo orden) de f en x."""
    n = x.size
    h = rel_step * (np.abs(x) + rel_step)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            xi = np.zeros(n)
            xi[i] = h[i]
            xj = np.zeros(n)
            xj[j] = h[j]
            fpp = f(x + xi + xj)
            fpm = f(x + xi - xj)
            fmp = f(x - xi + xj)
            fmm = f(x - xi - xj)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j])
    return H


def _natural_vector(
    theta: np.ndarray, K: int, n_cov: int = 0, dist: str = "normal", surprise: bool = False
) -> np.ndarray:
    """g(theta): concatena los parametros naturales (para el jacobiano del delta
    method). Incluye mu, v, alpha, gamma, beta, omega, la matriz P aplanada y,
    si aplica, nu, los beta del TVTP y delta (todos por el mismo camino para
    unificar el reporte de SE).
    """
    p = unpack(theta, K, n_cov=n_cov, dist=dist, surprise=surprise)
    blocks = [p[name] for name in _NATURAL_ORDER]
    blocks.append(p["P"].reshape(-1))
    if dist == "t":
        blocks.append(p["nu"])
    if n_cov > 0:
        blocks.append(p["beta_tvtp"].reshape(-1))
    if surprise:
        blocks.append(np.array([p["delta"]]))
    return np.concatenate(blocks)


def _numerical_jacobian(g, x: np.ndarray, rel_step: float = 1e-6) -> np.ndarray:
    """Jacobiano (m x n) de g: R^n -> R^m por diferencias centrales."""
    n = x.size
    g0 = g(x)
    m = g0.size
    J = np.empty((m, n))
    h = rel_step * (np.abs(x) + rel_step)
    for j in range(n):
        xp = x.copy()
        xp[j] += h[j]
        xm = x.copy()
        xm[j] -= h[j]
        J[:, j] = (g(xp) - g(xm)) / (2.0 * h[j])
    return J


def _standard_errors(
    theta: np.ndarray,
    r: np.ndarray,
    K: int,
    X_lagged: np.ndarray | None = None,
    dist: str = "normal",
    n_cov: int = 0,
    surprise_spec: dict | None = None,
    surprise: bool = False,
):
    """SE de los parametros naturales via hessiano invertido + metodo delta.

    cov_theta = inv(H)  con H = hessiano de la negloglik (informacion observada).
    cov_nat   = J cov_theta J^T   con J = d(natural)/d(theta).
    Devuelve (se_named_dict, se_P_matrix, hessian_ok).

    El hessiano es SIEMPRE el de la verosimilitud NO penalizada (l1_lambda=0
    aqui): con penalizacion activa los SE son la aproximacion condicional
    documentada en el docstring del modulo. Con surprise=True, delta entra en el
    hessiano y el jacobiano (su SE se reporta al final).
    """
    H = _numerical_hessian(
        lambda t: _neg_loglik(t, r, K, X_lagged=X_lagged, dist=dist, n_cov=n_cov,
                              surprise_spec=surprise_spec, surprise=surprise),
        theta,
    )
    hessian_ok = True
    try:
        cov_theta = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        hessian_ok = False
        cov_theta = np.linalg.pinv(H)

    diag = np.diag(cov_theta)
    if np.any(diag <= 0) or not np.all(np.isfinite(diag)):
        hessian_ok = False

    J = _numerical_jacobian(
        lambda t: _natural_vector(t, K, n_cov=n_cov, dist=dist, surprise=surprise), theta
    )
    cov_nat = J @ cov_theta @ J.T
    se_flat = np.sqrt(np.clip(np.diag(cov_nat), 0.0, None))

    # desempaquetar se_flat en el mismo orden que _natural_vector
    se: dict[str, np.ndarray] = {}
    idx = 0
    for name in _NATURAL_ORDER:
        se[name] = se_flat[idx : idx + K]
        idx += K
    se_P = se_flat[idx : idx + K * K].reshape(K, K)
    idx += K * K
    if dist == "t":
        se["nu"] = se_flat[idx : idx + K]
        idx += K
    if n_cov > 0:
        se["beta_tvtp"] = se_flat[idx : idx + K * (K - 1) * n_cov].reshape(K, K - 1, n_cov)
        idx += K * (K - 1) * n_cov
    if surprise:
        se["delta"] = se_flat[idx : idx + 1]

    return se, se_P, hessian_ok


def fit(
    r: np.ndarray,
    K: int,
    n_starts: int = 30,
    seed: int = 42,
    compute_se: bool = True,
    X_lagged: np.ndarray | None = None,
    dist: str = "normal",
    l1_lambda: float = 0.0,
    l1_smooth_eps: float | None = None,
    surprise_spec: dict | None = None,
) -> FitResult:
    """Ajusta el MS-GJR-GARCH K-regimenes por MLE con multistart.

    Parametros
    ----------
    r : (T,) retornos.
    K : numero de regimenes.
    n_starts : arranques aleatorios (R6 pide 20-50; default 30).
    seed : semilla fija, registrada en el resultado (R6).
    compute_se : si False, se salta el hessiano/delta method (SE vacios,
        hessian_ok=None). Util para barridos donde solo interesan los puntos
        estimados, no su incertidumbre.
    X_lagged : (T, n_cov) covariables del TVTP, fila t = x_{t-1} (contrato de
        rezago en models/tvtp.py; el rezago viene de la capa de features, R3).
        None = matriz de transicion constante (V0). Con surprise_spec, aqui van
        SOLO las covariables tecnicas/macro; la columna surprise_index la anade
        internamente el optimizador a partir de delta (ver abajo).
    dist : "normal" | "t" (t de Student con nu_k libre por regimen, V1).
    l1_lambda : penalizacion L1 sobre los beta del TVTP. El valor NO se elige
        aqui: viene de cv_l1_lambda (CV dentro del train) o de config.
    l1_smooth_eps : eps del suavizado de |x| (config v1.tvtp.l1_smooth_eps).
        Obligatorio si l1_lambda > 0 (cero constantes magicas en codigo).
    surprise_spec : capa de sorpresa (V2). dict con clave "si_of_delta": un
        callable delta -> columna (T,) de surprise_index YA rezagada (R3),
        rellenada y estandarizada. Cuando se pasa, delta = exp(d_raw) entra al
        vector de parametros y se estima por MLE del modelo completo (R7): la
        columna surprise_index se reconstruye de delta en cada evaluacion y se
        appendea como ULTIMA covariable del logit. n_cov efectivo = tecnicas + 1.
    """
    r = np.asarray(r, dtype=float)
    surprise = surprise_spec is not None
    n_tech = 0 if X_lagged is None else int(X_lagged.shape[1])
    # n_cov cuenta la columna de sorpresa (su beta vive en beta_tvtp); delta es un
    # escalar aparte que gobierna esa columna (ver params.n_params surprise=True).
    n_cov = n_tech + (1 if surprise else 0)
    if X_lagged is not None:
        X_lagged = np.ascontiguousarray(X_lagged, dtype=float)
        if X_lagged.shape[0] != r.shape[0]:
            raise ValueError("X_lagged no alinea con r.")
        if not np.all(np.isfinite(X_lagged)):
            raise ValueError("X_lagged contiene NaN/inf; alinear y limpiar antes de fit.")
    if surprise and n_cov == 0:
        raise ValueError("surprise_spec requiere al menos la columna de sorpresa (n_cov>=1).")
    if l1_lambda > 0.0 and l1_smooth_eps is None:
        raise ValueError("l1_lambda > 0 requiere l1_smooth_eps (de config, no un default magico).")
    eps = 0.0 if l1_smooth_eps is None else float(l1_smooth_eps)

    T = r.shape[0]
    rng = np.random.default_rng(seed)
    args = (r, K, X_lagged, dist, n_cov, l1_lambda, eps, surprise_spec, surprise)

    best = None
    logliks = []
    for _ in range(n_starts):
        x0 = _random_start(r, K, rng, n_cov=n_cov, dist=dist, surprise=surprise)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                _neg_loglik,
                x0,
                args=args,
                method="L-BFGS-B",
                options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-7},
            )
        if not np.isfinite(res.fun) or res.fun >= 1e11:
            continue
        logliks.append(-res.fun)
        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        raise RuntimeError("Ningun arranque convergio a un optimo valido.")

    # loglik NO penalizada en el optimo: es la que se reporta y la del BIC.
    best_loglik = -_neg_loglik(best.x, r, K, X_lagged=X_lagged, dist=dist, n_cov=n_cov,
                               surprise_spec=surprise_spec, surprise=surprise)
    logliks = np.array(logliks)
    # "Mismo optimo" = loglik a distancia despreciable del mejor. La tolerancia
    # escala con T porque la log-verosimilitud es una suma de T terminos: un
    # umbral absoluto fijo (p.ej. 1e-4) seria absurdamente estricto para T grande
    # (dos arranques en el mismo optimo difieren tipicamente en ~1e-2 sobre
    # T=5000) y demasiado laxo para T pequeno. 1e-5 por observacion separa el
    # optimo global de los maximos locales inferiores sin fragmentar el cluster.
    # El cluster se mide sobre el OBJETIVO optimizado (penalizado si lambda>0):
    # logliks guarda -res.fun, comparable con -best.fun, no con best_loglik.
    conv_tol = 1e-5 * T
    n_converged = int(np.sum(np.abs(logliks - (-best.fun)) < conv_tol))

    # R6: reportar cuantos arranques alcanzan el optimo global. Pocos => la
    # superficie es multimodal y el basin global es dificil de hallar; hay que
    # decirlo (no es prueba de mala identificacion estadistica, pero si una
    # senal de que conviene mas multistart antes de confiar en el punto).
    if n_converged < max(1, n_starts // 3):
        logger.warning(
            "MULTISTART: solo %d de %d arranques alcanzaron el optimo global "
            "(loglik=%.4f, tol=%.3g). Superficie multimodal con maximos locales "
            "inferiores; considerar mas arranques y revisar la estabilidad del punto.",
            n_converged,
            n_starts,
            best_loglik,
            conv_tol,
        )

    theta = best.x
    params = unpack(theta, K, n_cov=n_cov, dist=dist, surprise=surprise)
    if compute_se:
        se, se_P, hessian_ok = _standard_errors(
            theta, r, K, X_lagged=X_lagged, dist=dist, n_cov=n_cov,
            surprise_spec=surprise_spec, surprise=surprise,
        )
    else:
        se, se_P, hessian_ok = {}, None, None

    # BIC/AIC con la loglik NO penalizada y TODOS los parametros contados.
    # Con L1 activa el numero "efectivo" de parametros es menor (algunos beta
    # se contraen a ~0); contarlos todos es la eleccion CONSERVADORA contra el
    # propio modelo TVTP -- si aun asi gana en BIC, gana de verdad.
    k_free = n_params(K, n_cov=n_cov, dist=dist, surprise=surprise)
    bic = -2.0 * best_loglik + k_free * np.log(T)
    aic = -2.0 * best_loglik + 2.0 * k_free

    return FitResult(
        K=K,
        params=params,
        theta=theta,
        loglik=best_loglik,
        se=se,
        bic=bic,
        aic=aic,
        n_converged=n_converged,
        n_starts=n_starts,
        seed=seed,
        hessian_ok=hessian_ok,
        P=params["P"],
        se_P=se_P,
        dist=dist,
        n_cov=n_cov,
        l1_lambda=l1_lambda,
    )


def cv_l1_lambda(
    r_train: np.ndarray,
    X_train: np.ndarray,
    K: int,
    *,
    dist: str,
    l1_grid: list[float],
    val_frac: float,
    n_starts: int,
    seed: int,
    l1_smooth_eps: float,
) -> tuple[float, list[dict]]:
    """Elige el lambda de la penalizacion L1 por CV temporal DENTRO del train.

    LA REGLA QUE NO SE NEGOCIA: esta funcion recibe SOLO datos de entrenamiento
    del bloque. El test jamas entra aqui. Si el lambda se eligiera mirando el
    test, toda la validacion out-of-sample del proyecto quedaria invalidada
    (seria seleccion de hiperparametros sobre la muestra de evaluacion).

    Diseno: split temporal unico train_interno / validacion_interna (la ultima
    fraccion val_frac del bloque de train, respetando el orden temporal: nada de
    K-fold barajado, que en series de tiempo mezcla pasado y futuro). Para cada
    lambda de la malla se ajusta en el train interno (multistart reducido,
    config cv_n_starts) y se evalua la log-verosimilitud predictiva por
    observacion en la validacion interna, filtrando la ventana CONTINUA para
    arrastrar el estado (misma logica que el walk-forward). Gana el lambda con
    mayor loglik predictiva media; se devuelve tambien la tabla completa para el
    reporte (no se esconden los lambdas perdedores).
    """
    r_train = np.asarray(r_train, dtype=float)
    X_train = np.ascontiguousarray(X_train, dtype=float)
    T = r_train.shape[0]
    n_val = max(1, int(round(T * val_frac)))
    n_inner = T - n_val
    if n_inner < 2 * n_val:
        logger.warning(
            "CV de lambda: train interno (%d) pequeno frente a validacion (%d).",
            n_inner, n_val,
        )

    rows: list[dict] = []
    best_lambda, best_ll = None, -np.inf
    for lam in l1_grid:
        fr = fit(
            r_train[:n_inner],
            K,
            n_starts=n_starts,
            seed=seed,
            compute_se=False,
            X_lagged=X_train[:n_inner],
            dist=dist,
            l1_lambda=lam,
            l1_smooth_eps=l1_smooth_eps,
        )
        # Filtrado de la ventana continua train_interno+validacion con los
        # parametros del train interno: la validacion arrastra el estado, igual
        # que el test arrastra el del train en el walk-forward.
        ll_all = predictive_loglik_per_obs(r_train, fr.params, K, X_lagged=X_train)
        ll_val = float(ll_all[n_inner:].mean())
        rows.append(
            {
                "l1_lambda": lam,
                "val_loglik_per_obs": ll_val,
                "train_loglik_per_obs": float(ll_all[:n_inner].mean()),
                "n_converged": fr.n_converged,
            }
        )
        if ll_val > best_ll:
            best_ll, best_lambda = ll_val, lam

    return float(best_lambda), rows

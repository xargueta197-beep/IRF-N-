"""Parametrizacion sin restricciones del MS-GJR-GARCH (Haas et al. 2004).

El optimizador (L-BFGS-B) trabaja en R^n SIN restricciones. Todas las
restricciones del modelo se imponen por TRANSFORMACION, no por penalizacion ni
por correccion post-hoc. Este modulo es el unico lugar donde vive esa
transformacion: `unpack` va de R^n a los parametros naturales del modelo, y
`pack` es su inversa exacta.

--------------------------------------------------------------------------
POR QUE omega_k NO es un parametro libre
--------------------------------------------------------------------------
En un GJR-GARCH la varianza incondicional es, en estacionariedad,

    v_k = omega_k / (1 - (alpha_k + gamma_k/2 + beta_k))

(el gamma/2 sale de E[eps^2 * 1{eps<0}] = (1/2) E[sigma^2] para innovaciones
simetricas). Despejando:

    omega_k = v_k * (1 - kappa_k),   con kappa_k = alpha_k + gamma_k/2 + beta_k

Parametrizamos DIRECTAMENTE v_k (la varianza incondicional) y la persistencia
kappa_k, y dejamos que omega_k caiga por definicion. Si omega_k fuera libre, la
varianza incondicional seria una funcion enredada de cuatro parametros y la
restriccion de orden de R5 (abajo) tendria que imponerse sobre esa funcion, no
sobre un parametro directo. Al hacer v_k explicito, R5 se vuelve trivial de
imponer y v_k significa EXACTAMENTE lo que dice significar: la varianza
incondicional del regimen.

--------------------------------------------------------------------------
POR QUE el orden v_1 < v_2 < ... < v_K va AQUI y no post-hoc (R5)
--------------------------------------------------------------------------
    v_1 = exp(a_1)
    v_k = v_{k-1} + exp(a_k)   para k = 2..K

Como exp(.) > 0, cada v_k es estrictamente mayor que el anterior POR
CONSTRUCCION, para cualquier vector a en R^K. El optimizador no puede jamas
proponer un punto que viole el orden: no existe en el espacio de busqueda.

Si en cambio ordenaramos los regimenes despues de estimar (post-hoc), dos
bloques de walk-forward podrian converger a etiquetados distintos y el
"regimen 2" del bloque i no seria el mismo objeto que el "regimen 2" del bloque
i+1 (label switching). Imponer el orden en la parametrizacion mata el label
switching de raiz: el regimen de mayor varianza es SIEMPRE el ultimo indice, en
todos los bloques, sin post-proceso.

--------------------------------------------------------------------------
Reparto de la persistencia entre (alpha, gamma, beta)
--------------------------------------------------------------------------
    p_k = sigmoid(b_k)                      # persistencia kappa_k, en (0,1)
    (u_a, u_g, u_b) = softmax(c_k1, c_k2, 0)   # simplex de 3, gauge fija
    alpha_k = p_k * u_a
    gamma_k = 2 * p_k * u_g                 # el factor 2 hace kappa_k = p_k EXACTO
    beta_k  = p_k * u_b

Con esto kappa_k = alpha_k + gamma_k/2 + beta_k = p_k*(u_a + u_g + u_b) = p_k,
asi que omega_k = v_k*(1 - p_k) hace que v_k sea la varianza incondicional
verdadera. Sin el factor 2, kappa_k = p_k - gamma_k/2 y v_k dejaria de ser la
varianza incondicional (decision de diseno tomada con el director, R5).

Positividad y estacionariedad quedan garantizadas: alpha,beta >= 0; gamma >= 0
=> alpha+gamma >= 0 (positividad del GJR); kappa_k = p_k < 1 (estacionariedad).

--------------------------------------------------------------------------
Gauge fija del softmax
--------------------------------------------------------------------------
El softmax es invariante a sumar una constante a sus logits, asi que 3 logits
libres representarian el mismo simplex de infinitas formas (parametro redundante
que estorba al optimizador y rompe pack(unpack(x))==x). Fijamos la ultima
componente a 0 (categoria de referencia): 2 logits libres por regimen para el
reparto GARCH, y K-1 logits libres por fila para la matriz de transicion. Asi la
representacion vectorial es MINIMA y unica, y el round-trip es exacto.

--------------------------------------------------------------------------
Extensiones V1 (retrocompatibles: los defaults reproducen V0 exacto)
--------------------------------------------------------------------------
1. Innovaciones t de Student (dist="t"): un grado de libertad LIBRE por regimen,
       nu_k = 2 + exp(g_k)
   La cota nu_k > 2 no es cosmetica: garantiza varianza finita, que es lo que
   hace que sigma2_{k,t} SIGA siendo la varianza condicional verdadera y que v_k
   siga siendo la varianza incondicional (R5 intacto). La densidad se escala en
   msgarch.student_t_logpdf para que sigma2 sea la varianza, no el scale^2.
   Quien decide entre Normal y t es el BIC (V1), no nosotros.

2. TVTP (n_cov > 0): la matriz de transicion deja de ser constante,
       logit_ij(t) = d_ij + beta_ij' x_{t-1},   p_ij(t) = softmax_j(logit_ij(t))
   con la MISMA gauge que en V0: la columna K es la referencia, es decir
   beta_{i,K} = 0 y d_{i,K} = 0 para todo i (identificacion del logit
   multinomial, CLAUDE.md). Los interceptos d son EXACTAMENTE los logits `d` de
   V0: con x = 0 (covariables estandarizadas en su media de entrenamiento) el
   modelo TVTP colapsa al modelo de matriz constante. Eso significa que V0 esta
   ANIDADO en V1 y que "P" en el dict de salida es la matriz evaluada en
   condiciones medias -- la que usa la inicializacion estacionaria del filtro.

Layout del vector theta (los bloques V0 conservan su posicion; lo nuevo se
APPENDEA, para que theta de V0 sea un prefijo valido):
    [ mu(K) | a(K) | b(K) | c(2K) | d(K*(K-1)) | g(K si dist="t") | B(K*(K-1)*n_cov) ]
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p) - np.log1p(-p)


def _softmax_ref(logits_free: np.ndarray) -> np.ndarray:
    """softmax de [logits_free..., 0] (ultima categoria de referencia, gauge fija)."""
    full = np.concatenate([logits_free, [0.0]])
    m = full.max()
    e = np.exp(full - m)
    return e / e.sum()


def n_params(K: int, n_cov: int = 0, dist: str = "normal", surprise: bool = False) -> int:
    """Dimension del vector theta para K regimenes.

    mu(K) + a(K) + b(K) + c(2K) + transicion(K*(K-1))
    + g(K si dist="t") + B(K*(K-1)*n_cov si TVTP) + 1 (delta si surprise).

    Con surprise=True se anade UN escalar: el decaimiento delta del indice de
    sorpresa (delta = exp(d_raw), d_raw libre). n_cov ya CUENTA la columna
    surprise_index entre las covariables del logit (su beta vive en beta_tvtp);
    lo que delta anade es la tasa de decaimiento que genera esa columna (V2).
    """
    n = 5 * K + K * (K - 1)
    if dist == "t":
        n += K
    n += K * (K - 1) * n_cov
    if surprise:
        n += 1
    return n


def _slices(K: int, n_cov: int = 0, dist: str = "normal", surprise: bool = False) -> dict[str, slice]:
    sl = {
        "mu": slice(0, K),
        "a": slice(K, 2 * K),
        "b": slice(2 * K, 3 * K),
        "c": slice(3 * K, 5 * K),
        "d": slice(5 * K, 5 * K + K * (K - 1)),
    }
    pos = 5 * K + K * (K - 1)
    if dist == "t":
        sl["g"] = slice(pos, pos + K)
        pos += K
    if n_cov > 0:
        sl["B"] = slice(pos, pos + K * (K - 1) * n_cov)
        pos += K * (K - 1) * n_cov
    if surprise:
        sl["s"] = slice(pos, pos + 1)   # d_raw de delta = exp(d_raw)
    return sl


def unpack(
    theta: np.ndarray, K: int, n_cov: int = 0, dist: str = "normal", surprise: bool = False
) -> dict[str, np.ndarray]:
    """R^n -> parametros naturales del modelo.

    Devuelve un dict con arrays (todos de longitud K salvo los indicados):
        mu, v, p, alpha, gamma, beta, omega, P (KxK), d (K x K-1)
    y ademas, segun la especificacion:
        nu (K,)                  si dist="t"     -- grados de libertad, nu_k > 2
        beta_tvtp (K, K-1, n_cov) si n_cov > 0   -- betas del logit; la columna de
                                                    referencia K es 0 implicito
    "P" es softmax(d) por filas: con TVTP es la matriz en covariables MEDIAS
    (x=0 tras estandarizar); con n_cov=0 es la matriz constante de V0.
    """
    if dist not in ("normal", "t"):
        raise ValueError(f"dist debe ser 'normal' o 't', no {dist!r}")
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (n_params(K, n_cov, dist, surprise),):
        raise ValueError(
            f"theta debe tener {n_params(K, n_cov, dist, surprise)} entradas para "
            f"K={K}, n_cov={n_cov}, dist={dist}, surprise={surprise}; tiene {theta.shape}"
        )

    sl = _slices(K, n_cov, dist, surprise)
    mu = theta[sl["mu"]].copy()
    a = theta[sl["a"]]
    b = theta[sl["b"]]
    c = theta[sl["c"]].reshape(K, 2)

    # v estrictamente creciente por construccion (R5)
    v = np.cumsum(np.exp(a))

    # persistencia en (0,1)
    p = _sigmoid(b)

    alpha = np.empty(K)
    gamma = np.empty(K)
    beta = np.empty(K)
    for k in range(K):
        u_a, u_g, u_b = _softmax_ref(c[k])
        alpha[k] = p[k] * u_a
        gamma[k] = 2.0 * p[k] * u_g  # factor 2: kappa_k = p_k exacto
        beta[k] = p[k] * u_b

    omega = v * (1.0 - p)  # omega NO es libre

    if K == 1:
        d = np.zeros((1, 0))
        P = np.array([[1.0]])
    else:
        d = theta[sl["d"]].reshape(K, K - 1).copy()
        P = np.vstack([_softmax_ref(d[i]) for i in range(K)])

    out = {
        "mu": mu,
        "v": v,
        "p": p,
        "alpha": alpha,
        "gamma": gamma,
        "beta": beta,
        "omega": omega,
        "P": P,
        "d": d,
    }
    if dist == "t":
        # nu_k = 2 + exp(g_k) > 2: varianza finita SIEMPRE, para que sigma2 siga
        # siendo la varianza condicional y v_k la incondicional (R5 intacto).
        out["nu"] = 2.0 + np.exp(theta[sl["g"]])
    if n_cov > 0:
        # beta_{i,K} = 0 para todo i (identificacion): solo se almacenan las
        # K-1 columnas libres; la referencia queda implicita, igual que en d.
        out["beta_tvtp"] = theta[sl["B"]].reshape(K, K - 1, n_cov).copy()
    if surprise:
        # delta = exp(d_raw) > 0: tasa de decaimiento del indice de sorpresa. El
        # optimizador del modelo completo la mueve libremente en R (R7); la
        # transformacion garantiza delta > 0 sin restriccion de caja.
        out["delta"] = float(np.exp(theta[sl["s"]][0]))
    return out


def pack(
    params: dict[str, np.ndarray], K: int, n_cov: int = 0, dist: str = "normal",
    surprise: bool = False,
) -> np.ndarray:
    """Parametros naturales -> R^n. Inversa exacta de unpack.

    Lee mu, v, alpha, gamma, beta, P del dict (p y omega se derivan); con
    dist="t" lee ademas nu (>2) y con n_cov>0 lee beta_tvtp (K, K-1, n_cov).
    Requiere v estrictamente creciente y persistencia p = alpha + gamma/2 + beta
    en (0,1).
    """
    if dist not in ("normal", "t"):
        raise ValueError(f"dist debe ser 'normal' o 't', no {dist!r}")
    mu = np.asarray(params["mu"], dtype=float)
    v = np.asarray(params["v"], dtype=float)
    alpha = np.asarray(params["alpha"], dtype=float)
    gamma = np.asarray(params["gamma"], dtype=float)
    beta = np.asarray(params["beta"], dtype=float)
    P = np.asarray(params["P"], dtype=float)

    if np.any(np.diff(v) <= 0):
        raise ValueError(f"v debe ser estrictamente creciente (R5): {v}")

    # a: invierte v = cumsum(exp(a)) -> exp(a) = diferencias sucesivas
    incr = np.empty(K)
    incr[0] = v[0]
    incr[1:] = np.diff(v)
    if np.any(incr <= 0):
        raise ValueError(f"incrementos de v no positivos: {incr}")
    a = np.log(incr)

    # b: p = kappa = alpha + gamma/2 + beta
    p = alpha + gamma / 2.0 + beta
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError(f"persistencia p fuera de (0,1): {p}")
    b = _logit(p)

    # c: recupera los pesos del simplex y sus logits con referencia u_b
    c = np.empty((K, 2))
    for k in range(K):
        u_a = alpha[k] / p[k]
        u_g = (gamma[k] / 2.0) / p[k]
        u_b = beta[k] / p[k]
        # u_a + u_g + u_b = 1 por construccion; logits relativos a la referencia u_b
        c[k, 0] = np.log(u_a) - np.log(u_b)
        c[k, 1] = np.log(u_g) - np.log(u_b)

    parts = [mu, a, b, c.reshape(-1)]

    if K > 1:
        d = np.empty((K, K - 1))
        for i in range(K):
            ref = P[i, K - 1]
            d[i] = np.log(P[i, : K - 1]) - np.log(ref)
        parts.append(d.reshape(-1))

    if dist == "t":
        nu = np.asarray(params["nu"], dtype=float)
        if np.any(nu <= 2.0):
            raise ValueError(f"nu debe ser > 2 (varianza finita): {nu}")
        parts.append(np.log(nu - 2.0))  # g: invierte nu = 2 + exp(g)

    if n_cov > 0:
        B = np.asarray(params["beta_tvtp"], dtype=float)
        if B.shape != (K, K - 1, n_cov):
            raise ValueError(
                f"beta_tvtp debe tener forma {(K, K - 1, n_cov)}, tiene {B.shape}"
            )
        parts.append(B.reshape(-1))

    if surprise:
        delta = float(params["delta"])
        if delta <= 0.0:
            raise ValueError(f"delta debe ser > 0 (delta = exp(d_raw)): {delta}")
        parts.append(np.array([np.log(delta)]))   # d_raw = log(delta)

    return np.concatenate(parts)

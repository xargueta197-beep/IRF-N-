"""Logit multinomial de transicion: p_ij(t) modulada por x_{t-1} (TVTP, V1).

    p_ij(t) = exp(d_ij + beta'_ij x_{t-1}) / sum_m exp(d_im + beta'_im x_{t-1})

Identificacion: beta_{i,K} = 0 y d_{i,K} = 0 para todo i (la columna K es la
categoria de referencia, misma gauge que params.py usa para la matriz constante
de V0). Sin esa restriccion el logit multinomial esta sobreparametrizado: sumar
cualquier funcion h(x) a toda una fila de logits deja p_ij identica.

--------------------------------------------------------------------------------
EL CONTRATO DE ALINEACION TEMPORAL (R3) -- leer antes de tocar nada
--------------------------------------------------------------------------------
`X_lagged` fila t contiene x_{t-1}: covariables construidas SOLO con informacion
disponible al cierre de t-1. Ese rezago NO se aplica aqui: viene aplicado desde
la capa de features (features/technical.py con .shift(1) explicito; macro por
fecha de publicacion real + margen, ver features/macro.py) y esta registrado en
el lag_ledger. Este modulo consume X_lagged tal cual: la matriz devuelta en la
posicion t es la que el filtro de Hamilton usa para la PREDICCION del paso t,
xi_{t|t-1} = P(x_{t-1})' xi_{t-1|t-1}.

Aplicar el rezago dos veces (shift en features Y [t-1] aqui) seria x_{t-2}: no
es look-ahead pero si un dia entero de informacion tirada a la basura. Aplicarlo
cero veces es look-ahead y el proyecto muere. Por eso el contrato vive en UN
solo lugar (features) y este modulo lo documenta en su nombre de argumento.

--------------------------------------------------------------------------------
COVARIABLES ADMITIDAS (V1 tecnicas/macro; V2 anade sorpresa) -- este modulo es
AGNOSTICO al numero y significado de las covariables
--------------------------------------------------------------------------------
transition_matrices/transition_matrix_at no saben ni les importa QUE covariable
es cada columna de X: reciben (T, n_cov) y aplican el softmax del logit. Las
columnas de V1 son tecnicas (sma_gap, bb_width_z) y/o macro (slope_2s10y,
hy_oas_z). En V2 se anade `surprise_index` como UNA columna mas (no reemplaza a
las tecnicas; se controla con config/news.yaml: surprise_layer).

La columna surprise_index tiene una particularidad que este modulo NO ve pero
conviene documentar: depende del parametro LIBRE delta (decaimiento del indice de
sorpresa). Por eso, cuando la capa esta activa, esa columna NO es un input fijo:
el optimizador la reconstruye de delta en cada evaluacion de la verosimilitud
(models/estimate.py, argumento surprise_spec; features/surprise.surprise_feature
la genera ya rezagada, R3). Aqui llega como una columna mas de X_lagged, ya
ensamblada; el softmax la trata igual que a cualquier otra covariable.
"""

from __future__ import annotations

import numpy as np


def _row_softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax sobre el ultimo eje, estable (resta el maximo)."""
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)


def transition_matrix_at(d: np.ndarray, B: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Matriz de transicion (K, K) evaluada en UN vector de covariables x.

    d es (K, K-1) interceptos; B es (K, K-1, n_cov) betas del logit; x es
    (n_cov,). La columna K (referencia) tiene logit 0 por identificacion.
    Es la funcion que produce la "matriz de transicion condicional de hoy"
    de la pantalla 1 (evaluada en x_{asof}, la ultima covariable disponible).
    """
    d = np.asarray(d, dtype=float)
    B = np.asarray(B, dtype=float)
    x = np.asarray(x, dtype=float)
    K = d.shape[0]
    if K == 1:
        return np.array([[1.0]])
    logits_free = d + B @ x                              # (K, K-1)
    logits = np.concatenate([logits_free, np.zeros((K, 1))], axis=1)
    return _row_softmax(logits)


def transition_matrices(
    d: np.ndarray, B: np.ndarray, X_lagged: np.ndarray
) -> np.ndarray:
    """Trayectoria completa de matrices de transicion: (T, K, K).

    X_lagged es (T, n_cov) y su fila t es x_{t-1} (ver contrato en el docstring
    del modulo). La matriz [t] es la que usa la prediccion xi_{t|t-1} del filtro.

    Con B = 0 devuelve softmax(d) repetida: el modelo de matriz constante de V0
    exacto (V0 esta anidado en V1; test_tvtp_reduces_to_constant lo verifica).
    """
    d = np.asarray(d, dtype=float)
    B = np.asarray(B, dtype=float)
    X = np.asarray(X_lagged, dtype=float)
    if X.ndim != 2:
        raise ValueError("X_lagged debe ser (T, n_cov).")
    T = X.shape[0]
    K = d.shape[0]
    if K == 1:
        return np.ones((T, 1, 1))
    # logits libres: d_ij + beta'_ij x_{t-1}  -> (T, K, K-1)
    logits_free = d[None, :, :] + np.einsum("ijc,tc->tij", B, X)
    logits = np.concatenate([logits_free, np.zeros((T, K, 1))], axis=2)
    return _row_softmax(logits)

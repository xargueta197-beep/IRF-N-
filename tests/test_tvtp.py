"""Tests del TVTP (V1).

1. Anidamiento exacto: con beta_tvtp = 0 el filtro TVTP reproduce al filtro de
   matriz constante BIT A BIT (V0 anidado en V1). Si esto se rompe, cualquier
   comparacion M1 vs M2 queda invalidada.
2. Propiedades estructurales de transition_matrices (filas estocasticas,
   identificacion con columna de referencia).
3. Recuperacion (slow): simular con betas conocidos -> estimar -> signos y
   mejora de verosimilitud sobre el modelo constante.
"""

from __future__ import annotations

import numpy as np
import pytest

from irfn.models.estimate import fit
from irfn.models.hamilton import hamilton_filter
from irfn.models.msgarch import simulate
from irfn.models.params import pack, unpack
from irfn.models.tvtp import transition_matrices, transition_matrix_at

_BASE = {
    "mu": np.array([0.05, -0.10]),
    "v": np.array([0.5, 4.0]),
    "alpha": np.array([0.05, 0.08]),
    "gamma": np.array([0.06, 0.10]),
    "beta": np.array([0.85, 0.78]),
    "P": np.array([[0.97, 0.03], [0.06, 0.94]]),
}


def _ar1(T: int, rho: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + rng.normal(scale=0.3)
    return (x - x.mean()) / x.std()


def test_tvtp_reduces_to_constant():
    """beta_tvtp = 0 => filtro TVTP == filtro constante, exactamente."""
    K = 2
    theta = pack(_BASE, K)
    r, _ = simulate(theta, K, T=600, seed=1)
    params = unpack(theta, K)

    xf0, xp0, ll0 = hamilton_filter(r, params, K)

    params_tvtp = dict(params)
    params_tvtp["beta_tvtp"] = np.zeros((K, K - 1, 3))
    X = np.random.default_rng(2).normal(size=(600, 3))
    xf1, xp1, ll1 = hamilton_filter(r, params_tvtp, K, X_lagged=X)

    assert ll0 == pytest.approx(ll1, abs=1e-9)
    np.testing.assert_allclose(xf0, xf1, atol=1e-12)
    np.testing.assert_allclose(xp0, xp1, atol=1e-12)


def test_transition_matrices_structure():
    """Filas estocasticas para cualquier x; identificacion beta_{i,K}=0; el
    signo del beta mueve la probabilidad en la direccion correcta."""
    K = 3
    rng = np.random.default_rng(3)
    d = rng.normal(size=(K, K - 1))
    B = rng.normal(size=(K, K - 1, 2))
    X = rng.normal(size=(50, 2))

    P_path = transition_matrices(d, B, X)
    assert P_path.shape == (50, K, K)
    np.testing.assert_allclose(P_path.sum(axis=2), 1.0, atol=1e-12)
    assert np.all(P_path >= 0)

    # coherencia punto-a-punto con la version de un solo x
    P_10 = transition_matrix_at(d, B, X[10])
    np.testing.assert_allclose(P_10, P_path[10], atol=1e-12)

    # monotonia: subir el logit de la columna j (para la fila i) sube p_ij.
    d2 = np.zeros((2, 1))
    B2 = np.array([[[2.0]], [[0.0]]])       # fila 0: x empuja hacia la col 0
    lo = transition_matrix_at(d2, B2, np.array([-1.0]))
    hi = transition_matrix_at(d2, B2, np.array([+1.0]))
    assert hi[0, 0] > lo[0, 0]
    np.testing.assert_allclose(lo[1], hi[1], atol=1e-12)   # fila 1 insensible (B=0)


def test_hamilton_filter_contracts():
    """El filtro exige coherencia entre params y X_lagged (contrato explicito)."""
    K = 2
    theta = pack(_BASE, K)
    r, _ = simulate(theta, K, T=100, seed=4)
    params = unpack(theta, K)

    with pytest.raises(ValueError):
        hamilton_filter(r, params, K, X_lagged=np.zeros((100, 1)))  # sin beta_tvtp

    params_tvtp = dict(params)
    params_tvtp["beta_tvtp"] = np.zeros((K, K - 1, 1))
    with pytest.raises(ValueError):
        hamilton_filter(r, params_tvtp, K)                           # falta X_lagged
    with pytest.raises(ValueError):
        hamilton_filter(r, params_tvtp, K, X_lagged=np.zeros((99, 1)))  # desalineado


@pytest.mark.slow
def test_tvtp_recovery():
    """Simular TVTP con betas conocidos -> estimar -> signos correctos y
    verosimilitud mayor que el modelo constante sobre los mismos datos.

    No se exige clavar el valor puntual: con T=3000 y el logit saturable, la
    identificacion puntual de beta es debil (por eso existe la L1); lo que NO es
    negociable es el signo del efecto y que el TVTP domine en verosimilitud al
    constante cuando el TVTP es el proceso generador.
    """
    K = 2
    T = 3000
    x = _ar1(T, rho=0.95, seed=7)
    X = x[:, None]

    true = dict(_BASE)
    true["beta_tvtp"] = np.array([[[1.0]], [[-0.8]]])
    theta_true = pack(true, K, n_cov=1)
    r, _ = simulate(theta_true, K, T=T, seed=8, n_cov=1, X_lagged=X)

    fit_tvtp = fit(r, K, n_starts=12, seed=5, compute_se=False, X_lagged=X)
    fit_const = fit(r, K, n_starts=12, seed=5, compute_se=False)

    assert fit_tvtp.loglik > fit_const.loglik, "el TVTP verdadero debe dominar al constante"
    B_hat = fit_tvtp.params["beta_tvtp"]
    assert B_hat[0, 0, 0] > 0, f"signo de beta[0] mal recuperado: {B_hat.ravel()}"
    assert B_hat[1, 0, 0] < 0, f"signo de beta[1] mal recuperado: {B_hat.ravel()}"

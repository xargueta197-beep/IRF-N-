"""Invarianza de prefijo CON TVTP y t de Student (V1).

Es la version V1 del test mas importante del repo: con covariables en la matriz
de transicion aparecen dos superficies de look-ahead nuevas que V0 no tenia:
  1. el rezago de las covariables (si x_t entrara sin rezagar, ξ en fechas <= t
     cambiaria al truncar), y
  2. la estandarizacion (si media/std se recalcularan sobre la serie truncada,
     el pasado cambiaria al ver mas futuro).
Parametros y scaler se fijan del entrenamiento (misma logica que V0: aisla la
causalidad del filtro + covariables); ver audit.pit.prefix_invariance_check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from irfn.audit.pit import PREFIX_ATOL, prefix_invariance_check
from irfn.models.params import pack


def _theta_tvtp_t():
    params = {
        "mu": np.array([0.05, -0.10]),
        "v": np.array([0.5, 4.0]),
        "alpha": np.array([0.05, 0.08]),
        "gamma": np.array([0.06, 0.10]),
        "beta": np.array([0.85, 0.78]),
        "P": np.array([[0.97, 0.03], [0.06, 0.94]]),
        "nu": np.array([6.0, 8.0]),
        "beta_tvtp": np.array([[[0.8]], [[-0.6]]]),
    }
    return pack(params, K=2, n_cov=1, dist="t")


def test_prefix_invariance_tvtp():
    K, seed = 2, 42
    theta = _theta_tvtp_t()

    # covariable exogena AR(1) "ya rezagada" (fila t = x_{t-1} por contrato)
    rng = np.random.default_rng(seed)
    T = 1200
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.95 * x[t - 1] + rng.normal(scale=0.3)
    x = (x - x.mean()) / x.std()

    from irfn.models.msgarch import simulate

    r, _ = simulate(theta, K=K, T=T, seed=seed, dist="t", n_cov=1, X_lagged=x[:, None])
    index = pd.bdate_range("2015-01-01", periods=T)
    returns = pd.Series(r, index=index, name="r")
    X = pd.DataFrame({"x": x}, index=index)

    result = prefix_invariance_check(
        returns,
        K=K,
        seed=seed,
        n_starts=6,
        train_len=600,
        dates_to_test=10,
        X=X,
        dist="t",
    )

    assert result.attrs["passed"], (
        "INVARIANZA DE PREFIJO ROTA CON TVTP: hay look-ahead en covariables, "
        "scaler o filtro.\n" + result.to_string(index=False)
    )
    assert (result["max_abs_diff"] < PREFIX_ATOL).all()
    assert len(result) == 10

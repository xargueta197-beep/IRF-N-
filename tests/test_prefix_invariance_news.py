"""Invarianza de prefijo con la capa de noticias (V2) integrada al pipeline
completo (Hamilton + TVTP), analogo de test_prefix_invariance_tvtp.py.

Distincion con test_surprise.test_si_pit_prefix_invariance (que YA existe y
verifica que sigma_i/z_i/SI_t en si mismos son causales, truncando el
CALENDARIO de eventos): este test verifica la superficie de look-ahead del
PIPELINE alrededor de la covariable de sorpresa una vez que ya es una columna
de X -- el rezago (shift(1) de surprise_feature, aplicado UNA vez antes de
entrar aqui) y la estandarizacion (scaler fijo del entrenamiento) no deben
filtrar informacion futura, igual que con cualquier otra covariable TVTP. Este
modulo NO toca features/surprise.py ni models/estimate.py: solo los usa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from irfn.audit.pit import PREFIX_ATOL, prefix_invariance_check
from irfn.features.surprise import surprise_feature
from irfn.models.msgarch import simulate
from irfn.models.params import pack


def _theta_news():
    params = {
        "mu": np.array([0.04, -0.08]),
        "v": np.array([0.6, 3.5]),
        "alpha": np.array([0.04, 0.07]),
        "gamma": np.array([0.05, 0.09]),
        "beta": np.array([0.86, 0.79]),
        "P": np.array([[0.96, 0.04], [0.07, 0.93]]),
        "beta_tvtp": np.array([[[0.9]], [[-0.7]]]),   # (K, K-1, n_cov=1: surprise_index)
    }
    return pack(params, K=2, n_cov=1, dist="normal")


def _synthetic_surprise_column(T: int, seed: int, delta: float) -> np.ndarray:
    """SI_{t-1} (ya rezagada por surprise_feature) sobre un calendario de
    eventos macro sintetico ~cada 3 semanas, rellenada y estandarizada -- misma
    receta que tests/test_surprise_mle.py."""
    rng = np.random.default_rng(seed)
    target = pd.bdate_range("2015-01-02", periods=T)
    n_events = max(1, T // 15)
    ev_dates = target[:: max(1, T // n_events)][:n_events]
    contributions = rng.normal(0.0, 1.0, size=len(ev_dates))
    events = pd.Series(contributions, index=ev_dates)

    si = surprise_feature(events, target, delta=delta)
    si = si.fillna(0.0)
    std = si.std()
    si_std = (si - si.mean()) / (std if std > 0 else 1.0)
    return si_std.to_numpy(dtype=float)


def test_prefix_invariance_news():
    K, seed = 2, 7
    theta = _theta_news()
    T = 1400
    delta = np.log(2.0) / 25.0

    x = _synthetic_surprise_column(T, seed=seed, delta=delta)

    r, _ = simulate(theta, K=K, T=T, seed=seed, n_cov=1, X_lagged=x[:, None])
    index = pd.bdate_range("2015-01-02", periods=T)
    returns = pd.Series(r, index=index, name="r")
    X = pd.DataFrame({"surprise_index": x}, index=index)

    result = prefix_invariance_check(
        returns,
        K=K,
        seed=seed,
        n_starts=6,
        train_len=700,
        dates_to_test=10,
        X=X,
        dist="normal",
    )

    assert result.attrs["passed"], (
        "INVARIANZA DE PREFIJO ROTA CON LA CAPA DE NOTICIAS: hay look-ahead en "
        "la covariable surprise_index, el scaler o el filtro.\n"
        + result.to_string(index=False)
    )
    assert (result["max_abs_diff"] < PREFIX_ATOL).all()
    assert len(result) == 10

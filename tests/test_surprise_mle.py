"""test_delta_in_mle (5d): el optimizador del modelo completo recupera delta.

Toca estimate.py (por eso vive aparte de test_surprise.py). Simula un MS-GJR-GARCH
con TVTP cuya UNICA covariable es el indice de sorpresa SI_{t-1}(delta_true); luego
estima con delta LIBRE (delta = exp(d_raw)) y verifica que lo recupera. Es la
prueba de que delta se estima por MLE del modelo completo (R7), no se fija a mano.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from irfn.features.surprise import surprise_feature
from irfn.models.estimate import fit
from irfn.models.msgarch import simulate
from irfn.models.params import pack


def _events_and_target(seed=0, n_events=180, T_days=4200):
    """Eventos macro ~cada 3 semanas (c_i = w_i z_i) y calendario diario destino."""
    rng = np.random.default_rng(seed)
    target = pd.bdate_range("2005-01-03", periods=T_days)
    ev_dates = target[:: max(1, T_days // n_events)][:n_events]
    c = rng.normal(0.0, 1.0, size=len(ev_dates))          # contribuciones w_i*z_i
    events = pd.Series(c, index=ev_dates)
    return events, target


def _si_builder(events, target):
    """Callable delta -> columna SI_{t-1}(delta) RAW (sin estandarizar), rellenada.
    RAW a proposito: delta se identifica por la escala Y la forma del decaimiento;
    estandarizar por delta lavaria la escala. Es el x_{t-1} del logit (ya con
    shift(1) de surprise_feature)."""
    def si_of_delta(delta: float) -> np.ndarray:
        si = surprise_feature(events, target, delta=float(delta))
        return si.to_numpy(dtype=float)[: len(target)] * 1.0
    # rellenar el NaN del shift(1) con 0 dentro del callable:
    def si_filled(delta: float) -> np.ndarray:
        return np.nan_to_num(si_of_delta(delta), nan=0.0)
    return si_filled


@pytest.mark.slow
def test_delta_in_mle():
    events, target = _events_and_target(seed=1)
    si_of_delta = _si_builder(events, target)

    delta_true = np.log(2.0) / 20.0                        # vida media 20 dias
    T = len(target)

    # Covariable verdadera = SI_{t-1}(delta_true), RAW. El TVTP la usa para mover
    # la transicion; beta fuerte para que delta quede identificado.
    X_true = si_of_delta(delta_true).reshape(T, 1)

    # theta verdadero: K=2, Normal, n_cov=1. Varianzas bien separadas (regimenes
    # inferibles), persistencia GARCH baja (cerca de iid por regimen), betas
    # fuertes de la sorpresa sobre la transicion.
    params_true = {
        "mu": np.array([0.0, 0.0]),
        "v": np.array([1.0, 9.0]),
        "alpha": np.array([0.03, 0.03]),
        "gamma": np.array([0.02, 0.02]),
        "beta": np.array([0.04, 0.04]),
        "P": np.array([[0.88, 0.12], [0.12, 0.88]]),
        "beta_tvtp": np.array([[[-2.5]], [[-2.5]]]),       # (K, K-1, n_cov)
    }
    theta_true = pack(params_true, K=2, n_cov=1, dist="normal")
    r, _ = simulate(theta_true, K=2, T=T, seed=7, n_cov=1, X_lagged=X_true)

    # Estimacion: delta LIBRE via surprise_spec. n_cov efectivo = 1 (la sorpresa).
    fr = fit(
        r, K=2, n_starts=12, seed=3, compute_se=True,
        X_lagged=None, dist="normal",
        surprise_spec={"si_of_delta": si_of_delta},
    )
    delta_hat = fr.params["delta"]
    se_delta = float(fr.se["delta"][0]) if fr.se.get("delta") is not None else float("nan")

    # 1) recuperacion puntual razonable (la superficie es multimodal; tolerancia
    #    generosa en escala relativa) y 2) IC 95% cubre delta_true si el hessiano
    #    es fiable.
    assert delta_hat == pytest.approx(delta_true, rel=0.5), \
        f"delta_hat={delta_hat:.4f} lejos de delta_true={delta_true:.4f}"
    if fr.hessian_ok and np.isfinite(se_delta) and se_delta > 0:
        lo, hi = delta_hat - 1.96 * se_delta, delta_hat + 1.96 * se_delta
        assert lo <= delta_true <= hi, \
            f"delta_true={delta_true:.4f} fuera de IC95 [{lo:.4f}, {hi:.4f}]"

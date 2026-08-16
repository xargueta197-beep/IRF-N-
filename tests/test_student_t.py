"""Tests de las innovaciones t de Student (V1).

La eleccion Normal vs t la hace el BIC; estos tests verifican que la t este BIEN
IMPLEMENTADA para que esa eleccion sea entre dos modelos correctos:
  1. la densidad integra a 1 y coincide con scipy.stats.t reescalada;
  2. limite nu -> inf == Normal;
  3. sigma2 es la VARIANZA de verdad (el escalado (nu-2)/nu esta bien puesto);
  4. (slow) el MLE recupera nu sobre datos simulados con colas gordas.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from irfn.models.msgarch import normal_logpdf, simulate, student_t_logpdf
from irfn.models.params import pack


def test_student_t_matches_scipy():
    """Nuestra log-densidad == scipy.stats.t con loc/scale equivalentes."""
    rng = np.random.default_rng(0)
    T, K = 200, 2
    r = rng.normal(scale=2.0, size=T)
    mu = np.array([0.3, -0.5])
    sigma2 = np.abs(rng.normal(1.5, 0.3, size=(T, K)))
    nu = np.array([4.5, 12.0])

    ours = student_t_logpdf(r, mu, sigma2, nu)
    for k in range(K):
        scale = np.sqrt(sigma2[:, k] * (nu[k] - 2.0) / nu[k])
        ref = stats.t.logpdf(r, df=nu[k], loc=mu[k], scale=scale)
        np.testing.assert_allclose(ours[:, k], ref, atol=1e-10)


def test_student_t_limit_normal():
    """nu -> inf: la t escalada converge a la Normal con la misma varianza."""
    rng = np.random.default_rng(1)
    T, K = 100, 2
    r = rng.normal(size=T)
    mu = np.array([0.0, 0.1])
    sigma2 = np.abs(rng.normal(1.0, 0.2, size=(T, K)))

    lt = student_t_logpdf(r, mu, sigma2, np.array([1e8, 1e8]))
    ln = normal_logpdf(r, mu, sigma2)
    np.testing.assert_allclose(lt, ln, atol=1e-5)


def test_simulate_t_variance():
    """K=1, t con nu=6: la varianza muestral debe aproximar v (el escalado
    (nu-2)/nu esta bien; si faltara, la varianza saldria nu/(nu-2) ~ 1.5x)."""
    true = {
        "mu": np.array([0.0]),
        "v": np.array([2.0]),
        "alpha": np.array([0.05]),
        "gamma": np.array([0.04]),
        "beta": np.array([0.85]),
        "P": np.array([[1.0]]),
        "nu": np.array([6.0]),
    }
    theta = pack(true, K=1, dist="t")
    r, _ = simulate(theta, K=1, T=60000, seed=2, dist="t")
    assert r.var() == pytest.approx(2.0, rel=0.08), f"var={r.var():.3f}, esperada ~2.0"


@pytest.mark.slow
def test_student_t_recovery():
    """MLE con dist='t' sobre datos t simulados: recupera nu en un rango
    razonable y le gana en BIC a la Normal (colas gordas de verdad)."""
    from irfn.models.estimate import fit

    true = {
        "mu": np.array([0.02]),
        "v": np.array([1.5]),
        "alpha": np.array([0.06]),
        "gamma": np.array([0.05]),
        "beta": np.array([0.84]),
        "P": np.array([[1.0]]),
        "nu": np.array([5.0]),
    }
    theta = pack(true, K=1, dist="t")
    r, _ = simulate(theta, K=1, T=4000, seed=3, dist="t")

    ft = fit(r, K=1, n_starts=8, seed=1, compute_se=False, dist="t")
    fn = fit(r, K=1, n_starts=8, seed=1, compute_se=False)

    nu_hat = ft.params["nu"][0]
    assert 3.0 < nu_hat < 9.0, f"nu recuperada fuera de rango: {nu_hat:.2f} (verdadera 5.0)"
    assert ft.bic < fn.bic, "sobre datos t(5), la t debe ganarle a la Normal en BIC"

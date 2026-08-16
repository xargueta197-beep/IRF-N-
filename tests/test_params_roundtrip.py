"""Round-trip de la parametrizacion sin restricciones (params.py).

Con la gauge fija (softmax con categoria de referencia), la representacion
vectorial es minima y unica, asi que pack(unpack(theta)) == theta de forma
EXACTA para cualquier theta valido. Tambien verificamos las invariantes
estructurales que unpack garantiza por construccion (R5, positividad,
estacionariedad, filas estocasticas).
"""

from __future__ import annotations

import numpy as np
import pytest

from irfn.models.params import n_params, pack, unpack


@pytest.mark.parametrize("K", [1, 2, 3])
def test_pack_unpack_roundtrip(K):
    rng = np.random.default_rng(0)
    for _ in range(200):
        theta = rng.normal(scale=1.5, size=n_params(K))
        params = unpack(theta, K)
        theta2 = pack(params, K)
        np.testing.assert_allclose(theta2, theta, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("dist", ["normal", "t"])
@pytest.mark.parametrize("n_cov", [1, 2])
def test_pack_unpack_roundtrip_surprise(K, dist, n_cov):
    """Con la capa de sorpresa (V2), delta = exp(d_raw) se APPENDEA al vector y el
    round-trip sigue siendo exacto: pack(unpack(theta)) == theta."""
    rng = np.random.default_rng(2)
    for _ in range(100):
        theta = rng.normal(scale=1.2, size=n_params(K, n_cov=n_cov, dist=dist, surprise=True))
        params = unpack(theta, K, n_cov=n_cov, dist=dist, surprise=True)
        assert params["delta"] > 0.0                       # delta = exp(d_raw) > 0
        theta2 = pack(params, K, n_cov=n_cov, dist=dist, surprise=True)
        np.testing.assert_allclose(theta2, theta, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("K", [1, 2, 3])
def test_unpack_invariants(K):
    rng = np.random.default_rng(1)
    for _ in range(200):
        theta = rng.normal(scale=1.5, size=n_params(K))
        p = unpack(theta, K)

        # R5: varianza incondicional estrictamente creciente
        assert np.all(np.diff(p["v"]) > 0)

        # positividad del GJR
        assert np.all(p["alpha"] >= 0)
        assert np.all(p["beta"] >= 0)
        assert np.all(p["gamma"] >= 0)
        assert np.all(p["alpha"] + p["gamma"] >= 0)
        assert np.all(p["omega"] > 0)

        # persistencia kappa = alpha + gamma/2 + beta = p, estacionaria
        kappa = p["alpha"] + p["gamma"] / 2.0 + p["beta"]
        np.testing.assert_allclose(kappa, p["p"], atol=1e-12)
        assert np.all(kappa < 1.0)

        # omega = v*(1-p) hace que v sea la varianza incondicional verdadera
        implied_uncond = p["omega"] / (1.0 - kappa)
        np.testing.assert_allclose(implied_uncond, p["v"], atol=1e-10)

        # P es estocastica por filas
        assert p["P"].shape == (K, K)
        np.testing.assert_allclose(p["P"].sum(axis=1), np.ones(K), atol=1e-12)
        assert np.all(p["P"] >= 0)

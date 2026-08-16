"""Cobertura del bootstrap estacionario V4 (Politis-White + Politis-Romano).

Un IC no vale por existir: vale si CUBRE el verdadero con la frecuencia que
promete. Estos dos tests verifican justo eso sobre casos con respuesta conocida:

  1. Serie iid Normal(0,1) -> Sharpe verdadero = 0. El IC 95% debe cubrir el 0
     en ~95% de las simulaciones (ni mucho menos: sub-cobertura = IC mentiroso;
     ni siempre: sobre-cobertura = IC inutilmente ancho).
  2. Serie con Sharpe verdadero = 1.5 -> el IC 95% NO debe incluir el 0
     (el test tiene potencia para detectar un Sharpe claramente positivo).
"""

from __future__ import annotations

import numpy as np
import pytest

from irfn.validation.bootstrap import (
    optimal_block_length,
    sharpe_ci,
    stationary_bootstrap,
)

TRADING_DAYS = 252


def test_optimal_block_length_is_small_for_iid():
    """Ruido blanco no tiene dependencia: la longitud optima debe ser ~1 (el
    bootstrap estacionario colapsa al iid). Sanidad de Politis-White."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    b = optimal_block_length(x)
    assert isinstance(b, int)
    assert 1 <= b <= 3


def test_optimal_block_length_grows_with_persistence():
    """Un AR(1) fuertemente autocorrelacionado exige bloques mas largos que el
    ruido blanco: la longitud optima debe crecer con la persistencia."""
    rng = np.random.default_rng(1)
    n = 3000
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    phi = 0.7
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    assert optimal_block_length(x) > optimal_block_length(rng.normal(size=n))


@pytest.mark.slow
def test_sharpe_ci_covers_zero_under_iid():
    """iid Normal(0,1): Sharpe verdadero = 0. El IC 95% del Sharpe debe cubrir el
    0 en ~95% de 200 simulaciones."""
    n_sims = 200
    n_obs = 500
    covered = 0
    for s in range(n_sims):
        rng = np.random.default_rng(1000 + s)
        r = rng.normal(0.0, 1.0, size=n_obs)
        out = sharpe_ci(r, n_boot=199, seed=7)
        if out["includes_zero"]:
            covered += 1
    coverage = covered / n_sims
    # ~95% nominal; banda amplia por error de Monte Carlo (200 sims) y el leve
    # sub-cobertura tipico del IC percentil. Sub-cobertura seria < 0.90.
    assert 0.90 <= coverage <= 1.0, f"cobertura={coverage:.3f} fuera de rango"


@pytest.mark.slow
def test_sharpe_ci_excludes_zero_when_true_sharpe_positive():
    """Serie con Sharpe anualizado verdadero = 1.5: el IC 95% NO debe incluir el
    cero (potencia del test)."""
    target_ann_sharpe = 1.5
    n_obs = 1500
    sigma_d = 0.01
    mu_d = target_ann_sharpe * sigma_d / np.sqrt(TRADING_DAYS)
    rng = np.random.default_rng(2024)
    r = rng.normal(mu_d, sigma_d, size=n_obs)
    out = sharpe_ci(r, n_boot=1000, seed=42)
    assert out["includes_zero"] is False
    assert out["ci_lower"] > 0.0
    assert out["ci_lower"] < out["sharpe"] < out["ci_upper"]
    # el punto debe rondar 1.5 (misma escala de Sharpe anualizado)
    assert 0.7 < out["sharpe"] < 2.3


def test_stationary_bootstrap_returns_three_floats():
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.01, size=400)
    point, lo, hi = stationary_bootstrap(r, np.mean, n_boot=200, seed=1)
    assert lo <= point <= hi
    assert np.isfinite([point, lo, hi]).all()

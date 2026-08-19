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


# --------------------------------------------------------------------------- #
# Sprint de honestidad (2026-08-16): no publicar IC que el metodo no sostiene.
# --------------------------------------------------------------------------- #
def test_maxdd_ci_is_suppressed_but_point_survives():
    """F3: el bootstrap estacionario NO es valido sobre el maximo drawdown (un
    funcional de valor extremo). El punto se reporta; el IC debe ser None. Las
    demas metricas (Sharpe, etc.) SI llevan IC con datos suficientes."""
    from irfn.validation.bootstrap import bootstrap_regime_stats

    rng = np.random.default_rng(7)
    r = rng.normal(0.0003, 0.01, size=500)
    stats = bootstrap_regime_stats(
        r, n_boot=200, block_len=20, ci_level=0.90, seed=1, min_obs=30
    )
    dd_point, dd_lo, dd_hi = stats["maxdd"]
    assert np.isfinite(dd_point) and dd_point <= 0.0
    assert dd_lo is None and dd_hi is None  # sin IC espurio
    # una metrica valida SI trae IC
    sh_point, sh_lo, sh_hi = stats["sharpe"]
    assert sh_lo is not None and sh_hi is not None


def test_conditional_stats_suppresses_degenerate_regime_ci():
    """F4: un regimen 'absorbe-outliers' (E[D] < umbral) publica el punto de sus
    metricas pero NO su IC (sus dias son excursiones sueltas: el bootstrap por
    bloques finge precision). El regimen persistente conserva su IC."""
    from irfn.outputs.publish import conditional_stats

    rng = np.random.default_rng(11)
    n = 600
    r_pct = rng.normal(0.03, 1.0, size=n)          # log-retornos en %
    argmax = np.zeros(n, dtype=int)
    argmax[::2] = 1                                 # regimen 1 con muchos dias, pero degenerado por E[D]
    labels = ["persistente", "degenerado"]
    out = conditional_stats(
        r_pct, argmax, labels, "SPY",
        bootstrap_n_boot=200, bootstrap_block_len=20, bootstrap_ci_level=0.90,
        bootstrap_min_obs=30, bootstrap_seed=1,
        expected_durations=[12.0, 1.3],            # regimen 1: E[D] < 2.0 => degenerado
        degenerate_duration_days=2.0,
    )["SPY"]
    # persistente: IC presente en las metricas validas (Sharpe)
    assert out["persistente"]["sharpe"]["ci_low"] is not None
    # degenerado: TODAS las metricas sin IC (incluida Sharpe)
    for metric in ("mean_ann", "vol_ann", "sharpe", "maxdd"):
        assert out["degenerado"][metric]["ci_low"] is None
        assert out["degenerado"][metric]["ci_high"] is None
        assert "value" in out["degenerado"][metric]
    # degenerado (F2.c): las 3 metricas anualizadas suprimen tambien el PUNTO
    # (E[D]=1.3 < 2.0 -- no persiste, anualizar no tiene sentido aunque haya
    # 300 dias). vol_ann NO se suprime: no implica persistencia.
    for metric in ("mean_ann", "sharpe", "maxdd"):
        assert out["degenerado"][metric]["value"] is None
    assert out["degenerado"]["vol_ann"]["value"] is not None
    # persistente: no degenerado y con >= min_obs -> punto SI se publica
    for metric in ("mean_ann", "sharpe", "maxdd", "vol_ann"):
        assert out["persistente"][metric]["value"] is not None
    # n_obs viaja siempre, anualizable o no
    assert out["degenerado"]["sharpe"]["n_obs"] == 300
    assert out["persistente"]["sharpe"]["n_obs"] == 300


def test_conditional_stats_suppresses_low_occupancy_even_if_not_degenerate():
    """F2.c: un regimen con E[D] alta (persistente) pero pocas observaciones
    totales tambien suprime el PUNTO anualizado -- son dos criterios
    independientes (semantica vs. precision), ninguno sustituye al otro."""
    from irfn.outputs.publish import conditional_stats

    rng = np.random.default_rng(7)
    n = 100
    r_pct = rng.normal(0.03, 1.0, size=n)
    argmax = np.zeros(n, dtype=int)
    argmax[:20] = 1                                 # regimen 1: solo 20 obs, < min_obs=30
    labels = ["normal", "escaso"]
    out = conditional_stats(
        r_pct, argmax, labels, "SPY",
        bootstrap_n_boot=200, bootstrap_block_len=20, bootstrap_ci_level=0.90,
        bootstrap_min_obs=30, bootstrap_seed=1,
        expected_durations=[10.0, 15.0],            # ambos con E[D] alta: NINGUNO degenerado
        degenerate_duration_days=2.0,
    )["SPY"]
    assert out["escaso"]["sharpe"]["n_obs"] == 20
    for metric in ("mean_ann", "sharpe", "maxdd"):
        assert out["escaso"][metric]["value"] is None       # cae por ocupacion, no por E[D]
    assert out["escaso"]["vol_ann"]["value"] is not None
    for metric in ("mean_ann", "sharpe", "maxdd", "vol_ann"):
        assert out["normal"][metric]["value"] is not None   # 80 obs >= min_obs: se publica

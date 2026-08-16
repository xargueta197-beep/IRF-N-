"""Tests del kernel de Hawkes power-law (Omori/ETAS con marcas).

Rapidos: correctitud de la forma cerrada del compensador (vs integracion numerica),
formula del branching ratio, e identidad del re-escalamiento.
Slow: recuperacion de parametros conocidos por simulacion + MLE, y KS que NO
rechaza el modelo verdadero.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from irfn.models import hawkes_powerlaw as PL


def test_compensator_closed_form_matches_numerical():
    # La forma cerrada Lambda(T) = mu*T + (alpha/theta) sum_i s_i (c^-theta - (T-t_i+c)^-theta)
    # debe igualar sum_i alpha*s_i * Integral_0^{T-t_i} (u+c)^{-(1+theta)} du.
    times = np.array([0.3, 1.1, 2.7, 4.0])
    marks = np.array([0.5, 1.0, 0.2, 0.8])
    mu, alpha, c, theta, T = 0.4, 0.7, 0.05, 0.9, 6.0
    closed = PL.pl_compensator(times, marks, mu, alpha, c, theta, T)
    num = mu * T
    for ti, si in zip(times, marks):
        val, _ = integrate.quad(lambda u: (u + c) ** (-(1 + theta)), 0.0, T - ti)
        num += alpha * si * val
    assert np.isclose(closed, num, rtol=1e-6)


def test_branching_ratio_formula():
    # n = alpha*E[s]*c^{-theta}/theta ; ademas = alpha*E[s]*Integral_0^inf phi(u)du
    marks = np.array([0.4, 0.6, 0.5, 0.9, 0.2])
    alpha, c, theta = 0.5, 0.1, 0.7
    n = PL.pl_branching_ratio(alpha, c, theta, marks)
    mass, _ = integrate.quad(lambda u: (u + c) ** (-(1 + theta)), 0.0, np.inf)
    assert np.isclose(n, alpha * marks.mean() * mass, rtol=1e-6)


def test_lambda_and_loglik_finite():
    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(0, 50, size=200))
    marks = rng.uniform(0, 1, size=200)
    lam = PL.pl_lambda_at_events(times, marks, 1.0, 0.3, 0.05, 0.8)
    assert np.all(lam >= 1.0) and np.all(np.isfinite(lam))  # >= mu, sin auto-excitacion negativa
    ll = PL.pl_loglik(times, marks, 1.0, 0.3, 0.05, 0.8, 50.0)
    assert np.isfinite(ll)


def test_rescaling_identity_positive_increments():
    # tau_i = Lambda(t_i) debe ser creciente (compensador monotono) -> interarribos >= 0
    rng = np.random.default_rng(1)
    times = np.sort(rng.uniform(0, 100, size=300))
    marks = rng.uniform(0, 1, size=300)
    tau = PL.pl_time_rescaling(times, marks, {"mu": 0.5, "alpha": 0.2, "c": 0.05, "theta": 0.9})
    assert np.all(np.diff(tau) >= -1e-9)


@pytest.mark.slow
def test_powerlaw_recovery_and_ks():
    # Simular con parametros conocidos -> recuperar dentro del IC95 -> KS no rechaza.
    mu_t, c_t, theta_t = 0.5, 0.05, 0.8
    Es = 0.5
    n_target = 0.5
    alpha_t = n_target * theta_t * c_t ** theta_t / Es
    rng = np.random.default_rng(7)
    times, marks = PL.simulate_powerlaw_ogata_thinning(4000.0, mu_t, alpha_t, c_t, theta_t, rng)
    assert len(times) > 1000

    fit = PL.fit_powerlaw_mle(times, marks, T=4000.0, n_starts=12, seed=1)
    assert fit.hessian_ok
    truth = {"mu": mu_t, "alpha": alpha_t, "c": c_t, "theta": theta_t}
    for k, true_val in truth.items():
        lo = fit.params[k] - 1.96 * fit.se[k]
        hi = fit.params[k] + 1.96 * fit.se[k]
        assert lo <= true_val <= hi, f"{k}: verdadero {true_val} fuera de IC95 [{lo:.4f},{hi:.4f}]"
    assert abs(fit.branching_ratio - n_target) < 0.1

    ks = PL.pl_rescaling_ks(times, marks, fit.params)
    assert ks["passed"], f"KS rechaza el modelo ajustado (p={ks['p_value']})"
    ks_true = PL.pl_rescaling_ks(times, marks, truth)
    assert ks_true["passed"], f"KS rechaza el modelo VERDADERO (p={ks_true['p_value']})"

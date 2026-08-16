"""Hawkes con marcas (V3): exactitud de la verosimilitud, recuperacion de
parametros y bondad de ajuste por re-escalamiento temporal.

Tres capas de defensa:

  1. Tests EXACTOS y rapidos: la verosimilitud O(n) en dominio log reproduce el
     doble loop O(n^2) de referencia bit a bit (misma matematica, otra
     evaluacion); la identidad del re-escalamiento reproduce el compensador
     directo; el branching ratio usa E[s] (n = alpha*E[s]/beta, NO alpha/beta).
  2. Recuperacion (slow): simular con (mu, alpha, beta) conocidos via thinning
     de Ogata -> estimar -> los verdaderos caen dentro del IC 95%.
  3. KS de re-escalamiento sobre los datos SIMULADOS: debe pasar por
     construccion. Si no pasa aqui, el test o el estimador estan mal (no el
     modelo: los datos SON Hawkes exponencial).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from irfn.models.hawkes_mle import (
    branching_ratio,
    branching_ratio_ci,
    compensator,
    expected_cascade,
    expected_cascade_reported,
    fit_hawkes_mle,
    hawkes_loglik,
    lambda_on_grid,
    simulate_hawkes_ogata_thinning,
    time_rescaling,
)

TRUE = {"mu": 0.5, "alpha": 0.3, "beta": 0.8}   # n = 0.3*0.5/0.8 = 0.1875 < 1


# --------------------------------------------------------------------------- #
# Referencias O(n^2): la matematica del spec, sin ninguna optimizacion.
# --------------------------------------------------------------------------- #
def _loglik_bruteforce(t, s, mu, alpha, beta, T):
    ll = 0.0
    for i in range(len(t)):
        lam = mu + alpha * sum(
            s[j] * np.exp(-beta * (t[i] - t[j])) for j in range(i)
        )
        ll += np.log(lam)
    comp = mu * T + (alpha / beta) * sum(
        s[i] * (1.0 - np.exp(-beta * (T - t[i]))) for i in range(len(t))
    )
    return ll - comp


def _compensator_bruteforce_at(t, s, mu, alpha, beta, u):
    return mu * u + (alpha / beta) * sum(
        s[j] * (1.0 - np.exp(-beta * (u - t[j]))) for j in range(len(t)) if t[j] < u
    )


@pytest.fixture(scope="module")
def small_sample():
    rng = np.random.default_rng(7)
    times, marks = simulate_hawkes_ogata_thinning(T=200.0, **TRUE, rng=rng)
    assert len(times) > 50
    return times, marks


def test_loglik_matches_bruteforce(small_sample):
    times, marks = small_sample
    fast = hawkes_loglik(times, marks, **TRUE, T=200.0)
    slow = _loglik_bruteforce(times, marks, TRUE["mu"], TRUE["alpha"], TRUE["beta"], 200.0)
    assert np.isclose(fast, slow, rtol=0, atol=1e-8), (
        f"verosimilitud O(n) difiere de la referencia O(n^2): {fast} vs {slow}"
    )


def test_time_rescaling_matches_direct_compensator(small_sample):
    times, marks = small_sample
    tau = time_rescaling(times, marks, TRUE)
    for i in (0, 1, len(times) // 2, len(times) - 1):
        direct = _compensator_bruteforce_at(
            times, marks, TRUE["mu"], TRUE["alpha"], TRUE["beta"], times[i]
        )
        assert np.isclose(tau[i], direct, atol=1e-8), (
            f"tau_{i} difiere del compensador directo: {tau[i]} vs {direct}"
        )


def test_branching_ratio_uses_mean_mark():
    marks = np.array([0.2, 0.4, 0.6])          # E[s] = 0.4
    n = branching_ratio(alpha=1.0, beta=2.0, marks=marks)
    assert np.isclose(n, 1.0 * 0.4 / 2.0)      # = 0.2, NO alpha/beta = 0.5
    assert not np.isclose(n, 0.5)
    assert np.isclose(expected_cascade(n), 1.0 / (1.0 - 0.2))
    assert expected_cascade(1.0) == float("inf")   # n >= 1: explosivo, no publicable


def test_branching_ratio_ci_delta_method():
    # IC por metodo delta: analitico contra la formula, y coherencia (n dentro del IC).
    marks = np.array([0.2, 0.4, 0.6, 0.5, 0.3])          # E[s] = 0.4
    alpha, beta = 1.0, 2.0
    cov_ab = np.array([[0.01, 0.0], [0.0, 0.04]])        # Var(alpha)=0.01, Var(beta)=0.04
    out = branching_ratio_ci(alpha, beta, marks, cov_ab, ci_level=0.95)
    m = marks.mean()
    n = alpha * m / beta
    g = np.array([m / beta, -alpha * m / beta**2])
    var_params = g @ cov_ab @ g
    var_m = marks.var(ddof=1) / len(marks)
    se_expected = np.sqrt(var_params + (alpha / beta) ** 2 * var_m)
    assert np.isclose(out["n"], n)
    assert np.isclose(out["se"], se_expected)
    assert out["ci_low"] < n < out["ci_high"]
    # cov None (hessiano no definido positivo) -> n presente, IC en NaN, nunca inventado.
    nan_out = branching_ratio_ci(alpha, beta, marks, None)
    assert np.isclose(nan_out["n"], n) and np.isnan(nan_out["se"])


def test_expected_cascade_censoring_rule():
    # Lejos de la frontera: valor puntual 1/(1-n).
    val, bounded = expected_cascade_reported(n=0.30, ci_high=0.45, trigger=0.95)
    assert bounded and np.isclose(val, 1.0 / (1.0 - 0.30))
    # IC superior alcanza el disparador -> censurado ("no acotada").
    val, bounded = expected_cascade_reported(n=0.90, ci_high=0.96, trigger=0.95)
    assert val is None and not bounded
    # n >= 1 -> censurado aunque el IC no se pase.
    val, bounded = expected_cascade_reported(n=1.01, ci_high=1.05, trigger=0.95)
    assert val is None and not bounded


def test_fit_exposes_branching_ratio_ci():
    # El fit expone el IC del branching ratio (usado por la Parte B en publish).
    rng = np.random.default_rng(3)
    times, marks = simulate_hawkes_ogata_thinning(T=1500.0, mu=0.5, alpha=0.6, beta=2.0, rng=rng)
    fit = fit_hawkes_mle(times, marks, T=1500.0, n_starts=15, seed=1)
    assert np.isfinite(fit.branching_ratio_se)
    assert fit.branching_ratio_ci_low < fit.branching_ratio < fit.branching_ratio_ci_high


def test_lambda_on_grid_matches_definition(small_sample):
    times, marks = small_sample
    grid = np.array([0.0, 50.0, 100.0, 199.9])
    lam = lambda_on_grid(times, marks, TRUE, grid)
    for k, u in enumerate(grid):
        direct = TRUE["mu"] + TRUE["alpha"] * sum(
            marks[j] * np.exp(-TRUE["beta"] * (u - times[j]))
            for j in range(len(times)) if times[j] <= u
        )
        assert np.isclose(lam[k], direct, atol=1e-10)


def test_compensator_total_close_to_event_count(small_sample):
    """E[Lambda(T)] = E[N(T)]: con los parametros verdaderos, el compensador
    total debe ser comparable al numero de eventos (identidad de martingala)."""
    times, marks = small_sample
    comp = compensator(times, marks, **TRUE, T=200.0)
    n = len(times)
    assert abs(comp - n) < 4.0 * np.sqrt(n)


# --------------------------------------------------------------------------- #
# Recuperacion y KS (slow: simulacion larga + multistart completo)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_hawkes_recovery():
    """Simula con (mu, alpha, beta) conocidos, estima por MLE multistart y
    verifica que los verdaderos caen en el IC 95%. Corre el KS de
    re-escalamiento sobre los datos simulados: debe pasar por construccion."""
    rng = np.random.default_rng(42)
    event_times, marks = simulate_hawkes_ogata_thinning(T=2000.0, **TRUE, rng=rng)
    assert len(event_times) > 500, "simulacion demasiado corta para el test de recuperacion"

    fitted = fit_hawkes_mle(event_times, marks, T=2000.0, n_starts=20, seed=42)

    assert fitted.hessian_ok, "hessiano degenerado sobre datos simulados: estimador mal condicionado"
    for key, true_value in TRUE.items():
        assert fitted.ci_low[key] <= true_value <= fitted.ci_high[key], (
            f"{key}: verdadero={true_value} fuera del IC 95% "
            f"[{fitted.ci_low[key]:.4f}, {fitted.ci_high[key]:.4f}] "
            f"(estimado {fitted.params[key]:.4f})"
        )

    # Branching ratio recuperado: estacionario y cerca del verdadero.
    n_true = TRUE["alpha"] * marks.mean() / TRUE["beta"]
    assert fitted.stationary, f"n={fitted.branching_ratio} >= 1 sobre datos estacionarios simulados"
    assert abs(fitted.branching_ratio - n_true) < 0.15

    # KS de re-escalamiento con los parametros AJUSTADOS: Exp(1) por construccion.
    tau = time_rescaling(event_times, marks, fitted.params)
    interarrivals = np.diff(tau)
    ks_stat, ks_pvalue = stats.kstest(interarrivals, "expon")
    assert ks_pvalue > 0.05, (
        f"KS de re-escalamiento rechaza Exp(1) sobre datos SIMULADOS "
        f"(stat={ks_stat:.4f}, p={ks_pvalue:.4f}): el estimador o el test estan mal."
    )


@pytest.mark.slow
def test_rescaling_ks_rejects_wrong_model():
    """Control negativo: el KS debe RECHAZAR cuando los parametros estan muy mal
    (si pasara con cualquier cosa, el test de bondad de ajuste no mide nada)."""
    rng = np.random.default_rng(3)
    event_times, marks = simulate_hawkes_ogata_thinning(T=2000.0, **TRUE, rng=rng)
    wrong = {"mu": 5.0, "alpha": 0.01, "beta": 5.0}
    tau = time_rescaling(event_times, marks, wrong)
    _, p = stats.kstest(np.diff(tau), "expon")
    assert p < 0.05

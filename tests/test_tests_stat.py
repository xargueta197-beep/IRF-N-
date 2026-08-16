"""Tests de los tests estadisticos (V1): DM, PT y bootstrap de K.

Cada uno se verifica sobre casos sinteticos con respuesta CONOCIDA: un test
estadistico que no detecta una diferencia obvia (o que 'detecta' una que no
existe) es peor que no tener test, porque presta autoridad falsa al reporte.
"""

from __future__ import annotations

import numpy as np
import pytest

from irfn.validation.tests_stat import (
    bootstrap_lr_test,
    diebold_mariano,
    pesaran_timmermann,
)


def test_dm_detects_clear_difference():
    """A pierde sistematicamente menos que B => DM negativo y significativo."""
    rng = np.random.default_rng(0)
    T = 500
    base = rng.normal(1.0, 0.2, size=T)
    loss_a = base
    loss_b = base + 0.15 + rng.normal(0.0, 0.05, size=T)
    res = diebold_mariano(loss_a, loss_b)
    assert res["dm_stat"] < -3.0
    assert res["p_value"] < 0.01
    assert res["mean_diff"] < 0


def test_dm_no_difference():
    """Mismas perdidas en media => DM insignificante (no inventa señal)."""
    rng = np.random.default_rng(1)
    T = 500
    loss_a = rng.normal(1.0, 0.3, size=T)
    loss_b = rng.normal(1.0, 0.3, size=T)
    res = diebold_mariano(loss_a, loss_b)
    assert abs(res["dm_stat"]) < 2.5
    assert res["p_value"] > 0.01


def test_pt_perfect_and_random():
    rng = np.random.default_rng(2)
    T = 400
    y = rng.normal(size=T)

    perfect = pesaran_timmermann(y, y)                     # signo siempre correcto
    assert perfect["pt_stat"] > 5.0
    assert perfect["p_value"] < 1e-6

    noise = pesaran_timmermann(y, rng.normal(size=T))      # prediccion independiente
    assert noise["p_value"] > 0.01


def test_pt_degenerate_reported():
    """Prediccion de signo constante: el test degenera y se reporta tal cual."""
    rng = np.random.default_rng(3)
    y = rng.normal(size=200)
    res = pesaran_timmermann(y, np.ones(200))
    assert np.isnan(res["pt_stat"])
    assert "degenerado" in res["note"]


@pytest.mark.slow
def test_bootstrap_lr_runs_and_is_sane():
    """Sanidad del bootstrap parametrico: sobre datos generados con UN regimen,
    el test K=2 vs K=1 no debe rechazar con fuerza (p no minusculo); el p-value
    es valido y las replicas se completan. B pequeno: es un test de mecanica,
    no el experimento real (ese corre con config v1.ktest)."""
    from irfn.models.msgarch import simulate
    from irfn.models.params import pack

    true = {
        "mu": np.array([0.02]),
        "v": np.array([1.0]),
        "alpha": np.array([0.05]),
        "gamma": np.array([0.06]),
        "beta": np.array([0.85]),
        "P": np.array([[1.0]]),
    }
    r, _ = simulate(pack(true, K=1), K=1, T=600, seed=9)

    res = bootstrap_lr_test(
        r, K_null=1, K_alt=2, dist="normal",
        n_boot=9, n_starts_data=4, n_starts_boot=2, seed=11,
    )
    assert res["n_boot_ok"] >= 7
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["lr_obs"] >= 0.0
    # bajo H0 verdadero, el LR observado no deberia ser un outlier extremo del
    # bootstrap; con B=9 el p minimo es 0.1, asi que exigimos p >= 0.1
    assert res["p_value"] >= 0.1

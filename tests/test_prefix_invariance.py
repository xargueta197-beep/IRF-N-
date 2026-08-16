"""EL TEST ANTI-LOOK-AHEAD. El mas importante de los 6 (ver CLAUDE.md).

Correr el pipeline sobre data[:t] y sobre data[:T] con parametros FIJOS debe
producir exactamente el mismo ξ_{s|s} en las fechas s <= t. Si difiere, hay
informacion futura filtrandose. No hay excusa posible.

Se corre sobre datos SIMULADOS (msgarch.simulate con parametros conocidos), no
sobre descargas de red: el test no debe depender de yfinance ni de internet, y la
propiedad que verifica (causalidad del filtro) no necesita datos reales.

Ver el docstring de audit.pit.prefix_invariance_check para por que fijar los
parametros aisla exactamente la superficie de look-ahead correcta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from irfn.audit.pit import PREFIX_ATOL, prefix_invariance_check
from irfn.models.params import pack


def _known_theta_k2() -> np.ndarray:
    """theta de un MS-GJR-GARCH K=2 con parametros conocidos y bien separados."""
    def split(p, u_a, u_g, u_b):
        return p * u_a, 2.0 * p * u_g, p * u_b       # alpha, gamma, beta ; kappa=p

    a0, g0, b0 = split(0.95, 0.08, 0.10, 0.82)
    a1, g1, b1 = split(0.90, 0.10, 0.15, 0.75)
    params = {
        "mu": np.array([0.05, -0.10]),
        "v": np.array([0.5, 4.0]),                    # v_1 < v_2 (R5)
        "alpha": np.array([a0, a1]),
        "gamma": np.array([g0, g1]),
        "beta": np.array([b0, b1]),
        "P": np.array([[0.98, 0.02], [0.05, 0.95]]),
    }
    return pack(params, K=2)


def test_prefix_invariance():
    K = 2
    seed = 42
    theta = _known_theta_k2()

    from irfn.models.msgarch import simulate

    r, _states = simulate(theta, K=K, T=1200, seed=seed)
    index = pd.bdate_range("2015-01-01", periods=len(r))
    returns = pd.Series(r, index=index, name="r")

    result = prefix_invariance_check(
        returns,
        K=K,
        seed=seed,
        n_starts=6,          # basta: el test aisla la causalidad del filtro, no la calidad del fit
        train_len=600,
        dates_to_test=10,
    )

    # Toda fecha muestreada debe ser invariante a distancia < atol.
    assert result.attrs["passed"], (
        "INVARIANZA DE PREFIJO ROTA: hay look-ahead.\n"
        + result.to_string(index=False)
    )
    assert (result["max_abs_diff"] < PREFIX_ATOL).all()
    assert len(result) == 10


def test_block_reestimation_check_empty_is_vacuous_pass():
    """Con una lista vacia de bloques (p.ej. activo K=1: no hay escalera de
    walk-forward que re-estimar) el chequeo de R2 debe PASAR por vacuidad, no
    reventar. Regresion: antes `df["distinto"]` sobre un DataFrame sin columnas
    lanzaba KeyError('distinto') y tumbaba run_v3 para K=1 (2026-08-15)."""
    from irfn.audit.pit import block_reestimation_check

    df = block_reestimation_check([])
    assert df.attrs["passed"] is True
    assert df.attrs.get("vacuous") is True
    assert len(df) == 0
    assert "distinto" in df.columns  # columnas presentes aunque vacio

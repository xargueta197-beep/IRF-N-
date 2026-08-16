"""Tests del walk-forward economico pre-registrado (validation/economic.py).

Lo critico es la alineacion temporal: la posicion del dia t se decide con la
xi filtrada de t-1 (R3) y jamas puede ver la de t (look-ahead). El resto es
aritmetica de costos y equity verificable a mano.
"""

import numpy as np
import pandas as pd
import pytest

from irfn.validation.economic import economic_walkforward, strategy_returns


def _history(p_high, r, block_id=None):
    idx = pd.bdate_range("2024-01-01", periods=len(r))
    df = pd.DataFrame(
        {
            "r": r,
            "xi_filtered_0": [1.0 - p for p in p_high],
            "xi_filtered_1": p_high,
        },
        index=idx,
    )
    if block_id is not None:
        df["block_id"] = block_id
    return df


def test_hand_computed_five_days():
    """Caso de 5 dias verificado a mano (pre-registro seccion 7)."""
    # P(alta vol):  .1  .8  .8  .2  .1
    # senal (t-1): NaN  .1  .8  .8  .2   -> w: 1 1 0 0 1
    hist = _history([0.1, 0.8, 0.8, 0.2, 0.1], r=[1.0, 2.0, -3.0, -1.0, 0.5])
    out = strategy_returns(hist, p_threshold=0.5, w_reduced=0.0, cost_bps=2.0)

    assert list(out["w"]) == [1.0, 1.0, 0.0, 0.0, 1.0]
    # turnover: dia 3 (1->0) y dia 5 (0->1); costo 2 pb = 0.02% cada cambio
    assert list(out["turnover"]) == [0.0, 0.0, 1.0, 0.0, 1.0]
    expected = [1.0, 2.0, 0.0 - 0.02, 0.0, 0.5 - 0.02]
    assert np.allclose(out["r_strategy"], expected)


def test_no_lookahead():
    """Cambiar xi del dia t NO puede cambiar el retorno de la estrategia en t."""
    base = _history([0.1, 0.1, 0.1, 0.1, 0.1], r=[1.0, -1.0, 2.0, -2.0, 0.3])
    tampered = base.copy()
    tampered.loc[tampered.index[2], "xi_filtered_1"] = 0.99  # "hoy" se dispara

    a = strategy_returns(base, p_threshold=0.5, w_reduced=0.0, cost_bps=0.0)
    b = strategy_returns(tampered, p_threshold=0.5, w_reduced=0.0, cost_bps=0.0)
    # el dia 2 (indice 2) es identico: la senal de HOY no toca la posicion de HOY
    assert a["r_strategy"].iloc[2] == b["r_strategy"].iloc[2]
    # y el dia siguiente si difiere: la senal rige manana
    assert a["r_strategy"].iloc[3] != b["r_strategy"].iloc[3]


def test_always_invested_equals_buyhold():
    """Con w_reduced=1 la estrategia ES buy-and-hold y no carga ningun costo."""
    rng = np.random.default_rng(0)
    hist = _history(rng.uniform(0, 1, 100), r=rng.normal(0, 1, 100))
    out = strategy_returns(hist, p_threshold=0.5, w_reduced=1.0, cost_bps=2.0)
    assert np.allclose(out["r_strategy"], hist["r"])
    assert out["turnover"].sum() == 0.0


def test_first_day_starts_invested_no_cost():
    """El primer dia arranca comprado (w=1) sin costo de arranque."""
    hist = _history([0.9, 0.9], r=[1.0, 1.0])
    out = strategy_returns(hist, p_threshold=0.5, w_reduced=0.0, cost_bps=2.0)
    assert out["w"].iloc[0] == 1.0
    assert out["cost"].iloc[0] == 0.0


def test_economic_walkforward_contract():
    """El dict del artefacto tiene el contrato del pre-registro y el criterio
    de exito se evalua sobre el IC a cost_bps principal."""
    rng = np.random.default_rng(1)
    hist = _history(
        rng.uniform(0, 1, 300), r=rng.normal(0.05, 1.0, 300),
        block_id=np.repeat([0, 1, 2], 100),
    )
    res = economic_walkforward(
        hist, p_threshold=0.5, w_reduced=0.0, cost_bps=2.0,
        cost_bps_sensitivity=[0.0, 5.0], n_boot=199, seed=42,
    )
    assert res["rule"]["p_threshold"] == 0.5
    assert {c["cost_bps"] for c in res["cost_sensitivity"]} == {0.0, 5.0}
    assert len(res["per_block"]) == 3
    assert isinstance(res["success"], bool)
    lo = res["main"]["sharpe_diff_ci95"][0]
    assert res["success"] == bool(lo is not None and np.isfinite(lo) and lo > 0)

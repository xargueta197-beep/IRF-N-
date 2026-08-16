"""Tests de la capa de sorpresa macro (V2): sigma expanding, PIT de SI_t, w_i.

test_delta_in_mle (recuperacion de delta por el MLE del modelo completo) vive en
test_surprise_mle.py porque toca estimate.py; aqui solo la logica de features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from irfn.features.surprise import (
    estimate_impact_weights,
    expanding_surprise_z,
    surprise_index,
    weighted_events_from,
)


def _one_indicator_calendar(name, dates, actuals, consensus):
    return pd.DataFrame(
        {"indicator": name, "actual": actuals, "consensus": consensus},
        index=pd.DatetimeIndex(dates),
    )


def _synthetic_calendar(seed=0, n=60, min_gap_days=21):
    """Un indicador que publica cada ~mes, consenso siempre presente, sorpresas
    N(0, 3). Devuelve el calendario."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-15", periods=n, freq=f"{min_gap_days}D")
    consensus = rng.normal(100.0, 5.0, size=n)
    surprise = rng.normal(0.0, 3.0, size=n)
    actual = consensus + surprise
    return _one_indicator_calendar("CPI", dates, actual, consensus)


# --------------------------------------------------------------------------- #
# (5a) sigma_i EXPANDING: no usa datos posteriores a t
# --------------------------------------------------------------------------- #
def test_sigma_expanding_no_future():
    """z_i(t) calculado sobre el calendario completo == calculado sobre el
    calendario truncado en t. Si sigma_i usara datos futuros, diferiria."""
    cal = _synthetic_calendar(seed=1, n=50)
    min_obs = 12
    z_full, _ = expanding_surprise_z(cal, min_obs=min_obs)

    # Para cada fecha de release, recomputar con el calendario truncado en esa
    # fecha (inclusive) y exigir igualdad exacta del z de esa fecha.
    for t in cal.index[min_obs:]:          # tras el warm-up hay z no-NaN
        cal_trunc = cal.loc[:t]
        z_trunc, _ = expanding_surprise_z(cal_trunc, min_obs=min_obs)
        a = z_full.loc[t, "CPI"]
        b = z_trunc.loc[t, "CPI"]
        assert np.isfinite(a)
        assert a == pytest.approx(b, rel=1e-12, abs=1e-12), f"z difiere en {t}: {a} vs {b}"


def test_sigma_expanding_warmup_nan():
    """Antes de min_obs sorpresas validas, z_i es NaN (no se inventa una escala)."""
    cal = _synthetic_calendar(seed=2, n=40)
    min_obs = 12
    z, cov = expanding_surprise_z(cal, min_obs=min_obs)
    zc = z["CPI"].dropna()
    # las primeras min_obs-1 sorpresas no producen z (warm-up)
    assert cov["CPI"]["n_z_valid"] == 40 - (min_obs - 1)
    # y ninguna de las primeras min_obs-1 fechas tiene z
    first_valid_pos = min_obs - 1
    assert z["CPI"].iloc[:first_valid_pos].isna().all()
    assert np.isfinite(z["CPI"].iloc[first_valid_pos])


def test_missing_consensus_omitted_not_imputed():
    """Un release sin consenso se OMITE y se cuenta; jamas se imputa z=0."""
    dates = pd.date_range("2015-01-15", periods=30, freq="21D")
    rng = np.random.default_rng(3)
    consensus = rng.normal(100, 5, size=30)
    actual = consensus + rng.normal(0, 3, size=30)
    consensus[5] = np.nan          # un release sin consenso
    cal = _one_indicator_calendar("CPI", dates, actual, consensus)
    z, cov = expanding_surprise_z(cal, min_obs=12)
    assert cov["CPI"]["n_skipped_no_consensus"] == 1
    assert np.isnan(z["CPI"].iloc[5])          # omitido, no imputado a 0
    # el conteo de validas excluye el omitido
    assert cov["CPI"]["n_with_consensus"] == 29


# --------------------------------------------------------------------------- #
# (5b) PIT de SI_t: recortar el calendario en t no cambia SI_t
# --------------------------------------------------------------------------- #
def test_si_pit_prefix_invariance():
    """SI_t con el calendario truncado en t es identico al del pipeline completo.
    Es el prefix_invariance de la capa de noticias. Pesos y delta FIJOS para
    aislar la causalidad de SI (la de z ya la cubre test_sigma_expanding)."""
    cal = _synthetic_calendar(seed=4, n=48)
    min_obs = 12
    delta = np.log(2) / 25.0
    target = pd.date_range(cal.index[0], cal.index[-1], freq="B")

    z_full, _ = expanding_surprise_z(cal, min_obs=min_obs)
    w_fixed = {"CPI": {"w": 1.0}}
    ev_full = weighted_events_from(z_full, w_fixed)
    si_full = surprise_index(ev_full, target, delta=delta)

    # elegir varias fechas destino y recomputar con el calendario truncado
    for t in target[[100, 200, 300, len(target) - 1]]:
        cal_trunc = cal.loc[:t]
        z_tr, _ = expanding_surprise_z(cal_trunc, min_obs=min_obs)
        ev_tr = weighted_events_from(z_tr, w_fixed)
        si_tr = surprise_index(ev_tr, target[target <= t], delta=delta)
        assert si_tr.loc[t] == pytest.approx(si_full.loc[t], rel=1e-10, abs=1e-10), \
            f"SI difiere en {t}"


def test_si_decays_and_is_causal():
    """SI_t solo suma eventos pasados y decae; sin eventos futuros filtrandose."""
    delta = np.log(2) / 10.0
    ev = pd.Series([1.0], index=[pd.Timestamp("2020-01-10")])
    target = pd.date_range("2020-01-01", "2020-02-01", freq="D")
    si = surprise_index(ev, target, delta=delta)
    # antes del evento: 0; en el evento: 1; ~10 dias despues: ~0.5 (vida media 10)
    assert si.loc["2020-01-05"] == pytest.approx(0.0, abs=1e-12)
    assert si.loc["2020-01-10"] == pytest.approx(1.0, abs=1e-12)
    assert si.loc["2020-01-20"] == pytest.approx(0.5, rel=1e-6)


# --------------------------------------------------------------------------- #
# (5c) Recuperacion de w_i por la regresion de impacto
# --------------------------------------------------------------------------- #
def test_wi_recovery():
    """Simular |r| = a + w*|z| + eps con w conocido; recuperar dentro del IC 95%."""
    rng = np.random.default_rng(7)
    n = 400
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    z = pd.Series(rng.normal(0, 1, size=n), index=dates)     # z_i
    a_true, w_true, sigma_eps = 0.3, 0.8, 0.25
    abs_r = a_true + w_true * np.abs(z) + rng.normal(0, sigma_eps, size=n)
    abs_r = pd.Series(abs_r, index=dates)

    res = estimate_impact_weights(abs_r, {"CPI": z.abs()}, t_threshold=2.0)["CPI"]
    lo, hi = res["w"] - 1.96 * res["se"], res["w"] + 1.96 * res["se"]
    assert lo <= w_true <= hi, f"w_true={w_true} fuera de IC95 [{lo:.3f},{hi:.3f}]"
    assert res["distinguible_de_cero"] is True
    assert res["n_events"] == n


def test_wi_zero_marked_not_discarded():
    """Un indicador sin impacto real (w=0) se marca distinguible=False pero NO se
    descarta: sigue en el dict de salida (hallazgo, no problema)."""
    rng = np.random.default_rng(8)
    n = 300
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    z = pd.Series(rng.normal(0, 1, size=n), index=dates)
    abs_r = pd.Series(0.5 + rng.normal(0, 0.5, size=n), index=dates)   # sin dependencia de z
    res = estimate_impact_weights(abs_r, {"PMI": z.abs()}, t_threshold=2.0)
    assert "PMI" in res                       # no se descarta
    assert res["PMI"]["distinguible_de_cero"] is False

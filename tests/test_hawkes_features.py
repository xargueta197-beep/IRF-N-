"""Features de Hawkes (V3): causalidad de lambda_N diaria (invarianza al
truncamiento del feed de titulares), cobertura honesta (NaN fuera de la ventana
de datos), rezago R3 de lambda_N_z y atribucion de 3 vias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from irfn.features.hawkes_features import (
    GDELT_BIN_DAYS,
    compress_to_observed_time,
    dither_quantized_times,
    hawkes_feature,
    headline_event_times,
    lambda_daily,
)
from irfn.features.surprise import attribution_nway

PARAMS = {"mu": 5.0, "alpha": 0.8, "beta": 2.0}


@pytest.fixture
def scored_headlines():
    rng = np.random.default_rng(11)
    base = pd.Timestamp("2026-01-01", tz="UTC")
    ts = base + pd.to_timedelta(np.sort(rng.uniform(0, 90, size=400)), unit="D")
    return pd.DataFrame({"hora_titular": ts, "s": rng.uniform(0, 1, size=400)})


def test_lambda_daily_prefix_invariance(scored_headlines):
    """Truncar el feed de titulares en el dia t reproduce EXACTAMENTE lambda_N
    en fechas <= t: la intensidad solo usa titulares pasados (el analogo del
    chequeo PIT para la capa de Hawkes)."""
    times, marks, origin = headline_event_times(scored_headlines)
    target = pd.date_range("2026-01-05", "2026-03-25", freq="B")
    first = pd.Timestamp("2026-01-01", tz="UTC")
    last = pd.Timestamp("2026-03-31", tz="UTC")
    full = lambda_daily(times, marks, PARAMS, target, origin=origin,
                        coverage_first=first, coverage_last=last)

    cutoff = pd.Timestamp("2026-02-15", tz="UTC")
    trunc_df = scored_headlines[scored_headlines["hora_titular"] <= cutoff]
    t2, m2, _ = headline_event_times(trunc_df, origin=origin)
    target_trunc = target[target <= cutoff.tz_localize(None)]
    trunc = lambda_daily(t2, m2, PARAMS, target_trunc, origin=origin,
                         coverage_first=first, coverage_last=cutoff)

    # la vispera del corte incluye titulares del propio dia del corte hasta la
    # medianoche que lo cierra; comparar hasta el dia ANTERIOR al corte.
    comparable = target_trunc[target_trunc < cutoff.tz_localize(None).normalize()]
    diff = (full.loc[comparable] - trunc.loc[comparable]).abs().max()
    assert diff < 1e-12, f"lambda_N cambio al truncar el feed: max diff {diff}"


def test_lambda_daily_nan_outside_coverage(scored_headlines):
    """Fuera de [primer dia capturado, ultimo dia capturado] no se fabrica
    'calma basal': NaN, y el llamador alinea la muestra."""
    times, marks, origin = headline_event_times(scored_headlines)
    target = pd.date_range("2025-12-20", "2026-04-10", freq="B")
    first = pd.Timestamp("2026-01-01", tz="UTC")
    last = pd.Timestamp("2026-03-31", tz="UTC")
    lam = lambda_daily(times, marks, PARAMS, target, origin=origin,
                       coverage_first=first, coverage_last=last)
    assert lam[target < pd.Timestamp("2026-01-01")].isna().all()
    assert lam[target > pd.Timestamp("2026-03-31")].isna().all()
    inside = lam[(target >= pd.Timestamp("2026-01-01")) & (target <= pd.Timestamp("2026-03-31"))]
    assert inside.notna().all()
    assert (inside >= PARAMS["mu"]).all()          # mu_N es el piso de la intensidad


def test_hawkes_feature_applies_shift(scored_headlines):
    """R3: lambda_N_z en t es funcion de lambda_N hasta t-1 (shift(1) explicito)."""
    times, marks, origin = headline_event_times(scored_headlines)
    target = pd.date_range("2026-01-02", "2026-03-25", freq="B")
    lam = lambda_daily(times, marks, PARAMS, target, origin=origin,
                       coverage_first=pd.Timestamp("2026-01-01", tz="UTC"),
                       coverage_last=pd.Timestamp("2026-03-31", tz="UTC"))
    z_win = 20
    feat = hawkes_feature(lam, z_window=z_win)
    assert feat.name == "lambda_N_z"
    # el primer valor no-NaN del feature aparece un dia DESPUES del primer z valido
    from irfn.features.technical import rolling_zscore

    z_raw = rolling_zscore(lam, z_win)
    assert feat.first_valid_index() > z_raw.first_valid_index()
    aligned = (feat.dropna() - z_raw.shift(1).dropna()).abs().max()
    assert aligned < 1e-15


def test_attribution_nway_normalizes_and_flags_inactive():
    out = attribution_nway({"price": np.array([0.2, -0.1]), "surprise": 0.3, "hawkes": None})
    assert out["inactive_components"] == ["hawkes"]
    assert out["hawkes"] == 0.0
    total = out["price"] + out["surprise"] + out["hawkes"]
    assert np.isclose(total, 1.0)
    # sin movimiento en nada: 100% precio por convencion
    quiet = attribution_nway({"price": 0.0, "surprise": 0.0, "hawkes": 0.0})
    assert quiet["price"] == 1.0


def test_attribution_nway_requires_price():
    with pytest.raises(ValueError):
        attribution_nway({"surprise": 0.1})


def test_dither_quantized_times_properties():
    # Malla de 15 min con empates fuertes: 4 eventos por bin en 3 bins.
    grid = np.array([0.0, 0.0, 0.0, 0.0,
                     GDELT_BIN_DAYS, GDELT_BIN_DAYS, GDELT_BIN_DAYS, GDELT_BIN_DAYS,
                     2 * GDELT_BIN_DAYS, 2 * GDELT_BIN_DAYS, 2 * GDELT_BIN_DAYS, 2 * GDELT_BIN_DAYS])
    marks = np.arange(len(grid), dtype=float)

    t1, m1 = dither_quantized_times(grid, marks, seed=42)
    # 1) longitud preservada y salida ordenada de forma ascendente
    assert len(t1) == len(grid)
    assert np.all(np.diff(t1) >= 0)
    # 2) cada evento cae DENTRO de su bin observado [t_grid, t_grid + Delta): sin
    #    cruzar al bin siguiente (causal, sin look-ahead)
    assert t1.min() >= 0.0
    assert t1.max() < 3 * GDELT_BIN_DAYS
    # 3) de-empata: 12 timestamps unicos donde antes habia 3
    assert len(np.unique(t1)) == len(grid)
    # 4) determinista dado el seed; distinto seed -> distinto jitter
    t2, _ = dither_quantized_times(grid, marks, seed=42)
    assert np.array_equal(t1, t2)
    t3, _ = dither_quantized_times(grid, marks, seed=43)
    assert not np.array_equal(t1, t3)
    # 5) marcas siguen alineadas a su evento tras el reordenamiento
    assert sorted(m1.tolist()) == sorted(marks.tolist())


def test_compress_to_observed_time_excises_phantom_days():
    # Parte A: rango 2024-01-01..01-04 (offsets 0..3); 01-02 (offset 1) es fantasma.
    origin = pd.Timestamp("2024-01-01", tz="UTC")
    times = np.array([0.2, 2.5, 3.1])              # eventos en dias 0, 2, 3
    missing = ["2024-01-02"]                        # un dia sin titulares
    t_obs, T_obs = compress_to_observed_time(times, origin, missing)
    # dia 0: 0 fantasmas antes -> 0.2; dia 2: 1 antes -> 1.5; dia 3: 1 antes -> 2.1
    assert np.allclose(t_obs, [0.2, 1.5, 2.1])
    # 4 dias de rango menos 1 fantasma = 3 dias observados
    assert np.isclose(T_obs, 3.0)
    # la compresion preserva el orden (marcas siguen alineadas)
    assert np.all(np.diff(t_obs) > 0)


def test_compress_monotone_when_event_crosses_into_phantom_day():
    # Regresion: el dithering puede empujar un evento de las 23:45 cruzando la
    # medianoche a un dia FANTASMA. El reloj observado debe seguir siendo monotono
    # (anclar ese evento al fin del ultimo dia cubierto), no romper el orden.
    origin = pd.Timestamp("2024-01-01", tz="UTC")
    missing = ["2024-01-02"]                     # offset 1 es fantasma
    # evento A en dia 0 a las ~23:50 empujado a 1.002 (dentro del dia fantasma 1);
    # evento B en dia 2 (cubierto) a las 00:05.
    times = np.array([1.002, 2.003])
    t_obs, T_obs = compress_to_observed_time(times, origin, missing)
    assert np.all(np.diff(t_obs) > 0)            # MONOTONO pese al cruce al dia fantasma
    # A anclado a ~1.0 (fin del dia cubierto 0); B en el dia observado 1 (2 - 1 fantasma)
    assert t_obs[0] <= 1.0 + 1e-9
    assert np.isclose(t_obs[1], 1.003)


def test_compress_to_observed_time_no_gaps_is_identity():
    # Sin dias fantasma, el reloj observado == calendario y T = span.
    origin = pd.Timestamp("2024-03-01", tz="UTC")
    times = np.array([0.1, 1.4, 2.9, 4.0])
    t_obs, T_obs = compress_to_observed_time(times, origin, [])
    assert np.allclose(t_obs, times)
    assert np.isclose(T_obs, 5.0)    # offset maximo 4 -> 5 dias de span, sin fantasmas


def test_compress_reduces_branching_ratio_bias():
    # Verificacion de raiz (auditoria 2026-08-15): ajustar sobre el span inflado con
    # dias fantasma infla n; comprimir a tiempo observado lo devuelve cerca del verdadero.
    from irfn.models.hawkes_mle import fit_hawkes_mle, simulate_hawkes_ogata_thinning
    rng = np.random.default_rng(20260815)
    T_cov = 200.0
    times, marks = simulate_hawkes_ogata_thinning(T_cov, mu=0.5, alpha=1.0, beta=2.0, rng=rng)
    n_true = 1.0 * marks.mean() / 2.0                       # ~0.25
    # "span" inflado: como si hubieramos anotado 600 dias fantasma extra al final.
    fit_span = fit_hawkes_mle(times, marks, T=T_cov + 600.0, n_starts=20, seed=1)
    fit_obs = fit_hawkes_mle(times, marks, T=T_cov, n_starts=20, seed=1)
    # el span infla n hacia la frontera; el observado queda cerca del verdadero
    assert fit_span.branching_ratio > 0.9
    assert abs(fit_obs.branching_ratio - n_true) < 0.1
    assert fit_obs.branching_ratio < fit_span.branching_ratio

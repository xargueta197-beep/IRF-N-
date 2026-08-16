"""Tests de la capa de features (V1): causalidad, rezago y ledger.

La invarianza de prefijo de los FEATURES se testea truncando los PRECIOS crudos:
si technical_features(close[:t]) difiere de technical_features(close)[:t] en
alguna fecha <= t, alguna ventana movil esta mirando el futuro. Complementa al
chequeo PIT del filtro (que fija parametros y trunca retornos+covariables ya
construidas): entre ambos cubren el pipeline entero de punta a punta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from irfn.audit.pit import lag_ledger, v1_feature_registry
from irfn.data.alfred import point_in_time_series
from irfn.features.macro import macro_features
from irfn.features.technical import technical_features

_WINDOWS = dict(sma_short=20, sma_long=200, bb_window=20, bb_k=2.0, z_window=250)


def _fake_prices(T: int = 700, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.01, size=T)
    close = 100.0 * np.exp(np.cumsum(r))
    return pd.Series(close, index=pd.bdate_range("2018-01-01", periods=T), name="close")


def test_features_prefix_invariance():
    """Features sobre precios truncados == features truncados (causalidad)."""
    close = _fake_prices()
    full = technical_features(close, **_WINDOWS)
    rng = np.random.default_rng(1)
    for pos in rng.integers(500, len(close) - 1, size=8):
        trunc = technical_features(close.iloc[: pos + 1], **_WINDOWS)
        a = full.iloc[: pos + 1]
        pd.testing.assert_frame_equal(a, trunc, atol=1e-12, rtol=0)


def test_features_shift_is_one():
    """El valor fechado en t es el feature CRUDO de t-1 (R3): recalcular el
    crudo a mano en t-1 debe coincidir con la columna rezagada en t."""
    close = _fake_prices()
    feats = technical_features(close, **_WINDOWS)

    t = 400  # posicion arbitraria con ventanas calientes
    w = _WINDOWS
    sma_s = close.rolling(w["sma_short"]).mean()
    sma_l = close.rolling(w["sma_long"]).mean()
    std_s = close.rolling(w["sma_short"]).std()
    raw_tm1 = (sma_s - sma_l).iloc[t - 1] / std_s.iloc[t - 1]

    assert feats["sma_gap"].iloc[t] == pytest.approx(raw_tm1, abs=1e-12)
    # y NO es el crudo contemporaneo (si coincidiera, el shift no existe)
    raw_t = (sma_s - sma_l).iloc[t] / std_s.iloc[t]
    assert feats["sma_gap"].iloc[t] != pytest.approx(raw_t, abs=1e-15)


def test_lag_ledger_v1_verde():
    """Todas las covariables V1 registradas con shift >= 1 => ledger verde."""
    ledger = lag_ledger(v1_feature_registry(include_macro=True))
    assert ledger.attrs["passed"], ledger.to_string(index=False)
    covs = ledger[ledger["role"] == "covariable"]
    assert set(covs["feature"]) >= {"sma_gap", "bb_width_z", "slope_2s10y", "hy_oas_z"}


def test_point_in_time_series_publication_lag():
    """Un dato de enero publicado el 25 de febrero NO existe antes del 25 de
    febrero (mas margen). Las revisiones posteriores actualizan el valor solo
    desde SU fecha de publicacion. Esta es la semantica de R4 en miniatura."""
    # publicaciones en lunes habiles: 2024-02-26, 2024-03-11 y 2024-03-25
    vint = pd.DataFrame(
        {
            "obs_date": pd.to_datetime(["2024-01-31", "2024-01-31", "2024-02-29"]),
            "value": [100.0, 101.5, 103.0],       # ene inicial, ene REVISADO, feb
            "realtime_start": pd.to_datetime(["2024-02-26", "2024-03-11", "2024-03-25"]),
            "realtime_end": pd.to_datetime(["2024-03-10", "9999-12-31", "9999-12-31"]),
        }
    )
    s = point_in_time_series(vint, margin_days=1, name="M2_TEST")

    cal = pd.bdate_range("2024-02-01", "2024-04-05")
    aligned = s.reindex(s.index.union(cal)).ffill().reindex(cal)

    # antes de la publicacion (+margen) no hay dato
    assert np.isnan(aligned.loc["2024-02-23"])
    assert np.isnan(aligned.loc["2024-02-26"])            # publicado ese dia, margen 1
    assert aligned.loc["2024-02-27"] == 100.0             # primer dia usable
    # la revision de enero rige desde SU publicacion (+margen), no antes
    assert aligned.loc["2024-03-11"] == 100.0
    assert aligned.loc["2024-03-12"] == 101.5
    # el dato de febrero releva al de enero desde su propia publicacion (+margen)
    assert aligned.loc["2024-03-25"] == 101.5
    assert aligned.loc["2024-03-26"] == 103.0


def test_point_in_time_ignores_stale_revision():
    """Una revision TARDIA de una observacion vieja no pisa a la observacion
    mas reciente ya publicada (el 'ultimo valor conocido' avanza, no retrocede)."""
    vint = pd.DataFrame(
        {
            "obs_date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-01-31"]),
            "value": [100.0, 103.0, 99.0],        # revision de enero DESPUES de feb
            "realtime_start": pd.to_datetime(["2024-02-25", "2024-03-25", "2024-04-10"]),
            "realtime_end": pd.to_datetime(["9999-12-31", "9999-12-31", "9999-12-31"]),
        }
    )
    s = point_in_time_series(vint, margin_days=0, name="X")
    cal = pd.bdate_range("2024-03-20", "2024-04-20")
    aligned = s.reindex(s.index.union(cal)).ffill().reindex(cal)
    # tras el 10 de abril el "ultimo valor conocido" sigue siendo el de FEBRERO
    assert aligned.loc["2024-04-15"] == 103.0


def test_point_in_time_weekend_collision_no_duplicates():
    """Publicaciones en viernes, sabado y domingo colapsan sobre el mismo lunes
    al aplicar el margen en dias habiles. El indice de disponibilidad debe salir
    SIN duplicados (gana el ultimo estado publicado) o el reindex de
    macro_features revienta con 'cannot reindex on an axis with duplicate
    labels' (bug real observado con vintages ALFRED de DGS10/DGS2/BAMLH0A0HYM2,
    que traen realtime_start en fin de semana)."""
    # vie 2024-03-01, sab 2024-03-02, dom 2024-03-03: +1 habil => lunes 03-04
    vint = pd.DataFrame(
        {
            "obs_date": pd.to_datetime(["2024-02-29", "2024-03-01", "2024-03-02"]),
            "value": [1.0, 2.0, 3.0],
            "realtime_start": pd.to_datetime(["2024-03-01", "2024-03-02", "2024-03-03"]),
            "realtime_end": pd.to_datetime(["9999-12-31", "9999-12-31", "9999-12-31"]),
        }
    )
    s = point_in_time_series(vint, margin_days=1, name="X")
    assert not s.index.duplicated().any()
    # el valor usable el lunes es el del ULTIMO estado publicado (domingo)
    assert s.loc["2024-03-04"] == 3.0
    # y macro_features (el consumidor real) no debe reventar con esta serie
    cal = pd.bdate_range("2024-03-01", "2024-03-15")
    X = macro_features({"DGS10": s, "DGS2": s * 0.5}, cal, z_window=5)
    assert "slope_2s10y" in X.columns


def test_macro_features_alignment():
    """macro_features alinea por disponibilidad con ffill y sin shift extra."""
    cal = pd.bdate_range("2024-01-01", periods=300)
    avail = pd.bdate_range("2024-01-02", periods=299)     # disponible un dia despues
    dgs10 = pd.Series(np.linspace(4.0, 4.5, 299), index=avail, name="DGS10")
    dgs2 = pd.Series(np.linspace(4.5, 4.2, 299), index=avail, name="DGS2")
    oas = pd.Series(3.0 + np.sin(np.arange(299) / 20.0), index=avail, name="BAMLH0A0HYM2")

    X = macro_features(
        {"DGS10": dgs10, "DGS2": dgs2, "BAMLH0A0HYM2": oas}, cal, z_window=60
    )
    assert list(X.columns) == ["slope_2s10y", "hy_oas_z"]
    assert np.isnan(X["slope_2s10y"].iloc[0])             # nada disponible el dia 1
    expected = dgs10.iloc[0] - dgs2.iloc[0]
    assert X["slope_2s10y"].loc[avail[0]] == pytest.approx(expected)
    assert X["hy_oas_z"].notna().sum() > 200              # z caliente tras la ventana

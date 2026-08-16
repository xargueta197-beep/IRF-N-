"""Tests del prefijo historico recuperado (data/recovered.py).

El empalme Wayback + ALFRED debe ser un prefijo ESTRICTO (en el solape manda
ALFRED), con el lag de publicacion medido de las vintages reales, y la serie
point-in-time resultante no puede tener duplicados ni saltos de orden.
"""

import pandas as pd
import pytest

from irfn.data.alfred import point_in_time_series
from irfn.data.recovered import splice_recovered_prefix


def _alfred_vintages():
    """Vintages tipo BAMLH0A0HYM2: diarias, lag de publicacion 0, sin revision."""
    obs = pd.bdate_range("2023-07-17", periods=10)
    return pd.DataFrame(
        {
            "obs_date": obs,
            "value": [3.0 + i / 10 for i in range(10)],
            "realtime_start": obs,  # lag medido = 0, como la serie real
            "realtime_end": pd.Timestamp("9999-12-31"),
        }
    )


def _write_recovered(tmp_path, series_id, obs_dates, values):
    pd.DataFrame({"obs_date": obs_dates, "value": values}).to_parquet(
        tmp_path / f"{series_id}_wayback_20251104.parquet", index=False
    )


def test_splice_is_strict_prefix(tmp_path):
    """Solo entran obs anteriores a la primera de ALFRED; en el solape manda ALFRED."""
    alfred = _alfred_vintages()
    # el snapshot recuperado cubre 1996 y TAMBIEN pisa el rango ALFRED con un
    # valor distinto a proposito: ese tramo debe descartarse.
    rec_dates = pd.to_datetime(["1996-12-31", "1997-01-02", "2023-07-17", "2023-07-18"])
    _write_recovered(tmp_path, "SERIE_X", rec_dates, [9.9, 9.8, 999.0, 999.0])

    spliced, info = splice_recovered_prefix("SERIE_X", alfred, recovered_dir=tmp_path)

    assert info is not None
    assert info["prefix_n_obs"] == 2
    assert info["prefix_last_obs"] == "1997-01-02"
    # ninguna fila del prefijo cae en o despues del primer obs_date de ALFRED
    prefix_rows = spliced[spliced["obs_date"] < alfred["obs_date"].min()]
    assert len(prefix_rows) == 2
    # el valor de ALFRED en el solape sobrevive intacto (999.0 descartado)
    v = spliced.loc[spliced["obs_date"] == pd.Timestamp("2023-07-17"), "value"]
    assert list(v) == [3.0]


def test_splice_feeds_point_in_time_without_duplicates(tmp_path):
    """La serie PIT del empalme es continua, ordenada y sin duplicados de indice."""
    alfred = _alfred_vintages()
    rec_dates = pd.bdate_range("2023-06-01", periods=20)  # termina antes de ALFRED
    _write_recovered(tmp_path, "SERIE_X", rec_dates, [2.0] * 20)

    spliced, info = splice_recovered_prefix("SERIE_X", alfred, recovered_dir=tmp_path)
    assert info is not None
    s = point_in_time_series(spliced, margin_days=1, name="x")
    assert not s.index.duplicated().any()
    assert s.index.is_monotonic_increasing
    # el tramo recuperado aparece antes del tramo ALFRED
    assert s.loc[: "2023-07-14"].iloc[-1] == 2.0
    assert s.iloc[-1] == pytest.approx(3.9)


def test_no_snapshot_is_identity(tmp_path):
    """Sin snapshot recuperado, las vintages salen intactas e info=None."""
    alfred = _alfred_vintages()
    spliced, info = splice_recovered_prefix("SERIE_SIN_SNAPSHOT", alfred, recovered_dir=tmp_path)
    assert info is None
    pd.testing.assert_frame_equal(spliced, alfred)


def test_alfred_covering_everything_is_identity(tmp_path):
    """Si ALFRED ya cubre todo lo recuperado, no se antepone nada."""
    alfred = _alfred_vintages()
    rec_dates = pd.bdate_range("2023-07-18", periods=5)  # todo dentro de ALFRED
    _write_recovered(tmp_path, "SERIE_X", rec_dates, [1.0] * 5)
    spliced, info = splice_recovered_prefix("SERIE_X", alfred, recovered_dir=tmp_path)
    assert info is None
    pd.testing.assert_frame_equal(spliced, alfred)

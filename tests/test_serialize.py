"""Tests de outputs/serialize.py (helpers compartidos por run_pipeline y run_v3).

Verifican que, dado un WalkForwardResult y su calibracion, se producen
history.parquet y walkforward.json con la forma que la app espera.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from irfn.outputs.serialize import write_history_parquet, write_walkforward_json


def _fake_wf(K: int = 2):
    ts = pd.Timestamp
    block = SimpleNamespace(
        block_id=0,
        train_start=ts("2015-01-02"), test_start=ts("2019-01-02"), test_end=ts("2019-07-01"),
        n_train=1000, n_test=120,
        loglik_train_per_obs=-1.23, loglik_test_per_obs=-1.30, n_converged=18,
        kappa=np.array([0.9, 0.7]), P=np.array([[0.95, 0.05], [0.1, 0.9]]),
    )
    return SimpleNamespace(
        blocks=[block], n_blocks=1, K=K, seed=42, n_starts=20,
        block_boundaries=[ts("2019-01-02")],
    )


def _fake_oos(K: int = 2, n: int = 30):
    idx = pd.date_range("2019-01-02", periods=n, freq="B")
    data = {"r": np.random.default_rng(0).normal(0, 1, n)}
    for k in range(K):
        data[f"xi_filtered_{k}"] = 0.5
        data[f"xi_predicted_{k}"] = 0.5
    data["entropy"] = 0.6
    data["entropy_norm"] = 0.9
    data["argmax_idx"] = 0
    data["argmax"] = "baja volatilidad"
    data["block_id"] = 0
    data["loglik_obs"] = -1.0
    data["r_pred_mean"] = 0.01
    return pd.DataFrame(data, index=idx)


def test_write_walkforward_json(tmp_path):
    calib = {"log_loss": 0.179, "log_loss_baseline": 0.177, "brier": 0.05,
             "brier_baseline": 0.06, "ece": 0.02, "n_obs": 30, "target_note": "proxy"}
    out = tmp_path / "walkforward.json"
    write_walkforward_json(out, _fake_wf(), calib)
    obj = json.loads(out.read_text())
    assert obj["n_blocks"] == 1 and obj["K"] == 2 and obj["n_starts"] == 20
    assert obj["block_boundaries"] == ["2019-01-02"]
    assert obj["blocks"][0]["kappa"] == [0.9, 0.7]
    # el baseline honesto viaja tal cual dentro de calibration
    assert obj["calibration"]["log_loss_baseline"] == 0.177


def test_write_history_parquet(tmp_path):
    out = tmp_path / "history.parquet"
    write_history_parquet(out, _fake_oos())
    df = pd.read_parquet(out)
    assert "fecha" in df.columns and "equity" in df.columns
    assert "xi_filtered_0" in df.columns and "r_pred_mean" in df.columns
    # equity monotono en construccion desde exp(cumsum) -> primer valor > 0
    assert (df["equity"] > 0).all()
    assert len(df) == 30

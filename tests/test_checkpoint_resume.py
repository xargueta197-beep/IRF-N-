"""Los checkpoints de corridas largas (kselect/ablation) deben ser transparentes:
reanudar tras un corte tiene que dar EXACTAMENTE el mismo resultado que una
corrida sin cortes (misma semilla por replica/bloque, R6). Estos tests simulan
un corte a mitad de camino (una excepcion no capturada, como mataria al proceso
un corte real) y verifican que la segunda llamada retoma donde quedo y llega
al mismo resultado final que la corrida de referencia sin interrupciones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import irfn.validation.tests_stat as ts
import irfn.validation.walkforward as wfmod
from irfn.models.msgarch import simulate
from irfn.models.params import pack


class _SimulatedCrash(Exception):
    """Excepcion que NO es RuntimeError: bootstrap_lr_test solo atrapa RuntimeError
    (replica fallida), asi que esta se propaga tal como lo haria un corte real
    del proceso (SIGTERM/sistema suspendido), no una replica que simplemente
    no convergio."""


def test_bootstrap_lr_test_resume_after_crash_matches_uninterrupted(tmp_path, monkeypatch):
    true = {
        "mu": np.array([0.02]), "v": np.array([1.0]),
        "alpha": np.array([0.05]), "gamma": np.array([0.06]),
        "beta": np.array([0.85]), "P": np.array([[1.0]]),
    }
    r, _ = simulate(pack(true, K=1), K=1, T=600, seed=9)
    kwargs = dict(K_null=1, K_alt=2, dist="normal", n_boot=9,
                  n_starts_data=4, n_starts_boot=2, seed=11)

    direct = ts.bootstrap_lr_test(r, **kwargs)

    ckpt = tmp_path / "boot.pkl"
    real_fit = ts.fit
    calls = {"n": 0}

    def flaky_fit(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 9:   # revienta a mitad de la 3a replica del loop
            raise _SimulatedCrash("corte simulado")
        return real_fit(*a, **kw)

    monkeypatch.setattr(ts, "fit", flaky_fit)
    try:
        try:
            ts.bootstrap_lr_test(r, **kwargs, checkpoint_path=ckpt, checkpoint_every=1)
            assert False, "se esperaba que el corte simulado interrumpiera la corrida"
        except _SimulatedCrash:
            pass
    finally:
        monkeypatch.setattr(ts, "fit", real_fit)

    assert ckpt.exists(), "el checkpoint deberia existir tras el corte"

    resumed = ts.bootstrap_lr_test(r, **kwargs, checkpoint_path=ckpt, checkpoint_every=1)

    assert resumed["n_boot_ok"] == direct["n_boot_ok"]
    assert resumed["lr_boot_q50"] == direct["lr_boot_q50"]
    assert resumed["p_value"] == direct["p_value"]
    assert resumed["lr_obs"] == direct["lr_obs"]


def test_walk_forward_resume_after_crash_matches_uninterrupted(tmp_path, monkeypatch):
    true = {
        "mu": np.array([0.02, -0.01]), "v": np.array([0.5, 2.0]),
        "alpha": np.array([0.05, 0.05]), "gamma": np.array([0.06, 0.06]),
        "beta": np.array([0.85, 0.85]), "P": np.array([[0.98, 0.02], [0.05, 0.95]]),
    }
    r_arr, _ = simulate(pack(true, K=2), K=2, T=2200, seed=7)
    idx = pd.bdate_range("2013-01-02", periods=len(r_arr))
    returns = pd.Series(r_arr, index=idx)

    kwargs = dict(K=2, seed=5, n_starts=2, train_years=2, test_months=6, n_blocks_min=3)

    direct = wfmod.walk_forward(returns, **kwargs)

    ckpt = tmp_path / "wf.pkl"
    real_fit = wfmod.fit
    calls = {"n": 0}

    def flaky_fit(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:   # revienta al 2o bloque (el 1o ya quedo en checkpoint)
            raise _SimulatedCrash("corte simulado")
        return real_fit(*a, **kw)

    monkeypatch.setattr(wfmod, "fit", flaky_fit)
    try:
        try:
            wfmod.walk_forward(returns, **kwargs, checkpoint_path=ckpt)
            assert False, "se esperaba que el corte simulado interrumpiera la corrida"
        except _SimulatedCrash:
            pass
    finally:
        monkeypatch.setattr(wfmod, "fit", real_fit)

    assert ckpt.exists(), "el checkpoint deberia existir tras el corte"

    resumed = wfmod.walk_forward(returns, **kwargs, checkpoint_path=ckpt)

    assert resumed.n_blocks == direct.n_blocks
    for bd, br in zip(direct.blocks, resumed.blocks):
        assert bd.block_id == br.block_id
        np.testing.assert_array_equal(bd.theta, br.theta)
        assert bd.loglik_train_per_obs == br.loglik_train_per_obs
    pd.testing.assert_frame_equal(direct.oos_frame, resumed.oos_frame)

"""El guardian de R1. Publicar un payload que contenga xi_smoothed (o
cualquier otra clave de FORBIDDEN_KEYS) debe lanzar LookAheadViolation.

A diferencia de los otros 5 tests obligatorios, este SI corre en verde ya en
Sesion 0: publish.py y schema.py son infraestructura completa, no dependen
del modelo. No hay motivo honesto para marcarlo skip cuando la pieza que
prueba ya existe y funciona.
"""

from __future__ import annotations

import copy

import pytest

from irfn.outputs.publish import FORBIDDEN_KEYS, LookAheadViolation, publish

VALID_PAYLOAD = {
    "run_id": "a3f9c1e",
    "generated_at": "2026-07-11T06:00:00Z",
    "git_commit": "e91b2d4",
    "config_hash": "7c1f0000",
    "asof": "2026-07-10",
    "version": "V2",
    "model": {
        "K": 3,
        "spec": "MS-GJR-GARCH(1,1) Haas et al. (2004)",
        "tvtp": True,
        "covariates": ["sma_gap", "bb_width_z"],
        "news_layer": ["surprise"],
        "estimation": {"method": "MLE", "multistart": 30, "seed": 42, "converged": True},
        "news_layer_params": {
            "active": True,
            "delta": 0.0277,
            "delta_se": 0.004,
            "surprise_start_date": "2026-01-15",
            "indicators": [
                {"indicator": "CPI", "w": 1.4, "se": 0.3, "t_stat": 4.7,
                 "distinguible_de_cero": True, "n_events": 24},
            ],
            "coverage": {"CPI": {"n_releases": 24, "n_with_consensus": 24}},
            "blocker": None,
        },
        # Contrato V3: parametros de la capa de Hawkes (MLE + KS + branching).
        "hawkes_layer_params": {
            "active": True,
            "mu_N": 12.4, "alpha": 0.9, "beta": 2.1,
            "se_mu_N": 0.8, "se_alpha": 0.11, "se_beta": 0.25,
            "mean_mark": 0.55, "branching_ratio": 0.68,
            "branching_ratio_se": 0.03, "branching_ratio_ci_low": 0.62, "branching_ratio_ci_high": 0.74,
            "expected_cascade": 3.12, "expected_cascade_bounded": True,
            "stationary": True,
            "ks_stat": 0.021, "ks_pvalue": 0.14, "ks_passed": True,
            "n_events": 41250, "n_starts": 30, "starts_at_best": 27,
            "coverage": {"first_day": "2017-01-02", "last_day": "2026-07-12",
                         "n_days": 3450, "n_missing_days": 3, "n_censored_days": 12},
            "reflexive_threshold": 0.8,
            "blocker": None,
        },
    },
    "regime": {
        "labels": ["risk_on", "transicion", "risk_off"],
        "xi_filtered": [0.62, 0.29, 0.09],
        "entropy": 0.87,
        "entropy_max": 1.0986,
        "confidence": "media",
        "expected_duration_days": [14.2, 3.1, 9.8],
        "argmax": "risk_on",
        "xi_momentum_5d": [-0.08, 0.05, 0.03],
    },
    "transition_matrix_today": [
        [0.93, 0.05, 0.02],
        [0.31, 0.42, 0.27],
        [0.03, 0.11, 0.86],
    ],
    "news": {
        "surprise_index": -0.42,
        "lambda_N": 1.83,
        "lambda_N_z": 0.91,
        "branching_ratio": 0.68,
        "branching_ratio_ci_low": 0.62, "branching_ratio_ci_high": 0.74,
        "expected_cascade": 3.12, "expected_cascade_bounded": True,
        "attribution": {"surprise": 0.31, "hawkes": 0.12, "price": 0.57},
    },
    "conditional_stats": {
        "BTC": {"risk_on": {
            "mean_ann": {"value": 0.71, "ci_low": 0.55, "ci_high": 0.88, "n_obs": 210},
            "vol_ann": {"value": 0.48, "ci_low": 0.40, "ci_high": 0.57, "n_obs": 210},
            "sharpe": {"value": 1.48, "ci_low": 1.10, "ci_high": 1.90, "n_obs": 210},
            "maxdd": {"value": -0.22, "ci_low": None, "ci_high": None, "n_obs": 210},
        }},
    },
    "warnings": [],
    "validation_ref": "runs/a3f9c1e/validation.json",
    "disclaimer": "Indicador de investigacion. No es recomendacion de inversion.",
}


def test_valid_payload_publishes_without_error(tmp_path):
    out = tmp_path / "irfn.json"
    publish(copy.deepcopy(VALID_PAYLOAD), out)
    assert out.exists()


@pytest.mark.parametrize("forbidden_key", sorted(FORBIDDEN_KEYS))
def test_no_smoother_in_outputs(tmp_path, forbidden_key):
    payload = copy.deepcopy(VALID_PAYLOAD)
    payload["regime"][forbidden_key] = [0.1, 0.2, 0.7]

    with pytest.raises(LookAheadViolation):
        publish(payload, tmp_path / "irfn.json")

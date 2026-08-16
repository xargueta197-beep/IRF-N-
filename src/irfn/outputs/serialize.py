"""Serializacion del set OOS que lee la app: history.parquet + walkforward.json.

Extraido de scripts/run_pipeline.py (Fase 4) para que TANTO run_pipeline (V0)
COMO run_v3 (V3/V4) escriban estos dos artefactos con el MISMO codigo, sobre el
walk-forward de su propio modelo (R2: re-estimado por bloque, nunca heredado).
Cero logica de modelo aqui: solo toma un WalkForwardResult ya calculado y su
calibracion, y los vuelca a disco.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_walkforward_json(path: Path, wf, calib: dict) -> None:
    """Metricas por bloque + calibracion + fronteras -> walkforward.json.

    `wf` es un WalkForwardResult (validation/walkforward.py); `calib` es el dict
    de validation.calibration.summarize sobre el ξ PREDICHO OOS del mismo modelo
    (incluye log_loss del modelo vs log_loss_baseline = climatologia de regimen,
    el baseline honesto; jamas el favorecedor).
    """
    blocks = []
    for b in wf.blocks:
        blocks.append({
            "block_id": b.block_id,
            "train_start": str(b.train_start.date()),
            "test_start": str(b.test_start.date()),
            "test_end": str(b.test_end.date()),
            "n_train": b.n_train,
            "n_test": b.n_test,
            "loglik_train_per_obs": b.loglik_train_per_obs,
            "loglik_test_per_obs": b.loglik_test_per_obs,
            "n_converged": b.n_converged,
            "kappa": b.kappa.tolist(),
            "P": b.P.tolist(),
        })
    obj = {
        "n_blocks": wf.n_blocks,
        "K": wf.K,
        "seed": wf.seed,
        "n_starts": wf.n_starts,
        "block_boundaries": [str(d.date()) for d in wf.block_boundaries],
        "blocks": blocks,
        "calibration": calib,
    }
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_history_parquet(path: Path, oos) -> None:
    """Historia out-of-sample para la vista Historico: ξ filtrado + equity.

    `oos` es el oos_frame del walk-forward (una fila por dia OOS, con columnas
    xi_filtered_k, xi_predicted_k, entropy, entropy_norm, argmax_idx, argmax,
    block_id, loglik_obs, r_pred_mean, r). Se reconstruye un indice de equity
    acumulado desde el primer dia OOS (contexto visual del precio); nada mas.
    """
    df = oos.copy()
    r_dec = df["r"].to_numpy() / 100.0
    df["equity"] = np.exp(np.cumsum(r_dec))
    df.index.name = "fecha"
    df.reset_index().to_parquet(Path(path), index=False)

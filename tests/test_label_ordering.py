"""R5 en accion: tras CADA estimacion, v_1 < v_2. Se verifica en 20 semillas
distintas de datos simulados.

El orden es estructural (params.py construye v por sumas de exponenciales, asi
que v es creciente por definicion para cualquier theta). Este test es el blindaje
que confirma que ninguna ruta de estimacion produce una violacion, y que el
regimen de mayor varianza es SIEMPRE el ultimo indice -- lo que hace comparables
los regimenes entre bloques de walk-forward (mata el label switching).

Se corre con compute_se=False y series/arranques modestos: aqui interesa el orden
del punto estimado, no su incertidumbre.
"""

from __future__ import annotations

import numpy as np
import pytest

from irfn.models.estimate import fit
from irfn.models.msgarch import simulate
from irfn.models.params import pack

_TRUE = {
    "mu": np.array([0.0005, -0.0003]),
    "v": np.array([9.0e-5, 6.0e-4]),
    "alpha": np.array([0.05, 0.08]),
    "gamma": np.array([0.03, 0.10]),
    "beta": np.array([0.87, 0.80]),
    "P": np.array([[0.98, 0.02], [0.05, 0.95]]),
}


@pytest.mark.slow
def test_label_ordering():
    theta_true = pack(_TRUE, K=2)
    for seed in range(20):
        r, _ = simulate(theta_true, K=2, T=700, seed=1000 + seed)
        result = fit(r, K=2, n_starts=4, seed=seed, compute_se=False)
        v = result.params["v"]
        assert v[0] < v[1], f"semilla {seed}: v no ordenada tras estimar: {v}"

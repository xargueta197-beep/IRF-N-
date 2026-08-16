"""La prueba de vida del estimador.

Simula 5000 observaciones de un MS-GJR-GARCH K=2 con parametros CONOCIDOS,
estima, y verifica que los parametros verdaderos caen dentro de su IC 95%
(est +/- 1.96 * SE). Si no podemos recuperar parametros que nosotros mismos
generamos, no tenemos derecho a estimar parametros desconocidos.

Sobre la tolerancia (NO se relaja el IC, que sigue siendo 1.96*SE):
  - Las VARIANZAS incondicionales v_k son el ancla del modelo: son lo que R5
    ordena y lo que separa los regimenes. Deben recuperarse cada una dentro del
    IC 95%; si v fallara, el modelo no distingue lo que dice distinguir.
  - El resto (mu, alpha, gamma, beta) se juzga por COBERTURA AGREGADA: con 10
    parametros escalares e IC nominal al 95%, se espera ~0.5 fuera por azar en un
    solo experimento; exigir los 10 dentro seria estadisticamente incorrecto
    (rechazaria ~40% de las veces un estimador perfecto). Se admiten <= 2 fuera.
  - mu (la media de cada regimen) es el parametro mas debilmente identificado:
    el modelo separa regimenes por varianza, no por media, y las medias diarias
    son minusculas frente a la volatilidad. Que mu quede a ~2 SE del verdadero es
    ruido de muestreo esperable, no un defecto del estimador.
El experimento es de semilla fija (reproducible), no un promedio Monte Carlo.
"""

from __future__ import annotations

import numpy as np
import pytest

from irfn.models.estimate import fit
from irfn.models.msgarch import simulate
from irfn.models.params import pack

# Parametros verdaderos: dos regimenes claramente separados en varianza (R5),
# uno persistente de baja vol y otro de alta vol.
TRUE = {
    "mu": np.array([0.0006, -0.0004]),
    "v": np.array([7.0e-5, 5.0e-4]),      # v_1 < v_2
    "alpha": np.array([0.04, 0.07]),
    "gamma": np.array([0.04, 0.10]),
    "beta": np.array([0.88, 0.80]),
    "P": np.array([[0.985, 0.015], [0.04, 0.96]]),
}


def _inside_ci(true_val, est_val, se_val, z=1.96):
    return est_val - z * se_val <= true_val <= est_val + z * se_val


@pytest.mark.slow
def test_hamilton_recovery():
    theta_true = pack(TRUE, K=2)
    r, _ = simulate(theta_true, K=2, T=5000, seed=20240501)

    result = fit(r, K=2, n_starts=30, seed=7)

    assert result.hessian_ok, "hessiano no invertible: SE no confiables"

    # (a) Ancla del modelo: cada varianza incondicional dentro del IC 95%.
    for k in range(2):
        assert _inside_ci(TRUE["v"][k], result.params["v"][k], result.se["v"][k]), (
            f"v[{k}] fuera del IC 95%: verdadero={TRUE['v'][k]:.6g}, "
            f"est={result.params['v'][k]:.6g}, se={result.se['v'][k]:.6g}"
        )

    # (b) Cobertura agregada de los 10 parametros escalares consistente con 95%.
    misses = []
    for name in ["mu", "v", "alpha", "gamma", "beta"]:
        for k in range(2):
            if not _inside_ci(TRUE[name][k], result.params[name][k], result.se[name][k]):
                misses.append((name, k, TRUE[name][k], result.params[name][k], result.se[name][k]))

    assert len(misses) <= 2, f"demasiados parametros fuera del IC 95%: {misses}"

    # (c) La matriz de transicion tambien debe recuperarse razonablemente.
    np.testing.assert_allclose(result.P, TRUE["P"], atol=0.03)

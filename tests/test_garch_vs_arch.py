"""Ancla externa: con K=1 nuestra implementacion debe reproducir a la libreria
`arch` (GJR-GARCH(1,1), media constante, innovaciones Normales) dentro de
tolerancia numerica razonable, en loglik y en los parametros (omega, alpha,
gamma, beta, mu).

Con K=1 el filtro de Hamilton colapsa a un unico regimen: xi_filtered == 1
siempre, y la loglik es exactamente la de un GJR-GARCH estandar. Si esto no
coincide con `arch`, el nucleo esta mal antes de meter regimenes.

Discrepancia esperada y controlada: nosotros inicializamos sigma2_0 en la
varianza incondicional; `arch` usa un backcast exponencial. Sobre una serie
larga el transitorio se lava, asi que comparamos con tolerancia relativa en
loglik y tolerancia absoluta modesta en los parametros -- no se relaja la
tolerancia para tapar un bug; se dimensiona al efecto de arranque conocido.

`arch` se usa SOLO aqui como referencia; nunca en produccion (ver CLAUDE.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from irfn.models.estimate import fit
from irfn.models.msgarch import simulate
from irfn.models.params import pack

arch = pytest.importorskip("arch")


@pytest.mark.slow
def test_garch_vs_arch():
    # Serie GJR-GARCH de un solo regimen, en escala "porcentual" (~1) donde arch
    # trabaja comodo. K=1: P=[[1]], sin dinamica de estados.
    true = {
        "mu": np.array([0.03]),
        "v": np.array([1.0]),           # varianza incondicional ~ 1 (sd ~ 1)
        "alpha": np.array([0.06]),
        "gamma": np.array([0.08]),
        "beta": np.array([0.86]),
        "P": np.array([[1.0]]),
    }
    theta_true = pack(true, K=1)
    r, _ = simulate(theta_true, K=1, T=4000, seed=99)

    # --- nuestra implementacion ---
    ours = fit(r, K=1, n_starts=8, seed=3)

    # --- referencia arch: GJR-GARCH(1,1) = GARCH con o=1 ---
    from arch import arch_model

    am = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="normal", rescale=False)
    ref = am.fit(disp="off")

    # loglik: ancla principal, tolerancia relativa por el arranque distinto
    rel_ll = abs(ours.loglik - ref.loglikelihood) / abs(ref.loglikelihood)
    assert rel_ll < 5e-3, f"loglik difiere {rel_ll:.2e} (nuestro={ours.loglik:.3f}, arch={ref.loglikelihood:.3f})"

    # parametros: arch reporta 'mu', 'omega', 'alpha[1]', 'gamma[1]', 'beta[1]'
    p = ours.params
    assert abs(p["omega"][0] - ref.params["omega"]) < 0.05
    assert abs(p["alpha"][0] - ref.params["alpha[1]"]) < 0.03
    assert abs(p["gamma"][0] - ref.params["gamma[1]"]) < 0.03
    assert abs(p["beta"][0] - ref.params["beta[1]"]) < 0.03
    assert abs(p["mu"][0] - ref.params["mu"]) < 0.03

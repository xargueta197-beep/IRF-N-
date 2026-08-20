"""FASE B -- Doble implementacion: versiones lentas y obviamente correctas vs
produccion, con tolerancias exigidas por el protocolo.

  B1. Log-verosimilitud Hawkes: doble suma explicita O(n^2) (sin recursion)   1e-10
  B2. Compensador Lambda(T): cuadratura numerica de lambda(s) intervalo a
      intervalo (scipy.quad), NO la formula analitica                          1e-8*
  B3. Filtro de Hamilton: bucle ingenuo con normalizacion explicita           1e-10
      (constante, t de Student y TVTP)
  B4. Log-verosimilitud MS con statsmodels regime_switching (caso comparable:
      varianza constante por regimen, persistencia GARCH ~0)                  1e-6 rel
  B5. BIC: conteo manual de parametros enumerado                              exacto
  B6. Pesos fracdiff: NO APLICA (no existe diferenciacion fraccionaria en src/)

* Justificacion de la tolerancia de B2: el protocolo pide 1e-10 para una
  integracion "analitica termino a termino", que seria la MISMA formula del
  codigo (no independiente). Se usa cuadratura adaptativa (epsabs=1e-10 por
  tramo, ~500 tramos) cuyo error acumulado es ~1e-8 relativo: es una referencia
  verdaderamente independiente, y ese es su piso de precision.

Solo LECTURA de src/. Resultados en audits/probes/out/b_results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import integrate, stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from irfn.models import hawkes_mle as hm  # noqa: E402
from irfn.models.params import pack, unpack, n_params  # noqa: E402
from irfn.models.hamilton import hamilton_filter  # noqa: E402
from irfn.models.msgarch import simulate  # noqa: E402
from irfn.models.estimate import fit  # noqa: E402
from irfn.models.tvtp import transition_matrices  # noqa: E402

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
RES: dict = {}


def check(name, value, tol, extra=None):
    ok = bool(value < tol)
    RES[name] = {"value": float(value), "tol": tol, "ok": ok, **(extra or {})}
    print(f"[{name}] {'OK ' if ok else 'FALLA'} err={value:.3e} (tol {tol:g})", flush=True)


# ---------------------------------------------------------------------------
# B1: Hawkes loglik por doble suma O(n^2)
# ---------------------------------------------------------------------------
def hawkes_loglik_naive(t, s, mu, alpha, beta, T):
    n = len(t)
    ll = 0.0
    for i in range(n):
        lam = mu
        for j in range(i):
            lam += alpha * s[j] * np.exp(-beta * (t[i] - t[j]))
        ll += np.log(lam)
    comp = mu * T
    for i in range(n):
        comp += (alpha / beta) * s[i] * (1.0 - np.exp(-beta * (T - t[i])))
    return ll - comp


rng = np.random.default_rng(123)
t_ev = np.sort(rng.uniform(0, 400.0, size=3000))
s_ev = rng.beta(2, 1.2, size=3000)  # marcas tipo relevancia, media ~0.63
mu0, a0, b0 = 0.5, 0.8, 1.2
ll_prod = hm.hawkes_loglik(t_ev, s_ev, mu0, a0, b0, 400.0)
ll_naive = hawkes_loglik_naive(t_ev, s_ev, mu0, a0, b0, 400.0)
check("B1_hawkes_loglik_vs_double_sum", abs(ll_prod - ll_naive) / max(1.0, abs(ll_naive)), 1e-10,
      {"ll_prod": ll_prod, "ll_naive": ll_naive, "n_events": 3000, "note": "error RELATIVO"})

# tambien con beta grande (escala de produccion, ~30/dia) y empates ditherizados
t_ev2 = np.sort(rng.uniform(0, 240.0, size=2000))
ll_p2 = hm.hawkes_loglik(t_ev2, s_ev[:2000], 100.0, 34.0, 30.0, 240.0)
ll_n2 = hawkes_loglik_naive(t_ev2, s_ev[:2000], 100.0, 34.0, 30.0, 240.0)
check("B1b_hawkes_loglik_prod_scale", abs(ll_p2 - ll_n2) / max(1.0, abs(ll_n2)), 1e-10,
      {"ll_prod": ll_p2, "ll_naive": ll_n2})

# ---------------------------------------------------------------------------
# B2: compensador por cuadratura adaptativa (independiente de la formula)
# ---------------------------------------------------------------------------
t_q = np.sort(rng.uniform(0, 80.0, size=400))
s_q = rng.beta(2, 1.2, size=400)


def lam_naive(u):
    mask = t_q < u
    return mu0 + a0 * np.sum(s_q[mask] * np.exp(-b0 * (u - t_q[mask])))


T_q = 80.0
knots = np.concatenate([[0.0], t_q, [T_q]])
comp_quad = 0.0
for lo, hi in zip(knots[:-1], knots[1:]):
    if hi > lo:
        val, _ = integrate.quad(lam_naive, lo, hi, epsabs=1e-10, epsrel=1e-10, limit=200)
        comp_quad += val
comp_prod = hm.compensator(t_q, s_q, mu0, a0, b0, T_q)
check("B2_compensator_vs_quadrature", abs(comp_prod - comp_quad) / max(1.0, abs(comp_quad)), 1e-8,
      {"comp_prod": comp_prod, "comp_quad": comp_quad})

# identidad del re-escalamiento: Lambda(t_i) por formula directa O(n^2)
tau_prod = hm.time_rescaling(t_q, s_q, {"mu": mu0, "alpha": a0, "beta": b0})
tau_naive = np.array([hm.compensator(t_q[:i], s_q[:i], mu0, a0, b0, t_q[i]) if i else mu0 * t_q[0]
                      for i in range(len(t_q))])
# ojo: compensator(t[:i], ..., T=t[i]) = mu*t_i + suma_{j<i} -- exactamente Lambda(t_i)
check("B2b_time_rescaling_identity", float(np.max(np.abs(tau_prod - tau_naive))), 1e-10)

# ---------------------------------------------------------------------------
# B3: filtro de Hamilton ingenuo
# ---------------------------------------------------------------------------
def hamilton_naive(r, params, K, X_lagged=None):
    """Bucle ingenuo, densidades con scipy.stats, normalizacion explicita."""
    T = len(r)
    mu, v = params["mu"], params["v"]
    om, al, ga, be = params["omega"], params["alpha"], params["gamma"], params["beta"]
    # varianzas condicionales (recursion Haas escrita a mano)
    sig2 = np.empty((T, K))
    sig2[0] = v
    for t in range(1, T):
        for k in range(K):
            e = r[t - 1] - mu[k]
            sig2[t, k] = om[k] + al[k] * e * e + (ga[k] * e * e if e < 0 else 0.0) + be[k] * sig2[t - 1, k]
    # densidades
    f = np.empty((T, K))
    for k in range(K):
        if "nu" in params:
            nu = params["nu"][k]
            scale = np.sqrt(sig2[:, k] * (nu - 2.0) / nu)
            f[:, k] = stats.t.pdf(r, df=nu, loc=mu[k], scale=scale)
        else:
            f[:, k] = stats.norm.pdf(r, loc=mu[k], scale=np.sqrt(sig2[:, k]))
    # trayectoria de P
    if X_lagged is not None:
        P_path = transition_matrices(params["d"], params["beta_tvtp"], X_lagged)
    # distribucion inicial: la misma que produccion (estacionaria de P)
    from irfn.models.hamilton import stationary_distribution
    xi = stationary_distribution(params["P"])
    ll = 0.0
    xi_filt = np.empty((T, K))
    for t in range(T):
        if t > 0:
            P_t = P_path[t] if X_lagged is not None else params["P"]
            xi = P_t.T @ xi_filt[t - 1]
        num = xi * f[t]
        denom = num.sum()
        xi_filt[t] = num / denom
        ll += np.log(denom)
    return xi_filt, ll


TRUE = {
    "mu": np.array([0.05, -0.10]), "v": np.array([0.64, 9.00]),
    "alpha": np.array([0.05, 0.10]), "gamma": np.array([0.10, 0.10]),
    "beta": np.array([0.85, 0.70]), "P": np.array([[0.97, 0.03], [0.08, 0.92]]),
}
theta = pack(TRUE, K=2)
r_sim, _ = simulate(theta, K=2, T=800, seed=5)
p_nat = unpack(theta, K=2)
xi_p, _, ll_p = hamilton_filter(r_sim, p_nat, K=2)
xi_n, ll_n = hamilton_naive(r_sim, p_nat, K=2)
check("B3_hamilton_const_loglik", abs(ll_p - ll_n), 1e-10, {"ll_prod": ll_p, "ll_naive": ll_n})
check("B3_hamilton_const_xi", float(np.max(np.abs(xi_p - xi_n))), 1e-10)

# t de Student
TRUE_T = dict(TRUE, nu=np.array([6.0, 4.5]))
theta_t = pack(TRUE_T, K=2, dist="t")
r_t, _ = simulate(theta_t, K=2, T=800, seed=6, dist="t")
p_t = unpack(theta_t, K=2, dist="t")
xi_pt, _, ll_pt = hamilton_filter(r_t, p_t, K=2)
xi_nt, ll_nt = hamilton_naive(r_t, p_t, K=2)
check("B3_hamilton_student_loglik", abs(ll_pt - ll_nt), 1e-10, {"ll_prod": ll_pt, "ll_naive": ll_nt})
check("B3_hamilton_student_xi", float(np.max(np.abs(xi_pt - xi_nt))), 1e-10)

# TVTP con 2 covariables
rng2 = np.random.default_rng(9)
X = rng2.normal(size=(800, 2))
TRUE_TV = dict(TRUE, beta_tvtp=np.array([[[0.4, -0.2]], [[0.1, 0.3]]]))
theta_tv = pack(TRUE_TV, K=2, n_cov=2)
p_tv = unpack(theta_tv, K=2, n_cov=2)
xi_ptv, _, ll_ptv = hamilton_filter(r_sim, p_tv, K=2, X_lagged=X)
xi_ntv, ll_ntv = hamilton_naive(r_sim, p_tv, K=2, X_lagged=X)
check("B3_hamilton_tvtp_loglik", abs(ll_ptv - ll_ntv), 1e-10, {"ll_prod": ll_ptv, "ll_naive": ll_ntv})
check("B3_hamilton_tvtp_xi", float(np.max(np.abs(xi_ptv - xi_ntv))), 1e-10)

# ---------------------------------------------------------------------------
# B4: statsmodels regime_switching, caso comparable (varianza ~constante)
# ---------------------------------------------------------------------------
try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    # persistencia GARCH ~0 => sigma2_t ~ v constante por regimen
    eps_p = 1e-8
    CONST = {
        "mu": np.array([0.05, -0.10]), "v": np.array([0.64, 9.00]),
        "alpha": np.array([eps_p / 3, eps_p / 3]),
        "gamma": np.array([2 * eps_p / 3, 2 * eps_p / 3]),
        "beta": np.array([eps_p / 3, eps_p / 3]),
        "P": np.array([[0.97, 0.03], [0.08, 0.92]]),
    }
    theta_c = pack(CONST, K=2)
    r_c, _ = simulate(theta_c, K=2, T=1500, seed=8)
    p_c = unpack(theta_c, K=2)
    _, _, ll_c = hamilton_filter(r_c, p_c, K=2)

    mod = MarkovRegression(r_c, k_regimes=2, trend="c", switching_variance=True)
    # statsmodels: params = [p00, p10, const_0, const_1, sigma2_0, sigma2_1] (transformed)
    sm_params = np.r_[0.97, 0.08, 0.05, -0.10, 0.64, 9.00]
    ll_sm = float(mod.loglike(sm_params, transformed=True))
    # el transitorio sigma2_0 = v difiere del estacionario de statsmodels? ambos
    # usan la estacionaria de P para xi_0 y varianza constante: comparable directo.
    check("B4_msgarch_vs_statsmodels", abs(ll_c - ll_sm) / max(1.0, abs(ll_sm)), 1e-6,
          {"ll_prod": ll_c, "ll_statsmodels": ll_sm,
           "note": "caso comparable: persistencia GARCH 1e-8 (varianza constante por regimen)"})
except ImportError:
    RES["B4_msgarch_vs_statsmodels"] = {"ok": None, "note": "statsmodels NO instalado en el venv"}
    print("[B4] statsmodels no instalado -> NO VERIFICADO por esta via", flush=True)

# ---------------------------------------------------------------------------
# B5: BIC con conteo manual enumerado
# ---------------------------------------------------------------------------
r_small, _ = simulate(theta, K=2, T=600, seed=13)
enums = {
    # K=2 normal: mu1 mu2 | v1 v2 | kappa1 kappa2 | reparto (2 logits x 2 reg) | p11 p22
    ("K2", "normal"): ["mu1", "mu2", "v1", "v2", "kappa1", "kappa2",
                       "c11", "c12", "c21", "c22", "p11", "p22"],
    # K=1 t: mu | v | kappa | c1 c2 | nu   (sin matriz de transicion)
    ("K1", "t"): ["mu", "v", "kappa", "c1", "c2", "nu"],
    # K=2 t: los 12 de K2-normal + nu1 nu2
    ("K2", "t"): ["mu1", "mu2", "v1", "v2", "kappa1", "kappa2",
                  "c11", "c12", "c21", "c22", "p11", "p22", "nu1", "nu2"],
}
b5 = {}
for (klab, dist), names in enums.items():
    K = int(klab[1])
    k_manual = len(names)
    k_code = n_params(K, n_cov=0, dist=dist)
    fr = fit(r_small, K=K, n_starts=3, seed=1, compute_se=False, dist=dist)
    bic_manual = -2.0 * fr.loglik + k_manual * np.log(len(r_small))
    b5[f"{klab}_{dist}"] = {
        "k_manual": k_manual, "k_code": k_code, "match_k": bool(k_manual == k_code),
        "bic_code": fr.bic, "bic_manual": float(bic_manual),
        "match_bic": bool(abs(fr.bic - bic_manual) < 1e-9),
        "params_enumerados": names,
        "T_usada": len(r_small),
    }
    print(f"[B5 {klab} {dist}] k_manual={k_manual} k_code={k_code} "
          f"BIC match={b5[f'{klab}_{dist}']['match_bic']}", flush=True)
RES["B5_bic_manual"] = b5
RES["B5_nota_n"] = ("n del BIC = T total de la muestra (600 aqui); el filtro no descarta "
                    "warm-up (sigma2_0 = v es funcion de parametros), asi que T total es "
                    "el numero de contribuciones efectivas a la loglik: consistente.")

RES["B6_fracdiff"] = {"ok": None, "note": "NO APLICA: no existe fracdiff/ARFIMA en src/ (grep sin resultados)"}

(OUT / "b_results.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
n_fail = sum(1 for v in RES.values() if isinstance(v, dict) and v.get("ok") is False)
print(f"FASE B completa. Fallos: {n_fail}. -> audits/probes/out/b_results.json", flush=True)

"""FASE C -- Invariantes de probabilidad, con ASERCIONES ejecutables, sobre el
artefacto vigente (artifacts/latest, run 3b4f1e39b59c) y sobre corridas sinteticas.

C.1 Estructura de la cadena de Markov (parametrizacion + artefacto + TVTP)
C.2 Probabilidades filtradas y predictivas (artefacto real: filas suman 1,
    orientacion de la transpuesta verificada SOBRE LOS DATOS PUBLICADOS usando
    la P por bloque de walkforward.json)
C.3 Verosimilitud / estabilidad numerica / restricciones GARCH (por construccion
    + margenes del optimo publicado)
C.4 Calibracion de las probabilidades publicadas: descomposicion de Brier
    (Murphy) contra (a) el proxy del proyecto (argmax xi_{t|t}) y (b) un proxy
    OBSERVABLE externo (dia de |r| extremo). PIT de densidad: no reconstruible
    desde el artefacto (los bloques no persisten theta) -> se reporta en la
    seccion NO VERIFICADO del informe.

Sensibilidad a xi_0 (C.1): el filtro de produccion no expone xi_0; se usa la
reimplementacion ingenua de la Fase B (concordancia 1e-15 demostrada en
b_results.json) para perturbar xi_0 y medir Delta-LL.

Solo LECTURA de src/ y artifacts/. Resultados en audits/probes/out/c_results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from irfn.models.params import unpack, pack, n_params  # noqa: E402
from irfn.models.tvtp import transition_matrices  # noqa: E402
from irfn.models.hamilton import stationary_distribution, hamilton_filter  # noqa: E402
from irfn.models.msgarch import simulate, conditional_variances, obs_logpdf  # noqa: E402
from irfn.features.hawkes_features import hawkes_feature  # noqa: E402
from irfn.features.technical import rolling_zscore  # noqa: E402
from irfn.outputs.publish import publish, LookAheadViolation  # noqa: E402

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
RES: dict = {}
FAILS: list[str] = []


def record(name, ok, detail=None):
    RES[name] = {"ok": bool(ok), **({"detail": detail} if detail is not None else {})}
    if not ok:
        FAILS.append(name)
    print(f"[{name}] {'OK' if ok else 'FALLA'}" + (f" {detail}" if detail and not ok else ""), flush=True)


ART = ROOT / "artifacts" / "latest"
irfn = json.loads((ART / "irfn.json").read_text(encoding="utf-8"))
hist = pd.read_parquet(ART / "history.parquet")
wf = json.loads((ART / "walkforward.json").read_text(encoding="utf-8"))

# =============================================================================
# C.1 -- cadena de Markov
# =============================================================================
rng = np.random.default_rng(42)
worst_row_sum = 0.0
worst_bounds = 0.0
for _ in range(2000):
    K = int(rng.integers(1, 4))
    dist = "t" if rng.random() < 0.5 else "normal"
    n_cov = int(rng.integers(0, 3))
    theta = rng.normal(scale=2.0, size=n_params(K, n_cov=n_cov, dist=dist))
    p = unpack(theta, K, n_cov=n_cov, dist=dist)
    P = p["P"]
    worst_row_sum = max(worst_row_sum, float(np.max(np.abs(P.sum(axis=1) - 1.0))))
    worst_bounds = max(worst_bounds, float(max(np.max(P) - 1.0, -np.min(P), 0.0)))
    if n_cov > 0:
        X = rng.normal(size=(50, n_cov))
        Pp = transition_matrices(p["d"], p["beta_tvtp"], X)
        worst_row_sum = max(worst_row_sum, float(np.max(np.abs(Pp.sum(axis=2) - 1.0))))
        worst_bounds = max(worst_bounds, float(max(np.max(Pp) - 1.0, -np.min(Pp), 0.0)))
record("C1_filas_P_suman_1_(2000_thetas_aleatorios,+TVTP)", worst_row_sum < 1e-12,
       {"peor_desvio": worst_row_sum})
record("C1_P_en_[0,1]", worst_bounds <= 0.0, {"peor_exceso": worst_bounds})

P_today = np.array(irfn["transition_matrix_today"])
record("C1_P_today_filas_suman_1", float(np.max(np.abs(P_today.sum(axis=1) - 1.0))) < 1e-12,
       {"P": P_today.tolist()})
record("C1_P_today_sin_0_ni_1_exactos", bool(np.all((P_today > 0) & (P_today < 1))))

# xi_0: produccion usa la estacionaria de P (hamilton.py:103). Sensibilidad:
TRUE = {"mu": np.array([0.05, -0.10]), "v": np.array([0.64, 9.00]),
        "alpha": np.array([0.05, 0.10]), "gamma": np.array([0.10, 0.10]),
        "beta": np.array([0.85, 0.70]), "P": np.array([[0.97, 0.03], [0.08, 0.92]])}
theta0 = pack(TRUE, K=2)
r_sim, _ = simulate(theta0, K=2, T=2000, seed=3)
p_nat = unpack(theta0, K=2)


def naive_ll(r, params, K, xi0):
    sig2 = conditional_variances(r, params, K)
    logf = obs_logpdf(r, params, sig2)
    xi = np.asarray(xi0, dtype=float)
    ll = 0.0
    for t in range(len(r)):
        if t > 0:
            xi = params["P"].T @ xi_f
        num = xi * np.exp(logf[t] - logf[t].max())
        den = num.sum()
        xi_f = num / den
        ll += np.log(den) + logf[t].max()
    return ll


pi_st = stationary_distribution(p_nat["P"])
ll_st = naive_ll(r_sim, p_nat, 2, pi_st)
ll_unif = naive_ll(r_sim, p_nat, 2, np.array([0.5, 0.5]))
ll_e1 = naive_ll(r_sim, p_nat, 2, np.array([1.0 - 1e-9, 1e-9]))
_, _, ll_prod = hamilton_filter(r_sim, p_nat, K=2)
record("C1_xi0_es_estacionaria_y_coincide_con_prod", abs(ll_st - ll_prod) < 1e-8,
       {"ll_prod": ll_prod, "ll_estacionaria": ll_st})
RES["C1_sensibilidad_xi0"] = {
    "ll_estacionaria": ll_st, "delta_uniforme": ll_unif - ll_st, "delta_e1": ll_e1 - ll_st,
    "nota": "Delta-LL en unidades absolutas sobre T=2000; efecto acotado al transitorio inicial."}
print(f"[C1 xi0] dLL(unif)={ll_unif-ll_st:+.4e} dLL(e1)={ll_e1-ll_st:+.4e}", flush=True)

# R3: el rezago del feature es shift(1) REAL en el indice temporal
s = pd.Series(np.random.default_rng(1).normal(size=300),
              index=pd.date_range("2024-01-01", periods=300, freq="D"))
z = rolling_zscore(s, 60)
feat = hawkes_feature(s, z_window=60)
ok_lag = bool(np.allclose(feat.iloc[61:].to_numpy(), z.iloc[60:-1].to_numpy(), equal_nan=True))
record("C1_R3_shift1_real_en_indice_(hawkes_feature)", ok_lag)

# =============================================================================
# C.2 -- filtradas y predictivas del artefacto publicado
# =============================================================================
xi_f = hist[["xi_filtered_0", "xi_filtered_1"]].to_numpy()
xi_p = hist[["xi_predicted_0", "xi_predicted_1"]].to_numpy()
record("C2_xi_filtered_sin_NaN", bool(np.all(np.isfinite(xi_f))))
record("C2_xi_predicted_sin_NaN", bool(np.all(np.isfinite(xi_p))))
record("C2_xi_filtered_en_[0,1]_suma_1",
       bool(np.all((xi_f >= 0) & (xi_f <= 1))) and float(np.max(np.abs(xi_f.sum(1) - 1))) < 1e-9,
       {"peor_suma": float(np.max(np.abs(xi_f.sum(1) - 1)))})
record("C2_xi_predicted_en_[0,1]_suma_1",
       bool(np.all((xi_p >= 0) & (xi_p <= 1))) and float(np.max(np.abs(xi_p.sum(1) - 1))) < 1e-9,
       {"peor_suma": float(np.max(np.abs(xi_p.sum(1) - 1)))})

# Orientacion de la transpuesta SOBRE LOS DATOS PUBLICADOS: dentro de cada
# bloque del walk-forward, xi_pred[t] debe ser P_b' xi_filt[t-1] con la P del
# bloque persistida en walkforward.json. (M1: matriz constante por bloque.)
errs_T, errs_noT = [], []
block_ids = hist["block_id"].to_numpy()
for b in wf["blocks"]:
    P_b = np.array(b["P"])
    m = block_ids == b["block_id"]
    idx = np.where(m)[0]
    for t in idx[1:]:
        pred_T = P_b.T @ xi_f[t - 1]
        pred_noT = P_b @ xi_f[t - 1]
        errs_T.append(np.max(np.abs(pred_T - xi_p[t])))
        errs_noT.append(np.max(np.abs(pred_noT - xi_p[t])))
max_T, max_noT = float(np.max(errs_T)), float(np.max(errs_noT))
record("C2_orientacion_transpuesta_en_artefacto", max_T < 1e-9,
       {"max_err_con_P_transpuesta": max_T,
        "max_err_con_P_SIN_transponer_(control)": max_noT,
        "nota": "si la version sin transponer diera error ~0, la orientacion seria ambigua; debe ser grande"})

# R1: el guardian de publish rechaza claves prohibidas ANIDADAS
try:
    publish({"regime": {"nested": {"xi_smoothed": [0.1]}}}, OUT / "_r1_should_not_exist.json")
    r1_ok = False
except LookAheadViolation:
    r1_ok = True
record("C2_R1_publish_rechaza_clave_anidada", r1_ok and not (OUT / "_r1_should_not_exist.json").exists())

# =============================================================================
# C.3 -- restricciones GARCH: por construccion + margen del optimo publicado
# =============================================================================
viol = 0
closest = {"p_max": 0.0, "min_omega": np.inf}
for _ in range(2000):
    K = int(rng.integers(1, 4))
    theta = rng.normal(scale=2.5, size=n_params(K))
    p = unpack(theta, K)
    kappa = p["alpha"] + p["gamma"] / 2.0 + p["beta"]
    if (np.any(p["omega"] <= 0) or np.any(p["alpha"] < 0) or np.any(p["gamma"] < 0)
            or np.any(p["beta"] < 0) or np.any(kappa >= 1.0)):
        viol += 1
    closest["p_max"] = max(closest["p_max"], float(kappa.max()))
    closest["min_omega"] = min(closest["min_omega"], float(p["omega"].min()))
record("C3_restricciones_GARCH_por_construccion_(2000_thetas)", viol == 0, closest)

# margen del optimo publicado: kappa por bloque del walk-forward
kappas = np.array([b["kappa"] for b in wf["blocks"]])
RES["C3_kappa_bloques_publicados"] = {
    "max_kappa": float(kappas.max()), "min_kappa": float(kappas.min()),
    "dist_a_1_minima": float(1.0 - kappas.max()),
    "nota": "kappa < 1 estricto en todos los bloques; cercania a 1 = persistencia alta tipica GARCH"}
print(f"[C3] kappa bloques: max={kappas.max():.6f} (dist a 1: {1-kappas.max():.2e})", flush=True)

# =============================================================================
# C.4 -- calibracion de las probabilidades OOS publicadas
# =============================================================================
def brier_decomposition(p1, y, n_bins=10):
    """Murphy (1973): Brier = reliability - resolution + uncertainty (binaria)."""
    p1 = np.asarray(p1, float); y = np.asarray(y, float)
    bins = np.clip((p1 * n_bins).astype(int), 0, n_bins - 1)
    ybar = y.mean()
    rel = res = 0.0
    curve = []
    for b in range(n_bins):
        m = bins == b
        nk = int(m.sum())
        if nk == 0:
            curve.append({"bin": b, "n": 0}); continue
        pk, ok = float(p1[m].mean()), float(y[m].mean())
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - ybar) ** 2
        curve.append({"bin": b, "n": nk, "mean_p": pk, "freq_obs": ok})
    T = len(y)
    rel /= T; res /= T
    unc = float(ybar * (1 - ybar))
    return {"brier": float(np.mean((p1 - y) ** 2)), "reliability": rel, "resolution": res,
            "uncertainty": unc, "check_identity": rel - res + unc, "base_rate": float(ybar),
            "curve": curve}


p1 = xi_p[:, 1]
# (a) proxy del proyecto: argmax xi_{t|t}
y_proj = (hist["argmax_idx"].to_numpy() == 1).astype(float)
RES["C4_brier_vs_proxy_proyecto"] = brier_decomposition(p1, y_proj)
# (b) proxy OBSERVABLE externo: |r_t| en el quintil superior de |r| (diagnostico;
# umbral de muestra completa, solo para la auditoria, no publicable)
r_abs = np.abs(hist["r"].to_numpy())
thr = np.quantile(r_abs, 0.80)
y_ext = (r_abs > thr).astype(float)
RES["C4_brier_vs_proxy_observable_quintil|r|"] = brier_decomposition(p1, y_ext)
for k in ("C4_brier_vs_proxy_proyecto", "C4_brier_vs_proxy_observable_quintil|r|"):
    d = RES[k]
    print(f"[{k}] brier={d['brier']:.4f} rel={d['reliability']:.4f} "
          f"res={d['resolution']:.4f} unc={d['uncertainty']:.4f} base={d['base_rate']:.3f}", flush=True)
RES["C4_pit_densidad"] = {
    "ok": None,
    "nota": ("NO VERIFICABLE desde el artefacto: walkforward.json no persiste theta por "
             "bloque, la densidad predictiva OOS no es reconstruible sin re-correr el "
             "walk-forward (computo pesado, fuera del presupuesto de esta pasada). "
             "El PIT del bloque reconstruido en F.4 cubre parcialmente este hueco.")}

(OUT / "c_results.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
print(f"FASE C completa. Fallos: {len(FAILS)} {FAILS}", flush=True)

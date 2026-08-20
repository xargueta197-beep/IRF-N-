"""FASE A.2 -- Recuperacion de parametros del MS-GJR-GARCH sobre verdad conocida.

Protocolo (AUDIT_MATH_v1):
  1. Simular MS-GJR-GARCH de 2 regimenes con P conocida y separacion clara
     (sigma1~0.8%, sigma2~3% diarios; p11=0.97, p22=0.92), con el simulador de
     produccion (msgarch.simulate) y la verdad empaquetada con params.pack.
  2. Estimar con el codigo de produccion bajo R6 (fit, multistart 20, semilla fija).
  3. Reportar recuperacion de P, parametros GARCH por regimen y precision de
     clasificacion de xi_{t|t} contra el regimen verdadero (AUC + tasa de acierto).
  4. PRUEBA DECISIVA de rango de P: verificar si en sintetico bien condicionado el
     estimador recupera rango 2 (filas distintas). Metricas: sigma_2/sigma_1 de la
     SVD de P y |p11 - p21|.
  5. Caso adversarial: P casi absorbente (p11=0.999) -> comportamiento del regimen
     degenerado.
  + 8 replicas cortas (n_starts=8) para sesgo direccional (la replica completa de
    50 no es exigida por el protocolo para A.2; se documenta el numero usado).

Solo LECTURA de src/. Resultados en audits/probes/out/a2_results.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from irfn.models.params import pack  # noqa: E402
from irfn.models.msgarch import simulate  # noqa: E402
from irfn.models.estimate import fit  # noqa: E402
from irfn.models.hamilton import hamilton_filter  # noqa: E402

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# Verdad: retornos en % diarios. v = varianza incondicional por regimen.
TRUE = {
    "mu": np.array([0.05, -0.10]),
    "v": np.array([0.64, 9.00]),            # sigma 0.8% y 3.0%
    "alpha": np.array([0.05, 0.10]),
    "gamma": np.array([0.10, 0.10]),        # kappa1=0.95, kappa2=0.85
    "beta": np.array([0.85, 0.70]),
    "P": np.array([[0.97, 0.03], [0.08, 0.92]]),
}
THETA_TRUE = pack(TRUE, K=2)


def auc_rank(score: np.ndarray, y: np.ndarray) -> float:
    """AUC por el estadistico de rangos de Mann-Whitney (sin sklearn)."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def p_rank_metrics(P: np.ndarray) -> dict:
    sv = np.linalg.svd(P, compute_uv=False)
    return {"sv_ratio": float(sv[1] / sv[0]), "row_gap_p11_p21": float(abs(P[0, 0] - P[1, 0]))}


def classify(r, params, states):
    xi_filt, _, _ = hamilton_filter(r, params, K=2)
    hit = float(np.mean(xi_filt.argmax(axis=1) == states))
    auc = auc_rank(xi_filt[:, 1], (states == 1).astype(int))
    return hit, auc


def main():
    results = {}

    # --- 1-4: caso principal, T=5000, R6 multistart 20 ------------------------
    t0 = time.time()
    r, states = simulate(THETA_TRUE, K=2, T=5000, seed=7)
    fr = fit(r, K=2, n_starts=20, seed=42, compute_se=True)
    within3 = {}
    for name in ("mu", "v", "alpha", "gamma", "beta"):
        est, se, tv = fr.params[name], fr.se.get(name), TRUE[name]
        within3[name] = [bool(np.isfinite(s) and abs(e - t) <= 3 * s) for e, s, t in zip(est, se, tv)]
    P_ok = [[bool(np.isfinite(fr.se_P[i, j]) and abs(fr.P[i, j] - TRUE["P"][i, j]) <= 3 * fr.se_P[i, j])
             for j in range(2)] for i in range(2)]
    hit, auc = classify(r, fr.params, states)
    results["main"] = {
        "T": 5000, "loglik": fr.loglik, "n_converged": fr.n_converged,
        "hessian_ok": fr.hessian_ok,
        "est": {k: fr.params[k].tolist() for k in ("mu", "v", "alpha", "gamma", "beta", "omega")},
        "se": {k: fr.se[k].tolist() for k in ("mu", "v", "alpha", "gamma", "beta")},
        "P_est": fr.P.tolist(), "P_se": fr.se_P.tolist(), "P_true": TRUE["P"].tolist(),
        "within_3se": within3, "P_within_3se": P_ok,
        "rank_true": p_rank_metrics(TRUE["P"]), "rank_est": p_rank_metrics(fr.P),
        "classification": {"hit_rate": hit, "auc": auc},
        "wall_s": round(time.time() - t0, 1),
    }
    print(f"[main] loglik={fr.loglik:.2f} conv={fr.n_converged}/20 "
          f"P_est={np.round(fr.P,4).tolist()} rank_est={results['main']['rank_est']} "
          f"hit={hit:.3f} auc={auc:.3f} within3se={within3} P_ok={P_ok} "
          f"t={results['main']['wall_s']}s", flush=True)

    # --- 5: caso adversarial casi absorbente ---------------------------------
    t0 = time.time()
    TRUE_ABS = dict(TRUE)
    TRUE_ABS["P"] = np.array([[0.999, 0.001], [0.08, 0.92]])
    theta_abs = pack(TRUE_ABS, K=2)
    r2, states2 = simulate(theta_abs, K=2, T=5000, seed=11)
    fr2 = fit(r2, K=2, n_starts=12, seed=42, compute_se=False)
    occup = float(np.mean(states2 == 1))
    hit2, auc2 = classify(r2, fr2.params, states2)
    results["absorbing"] = {
        "P_true": TRUE_ABS["P"].tolist(), "occupancy_regime2_true": occup,
        "P_est": fr2.P.tolist(), "est_v": fr2.params["v"].tolist(),
        "n_converged": fr2.n_converged,
        "classification": {"hit_rate": hit2, "auc": auc2},
        "expected_duration_est": [float(1.0 / max(1e-12, 1.0 - fr2.P[k, k])) for k in range(2)],
        "wall_s": round(time.time() - t0, 1),
    }
    print(f"[absorbing] occup2={occup:.4f} P_est={np.round(fr2.P,4).tolist()} "
          f"hit={hit2:.3f} auc={auc2:.3f} t={results['absorbing']['wall_s']}s", flush=True)

    # --- replicas cortas: direccion del sesgo --------------------------------
    t0 = time.time()
    rows = []
    for rep in range(8):
        rr, ss = simulate(THETA_TRUE, K=2, T=5000, seed=100 + rep)
        f = fit(rr, K=2, n_starts=8, seed=rep, compute_se=False)
        h, a = classify(rr, f.params, ss)
        rows.append({"P": f.P.tolist(), "v": f.params["v"].tolist(), "hit": h, "auc": a,
                     "p11": float(f.P[0, 0]), "p22": float(f.P[1, 1]),
                     "sv_ratio": p_rank_metrics(f.P)["sv_ratio"]})
    p11s = np.array([x["p11"] for x in rows]); p22s = np.array([x["p22"] for x in rows])
    v1s = np.array([x["v"][0] for x in rows]); v2s = np.array([x["v"][1] for x in rows])
    results["replicas"] = {
        "n_rep": 8, "n_starts": 8,
        "p11": {"true": 0.97, "mean": float(p11s.mean()), "sd": float(p11s.std(ddof=1))},
        "p22": {"true": 0.92, "mean": float(p22s.mean()), "sd": float(p22s.std(ddof=1))},
        "v1": {"true": 0.64, "mean": float(v1s.mean()), "sd": float(v1s.std(ddof=1))},
        "v2": {"true": 9.00, "mean": float(v2s.mean()), "sd": float(v2s.std(ddof=1))},
        "hit_mean": float(np.mean([x["hit"] for x in rows])),
        "auc_mean": float(np.mean([x["auc"] for x in rows])),
        "sv_ratio_mean": float(np.mean([x["sv_ratio"] for x in rows])),
        "rows": rows, "wall_s": round(time.time() - t0, 1),
    }
    print(f"[replicas] p11 {p11s.mean():.4f}+-{p11s.std(ddof=1):.4f} (0.97) "
          f"p22 {p22s.mean():.4f}+-{p22s.std(ddof=1):.4f} (0.92) "
          f"v1 {v1s.mean():.3f} (0.64) v2 {v2s.mean():.3f} (9.0) "
          f"auc={results['replicas']['auc_mean']:.3f} t={results['replicas']['wall_s']}s", flush=True)

    (OUT / "a2_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print("OK -> audits/probes/out/a2_results.json", flush=True)


if __name__ == "__main__":
    main()

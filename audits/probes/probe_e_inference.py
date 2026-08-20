"""FASE E -- Inferencia y comparacion de modelos.

E.1  BIC: re-verificacion aritmetica de las 8 filas de v1_kselect.json
     (bic = -2*loglik + k_free*ln(n_obs)) y k_free vs conteo manual (Fase B5).
E.2  "Hansen": la implementacion REAL es un bootstrap LR parametrico
     (tests_stat.bootstrap_lr_test); cuantificacion del sesgo por presupuesto
     de multistart ASIMETRICO (LR_obs con 20 arranques vs replicas con 6):
     se simulan replicas del nulo K=1 ajustado a la muestra del kselect y se
     compara el LR de cada replica con 6 vs 20 arranques en el modelo K=2.
     Si LR(20) > LR(6) sistematicamente, el p publicado esta sesgado a la baja
     (anti-conservador). Ademas: con B=49, el p minimo alcanzable es 1/50=0.02,
     que es exactamente el p reportado (granularidad, no evidencia fuerte).
E.3  Diebold-Mariano M2 vs M1 (p=0.106): recomputo INDEPENDIENTE desde los dos
     history.parquet persistidos (runs 3b4f1e39b59c=M1 y 7c44a7fac16d=M2),
     con NW+HLN propios y sensibilidad al numero de rezagos.
E.4  Bootstrap: cobertura empirica del IC 95% de sharpe_ci (bootstrap
     estacionario + Politis-White) sobre AR(1) con Sharpe conocido.
     + verificacion de que la supresion A2 (maxdd sin IC; regimen degenerado
     sin IC) esta ACTIVA en el artefacto publicado.

Solo LECTURA. Resultados en audits/probes/out/e_results.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from irfn.models.params import n_params, pack  # noqa: E402
from irfn.models.estimate import fit  # noqa: E402
from irfn.models.msgarch import simulate  # noqa: E402
from irfn.validation.bootstrap import sharpe_ci  # noqa: E402

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
RES: dict = {}

# =============================================================================
# E.1 -- BIC del kselect
# =============================================================================
ks = json.loads((ROOT / "artifacts/analysis/v1_kselect.json").read_text(encoding="utf-8"))
n_obs = ks["n_obs"]
rows = []
for row in ks["bic_table"]:
    k_manual = n_params(row["K"], dist=row["dist"])
    bic_re = -2.0 * row["loglik"] + row["k_free"] * np.log(n_obs)
    rows.append({
        "K": row["K"], "dist": row["dist"], "k_free": row["k_free"], "k_manual": k_manual,
        "k_ok": bool(row["k_free"] == k_manual),
        "bic_archivo": row["bic"], "bic_recalc": float(bic_re),
        "bic_ok": bool(abs(row["bic"] - bic_re) < 1e-6),
        "n_converged": row["n_converged"],
    })
RES["E1_bic_kselect"] = {"n_obs": n_obs, "rows": rows,
                         "todo_ok": all(r["k_ok"] and r["bic_ok"] for r in rows),
                         "ganador_archivo": ks["winner"]}
print(f"[E1] BIC kselect: {'OK' if RES['E1_bic_kselect']['todo_ok'] else 'FALLA'} "
      f"ganador={ks['winner']}", flush=True)

# =============================================================================
# E.3 -- DM M2 vs M1 desde artefactos persistidos (independiente)
# =============================================================================
h1 = pd.read_parquet(ROOT / "artifacts/runs/3b4f1e39b59c/history.parquet")  # M1
h2 = pd.read_parquet(ROOT / "artifacts/runs/7c44a7fac16d/history.parquet")  # M2
m = pd.merge(h1[["fecha", "loglik_obs"]], h2[["fecha", "loglik_obs"]],
             on="fecha", suffixes=("_m1", "_m2"))
d = (-m["loglik_obs_m2"].to_numpy()) - (-m["loglik_obs_m1"].to_numpy())  # loss M2 - loss M1
T = len(d)


def dm_own(d, n_lags, h=1):
    d_bar = d.mean()
    dc = d - d_bar
    v = float(dc @ dc) / T
    for j in range(1, n_lags + 1):
        v += 2.0 * (1.0 - j / (n_lags + 1.0)) * float(dc[j:] @ dc[:-j]) / T
    dm = d_bar / np.sqrt(v / T)
    hln = np.sqrt((T + 1.0 - 2.0 * h + h * (h - 1.0) / T) / T)
    return float(dm * hln), 2.0 * float(stats.t.sf(abs(dm * hln), df=T - 1))


lag_rule = int(np.floor(1.5 * T ** (1 / 3)))
sens = {}
for L in (0, 5, lag_rule, 40):
    s, p = dm_own(d, L)
    sens[f"lags_{L}"] = {"dm": s, "p": p}
RES["E3_dm_m2_vs_m1"] = {
    "n_obs_emparejadas": T, "mean_diff_loss": float(d.mean()),
    "reportado": {"dm": 1.617, "p": 0.106},
    "recomputo": sens, "lag_regla_proyecto": lag_rule,
    "nota": ("perdida = -loglik predictiva por observacion, h=1 (sin ventanas "
             "solapadas: la premisa 'h>1' del encargo no aplica a este pipeline); "
             "d>0 => M2 pierde mas que M1"),
}
print(f"[E3] DM propio: {json.dumps(sens)} (reportado 1.617/0.106, T={T})", flush=True)

# =============================================================================
# E.2 -- asimetria de multistart en el bootstrap LR
# =============================================================================
t0 = time.time()
px = pd.read_parquet(ROOT / "data/raw/close_SPY_2010-01-01.parquet")
px = px.sort_index()
col = px.columns[0]
r_all = 100.0 * np.log(px[col]).diff().dropna()
r_all.index = pd.to_datetime(r_all.index)
r_ks = r_all.loc["2013-01-02":"2026-07-10"].to_numpy()
RES["E2_muestra"] = {"n": int(len(r_ks)), "n_esperado_kselect": n_obs,
                     "coincide": bool(len(r_ks) == n_obs)}
print(f"[E2] muestra reconstruida n={len(r_ks)} (kselect {n_obs})", flush=True)

f_null = fit(r_ks, K=1, n_starts=20, seed=42, compute_se=False)
theta_null = pack(f_null.params, K=1)
reps = []
for b in range(4):
    r_b, _ = simulate(theta_null, K=1, T=len(r_ks), seed=42 + 1000 + b)
    f0_6 = fit(r_b, K=1, n_starts=6, seed=42 + b, compute_se=False)
    f1_6 = fit(r_b, K=2, n_starts=6, seed=42 + b, compute_se=False)
    f1_20 = fit(r_b, K=2, n_starts=20, seed=42 + b, compute_se=False)
    lr6 = max(0.0, 2.0 * (f1_6.loglik - f0_6.loglik))
    lr20 = max(0.0, 2.0 * (f1_20.loglik - f0_6.loglik))
    reps.append({"replica": b, "lr_6starts": lr6, "lr_20starts": lr20,
                 "delta": lr20 - lr6})
    print(f"[E2] replica {b}: LR(6)={lr6:.3f} LR(20)={lr20:.3f} delta={lr20-lr6:+.3f}", flush=True)
RES["E2_asimetria_multistart"] = {
    "replicas": reps,
    "delta_medio": float(np.mean([x["delta"] for x in reps])),
    "p_minimo_alcanzable_B49": 1.0 / 50.0,
    "p_reportado": 0.02,
    "nota": ("delta>0 => las replicas con 6 arranques subestiman el LR nulo => "
             "la cola nula queda corta => p sesgado a la BAJA (anti-conservador). "
             "Ademas p=0.02 = p minimo alcanzable con B=49: LR_obs supero a las 49 "
             "replicas; la evidencia es 'p<=0.02', no un p fino."),
    "wall_s": round(time.time() - t0, 1),
}

# =============================================================================
# E.4 -- cobertura del IC de sharpe_ci + supresion A2 en el artefacto
# =============================================================================
t0 = time.time()
rng = np.random.default_rng(7)
phi, mu_r, sd_r = 0.2, 0.03, 1.0
sd_eps = sd_r * np.sqrt(1 - phi**2)
true_sharpe = mu_r / sd_r * np.sqrt(252)
cover = 0
n_rep = 100
for rep in range(n_rep):
    e = rng.normal(0, sd_eps, size=1500)
    x = np.empty(1500)
    x[0] = mu_r + e[0]
    for t in range(1, 1500):
        x[t] = mu_r + phi * (x[t - 1] - mu_r) + e[t]
    ci = sharpe_ci(x, n_boot=299, seed=rep)
    if ci["ci_lower"] <= true_sharpe <= ci["ci_upper"]:
        cover += 1
RES["E4_cobertura_sharpe_ci"] = {
    "modelo": "AR(1) phi=0.2, Sharpe anual verdadero", "sharpe_true": float(true_sharpe),
    "coverage_95": cover / n_rep, "n_rep": n_rep, "n_boot": 299,
    "wall_s": round(time.time() - t0, 1),
}
print(f"[E4] cobertura IC95 sharpe_ci = {cover}/{n_rep} ({time.time()-t0:.0f}s)", flush=True)

irfn = json.loads((ROOT / "artifacts/latest/irfn.json").read_text(encoding="utf-8"))
cs = irfn["conditional_stats"]
asset = list(cs.keys())[0]
labels = irfn["regime"]["labels"]
edd = irfn["regime"]["expected_duration_days"]
checks = {"maxdd_sin_ic_en_todos": True, "degenerado_sin_ic_en_nada": True}
for i, lab in enumerate(labels):
    entry = cs[asset][lab]
    if entry["maxdd"]["ci_low"] is not None or entry["maxdd"]["ci_high"] is not None:
        checks["maxdd_sin_ic_en_todos"] = False
    if edd[i] < 2.0:
        for met in entry.values():
            if met["ci_low"] is not None or met["ci_high"] is not None:
                checks["degenerado_sin_ic_en_nada"] = False
RES["E4_supresion_A2_artefacto"] = {**checks, "expected_durations": edd,
                                    "ok": all(checks.values())}
print(f"[E4] supresion A2 en artefacto: {checks}", flush=True)

(OUT / "e_results.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
print("FASE E completa -> audits/probes/out/e_results.json", flush=True)

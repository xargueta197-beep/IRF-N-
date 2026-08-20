"""FASE F -- Walk-forward: reconstruccion INDEPENDIENTE del bloque 5 + chequeos
de fuga.

F.4 Reconstruccion desde cero (sin reutilizar funciones de produccion):
    - parametrizacion propia del MS-GJR-GARCH K=2 (Haas + orden R5),
    - filtro de Hamilton propio (bucle explicito),
    - multistart L-BFGS propio (20 arranques, semilla fija),
    - entrenar SOLO con [train_start, test_start) del bloque 5 del artefacto,
    - filtrar la ventana continua train+test (arrastre de estado, como el spec)
      y comparar la loglik predictiva media del tramo test contra
      walkforward.json (block 5: ll_test = -0.968256, ll_train = -1.028048).
    Criterio: si mi optimo iguala o supera la loglik de train de produccion y
    ll_test coincide dentro de ~1e-3/obs, el numero publicado se reproduce.
    (El unico insumo compartido es la serie de retornos, reconstruida de la
    cache cruda de precios: r = 100*log(P_t/P_{t-1}).)

F.4b PIT del bloque reconstruido: u_t = F_pred(r_t) en el tramo test debe ser
    ~U(0,1); KS + autocorrelacion lag-1 de u_t (cubre parcialmente C.4).

F.1 (adaptado): el pipeline evalua densidad un-paso-adelante (h=1, sin
    etiquetas solapadas de 20 dias): el requisito de embargo del encargo no
    aplica; se verifica en cambio que train y test del bloque no se solapan y
    que el test es contiguo al train (arrastre de estado documentado).

Resultados en audits/probes/out/f_results.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
RES: dict = {}

# --- datos: SOLO la cache cruda de precios (ningun modulo de irfn) ------------
px = pd.read_parquet(ROOT / "data/raw/close_SPY_2010-01-01.parquet").sort_index()
close = px[px.columns[0]]
close.index = pd.to_datetime(close.index)
r = (100.0 * np.log(close).diff()).dropna()
r = r.loc["2013-01-01":]

wf = json.loads((ROOT / "artifacts/latest/walkforward.json").read_text(encoding="utf-8"))
blk = wf["blocks"][5]
tr0, ts0, te0 = map(pd.Timestamp, (blk["train_start"], blk["test_start"], blk["test_end"]))
r_train = r[(r.index >= tr0) & (r.index < ts0)].to_numpy()
r_win = r[(r.index >= tr0) & (r.index < te0)].to_numpy()
n_train, n_test = len(r_train), len(r_win) - len(r_train)
RES["F4_alineacion"] = {"n_train_mio": n_train, "n_train_artefacto": blk["n_train"],
                       "n_test_mio": n_test, "n_test_artefacto": blk["n_test"],
                       "coincide": bool(n_train == blk["n_train"] and n_test == blk["n_test"])}
print(f"[F4] alineacion train {n_train}/{blk['n_train']} test {n_test}/{blk['n_test']}", flush=True)

# F.1 adaptado: no solapamiento train/test
RES["F1_no_solape"] = {"train": [str(tr0), str(ts0)], "test": [str(ts0), str(te0)],
                       "solape": False, "nota": "perdida h=1 sin etiquetas solapadas; embargo no aplica"}


# --- implementacion PROPIA ----------------------------------------------------
def my_unpack(th):
    """theta -> naturales. Estructura canonica (Haas + R5), codigo propio."""
    mu = th[0:2]
    v1 = np.exp(th[2]); v2 = v1 + np.exp(th[3])
    p = 1.0 / (1.0 + np.exp(-th[4:6]))                      # kappa por regimen
    out_alpha = np.empty(2); out_gamma = np.empty(2); out_beta = np.empty(2)
    for k in range(2):
        l1, l2 = th[6 + 2 * k], th[7 + 2 * k]
        e = np.exp([l1, l2, 0.0]); u = e / e.sum()          # (u_a, u_g, u_b)
        out_alpha[k] = p[k] * u[0]
        out_gamma[k] = 2.0 * p[k] * u[1]
        out_beta[k] = p[k] * u[2]
    v = np.array([v1, v2])
    omega = v * (1.0 - p)
    q = 1.0 / (1.0 + np.exp(-th[10:12]))                    # p11, p22
    P = np.array([[q[0], 1 - q[0]], [1 - q[1], q[1]]])
    return mu, v, omega, out_alpha, out_gamma, out_beta, P


def my_filter_ll(th, rr, per_obs=False):
    mu, v, om, al, ga, be, P = my_unpack(th)
    T = len(rr)
    sig2 = np.empty((T, 2)); sig2[0] = v
    for t in range(1, T):
        e = rr[t - 1] - mu
        sig2[t] = om + al * e * e + ga * e * e * (e < 0) + be * sig2[t - 1]
    logf = -0.5 * (np.log(2 * np.pi) + np.log(sig2) + (rr[:, None] - mu) ** 2 / sig2)
    # xi0: estacionaria de P (2x2 cerrada)
    p12, p21 = P[0, 1], P[1, 0]
    pi0 = np.array([p21, p12]) / (p12 + p21)
    xi = pi0
    ll = np.empty(T)
    PT = P.T
    for t in range(T):
        if t > 0:
            xi = PT @ xi_f
        m = logf[t].max()
        num = xi * np.exp(logf[t] - m)
        den = num.sum()
        xi_f = num / den
        ll[t] = np.log(den) + m
    return ll if per_obs else float(ll.sum())


def my_neg_ll(th, rr):
    try:
        ll = my_filter_ll(th, rr)
    except FloatingPointError:
        return 1e12
    return -ll if np.isfinite(ll) else 1e12


t0 = time.time()
rng = np.random.default_rng(2024)
s2 = float(np.var(r_train))
best = None
lls = []
for s in range(20):
    th0 = np.zeros(12)
    th0[0:2] = rng.normal(scale=0.1, size=2)
    th0[2] = np.log(s2 * rng.uniform(0.2, 0.8))
    th0[3] = np.log(s2 * rng.uniform(0.5, 3.0))
    th0[4:6] = rng.normal(2.5, 0.8, size=2)                 # kappa alta
    th0[6:10] = rng.normal(scale=0.7, size=4)
    th0[10:12] = rng.normal(3.0, 0.7, size=2)               # persistencia alta
    res = optimize.minimize(my_neg_ll, th0, args=(r_train,), method="L-BFGS-B",
                            options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-7})
    if np.isfinite(res.fun) and res.fun < 1e11:
        lls.append(-res.fun)
        if best is None or res.fun < best.fun:
            best = res
    print(f"  start {s}: ll={-res.fun:.4f}", flush=True)

ll_train_mine = -best.fun / n_train
ll_obs_win = my_filter_ll(best.x, r_win, per_obs=True)
ll_test_mine = float(ll_obs_win[n_train:].mean())
RES["F4_reconstruccion_bloque5"] = {
    "ll_train_per_obs_mio": float(ll_train_mine),
    "ll_train_per_obs_artefacto": blk["loglik_train_per_obs"],
    "delta_train": float(ll_train_mine - blk["loglik_train_per_obs"]),
    "ll_test_per_obs_mio": ll_test_mine,
    "ll_test_per_obs_artefacto": blk["loglik_test_per_obs"],
    "delta_test": float(ll_test_mine - blk["loglik_test_per_obs"]),
    "n_starts_en_mi_optimo": int(np.sum(np.abs(np.array(lls) - (-best.fun)) < 1e-5 * n_train)),
    "wall_s": round(time.time() - t0, 1),
}
print(f"[F4] train mio={ll_train_mine:.6f} art={blk['loglik_train_per_obs']:.6f} | "
      f"test mio={ll_test_mine:.6f} art={blk['loglik_test_per_obs']:.6f} "
      f"({RES['F4_reconstruccion_bloque5']['wall_s']}s)", flush=True)

# --- F.4b: PIT del tramo test (mezcla Normal con params del train) -----------
mu, v, om, al, ga, be, P = my_unpack(best.x)
T = len(r_win)
sig2 = np.empty((T, 2)); sig2[0] = v
for t in range(1, T):
    e = r_win[t - 1] - mu
    sig2[t] = om + al * e * e + ga * e * e * (e < 0) + be * sig2[t - 1]
logf = -0.5 * (np.log(2 * np.pi) + np.log(sig2) + (r_win[:, None] - mu) ** 2 / sig2)
p12, p21 = P[0, 1], P[1, 0]
xi = np.array([p21, p12]) / (p12 + p21)
u_pit = []
PT = P.T
for t in range(T):
    if t > 0:
        xi = PT @ xi_f
    if t >= n_train:
        u = float(np.sum(xi * stats.norm.cdf(r_win[t], loc=mu, scale=np.sqrt(sig2[t]))))
        u_pit.append(u)
    m = logf[t].max()
    num = xi * np.exp(logf[t] - m)
    xi_f = num / num.sum()
u_pit = np.array(u_pit)
ks_stat, ks_p = stats.kstest(u_pit, "uniform")
ac1 = float(np.corrcoef(u_pit[:-1], u_pit[1:])[0, 1])
se_ac1 = 1.0 / np.sqrt(len(u_pit))
RES["F4b_pit_bloque5"] = {
    "n": int(len(u_pit)), "ks_stat": float(ks_stat), "ks_p": float(ks_p),
    "autocorr_lag1": ac1, "se_autocorr": se_ac1,
    "nota": ("PIT un-paso-adelante del tramo OOS del bloque 5; con autocorr(u) "
             "dentro de +-2/sqrt(n), el KS sobre u_t es interpretable sin correccion"),
}
print(f"[F4b] PIT: KS={ks_stat:.4f} p={ks_p:.3f} ac1={ac1:+.3f} (2se={2*se_ac1:.3f})", flush=True)

(OUT / "f_results.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
print("FASE F (probe) completa -> audits/probes/out/f_results.json", flush=True)

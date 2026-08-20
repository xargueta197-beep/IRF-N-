"""FASE D -- Hawkes: verificaciones numericas linea a linea sobre el CORPUS REAL
y el artefacto publicado (run 3b4f1e39b59c).

D.1 Unidades: reproduce la preparacion de produccion (load_headlines ->
    score_headlines [cache] -> headline_event_times [dias] -> dithering seed=42
    -> compress_to_observed_time) y verifica que el re-ajuste con el MLE de
    produccion reproduce los parametros publicados (tambien sirve a Fase H).
D.2 Formula de n para el kernel implementado phi(u,m)=alpha*m*exp(-beta*u):
    integral analitica = alpha*m/beta por evento => n = alpha*E[s]/beta.
    Verificacion aritmetica del n publicado desde (alpha, beta, mean_mark).
D.5 Empates de timestamp en el corpus real, antes y despues del dithering.
D.6 IC de n: metodo delta (publicado) vs verosimilitud PERFILADA sobre el corpus
    real (perfil en n, optimizando (mu, beta) con alpha = n*beta/E[s]).
D.7 KS: D critico al 5% para N~95k (asintotico 1.3581/sqrt(N)) y encuadre del
    tamano de efecto; nota Lilliefors (parametros estimados de la misma muestra).
D.4 Efecto de borde t0 (sin historia previa): sesgo cuantificado en sintetico
    con kernel LENTO (beta=1.2/d, peor caso) y a escala de produccion.

Solo LECTURA. Resultados en audits/probes/out/d_results.json.
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
sys.path.insert(0, str(ROOT / "src"))

from irfn.models import hawkes_mle as hm  # noqa: E402
from irfn.features import hawkes_features as hf  # noqa: E402
from irfn.data.headlines import load_headlines  # noqa: E402
from irfn.features.relevance import score_headlines  # noqa: E402

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
RES: dict = {}

irfn = json.loads((ROOT / "artifacts/latest/irfn.json").read_text(encoding="utf-8"))
hk = irfn["model"]["hawkes_layer_params"]

# --- D.2: aritmetica del n publicado -----------------------------------------
n_recalc = hk["alpha"] * hk["mean_mark"] / hk["beta"]
casc_recalc = 1.0 / (1.0 - n_recalc)
RES["D2_n_publicado"] = {
    "alpha": hk["alpha"], "beta": hk["beta"], "mean_mark": hk["mean_mark"],
    "n_recalc": n_recalc, "n_publicado": hk["branching_ratio"],
    "coincide": bool(abs(n_recalc - hk["branching_ratio"]) < 1e-12),
    "cascada_recalc": casc_recalc, "cascada_publicada": hk["expected_cascade"],
    "cascada_coincide": bool(abs(casc_recalc - hk["expected_cascade"]) < 1e-12),
}
print(f"[D2] n recalc={n_recalc:.10f} pub={hk['branching_ratio']:.10f} "
      f"ok={RES['D2_n_publicado']['coincide']}", flush=True)

# --- D.7: D critico del KS ----------------------------------------------------
N = hk["n_events"] - 1  # interarribos
D_obs = hk["ks_stat"]
D_crit = 1.3581 / np.sqrt(N)
RES["D7_ks"] = {
    "N_interarribos": N, "D_obs": D_obs, "D_crit_5pct": float(D_crit),
    "ratio_obs_sobre_crit": float(D_obs / D_crit),
    "nota_lilliefors": ("kstest se aplica con parametros ESTIMADOS de la misma muestra: "
                        "el p-valor nominal no es exacto (efecto tipo Lilliefors). Con "
                        "D/D_crit ~ 6.6 la conclusion (rechazo del kernel exponencial) no "
                        "cambia, pero el p exacto no es interpretable literalmente."),
    "encuadre_efecto": ("D=0.0289 = desviacion maxima de 2.9 puntos porcentuales en la CDF "
                        "de interarribos re-escalados; el rechazo lo domina N~95k."),
}
print(f"[D7] D_obs={D_obs:.4f} D_crit(5%)={D_crit:.5f} ratio={D_obs/D_crit:.1f}", flush=True)

# --- D.1 + D.5 + D.6: corpus real --------------------------------------------
t0 = time.time()
headlines = load_headlines()
cov = headlines.attrs["coverage"]
scored = score_headlines(headlines, model_id="ProsusAI/finbert", batch_size=32)
times_raw, marks, origin = hf.headline_event_times(scored)
n_ev = len(times_raw)
uniq = len(np.unique(times_raw))
ties_frac = 1.0 - uniq / n_ev
times_d, marks_d = hf.dither_quantized_times(times_raw, marks, seed=42)
uniq_d = len(np.unique(times_d))
RES["D5_empates"] = {
    "n_eventos": n_ev, "timestamps_unicos_pre_dither": uniq,
    "frac_eventos_empatados_pre_dither": float(ties_frac),
    "timestamps_unicos_post_dither": uniq_d,
    "empates_post_dither": int(n_ev - uniq_d),
}
print(f"[D5] eventos={n_ev} empatados pre-dither={ties_frac:.3f} "
      f"post-dither unicos={uniq_d}/{n_ev} (prep {time.time()-t0:.0f}s)", flush=True)

times_obs, T_obs = hf.compress_to_observed_time(times_d, origin, cov["missing_days"])
fit = hm.fit_hawkes_mle(times_obs, marks_d, T=T_obs, n_starts=30, seed=42)
RES["D1_reproduccion_fit_publicado"] = {
    "T_obs": T_obs,
    "est": fit.params, "pub": {"mu_N": hk["mu_N"], "alpha": hk["alpha"], "beta": hk["beta"]},
    "n_est": fit.branching_ratio, "n_pub": hk["branching_ratio"],
    "coincide_1e-6_rel": bool(
        abs(fit.params["mu"] - hk["mu_N"]) / hk["mu_N"] < 1e-6
        and abs(fit.params["alpha"] - hk["alpha"]) / hk["alpha"] < 1e-6
        and abs(fit.params["beta"] - hk["beta"]) / hk["beta"] < 1e-6),
    "unidades": "times en DIAS desde origin => mu_N y beta en 1/dia; n adimensional",
}
print(f"[D1] refit mu={fit.params['mu']:.4f} alpha={fit.params['alpha']:.4f} "
      f"beta={fit.params['beta']:.4f} n={fit.branching_ratio:.6f} "
      f"(pub {hk['mu_N']:.4f}/{hk['alpha']:.4f}/{hk['beta']:.4f}/{hk['branching_ratio']:.6f}) "
      f"match={RES['D1_reproduccion_fit_publicado']['coincide_1e-6_rel']}", flush=True)

# --- D.6: IC de n por verosimilitud perfilada --------------------------------
E_s = float(marks_d.mean())
ll_hat = fit.loglik


def profile_ll(n_fix):
    def nll(theta):
        mu, beta = np.exp(theta)
        alpha = n_fix * beta / E_s
        ll = hm.hawkes_loglik(times_obs, marks_d, mu, alpha, beta, T_obs)
        return -ll if np.isfinite(ll) else 1e30
    x0 = np.log([fit.params["mu"], fit.params["beta"]])
    res = optimize.minimize(nll, x0, method="L-BFGS-B")
    return -res.fun


t0 = time.time()
grid = np.linspace(hk["branching_ratio"] - 0.020, hk["branching_ratio"] + 0.020, 21)
prof = np.array([profile_ll(n) for n in grid])
dev = 2.0 * (ll_hat - prof)      # ~ chi2_1 bajo H0
chi_05 = stats.chi2.ppf(0.95, 1)
inside = grid[dev <= chi_05]
ci_prof = [float(inside.min()), float(inside.max())] if len(inside) else [np.nan, np.nan]
RES["D6_ic_perfilado_vs_delta"] = {
    "ci_delta_publicado": [hk["branching_ratio_ci_low"], hk["branching_ratio_ci_high"]],
    "ci_perfilado_95": ci_prof,
    "grid": grid.tolist(), "deviance": dev.tolist(),
    "wall_s": round(time.time() - t0, 1),
    "nota": ("perfil en n con alpha=n*beta/E[s]; E[s] tratado como constante (su "
             "varianza muestral con N=95k es despreciable frente a la de (alpha,beta))"),
}
print(f"[D6] IC delta=[{hk['branching_ratio_ci_low']:.4f},{hk['branching_ratio_ci_high']:.4f}] "
      f"perfilado=[{ci_prof[0]:.4f},{ci_prof[1]:.4f}] ({RES['D6_ic_perfilado_vs_delta']['wall_s']}s)", flush=True)

# --- D.4: efecto de borde t0 --------------------------------------------------
def edge_bias(mu, beta, n_true, T_total, burn_days, seed, n_starts=8):
    alpha = n_true * beta / E_s
    rng = np.random.default_rng(seed)
    samp = lambda g: float(marks_d[g.integers(0, len(marks_d))])  # noqa: E731
    t_all, s_all = hm.simulate_hawkes_ogata_thinning(T_total, mu, alpha, beta, rng, mark_sampler=samp)
    # ventana [burn, T]: la historia previa se DESCARTA y el reloj se reinicia,
    # exactamente el supuesto lambda(t0)=mu del estimador de produccion
    m = t_all >= burn_days
    t_win = t_all[m] - burn_days
    s_win = s_all[m]
    f_edge = hm.fit_hawkes_mle(t_win, s_win, T=T_total - burn_days, n_starts=n_starts, seed=seed)
    f_full = hm.fit_hawkes_mle(t_all, s_all, T=T_total, n_starts=n_starts, seed=seed)
    return {"n_true": n_true, "n_ventana_sin_historia": f_full.branching_ratio,
            "n_ventana_borde": f_edge.branching_ratio,
            "delta_n_por_borde": f_edge.branching_ratio - f_full.branching_ratio,
            "eventos_ventana": f_edge.n_events}


RES["D4_borde_kernel_lento"] = edge_bias(mu=0.5, beta=1.2, n_true=0.85, T_total=6000.0,
                                         burn_days=3000.0, seed=21)
RES["D4_borde_escala_prod"] = edge_bias(mu=100.0, beta=284.0, n_true=0.74, T_total=240.0,
                                        burn_days=120.0, seed=22)
for k in ("D4_borde_kernel_lento", "D4_borde_escala_prod"):
    d = RES[k]
    print(f"[{k}] n_true={d['n_true']} full={d['n_ventana_sin_historia']:.4f} "
          f"borde={d['n_ventana_borde']:.4f} delta={d['delta_n_por_borde']:+.4f}", flush=True)

# --- D.3: regla de censura activa en serializacion (verificacion estatica ya
# hecha por lectura: run_v3.py:569-574 llama expected_cascade_reported con el
# trigger de config; el artefacto publica expected_cascade_bounded=True) --------
RES["D3_censura_activa"] = {
    "llamada_en_serializacion": "scripts/run_v3.py:572",
    "trigger_config": 0.95,
    "artefacto": {"ci_high": hk["branching_ratio_ci_high"],
                  "bounded_publicado": hk["expected_cascade_bounded"]},
    "coherente": bool((hk["branching_ratio_ci_high"] < 0.95) == hk["expected_cascade_bounded"]),
}

(OUT / "d_results.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
print("FASE D completa -> audits/probes/out/d_results.json", flush=True)

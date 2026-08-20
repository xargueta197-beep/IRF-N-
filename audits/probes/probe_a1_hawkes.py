"""FASE A.1 -- Recuperacion de parametros del Hawkes marcado sobre verdad conocida.

Protocolo (AUDIT_MATH_v1):
  1. Simular por thinning (Ogata) un Hawkes exponencial marcado con parametros
     fijados a mano, marcas con la distribucion EMPIRICA de FinBERT (bootstrap
     del corpus real artifacts/latest/headline_rug.parquet, E[s]~0.636).
     NOTA sobre los valores del encargo: el doc fija mu=0.5, alpha=0.3, beta=1.2
     "=> n=0.25", que es la cuenta SIN marcas (alpha/beta). El canon del proyecto
     (y la fisica del modelo marcado) es n = alpha*E[s]/beta, asi que para que la
     VERDAD sea n=0.25 exacto se ajusta alpha = 0.25*beta/E[s_emp] manteniendo
     mu=0.5 y beta=1.2. El objetivo del criterio (n verdadero = 0.25) se respeta.
  2. Alimentar la serie al estimador DE PRODUCCION (fit_hawkes_mle), sin
     adaptadores.
  3. Criterio: cada parametro dentro de +-3 SE del verdadero; n dentro de
     +-0.03 del objetivo.
  4. Repetir con n=0.85 (cerca de frontera) y con SOPORTE CENSURADO (dias
     completos eliminados; escenario T_span vs T_observado), fit por la ruta de
     produccion compress_to_observed_time y tambien por la ruta ingenua T=span
     para exhibir la direccion del sesgo que el fix del 2026-08-15 corrige.
  5. 50 replicas por escenario: sesgo y RMSE por parametro (sesgo sistematico
     > 5% = hallazgo CRITICO).

Solo LECTURA de src/. Escribe resultados en audits/probes/out/a1_results.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from irfn.models import hawkes_mle as hm                      # noqa: E402
from irfn.features.hawkes_features import compress_to_observed_time  # noqa: E402

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

EMP_MARKS = pd.read_parquet(ROOT / "artifacts/latest/headline_rug.parquet")["s"].to_numpy()
E_S = float(EMP_MARKS.mean())


def emp_mark_sampler(rng: np.random.Generator) -> float:
    return float(EMP_MARKS[rng.integers(0, len(EMP_MARKS))])


def run_single(label, mu, beta, n_true, T, seed, n_starts=20):
    alpha = n_true * beta / E_S
    rng = np.random.default_rng(seed)
    t0 = time.time()
    times, marks = hm.simulate_hawkes_ogata_thinning(T, mu, alpha, beta, rng, mark_sampler=emp_mark_sampler)
    fit = hm.fit_hawkes_mle(times, marks, T=T, n_starts=n_starts, seed=seed)
    truth = {"mu": mu, "alpha": alpha, "beta": beta}
    within3 = {}
    for k in ("mu", "alpha", "beta"):
        se = fit.se[k]
        within3[k] = bool(np.isfinite(se) and abs(fit.params[k] - truth[k]) <= 3 * se)
    res = {
        "label": label,
        "truth": {**truth, "n": n_true, "E_s": E_S},
        "n_events": fit.n_events,
        "T": T,
        "est": fit.params,
        "se": fit.se,
        "n_hat": fit.branching_ratio,
        "n_se": fit.branching_ratio_se,
        "n_ci": [fit.branching_ratio_ci_low, fit.branching_ratio_ci_high],
        "within_3se": within3,
        "n_within_003": bool(abs(fit.branching_ratio - n_true) <= 0.03),
        "starts_at_best": fit.starts_at_best,
        "n_converged": fit.n_converged,
        "hessian_ok": fit.hessian_ok,
        "wall_s": round(time.time() - t0, 1),
    }
    print(f"[{label}] N={fit.n_events} n_hat={fit.branching_ratio:.4f} (true {n_true}) "
          f"mu={fit.params['mu']:.4f}({mu}) alpha={fit.params['alpha']:.4f}({alpha:.4f}) "
          f"beta={fit.params['beta']:.4f}({beta}) within3se={within3} "
          f"n_ok={res['n_within_003']} t={res['wall_s']}s", flush=True)
    return res


def run_replicas(label, mu, beta, n_true, T, n_rep=50, n_starts=6):
    alpha = n_true * beta / E_S
    truth = {"mu": mu, "alpha": alpha, "beta": beta, "n": n_true}
    rows = []
    t0 = time.time()
    for rep in range(n_rep):
        rng = np.random.default_rng(10_000 + rep)
        times, marks = hm.simulate_hawkes_ogata_thinning(T, mu, alpha, beta, rng, mark_sampler=emp_mark_sampler)
        fit = hm.fit_hawkes_mle(times, marks, T=T, n_starts=n_starts, seed=rep)
        rows.append({"mu": fit.params["mu"], "alpha": fit.params["alpha"],
                     "beta": fit.params["beta"], "n": fit.branching_ratio,
                     "n_se": fit.branching_ratio_se,
                     "cover_n": bool(np.isfinite(fit.branching_ratio_se)
                                     and abs(fit.branching_ratio - n_true) <= 1.96 * fit.branching_ratio_se),
                     "N": fit.n_events})
    df = pd.DataFrame(rows)
    summary = {"label": label, "truth": truth, "n_rep": n_rep,
               "mean_events": float(df["N"].mean()), "wall_s": round(time.time() - t0, 1)}
    for k in ("mu", "alpha", "beta", "n"):
        est = df[k].to_numpy()
        tv = truth[k]
        bias = float(est.mean() - tv)
        summary[k] = {
            "true": tv, "mean": float(est.mean()), "bias": bias,
            "bias_pct": float(100.0 * bias / tv),
            "rmse": float(np.sqrt(np.mean((est - tv) ** 2))),
            "sd": float(est.std(ddof=1)),
        }
    summary["ci95_coverage_n"] = float(df["cover_n"].mean())
    print(f"[{label}] replicas={n_rep} bias%: mu={summary['mu']['bias_pct']:.2f} "
          f"alpha={summary['alpha']['bias_pct']:.2f} beta={summary['beta']['bias_pct']:.2f} "
          f"n={summary['n']['bias_pct']:.2f} | RMSE n={summary['n']['rmse']:.4f} "
          f"| cobertura IC95(n)={summary['ci95_coverage_n']:.2f} | t={summary['wall_s']}s", flush=True)
    return summary


def run_censored(label, mu, beta, n_true, span_days, frac_observed, seed, n_starts=12):
    """Simula sobre el calendario completo [0, span]; censura dias completos
    (se conservan el primero y el ultimo para anclar el rango, como en el corpus
    real, cuyo primer/ultimo dia estan capturados); ajusta por (a) la ruta
    INGENUA T=span y (b) la ruta de produccion compress_to_observed_time."""
    alpha = n_true * beta / E_S
    rng = np.random.default_rng(seed)
    times, marks = hm.simulate_hawkes_ogata_thinning(span_days, mu, alpha, beta, rng, mark_sampler=emp_mark_sampler)
    all_days = np.arange(int(span_days))
    interior = all_days[1:-1]
    n_keep_interior = int(round(frac_observed * len(all_days))) - 2
    keep_interior = rng.choice(interior, size=n_keep_interior, replace=False)
    observed = np.sort(np.concatenate([[all_days[0], all_days[-1]], keep_interior]))
    missing = np.setdiff1d(all_days, observed)
    day_of = np.floor(times).astype(int)
    keep_mask = np.isin(day_of, observed)
    t_c, s_c = times[keep_mask], marks[keep_mask]

    fit_naive = hm.fit_hawkes_mle(t_c, s_c, T=float(span_days), n_starts=n_starts, seed=seed)

    origin = pd.Timestamp("2020-01-01")
    missing_iso = [(origin + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in missing]
    t_obs, T_obs = compress_to_observed_time(t_c, origin, missing_iso)
    fit_prod = hm.fit_hawkes_mle(t_obs, s_c, T=T_obs, n_starts=n_starts, seed=seed)

    res = {
        "label": label,
        "truth": {"mu": mu, "alpha": alpha, "beta": beta, "n": n_true},
        "span_days": span_days, "observed_days": int(len(observed)),
        "events_total": int(len(times)), "events_kept": int(len(t_c)),
        "naive": {"T": float(span_days), "est": fit_naive.params, "n_hat": fit_naive.branching_ratio,
                  "n_se": fit_naive.branching_ratio_se},
        "prod": {"T_obs": T_obs, "est": fit_prod.params, "n_hat": fit_prod.branching_ratio,
                 "n_se": fit_prod.branching_ratio_se,
                 "n_ci": [fit_prod.branching_ratio_ci_low, fit_prod.branching_ratio_ci_high]},
        "n_within_003_prod": bool(abs(fit_prod.branching_ratio - n_true) <= 0.03),
    }
    print(f"[{label}] true n={n_true} | naive(T=span) n={fit_naive.branching_ratio:.4f} "
          f"mu={fit_naive.params['mu']:.3f} | prod(T_obs) n={fit_prod.branching_ratio:.4f} "
          f"mu={fit_prod.params['mu']:.3f} (mu true {mu}) | prod ok={res['n_within_003_prod']}", flush=True)
    return res


def main():
    results = {}
    # 1) caso base n=0.25 lejos de frontera, ~20000 eventos esperados
    #    tasa = mu/(1-n) = 0.6667/d -> T = 30000 d
    results["single_n025"] = run_single("single n=0.25", mu=0.5, beta=1.2, n_true=0.25, T=30000.0, seed=1)
    # 2) cerca de frontera n=0.85: tasa = 3.33/d -> T = 6000 d
    results["single_n085"] = run_single("single n=0.85", mu=0.5, beta=1.2, n_true=0.85, T=6000.0, seed=2)
    # 3) censura de soporte, escala de PRODUCCION (mu~100/d, beta~30/d, n~0.74;
    #    span 998 d, ~24% observado: replica el escenario real 240/998)
    results["censored_prod_scale"] = run_censored(
        "censura escala prod", mu=100.0, beta=30.0, n_true=0.74, span_days=998, frac_observed=0.24, seed=3)
    # 3b) censura con kernel LENTO (beta=1.2/d: la excitacion cruza los huecos;
    #     mide el limite de validez de la compresion)
    results["censored_slow_beta"] = run_censored(
        "censura beta lenta", mu=0.5, beta=1.2, n_true=0.85, span_days=6000, frac_observed=0.40, seed=4)
    # 5) 50 replicas por escenario: sesgo y RMSE
    results["replicas_n025"] = run_replicas("replicas n=0.25", mu=0.5, beta=1.2, n_true=0.25, T=30000.0)
    results["replicas_n085"] = run_replicas("replicas n=0.85", mu=0.5, beta=1.2, n_true=0.85, T=6000.0)

    (OUT / "a1_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print("OK -> audits/probes/out/a1_results.json", flush=True)


if __name__ == "__main__":
    main()

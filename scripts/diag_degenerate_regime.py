"""@diagnostic_only -- Diagnostico del regimen degenerado (absorbe-outliers).

Fase 7 de la remediacion. NO publica, NO toca artifacts/latest/. Solo estima y
mide, para decidir con evidencia si el estado de 'alta volatilidad' que dura ~1
dia es (a) un artefacto del multistart, (b) estructural del K=2/Normal, o (c)
algo que un remedio (Student-t o K=3) corrige.

Pruebas:
  1. K=2 Normal con 50 arranques (MAS que el R6=30 publicado): si el ganador
     sigue degenerado, no es cuestion de mas arranques -> es el optimo global.
  2. K=2 Student-t: fat tails podrian absorber los outliers en la cola en vez de
     inventar un regimen-pico.
  3. K=3 Normal: un tercer regimen podria volverse el estado persistente de alta
     vol y liberar al segundo de su papel de absorbe-outliers.

Metricas por regimen: kappa = alpha+gamma/2+beta (persistencia GARCH), E[D] =
1/(1-p_kk) (dias), % de dias asignados (argmax), vol anual empirica de esos dias.
Un regimen degenerado tiene E[D]~1 y %dias muy bajo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irfn.data.prices import load_close  # noqa: E402
from irfn.pipeline import run_pipeline  # noqa: E402

REPORTS = ROOT / "reports"
SEED = 42
DEGEN_EDD = 2.0  # E[D] < esto = degenerado (mismo umbral que la app)


def _load_returns() -> pd.Series:
    close = load_close("SPY", "2010-01-01")  # cache fresco (hasta asof vigente)
    r = 100.0 * np.log(close).diff().dropna()
    return r.loc["2013-01-01":]


def _regime_table(res, r: pd.Series) -> list[dict]:
    fit = res.fit
    P = np.asarray(fit.P, dtype=float)
    K = P.shape[0]
    am = res.frame["argmax_idx"].to_numpy()
    rows = []
    for k in range(K):
        kappa = float(fit.params["alpha"][k] + fit.params["gamma"][k] / 2 + fit.params["beta"][k])
        edd = float(1.0 / (1.0 - min(P[k, k], 1 - 1e-12)))
        sel = am == k
        frac = float(np.mean(sel))
        vol_ann = float(np.std(r.to_numpy()[sel]) * np.sqrt(252) / 100.0) if sel.any() else float("nan")
        rows.append({"regime": k, "kappa": kappa, "E[D]_dias": edd,
                     "pct_dias": 100 * frac, "vol_anual_emp": vol_ann,
                     "degenerado": edd < DEGEN_EDD})
    return rows


def _run(label: str, r: pd.Series, K: int, dist: str, n_starts: int) -> dict:
    print(f"\n=== {label}: K={K} dist={dist} n_starts={n_starts} ===")
    res = run_pipeline(r, K=K, seed=SEED, n_starts=n_starts, compute_se=False, dist=dist)
    ll = float(res.fit.loglik) if hasattr(res.fit, "loglik") else float("nan")
    tbl = _regime_table(res, r)
    for row in tbl:
        print(f"  reg {row['regime']}: kappa={row['kappa']:.4f}  E[D]={row['E[D]_dias']:.2f}d  "
              f"%dias={row['pct_dias']:.1f}  vol_anual={row['vol_anual_emp']:.2%}  "
              f"{'<== DEGENERADO' if row['degenerado'] else ''}")
    print(f"  n_converged/n_starts al optimo: {res.fit.n_converged}/{res.fit.n_starts}  loglik={ll:.2f}")
    return {"label": label, "K": K, "dist": dist, "n_starts": n_starts,
            "loglik": ll, "n_converged": res.fit.n_converged, "n_starts_used": res.fit.n_starts,
            "regimes": tbl, "any_degenerate": any(x["degenerado"] for x in tbl)}


def main() -> None:
    r = _load_returns()
    print(f"muestra SPY: {r.index[0].date()}..{r.index[-1].date()} ({len(r)} obs)")
    results = [
        _run("1) K=2 Normal (50 arranques, > R6)", r, K=2, dist="normal", n_starts=50),
        _run("2) K=2 Student-t (colas gordas)", r, K=2, dist="t", n_starts=40),
        _run("3) K=3 Normal (regimen extra)", r, K=3, dist="normal", n_starts=40),
    ]

    lines = ["# Diagnostico del regimen degenerado (absorbe-outliers) -- Fase 7\n",
             f"Muestra SPY {r.index[0].date()}..{r.index[-1].date()} ({len(r)} obs). "
             f"Semilla {SEED}. @diagnostic_only: no publica.\n",
             "Un regimen es 'degenerado' (absorbe-outliers) si E[D]=1/(1-p_kk) < "
             f"{DEGEN_EDD} dias: aparece un dia suelto para capturar un outlier y no persiste.\n"]
    for res in results:
        lines.append(f"## {res['label']}\n")
        lines.append("| regimen | kappa | E[D] (dias) | % dias | vol anual emp | degenerado |")
        lines.append("| --: | --: | --: | --: | --: | :--: |")
        for row in res["regimes"]:
            lines.append(f"| {row['regime']} | {row['kappa']:.4f} | {row['E[D]_dias']:.2f} | "
                         f"{row['pct_dias']:.1f}% | {row['vol_anual_emp']:.1%} | "
                         f"{'SI' if row['degenerado'] else 'no'} |")
        lines.append(f"\n- loglik={res['loglik']:.2f}, convergieron al optimo "
                     f"{res['n_converged']}/{res['n_starts_used']} arranques. "
                     f"Regimen degenerado presente: **{'SI' if res['any_degenerate'] else 'NO'}**.\n")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "diag_degenerate_regime.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreporte -> reports/diag_degenerate_regime.md")


if __name__ == "__main__":
    main()

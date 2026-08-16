"""Demo V0: baja SPY con yfinance, ajusta el MS-GJR-GARCH K=2 (matriz de
transicion constante, innovaciones Normales) e imprime parametros con errores
estandar, loglik, BIC y la probabilidad filtrada xi_{t|t} de los ultimos 10 dias.

Esto NO es el pipeline de produccion (no publica artefactos, no corre
walk-forward). Es una demostracion end-to-end del nucleo para inspeccion visual.
Retornos en escala porcentual (log-retorno * 100) para que los numeros sean
legibles.

Uso:
    python scripts/demo_v0.py
    python scripts/demo_v0.py --start 2010-01-01 --n-starts 30 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from irfn.models.estimate import fit  # noqa: E402


def _load_spy_returns(start: str) -> tuple[np.ndarray, "pd.DatetimeIndex"]:
    import pandas as pd  # noqa: F401
    import yfinance as yf

    df = yf.download("SPY", start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("yfinance no devolvio datos para SPY (ver reports/data_audit.md).")
    close = df["Close"].squeeze()
    log_ret = np.log(close).diff().dropna() * 100.0  # escala porcentual
    return log_ret.to_numpy(), log_ret.index


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo V0: MS-GJR-GARCH K=2 sobre SPY.")
    parser.add_argument("--start", default="2010-01-01", help="fecha de inicio (YYYY-MM-DD).")
    parser.add_argument("--n-starts", type=int, default=30, help="arranques del multistart (R6).")
    parser.add_argument("--seed", type=int, default=42, help="semilla fija (R6).")
    args = parser.parse_args()

    r, index = _load_spy_returns(args.start)
    print(f"SPY: {len(r)} retornos diarios, {index[0].date()} a {index[-1].date()}")
    print(f"Ajustando K=2 con {args.n_starts} arranques (semilla {args.seed})...\n")

    res = fit(r, K=2, n_starts=args.n_starts, seed=args.seed)

    labels = ["regimen 0 (baja vol)", "regimen 1 (alta vol)"]
    print("=" * 68)
    print(f"loglik = {res.loglik:.3f}   BIC = {res.bic:.3f}   AIC = {res.aic:.3f}")
    print(f"arranques al mismo optimo: {res.n_converged}/{res.n_starts}   "
          f"hessiano_ok: {res.hessian_ok}   semilla: {res.seed}")
    print("=" * 68)

    def line(name: str, vals, ses):
        cells = "  ".join(f"{v:>12.6g} (+/-{s:.2g})" for v, s in zip(vals, ses))
        print(f"  {name:<7}{cells}")

    print(f"  {'':<7}{labels[0]:>26}  {labels[1]:>26}")
    for name in ["mu", "v", "omega", "alpha", "gamma", "beta"]:
        line(name, res.params[name], res.se.get(name, [float('nan')] * 2))

    print("\n  Persistencia kappa = alpha + gamma/2 + beta:")
    kappa = res.params["alpha"] + res.params["gamma"] / 2 + res.params["beta"]
    print(f"    {kappa[0]:.4f}   {kappa[1]:.4f}")

    print("\n  Matriz de transicion P (constante):")
    for i in range(2):
        print(f"    [{res.P[i, 0]:.4f}  {res.P[i, 1]:.4f}]")

    # Notas interpretativas honestas: en V0 (K=2, P constante, Normal) es comun
    # que un regimen colapse a un estado transitorio casi integrado (kappa -> 1)
    # que absorbe outliers en vez de ser un regimen persistente de alta vol. Eso
    # NO es un bug del estimador (el test de recuperacion lo prueba sobre datos
    # simulados limpios); es una limitacion conocida de esta especificacion
    # minima, que V1 ataca con innovaciones t de Student, TVTP y seleccion de K.
    notas = []
    if res.hessian_ok is False:
        notas.append(
            "hessiano_ok=False: el optimo esta en/cerca de un borde y algunas SE "
            "no son fiables (apareceran como ~0). Parametro(s) no identificado(s) ahi."
        )
    if np.any(kappa > 0.999):
        reg = int(np.argmax(kappa))
        notas.append(
            f"regimen {reg} casi integrado (kappa={kappa[reg]:.4f}): actua como "
            f"estado transitorio absorbe-outliers, no como regimen persistente. "
            f"Su varianza incondicional v esta mal definida en el limite kappa->1."
        )
    if res.n_converged < max(1, res.n_starts // 3):
        notas.append(
            f"solo {res.n_converged}/{res.n_starts} arranques hallaron el optimo "
            f"global: superficie multimodal, conviene mas multistart."
        )
    if notas:
        print("\n  Notas (lectura honesta del ajuste V0):")
        for n in notas:
            print(f"    - {n}")

    # xi_{t|t} de los ultimos 10 dias
    from irfn.models.hamilton import hamilton_filter

    xi_filt, _, _ = hamilton_filter(r, res.params, K=2)
    print("\n  xi_{t|t} (prob. filtrada) - ultimos 10 dias:")
    print(f"    {'fecha':<12}{'P(reg 0)':>10}{'P(reg 1)':>10}   argmax")
    for d in range(-10, 0):
        row = xi_filt[d]
        am = "alta vol" if row[1] >= row[0] else "baja vol"
        print(f"    {str(index[d].date()):<12}{row[0]:>10.4f}{row[1]:>10.4f}   {am}")


if __name__ == "__main__":
    main()

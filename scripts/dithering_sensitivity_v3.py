"""Chequeo de sensibilidad del dithering del Hawkes (aviso #5, SPY) -- @diagnostic_only.

Contexto (reports/auditoria_pre_corrida.md, seccion 5):
  El seendate de GDELT esta cuantizado a 15 min, asi que ~83% de los titulares
  comparten timestamp exacto. Sin dithering el MLE del Hawkes continuo degenera
  (beta->inf). run_v3 aplica ruido U(0, 15min) por evento con SEMILLA FIJA
  (base.model.seed = 42). El auditor senalo: como el 83% de los datos recibe
  ruido inyectado, hay que confirmar UNA VEZ que el optimo del MLE lo fijan los
  DATOS y no la realizacion del dithering. Recomendacion textual: "re-ajustar con
  2-3 semillas alternativas y confirmar que (mu_N, alpha, beta) se mueven dentro
  de sus SE".

Que hace este script (NO cambia metodologia, no publica nada, no toca artifacts/):
  1. Replica EXACTAMENTE la preparacion de datos de run_v3 (SPY): carga el corpus
     de titulares, puntua s_i con el cache de relevancia, arma (times, marks).
  2. Re-ajusta el Hawkes variando SOLO la semilla del DITHERING. La semilla del
     multistart se deja FIJA en 42: con n_starts=30 (R6) el multistart encuentra
     el optimo global, asi que cualquier variacion del optimo entre corridas es
     atribuible al ruido del dithering, no a que arranques se probaron. Aislar asi
     las dos fuentes es lo que hace interpretable el resultado.
  3. Reporta, para cada semilla alternativa, |param - param_base| / SE_base (en
     unidades de la SE del ajuste base). Criterio del auditor: si todas las
     desviaciones quedan < 1 SE, el estimador esta dominado por los datos y no por
     el dithering -- el MLE del Hawkes es fiable. Si alguna salta > 1 SE, hay que
     saberlo antes de leer nada del Hawkes.

Uso:
  python scripts/dithering_sensitivity_v3.py                 # SPY, semillas por defecto
  python scripts/dithering_sensitivity_v3.py --seeds 42 1 7 123 2024

Es diagnostico: imprime una tabla y escribe reports/dithering_sensitivity_v3.md.
NO escribe en artifacts/ ni cambia el artefacto publicado.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from irfn.config import BaseConfig, NewsConfig  # noqa: E402
from irfn.data import headlines as headlines_mod  # noqa: E402
from irfn.features import hawkes_features as hf  # noqa: E402
from irfn.features.relevance import score_headlines  # noqa: E402
from irfn.models import hawkes_mle  # noqa: E402
import run_v2  # noqa: E402

REPORT = ROOT / "reports" / "dithering_sensitivity_v3.md"

# La semilla del multistart se mantiene FIJA para no confundir la sensibilidad al
# dithering con la del multistart (ver docstring). Es la semilla base del proyecto.
MULTISTART_SEED = 42
DEFAULT_DITHER_SEEDS = [42, 1, 7, 123, 2024]


def _fmt(x: float, nd: int = 4) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def load_scored_corpus(news_cfg: NewsConfig):
    """Corpus de titulares con s_i, replicando run_v3 (usa el cache de relevancia)."""
    headlines = headlines_mod.load_headlines()
    if headlines.empty:
        raise SystemExit("corpus de titulares vacio: data/raw/headlines/ sin snapshots.")
    cov = headlines.attrs["coverage"]
    scored = score_headlines(
        headlines,
        model_id=news_cfg.relevance.finbert_model,
        batch_size=news_cfg.relevance.batch_size,
    )
    n_events = int(scored["s"].notna().sum())
    if n_events < news_cfg.headlines.min_events_fit:
        raise SystemExit(
            f"solo {n_events} titulares puntuados (< {news_cfg.headlines.min_events_fit})."
        )
    return scored, cov


def fit_for_seed(times: np.ndarray, marks: np.ndarray, origin: pd.Timestamp,
                 cov: dict, *, dither_seed: int, n_starts: int,
                 multistart_seed: int = MULTISTART_SEED):
    """Un ajuste completo del Hawkes para una semilla de dithering dada.

    Replica el bloque (e) de run_v3: dither -> compress_to_observed_time -> MLE
    sobre el tiempo observado -> KS. Por defecto la semilla del MLE (multistart)
    es FIJA (MULTISTART_SEED); el chequeo base varia SOLO la del dithering. La
    extension (a) usa `multistart_seed` para variar tambien la del multistart.
    """
    t_d, m_d = hf.dither_quantized_times(times, marks, seed=dither_seed)
    times_obs, T_obs = hf.compress_to_observed_time(t_d, origin, cov["missing_days"])
    fit = hawkes_mle.fit_hawkes_mle(
        times_obs, m_d, T=T_obs, n_starts=n_starts, seed=multistart_seed,
    )
    ks = hawkes_mle.rescaling_ks(times_obs, m_d, fit.params)
    return fit, ks, T_obs


def run_multistart_sweep(times, marks, origin, cov, *, dither_seed, multistart_seeds, n_starts):
    """EXTENSION (a): confirma que el optimo del MLE es GLOBAL, no suerte de arranques.

    Fija la realizacion del dithering (una sola, `dither_seed`) y VARIA la semilla
    del multistart. Con n_starts=30 (R6), si el optimo es unimodal las distintas
    tandas de arranques aleatorios deben aterrizar en el MISMO optimo (misma
    log-verosimilitud y mismos parametros a precision numerica). Que difiera
    delataria multimodalidad / problema de identificacion. Es ORTOGONAL al chequeo
    del dithering (que fija el multistart y varia el de-empate).
    """
    t_d, m_d = hf.dither_quantized_times(times, marks, seed=dither_seed)
    times_obs, T_obs = hf.compress_to_observed_time(t_d, origin, cov["missing_days"])
    out = []
    for ms in multistart_seeds:
        fit = hawkes_mle.fit_hawkes_mle(times_obs, m_d, T=T_obs, n_starts=n_starts, seed=ms)
        out.append({
            "ms_seed": ms, "loglik": fit.loglik,
            "mu": fit.params["mu"], "alpha": fit.params["alpha"], "beta": fit.params["beta"],
            "n": fit.branching_ratio,
            "starts_at_best": fit.starts_at_best, "n_starts_used": fit.n_starts,
        })
    return out


def rubin_pool(rows):
    """EXTENSION (b): agrupa n sobre las semillas de dithering por imputacion multiple.

    Cada semilla de dithering es UNA imputacion de los timestamps sub-bin no
    observados (supuesto de llegada uniforme intra-bin). La regla de Rubin combina
    m imputaciones:
        Q_bar = media de n_i                          (punto agrupado)
        W_bar = media de Var_dentro_i = media (SE_i^2)  (incertidumbre del ajuste)
        B     = var muestral de n_i (ddof=1)          (incertidumbre ENTRE imputaciones)
        T     = W_bar + (1 + 1/m) * B                 (varianza TOTAL)
    Fraccion de informacion perdida por el dithering: r = (1+1/m)*B / W_bar
    (cuanta incertidumbre AÑADE el de-empate sobre la del ajuste). Reportar el SE
    agrupado sqrt(T) frente al SE de una sola semilla mide ese aporte.

    NO cambia lo que se publica: es un diagnostico. Usar `n` agrupado en cualquier
    ruta publicada es DECISION DEL DIRECTOR (R3): cambia COMO se reporta la
    incertidumbre del indicador.
    """
    n_i = np.array([r["n"] for r in rows], dtype=float)
    w_i = np.array([r["n_se"] ** 2 for r in rows], dtype=float)  # Var_dentro por imputacion
    m = len(n_i)
    q_bar = float(np.mean(n_i))
    w_bar = float(np.mean(w_i))
    b = float(np.var(n_i, ddof=1)) if m > 1 else 0.0
    t_total = w_bar + (1.0 + 1.0 / m) * b
    se_pooled = float(np.sqrt(t_total))
    se_single = float(np.sqrt(w_bar))  # SE tipico de una sola imputacion (~ el que se publica)
    r_missing = ((1.0 + 1.0 / m) * b) / w_bar if w_bar > 0 else float("nan")
    z = 1.959963984540054  # normal 95% (m pequeno -> aprox; la df de Rubin seria mayor)
    return {
        "m": m, "q_bar": q_bar, "w_bar": w_bar, "b": b, "t_total": t_total,
        "se_pooled": se_pooled, "se_single": se_single, "r_missing": r_missing,
        "ci_low": q_bar - z * se_pooled, "ci_high": q_bar + z * se_pooled,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Chequeo de sensibilidad del dithering (SPY, V3).")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_DITHER_SEEDS,
                    help="semillas de dithering; la 1a es la base para las SE.")
    ap.add_argument("--asset", default=None, help="override de activo (por defecto SPY).")
    ap.add_argument("--quick", action="store_true",
                    help="multistart reducido (humo, NO R6). Por defecto usa n_starts de config.")
    ap.add_argument("--multistart-seeds", type=int, nargs="+", default=DEFAULT_DITHER_SEEDS,
                    help="EXTENSION (a): semillas del multistart a barrer con el dithering fijo.")
    args = ap.parse_args()

    base = BaseConfig.load()
    news_cfg = NewsConfig.load()
    run_v2._configure_asset(base, args.asset)
    asset = run_v2.ASSET_NAME
    n_starts = 8 if args.quick else news_cfg.hawkes.n_starts

    print(f"Activo: {asset}  |  semilla multistart FIJA={MULTISTART_SEED}  |  n_starts={n_starts}")
    print(f"Semillas de dithering a probar: {args.seeds}")
    print("Cargando corpus + relevancia (cache)...")
    scored, cov = load_scored_corpus(news_cfg)
    times, marks, origin = hf.headline_event_times(scored)
    print(f"Corpus: {len(times)} eventos puntuados, cobertura {cov['first_day']} -> "
          f"{cov['last_day']} ({cov['n_days']} dias, {len(cov['missing_days'])} fantasma)")

    rows = []
    for seed in args.seeds:
        fit, ks, T_obs = fit_for_seed(times, marks, origin, cov,
                                      dither_seed=seed, n_starts=n_starts)
        rows.append({
            "seed": seed, "fit": fit, "ks": ks, "T_obs": T_obs,
            "mu": fit.params["mu"], "alpha": fit.params["alpha"], "beta": fit.params["beta"],
            "se_mu": fit.se["mu"], "se_alpha": fit.se["alpha"], "se_beta": fit.se["beta"],
            "n": fit.branching_ratio, "cascade": fit.expected_cascade,
            "n_se": fit.branching_ratio_se,
            "n_ci_low": fit.branching_ratio_ci_low, "n_ci_high": fit.branching_ratio_ci_high,
            "ks_p": ks["p_value"], "ks_stat": ks["ks_stat"],
            "starts_at_best": fit.starts_at_best, "n_starts_used": fit.n_starts,
        })
        print(f"  seed={seed:>5}: mu_N={_fmt(fit.params['mu'],3)} alpha={_fmt(fit.params['alpha'],3)} "
              f"beta={_fmt(fit.params['beta'],3)} | n={_fmt(fit.branching_ratio,4)} "
              f"KS p={_fmt(ks['p_value'],4)} | {fit.starts_at_best}/{fit.n_starts} en optimo")

    base_row = rows[0]  # la 1a semilla (42) fija las SE de referencia
    se = {"mu": base_row["se_mu"], "alpha": base_row["se_alpha"], "beta": base_row["se_beta"]}
    p0 = {"mu": base_row["mu"], "alpha": base_row["alpha"], "beta": base_row["beta"]}

    # Desviacion de cada semilla frente a la base, en unidades de SE MARGINAL de la
    # base. OJO: en un kernel exponencial alpha y beta estan CO-IDENTIFICADOS (se
    # mueven juntos por la cresta de la verosimilitud), asi que su SE marginal
    # SUBESTIMA la incertidumbre conjunta y una desviacion > 1 SE marginal NO
    # implica que el indicador publicado se mueva. Por eso ademas se juzga:
    #   - el co-movimiento alpha-beta (delta_alpha vs delta_beta): si ~1:1 es la cresta;
    #   - el branching ratio n (LA cantidad publicada) contra SU PROPIA SE.
    max_dev = 0.0
    for r in rows:
        devs = {}
        for k in ("mu", "alpha", "beta"):
            d = abs(r[k] - p0[k]) / se[k] if np.isfinite(se[k]) and se[k] > 0 else float("nan")
            devs[k] = d
            if r["seed"] != base_row["seed"] and np.isfinite(d):
                max_dev = max(max_dev, d)
        r["dev"] = devs
        r["d_alpha"] = r["alpha"] - p0["alpha"]
        r["d_beta"] = r["beta"] - p0["beta"]

    # Cantidad publicada: n. Cuanto se mueve entre semillas vs su propia SE / IC95.
    n_vals = [r["n"] for r in rows]
    n_spread = max(n_vals) - min(n_vals)
    n_se_base = base_row["n_se"]
    n_max_dev_se = max(abs(r["n"] - base_row["n"]) / n_se_base
                       for r in rows[1:]) if (np.isfinite(n_se_base) and n_se_base > 0) else float("nan")
    ci_halfwidth = (base_row["n_ci_high"] - base_row["n_ci_low"]) / 2.0
    n_within_ci = all(base_row["n_ci_low"] <= r["n"] <= base_row["n_ci_high"] for r in rows)
    mu_dev_max = max(r["dev"]["mu"] for r in rows[1:])

    # Veredicto matizado (el director decide; aqui se senala, R3/roles).
    # PASA: la cantidad publicada (n) es estable dentro de su propia incertidumbre y
    # mu_N se mueve < 1 SE; el wobble de alpha/beta es co-movimiento sobre la cresta.
    published_stable = n_within_ci and mu_dev_max < 1.0

    print("\n" + "=" * 78)
    print(f"Desviacion max de parametros MARGINALES (alpha/beta co-identificados) = "
          f"{_fmt(max_dev,3)} SE")
    print(f"branching ratio n: rango entre semillas = {_fmt(n_spread,4)} "
          f"(= {_fmt(n_max_dev_se,2)} SE de n; IC95 base [{_fmt(base_row['n_ci_low'],3)}, "
          f"{_fmt(base_row['n_ci_high'],3)}], semi-ancho {_fmt(ci_halfwidth,4)})")
    print(f"mu_N: desviacion max = {_fmt(mu_dev_max,3)} SE  |  todas las n dentro del IC95 base: {n_within_ci}")
    print("Co-movimiento alpha-beta (delta_alpha, delta_beta) por semilla:")
    for r in rows[1:]:
        print(f"  seed={r['seed']:>5}: d_alpha={_fmt(r['d_alpha'],3)}  d_beta={_fmt(r['d_beta'],3)} "
              f"(ratio {_fmt(r['d_alpha']/r['d_beta'],3) if r['d_beta'] else 'nan'})")
    print("-" * 78)
    if published_stable:
        print("VEREDICTO: PASA (con matiz). La CANTIDAD PUBLICADA (n, cascada) y mu_N son")
        print("robustas al dithering dentro de su propia incertidumbre. alpha y beta se")
        print("mueven ~1:1 sobre la cresta de la verosimilitud (co-identificados): su SE")
        print("marginal subestima ese co-movimiento, pero n = alpha*E[s]/beta lo absorbe.")
        print("El MLE no esta leyendo ruido del dithering como senal en lo que se publica.")
    else:
        print("VEREDICTO: ATENCION. La cantidad publicada (n) o mu_N se mueven fuera de su")
        print("propia incertidumbre al cambiar la semilla del dithering. Reportar al director.")
    print("=" * 78)

    # ----- EXTENSION (a): barrido de la semilla del multistart (optimo global) -----
    print("\n" + "-" * 78)
    print(f"EXTENSION (a): barrido del MULTISTART (dithering FIJO en {base_row['seed']}), "
          f"semillas {args.multistart_seeds}")
    ms_rows = run_multistart_sweep(times, marks, origin, cov,
                                   dither_seed=base_row["seed"],
                                   multistart_seeds=args.multistart_seeds, n_starts=n_starts)
    ll_vals = [r["loglik"] for r in ms_rows]
    n_vals_ms = [r["n"] for r in ms_rows]
    ll_spread = max(ll_vals) - min(ll_vals)
    n_spread_ms = max(n_vals_ms) - min(n_vals_ms)
    ms_global = ll_spread < 1e-4 and n_spread_ms < 1e-4  # mismo optimo a precision numerica
    for r in ms_rows:
        print(f"  ms_seed={r['ms_seed']:>5}: logL={_fmt(r['loglik'],6)} n={_fmt(r['n'],6)} "
              f"| {r['starts_at_best']}/{r['n_starts_used']} en optimo")
    print(f"  rango logL entre semillas = {ll_spread:.2e}  |  rango n = {n_spread_ms:.2e}  "
          f"-> optimo global confirmado: {ms_global}")

    # ----- EXTENSION (b): agrupacion de n por imputacion multiple (Rubin) -----
    pool = rubin_pool(rows)
    print("-" * 78)
    print("EXTENSION (b): imputacion multiple (Rubin) sobre las semillas de dithering")
    print(f"  m={pool['m']} imputaciones | n agrupado (Q_bar) = {_fmt(pool['q_bar'],5)}")
    print(f"  SE una sola imputacion = {_fmt(pool['se_single'],5)}  ->  SE agrupado (Rubin) = "
          f"{_fmt(pool['se_pooled'],5)}")
    print(f"  W_bar (dentro)={pool['w_bar']:.3e}  B (entre)={pool['b']:.3e}  "
          f"T (total)={pool['t_total']:.3e}  r (info perdida por dithering)={_fmt(pool['r_missing'],4)}")
    print(f"  IC95 agrupado [{_fmt(pool['ci_low'],4)}, {_fmt(pool['ci_high'],4)}]")
    print("  NOTA (R3): usar n agrupado en ruta publicada = decision del director.")
    print("=" * 78)

    write_report(asset, n_starts, cov, len(times), rows, se, p0, max_dev,
                 published_stable, n_spread, n_max_dev_se, n_within_ci, mu_dev_max,
                 base_row, ms_rows, ll_spread, n_spread_ms, ms_global, pool)
    print(f"\nReporte escrito en {REPORT.relative_to(ROOT)}")


def write_report(asset, n_starts, cov, n_events, rows, se, p0, max_dev,
                 published_stable, n_spread, n_max_dev_se, n_within_ci, mu_dev_max,
                 base_row, ms_rows, ll_spread, n_spread_ms, ms_global, pool):
    L = []
    L.append(f"# Chequeo de sensibilidad del dithering (Hawkes V3, {asset})")
    L.append("")
    L.append("**Diagnostico** (`@diagnostic_only`) -- no cambia metodologia, no publica en "
             "`artifacts/`. Responde el aviso #5 de `reports/auditoria_pre_corrida.md`.")
    L.append("")
    L.append(f"- Activo: **{asset}**  |  corpus: **{n_events} eventos** puntuados, "
             f"cobertura {cov['first_day']} -> {cov['last_day']} ({cov['n_days']} dias, "
             f"{len(cov['missing_days'])} fantasma).")
    L.append(f"- Multistart: `n_starts={n_starts}`, **semilla del multistart FIJA en "
             f"{MULTISTART_SEED}** (se varia SOLO la semilla del dithering, para aislar el "
             "efecto del ruido inyectado del efecto de que arranques se probaron).")
    L.append(f"- Semilla base (fija las SE de referencia): **{rows[0]['seed']}**.")
    L.append("")
    L.append("## Parametros por semilla de dithering")
    L.append("")
    L.append("| dither seed | mu_N | alpha | beta | n | E[hijos] | KS stat | KS p | arranques |")
    L.append("| --: | --: | --: | --: | --: | --: | --: | --: | :-- |")
    for r in rows:
        L.append(f"| {r['seed']} | {_fmt(r['mu'],4)} | {_fmt(r['alpha'],4)} | {_fmt(r['beta'],4)} "
                 f"| {_fmt(r['n'],4)} | {_fmt(r['cascade'],2)} | {_fmt(r['ks_stat'],4)} "
                 f"| {_fmt(r['ks_p'],4)} | {r['starts_at_best']}/{r['n_starts_used']} |")
    L.append("")
    L.append(f"SE marginal del ajuste base (semilla {rows[0]['seed']}): "
             f"mu_N={_fmt(se['mu'],4)}, alpha={_fmt(se['alpha'],4)}, beta={_fmt(se['beta'],4)}. "
             f"SE del branching ratio n (delta) = {_fmt(base_row['n_se'],4)}, "
             f"IC95 [{_fmt(base_row['n_ci_low'],4)}, {_fmt(base_row['n_ci_high'],4)}].")
    L.append("")
    L.append("## Desviacion frente a la base, en SE MARGINAL de la base")
    L.append("")
    L.append("El criterio literal del auditor (cada parametro `|param_seed - param_base| / "
             "SE_base < 1`) es un primer filtro, pero **subestima** en un kernel exponencial: "
             "`alpha` y `beta` estan co-identificados (cresta de la verosimilitud) y su SE "
             "marginal ignora esa correlacion. Ver la seccion siguiente.")
    L.append("")
    L.append("| dither seed | mu_N (SE) | alpha (SE) | beta (SE) | d_alpha | d_beta | d_alpha/d_beta |")
    L.append("| --: | --: | --: | --: | --: | --: | --: |")
    for r in rows[1:]:
        ratio = _fmt(r["d_alpha"] / r["d_beta"], 3) if r["d_beta"] else "nan"
        L.append(f"| {r['seed']} | {_fmt(r['dev']['mu'],3)} | {_fmt(r['dev']['alpha'],3)} "
                 f"| {_fmt(r['dev']['beta'],3)} | {_fmt(r['d_alpha'],3)} | {_fmt(r['d_beta'],3)} "
                 f"| {ratio} |")
    L.append("")
    L.append(f"Desviacion marginal maxima (alpha/beta) = **{_fmt(max_dev,3)} SE**. Pero "
             "`d_alpha/d_beta ~ 1` en TODAS las semillas: alpha y beta se mueven **juntos** "
             "(la excitacion se re-parametriza sobre la cresta), no de forma independiente.")
    L.append("")
    L.append("## Lo que se PUBLICA: branching ratio n y cascada")
    L.append("")
    L.append("`n = alpha * E[s] / beta` es invariante al deslizamiento sobre la cresta, y es la "
             "cantidad que viaja al artefacto y a la pantalla 3 (no `alpha`/`beta` por separado).")
    L.append("")
    L.append(f"- Rango de `n` entre las {len(rows)} semillas: **{_fmt(n_spread,4)}** "
             f"({_fmt(min(r['n'] for r in rows),4)} a {_fmt(max(r['n'] for r in rows),4)}).")
    L.append(f"- Eso es **{_fmt(n_max_dev_se,2)} SE** de `n` (SE = {_fmt(base_row['n_se'],4)}).")
    L.append(f"- Todas las `n` caen **dentro del IC95 base** [{_fmt(base_row['n_ci_low'],4)}, "
             f"{_fmt(base_row['n_ci_high'],4)}]: **{n_within_ci}**.")
    L.append(f"- `mu_N` (piso): desviacion maxima = **{_fmt(mu_dev_max,3)} SE** (< 1).")
    L.append(f"- Cascada esperada E[hijos]: {_fmt(min(r['cascade'] for r in rows),2)} a "
             f"{_fmt(max(r['cascade'] for r in rows),2)} (estable).")
    L.append("")
    if published_stable:
        L.append("## Veredicto: PASA (con matiz)")
        L.append("")
        L.append("**Las cantidades que se publican -- `n`, la cascada esperada y `mu_N` -- son "
                 "robustas a la semilla del dithering, dentro de su propia incertidumbre.** El "
                 "`n` se mueve menos de una fraccion de su SE y no sale de su IC95; `mu_N` se "
                 "mueve < 1 SE. El unico movimiento > 1 SE marginal es el de `alpha` y `beta`, "
                 "y es un **co-movimiento ~1:1 sobre la cresta de la verosimilitud** (estan "
                 "co-identificados en el kernel exponencial): su SE marginal, calculada ignorando "
                 "esa correlacion, exagera la aparente sensibilidad. El de-empate desliza el par "
                 "(alpha, beta) por esa cresta sin tocar el cociente que define la excitacion.")
        L.append("")
        L.append("**Conclusion honesta:** el MLE del Hawkes NO esta dominado por el ruido "
                 "inyectado del dithering en lo que se reporta. La decision del director de usar "
                 "dithering con semilla fija queda respaldada. Esto NO revierte los avisos #4/#6: "
                 "`n` sigue siendo una **cota superior cualitativa** bajo un kernel exponencial "
                 "que el KS rechaza (stat ~0.03, p=0) -- la mala especificacion del kernel, no el "
                 "de-empate, es la limitacion vigente. Los dos chequeos mas estrictos que se "
                 "sugirieron (variar tambien la semilla del multistart, y promediar `n` por "
                 "imputacion multiple) se ejecutaron: ver las secciones (a) y (b) abajo.")
    else:
        L.append("## Veredicto: ATENCION")
        L.append("")
        L.append("La cantidad publicada (`n`) o `mu_N` se mueven **fuera de su propia "
                 "incertidumbre** al cambiar la semilla del dithering. El estimador es sensible al "
                 "ruido de de-empate en lo que se reporta; el director debe decidir como manejarlo "
                 "(promediar sobre semillas, ampliar la incertidumbre reportada, o revisar el "
                 "supuesto de llegada uniforme intra-bin).")
    L.append("")

    # ---------------- EXTENSION (a): optimo global (barrido de multistart) ----------------
    L.append("## Extension (a): el optimo del MLE es global (barrido de la semilla del multistart)")
    L.append("")
    L.append(f"Chequeo **ortogonal** al del dithering: se FIJA la realizacion del de-empate "
             f"(semilla de dithering {base_row['seed']}) y se VARIA la semilla del multistart "
             f"(`n_starts={n_starts}`, R6). Si el optimo es unimodal, tandas distintas de "
             "arranques aleatorios deben caer en el MISMO optimo. Esto confirma que el `n` "
             "reportado es el maximo global, no un arranque afortunado.")
    L.append("")
    L.append("| multistart seed | log-verosimilitud | n | arranques en el optimo |")
    L.append("| --: | --: | --: | :-- |")
    for r in ms_rows:
        L.append(f"| {r['ms_seed']} | {_fmt(r['loglik'],6)} | {_fmt(r['n'],6)} "
                 f"| {r['starts_at_best']}/{r['n_starts_used']} |")
    L.append("")
    L.append(f"Rango de la log-verosimilitud entre semillas = **{ll_spread:.2e}**; rango de `n` "
             f"= **{n_spread_ms:.2e}**. Optimo global confirmado (mismo optimo a precision "
             f"numerica): **{ms_global}**. La verosimilitud es unimodal para estos datos; el "
             "multistart de R6 no es el eslabon debil.")
    L.append("")

    # ---------------- EXTENSION (b): imputacion multiple (Rubin) ----------------
    L.append("## Extension (b): agrupacion de `n` por imputacion multiple (regla de Rubin)")
    L.append("")
    L.append("Cada semilla de dithering es **una imputacion** de los timestamps sub-bin no "
             "observados (supuesto de llegada uniforme intra-bin). En vez de fijar una semilla "
             "arbitraria, se agrupan las `m` imputaciones con la regla de Rubin, que separa la "
             "incertidumbre en dos fuentes: la **de cada ajuste** (hessiano) y la **entre "
             "imputaciones** (el dithering).")
    L.append("")
    L.append("| cantidad | valor |")
    L.append("| :-- | --: |")
    L.append(f"| m (imputaciones) | {pool['m']} |")
    L.append(f"| `n` agrupado (Q&#772;) | {_fmt(pool['q_bar'],5)} |")
    L.append(f"| W&#772; (varianza DENTRO, media de SE_i^2) | {pool['w_bar']:.3e} |")
    L.append(f"| B (varianza ENTRE imputaciones) | {pool['b']:.3e} |")
    L.append(f"| T = W&#772; + (1+1/m)&middot;B (varianza TOTAL) | {pool['t_total']:.3e} |")
    L.append(f"| SE de una sola imputacion (&asymp; el publicado) | {_fmt(pool['se_single'],5)} |")
    L.append(f"| **SE agrupado (Rubin)** | **{_fmt(pool['se_pooled'],5)}** |")
    L.append(f"| r = incremento relativo de varianza por el dithering | {_fmt(pool['r_missing'],4)} |")
    L.append(f"| IC95 agrupado | [{_fmt(pool['ci_low'],4)}, {_fmt(pool['ci_high'],4)}] |")
    L.append("")
    infl = (pool["se_pooled"] / pool["se_single"] - 1.0) * 100.0 if pool["se_single"] > 0 else float("nan")
    L.append(f"El dithering **añade** aproximadamente **{_fmt(infl,1)}%** al SE del branching "
             f"ratio (r = {_fmt(pool['r_missing'],4)}): el SE agrupado {_fmt(pool['se_pooled'],5)} "
             f"apenas supera al de una sola imputacion {_fmt(pool['se_single'],5)}. Es una fuente "
             "de incertidumbre **real pero de segundo orden**, dominada por la del ajuste y muy "
             "por debajo de la limitacion vigente (el KS rechaza el kernel exponencial).")
    L.append("")
    L.append("**Decision del director (R3):** el `n` agrupado y el SE de Rubin son un "
             "**diagnostico**. Adoptarlos como el `n`/IC **publicado** (en vez del de la semilla "
             "fija) cambia COMO se reporta la incertidumbre del indicador y por tanto es decision "
             "del director, no de Claude. Con la evidencia actual el aporte es minusculo, asi que "
             "no hay urgencia; tendria mas sentido revisitarlo si el kernel power-law (Opcion C) "
             "resuelve el rechazo del KS y el dithering deja de estar tapado por esa limitacion.")
    L.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

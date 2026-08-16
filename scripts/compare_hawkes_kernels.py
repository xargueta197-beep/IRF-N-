"""Comparacion REAL exp vs power-law del kernel de Hawkes sobre una ventana
contigua del corpus de titulares (misma data para ambos = comparacion justa).

Motivacion: el KS rechazo el kernel exponencial (guia 6.6). Se ajustan ambos
kernels sobre la MISMA ventana y se comparan KS, AIC y branching ratio. La ventana
es contigua y de tamano manejable porque el power-law es O(n^2) (inviable exacto
sobre los ~95k eventos completos; se documenta esa restriccion de computo).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irfn.config import BaseConfig, NewsConfig  # noqa: E402
from irfn.data import headlines as H  # noqa: E402
from irfn.features import hawkes_features as hf  # noqa: E402
from irfn.features.relevance import score_headlines  # noqa: E402
from irfn.models import hawkes_mle as EXP  # noqa: E402
from irfn.models import hawkes_powerlaw as PL  # noqa: E402

WIN_START, WIN_END = "2026-05-01", "2026-06-01"
N_STARTS = 20

base = BaseConfig.load()
news = NewsConfig.load()
seed = base.model.seed

hl = H.load_headlines()
scored = score_headlines(hl, model_id=news.relevance.finbert_model, batch_size=news.relevance.batch_size)
ts = pd.to_datetime(scored["hora_titular"]).dt.tz_convert("UTC")
win = scored[(ts >= WIN_START) & (ts < WIN_END)].copy()

times, marks, origin = hf.headline_event_times(win)
times, marks = hf.dither_quantized_times(times, marks, seed=seed)
T = float((pd.Timestamp(WIN_END, tz="UTC") - origin).total_seconds() / 86400.0)
T = max(T, float(times.max()))
print(f"ventana {WIN_START}..{WIN_END}: {len(times)} eventos, T={T:.2f} dias", flush=True)

# --- exponencial (O(n), rapido) ---
t0 = time.time()
fe = EXP.fit_hawkes_mle(times, marks, T=T, n_starts=N_STARTS, seed=seed)
kse = EXP.rescaling_ks(times, marks, fe.params)
aicE = 2 * 3 - 2 * fe.loglik
print(f"[EXP]  {time.time()-t0:.0f}s | mu={fe.params['mu']:.3f} alpha={fe.params['alpha']:.2f} "
      f"beta={fe.params['beta']:.2f} | n={fe.branching_ratio:.4f} | KS p={kse['p_value']:.4g} | "
      f"logL={fe.loglik:.1f} AIC={aicE:.1f} | {fe.starts_at_best}/{fe.n_starts}", flush=True)

# --- power-law (O(n^2), numba) ---
_ = PL._pl_excess(times[:50].copy(), marks[:50].copy(), 0.05, 1.8)  # warmup JIT
t0 = time.time()
fp = PL.fit_powerlaw_mle(times, marks, T=T, n_starts=N_STARTS, seed=seed)
ksp = PL.pl_rescaling_ks(times, marks, fp.params)
print(f"[PL]   {time.time()-t0:.0f}s | mu={fp.params['mu']:.3f} alpha={fp.params['alpha']:.4f} "
      f"c={fp.params['c']:.4f} theta={fp.params['theta']:.4f} | n={fp.branching_ratio:.4f} | "
      f"KS p={ksp['p_value']:.4g} | logL={fp.loglik:.1f} AIC={fp.aic:.1f} | "
      f"{fp.starts_at_best}/{fp.n_starts} hess_ok={fp.hessian_ok}", flush=True)

print("\n=== VEREDICTO (misma ventana, comparacion justa) ===", flush=True)
print(f"KS: exp p={kse['p_value']:.4g} vs power-law p={ksp['p_value']:.4g}", flush=True)
print(f"AIC: exp {aicE:.1f} vs power-law {fp.aic:.1f}  (menor = mejor)  dAIC={aicE-fp.aic:+.1f}", flush=True)
print(f"branching n: exp {fe.branching_ratio:.4f} vs power-law {fp.branching_ratio:.4f}", flush=True)
mejor_ks = "power-law" if (ksp["p_value"] or 0) > (kse["p_value"] or 0) else "exponencial"
mejor_aic = "power-law" if fp.aic < aicE else "exponencial"
print(f"-> KS favorece: {mejor_ks} | AIC favorece: {mejor_aic}", flush=True)
print(f"-> power-law {'PASA' if ksp['passed'] else 'NO pasa'} el KS (p>0.05); "
      f"exponencial {'PASA' if kse['passed'] else 'NO pasa'}.", flush=True)

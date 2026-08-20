"""FASE H -- Reconciliacion codigo <-> artefacto <-> panel (run 3b4f1e39b59c).

H.1 Recalculo de numeros publicados desde el artefacto en disco:
    - E[D] por regimen = 1/(1 - P_ii) desde transition_matrix_today
    - entropia de xi_filtered de hoy (y entropy_max = ln K)
    - n y cascada desde (alpha, beta, mean_mark) [ya verificado en D2; se repite]
    - vol_ann del regimen 'baja volatilidad' desde history.parquet (r, argmax)
      vs conditional_stats publicado
H.2 Sincronia run_id/asof entre irfn.json, manifest.json, validation.json del
    panel (validates_run_id / published_run_id / stale) y panel/irfn.json.
H.3 Etiquetas de version/modelo del panel = M1 (tvtp=false, covariates=[]).
H.4 R9 se verifica aparte con grep sobre panel/ (aritmetica en la interfaz).
H.5 Desfase documental CLAUDE.md/ESTADO: verificado aparte por git log.
+   no look-ahead: ultima fecha de history <= asof.

Resultados en audits/probes/out/h_results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
RES: dict = {}
FAILS = []


def record(name, ok, detail=None):
    RES[name] = {"ok": bool(ok), **({"detail": detail} if detail is not None else {})}
    if not ok:
        FAILS.append(name)
    print(f"[{name}] {'OK' if ok else 'FALLA'}" + (f" {detail}" if detail is not None else ""), flush=True)


art = json.loads((ROOT / "artifacts/latest/irfn.json").read_text(encoding="utf-8"))
man = json.loads((ROOT / "artifacts/latest/manifest.json").read_text(encoding="utf-8"))
hist = pd.read_parquet(ROOT / "artifacts/latest/history.parquet")
pan_irfn = json.loads((ROOT / "panel/public/data/irfn.json").read_text(encoding="utf-8"))
pan_hist = json.loads((ROOT / "panel/public/data/history.json").read_text(encoding="utf-8"))
pan_val = json.loads((ROOT / "panel/public/data/validation.json").read_text(encoding="utf-8"))

# --- H.2 sincronia -----------------------------------------------------------
record("H2_runid_artefacto_manifest", art["run_id"] == man["run_id"],
       {"irfn": art["run_id"], "manifest": man["run_id"]})
record("H2_runid_panel_irfn", pan_irfn.get("run_id") == art["run_id"],
       {"panel": pan_irfn.get("run_id")})
record("H2_validacion_panel",
       pan_val.get("published_run_id") == art["run_id"] and pan_val.get("stale") is False,
       {"validates": pan_val.get("validates_run_id"), "published": pan_val.get("published_run_id"),
        "stale": pan_val.get("stale")})
record("H2_asof_panel", pan_irfn.get("asof") == art["asof"],
       {"panel": pan_irfn.get("asof"), "art": art["asof"]})

# --- H.3 etiquetas M1 --------------------------------------------------------
m = pan_irfn["model"]
record("H3_panel_es_M1", m["tvtp"] is False and m["covariates"] == [] and m["K"] == 2,
       {"tvtp": m["tvtp"], "covariates": m["covariates"], "K": m["K"],
        "version": pan_irfn.get("version")})

# --- H.1 recalculos ----------------------------------------------------------
P = np.array(art["transition_matrix_today"])
edd_re = [float(1.0 / (1.0 - P[k, k])) for k in range(2)]
edd_pub = art["regime"]["expected_duration_days"]
record("H1_expected_duration", bool(np.allclose(edd_re, edd_pub, rtol=1e-9)),
       {"recalc": edd_re, "pub": edd_pub})

xi = np.array(art["regime"]["xi_filtered"])
Hent = float(-np.sum(xi * np.log(np.clip(xi, 1e-300, None))))
record("H1_entropia_hoy", abs(Hent - art["regime"]["entropy"]) < 1e-9,
       {"recalc": Hent, "pub": art["regime"]["entropy"]})
record("H1_entropy_max", abs(art["regime"]["entropy_max"] - np.log(2)) < 1e-12)

hk = art["model"]["hawkes_layer_params"]
n_re = hk["alpha"] * hk["mean_mark"] / hk["beta"]
record("H1_branching_ratio", abs(n_re - art["news"]["branching_ratio"]) < 1e-12,
       {"recalc": n_re})
record("H1_cascada", abs(1.0 / (1.0 - n_re) - art["news"]["expected_cascade"]) < 1e-12)

# vol_ann del regimen 0 desde history.parquet vs conditional_stats
asset = list(art["conditional_stats"].keys())[0]
lab0 = art["regime"]["labels"][0]
r_dec = hist["r"].to_numpy() / 100.0
sel0 = hist["argmax_idx"].to_numpy() == 0
vol_ann_re = float(r_dec[sel0].std(ddof=1) * np.sqrt(252))
vol_pub = art["conditional_stats"][asset][lab0]["vol_ann"]["value"]
record("H1_vol_ann_regimen0", abs(vol_ann_re - vol_pub) / vol_pub < 0.02,
       {"recalc": vol_ann_re, "pub": vol_pub,
        "nota": "tolerancia 2%: el punto publicado es la media bootstrap? no: es el puntual; ddof puede diferir"})

# panel history vs parquet (ultima fila)
last_pq = hist.iloc[-1]
last_pj = pan_hist[-1] if isinstance(pan_hist, list) else pan_hist["rows"][-1]
keys = [k for k in ("xi_filtered_1", "entropy_norm") if k in last_pj]
ok_hist = str(last_pj.get("fecha", last_pj.get("date")))[:10] == str(last_pq["fecha"])[:10] and all(
    abs(float(last_pj[k]) - float(last_pq[k])) < 1e-9 for k in keys)
record("H1_panel_history_ultima_fila", ok_hist,
       {"panel": {k: last_pj.get(k) for k in ['fecha'] + keys}, "parquet_fecha": str(last_pq['fecha'])})

# no look-ahead
record("H_no_lookahead_history", str(hist["fecha"].max())[:10] <= art["asof"],
       {"history_end": str(hist["fecha"].max())[:10], "asof": art["asof"]})

(OUT / "h_results.json").write_text(json.dumps(RES, indent=2, default=float), encoding="utf-8")
print(f"FASE H (probe) completa. Fallos: {len(FAILS)} {FAILS}", flush=True)

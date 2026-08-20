"""Capa cripto-nativa BTC -- GARCH-X con drivers exogenos en la varianza.

Pre-registro: reports/diseno_capa_cripto_btc_2026-08-19.md (director 2026-08-19).
Pregunta por driver: ¿mejora la densidad predictiva de la varianza de BTC sobre
el baseline K=1 GJR-GARCH-t?

Drivers y su TRANSFORM pre-especificado (rezagado R3, estandarizado con media 0
para identificar theta ortogonal a omega):
  - volume  : log(volumen)                 (liquidez; nivel)
  - fng     : |FearGreed - 50|             (sentimiento; EXTREMIDAD: miedo Y codicia
                                            extremos preceden turbulencia)
  - funding : |funding rate|               (apalancamiento; magnitud = riesgo de
                                            liquidacion en cualquier direccion)

Uso:
    python scripts/run_garchx_btc.py --driver fng           # in-sample
    python scripts/run_garchx_btc.py --driver fng --wf      # + walk-forward OOS
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irfn.config import BaseConfig  # noqa: E402
from irfn.data import prices as P  # noqa: E402
from irfn.models import garchx as G  # noqa: E402

RAW = ROOT / "data" / "raw"


def _btc_meta():
    import yaml
    assets = yaml.safe_load((ROOT / "config" / "assets.yaml").read_text(encoding="utf-8"))["assets"]
    btc = next(a for a in assets if a["name"] == "BTC")
    return btc["symbol"], btc["start_date"]


def _fetch_fng() -> pd.Series:
    """Fear & Greed Index diario (alternative.me), 2018->hoy. 0=miedo, 100=codicia."""
    cache = RAW / "fng_alternative_me.parquet"
    if cache.exists():
        s = pd.read_parquet(cache)["fng"]; s.index = pd.to_datetime(s.index); return s
    d = json.loads(urllib.request.urlopen(
        "https://api.alternative.me/fng/?limit=0&format=json", timeout=30).read())
    ts = [pd.to_datetime(int(e["timestamp"]), unit="s") for e in d["data"]]
    val = [float(e["value"]) for e in d["data"]]
    s = pd.Series(val, index=pd.DatetimeIndex(ts), name="fng").sort_index()
    s = s[~s.index.duplicated(keep="last")]
    cache.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(cache)
    return s


def _fetch_funding(symbol: str) -> pd.Series:
    """Funding rate de perpetuos de Binance (fapi), agregado a diario (suma de los
    3 fundings de 8h). ~2019/2020->hoy. Sin API key."""
    cache = RAW / f"funding_{symbol}.parquet"
    if cache.exists():
        s = pd.read_parquet(cache)["funding"]; s.index = pd.to_datetime(s.index); return s
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    rows, start_ms = [], int(pd.Timestamp("2019-09-01", tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    import time as _t
    while start_ms < end_ms:
        q = f"{url}?symbol={symbol}&startTime={start_ms}&limit=1000"
        page = json.loads(urllib.request.urlopen(q, timeout=30).read())
        if not page:
            break
        rows.extend(page)
        nxt = int(page[-1]["fundingTime"]) + 1
        if nxt <= start_ms:
            break
        start_ms = nxt
        if len(page) < 1000:
            break
        _t.sleep(0.2)
    idx = pd.to_datetime([int(r["fundingTime"]) for r in rows], unit="ms")
    fr = pd.Series([float(r["fundingRate"]) for r in rows], index=idx, name="funding")
    daily = fr.groupby(fr.index.normalize()).sum()
    daily.name = "funding"
    cache.parent.mkdir(parents=True, exist_ok=True)
    daily.to_frame().to_parquet(cache)
    return daily


def _load(driver: str):
    symbol, start = _btc_meta()
    close = P.load_close(symbol, start, source="binance")
    r = 100.0 * np.log(close / close.shift(1))
    if driver == "volume":
        raw = np.log(P.load_volume(symbol, start))
    elif driver == "fng":
        raw = (_fetch_fng() - 50.0).abs()          # extremidad
    elif driver == "funding":
        raw = _fetch_funding(symbol).abs()          # magnitud
    else:
        raise ValueError(driver)
    raw.name = "driver"
    df = pd.DataFrame({"r": r}).join(raw, how="inner").dropna()
    return df, symbol


def _standardize_lag(raw: np.ndarray, mu: float, sd: float) -> np.ndarray:
    xs = (raw - mu) / sd
    xl = np.empty_like(xs); xl[0] = 0.0; xl[1:] = xs[:-1]      # R3: x_{t-1}
    return xl


def _fit_pair(r, x_lag, seed, n_starts):
    b = G.fit_garchx(r, None, n_starts=n_starts, seed=seed, with_exog=False, compute_se=False)
    x = G.fit_garchx(r, x_lag, n_starts=n_starts, seed=seed, with_exog=True)
    return b, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", choices=["volume", "fng", "funding"], required=True)
    ap.add_argument("--wf", action="store_true")
    ap.add_argument("--starts", type=int, default=20)
    args = ap.parse_args()

    df, symbol = _load(args.driver)
    r = df["r"].to_numpy()
    raw = df["driver"].to_numpy()
    T = len(r)
    print(f"BTC {symbol} driver={args.driver}: {T} obs, {df.index[0].date()} -> {df.index[-1].date()}")

    # ---- FULL-SAMPLE ----
    x_lag = _standardize_lag(raw, raw.mean(), raw.std())
    base, xfit = _fit_pair(r, x_lag, seed=42, n_starts=args.starts)
    theta, se = xfit.params["theta"], xfit.se["theta"]
    tstat = theta / se if se and np.isfinite(se) else float("nan")
    LR = 2.0 * (xfit.loglik - base.loglik)
    p_lr = float(chi2.sf(max(LR, 0.0), df=1))
    print("\n=== FULL-SAMPLE (in-sample) ===")
    print(f"  baseline logL = {base.loglik:.2f}")
    print(f"  garchx   logL = {xfit.loglik:.2f}   (theta={theta:+.4f}, SE={se:.4f}, t={tstat:+.2f})")
    print(f"  LR (chi2_1) = {LR:.2f}  p = {p_lr:.4f}  -> "
          f"{'theta SIGNIFICATIVO' if abs(tstat) > 2 else 'theta NO signif'}, "
          f"{'LR rechaza baseline' if p_lr < 0.05 else 'LR NO rechaza'}")

    if not args.wf:
        print("\n(sin --wf: falta el veredicto OOS, que es el que decide, R8.)")
        return

    wf = BaseConfig.load().walkforward
    train_n = int(wf.train_years * 365.25); test_n = int(wf.test_months * 30.44)
    lb_all, lx_all = [], []
    start, blk = train_n, 0
    while start + test_n <= T and blk < max(wf.n_blocks, 6) + 8:
        tr = slice(0, start); te = slice(start, start + test_n)
        xl = _standardize_lag(raw, raw[tr].mean(), raw[tr].std())   # estandariza con TRAIN
        rb, rx = _fit_pair(r[tr], xl[tr], seed=42, n_starts=max(8, args.starts // 2))
        lb_all.append(G.predictive_logscore(rb.params, r, np.zeros(T))[te])
        lx_all.append(G.predictive_logscore(rx.params, r, xl)[te])
        blk += 1; start += test_n
    lb = np.concatenate(lb_all); lx = np.concatenate(lx_all)
    d = lx - lb
    dm = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    p_dm = 2.0 * (1.0 - norm.cdf(abs(dm)))
    print(f"\n=== WALK-FORWARD OOS ({blk} bloques, n_oos={len(d)}) ===")
    print(f"  log-score medio: baseline {lb.mean():.4f}  garchx {lx.mean():.4f}  dif {d.mean():+.5f}")
    print(f"  Diebold-Mariano = {dm:+.2f}  p = {p_dm:.4f}")
    print(f"  VEREDICTO OOS ({args.driver}): {'MEJORA' if (d.mean() > 0 and p_dm < 0.05) else 'NO mejora'} "
          "(criterio pre-registrado: DM>0 y p<0.05).")


if __name__ == "__main__":
    main()

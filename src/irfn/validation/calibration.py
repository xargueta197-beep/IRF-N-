"""Brier score, log-loss, reliability diagram sobre xi_predicted (no filtrada).

--------------------------------------------------------------------------------
Por que se evalua ξ_{t|t-1} (PREDICHA) y NUNCA ξ_{t|t} (FILTRADA)
--------------------------------------------------------------------------------
La calibracion pregunta: "cuando el modelo dijo 70% de probabilidad de regimen k
ANTES de ver r_t, acerto el 70% de las veces?". La prediccion honesta es la que
se hace con informacion hasta t-1: ξ_{t|t-1}. La FILTRADA ξ_{t|t} ya incorporo
r_t; evaluarla es calificarse el examen con la respuesta ya vista -> siempre se
vera bien calibrada por construccion, y no mide capacidad predictiva alguna. Por
eso todas las funciones de este modulo reciben ξ_PREDICHA.

--------------------------------------------------------------------------------
El objetivo "realizado" cuando el regimen es LATENTE (honestidad, no truco)
--------------------------------------------------------------------------------
El regimen S_t no se observa: no hay etiqueta verdadera. Como proxy del estado
realizado usamos y_t = argmax_k ξ_{t|t}(k), la mejor lectura del estado que
permite TODA la informacion hasta t (incluida r_t). Es un proxy, no la verdad, y
hay que decirlo: mide consistencia entre lo que el modelo predijo ayer para hoy y
lo que concluyo hoy con el dato de hoy. No es prueba de que los regimenes existan.

Como complemento SIN etiquetas latentes se ofrece `pit_returns`: la transformada
integral de probabilidad de los retornos bajo la densidad predictiva de la mezcla.
Si el modelo predictivo esta bien calibrado, esos valores son Uniforme(0,1). Esa
calibracion es sobre algo OBSERVABLE (el retorno), sin proxies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _as_2d(xi_pred: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi_pred, dtype=float)
    if xi.ndim != 2:
        raise ValueError("xi_pred debe ser (T, K).")
    return xi


def onehot_from_filtered(xi_filtered: np.ndarray) -> np.ndarray:
    """y_t one-hot del estado realizado-proxy = argmax_k ξ_{t|t}(k). Ver docstring
    del modulo: es un PROXY del estado latente, no la verdad."""
    xi = _as_2d(xi_filtered)
    T, K = xi.shape
    y = np.zeros((T, K))
    y[np.arange(T), xi.argmax(axis=1)] = 1.0
    return y


def brier_score(xi_pred: np.ndarray, y_onehot: np.ndarray) -> float:
    """Brier multiclase = (1/T) Σ_t Σ_k (ξ_{t|t-1}(k) - y_t(k))². Menor es mejor;
    0 = prediccion perfecta, y su rango es [0, 2]. Baseline de referencia: predecir
    siempre las frecuencias marginales (ver `brier_baseline`)."""
    p = _as_2d(xi_pred)
    y = _as_2d(y_onehot)
    return float(((p - y) ** 2).sum(axis=1).mean())


def brier_baseline(y_onehot: np.ndarray) -> float:
    """Brier de la prediccion climatologica: la frecuencia marginal constante de
    cada clase. Es el listo minimo que hay que batir para que la prediccion
    condicional aporte algo."""
    y = _as_2d(y_onehot)
    freq = y.mean(axis=0, keepdims=True)          # (1, K)
    return float(((freq - y) ** 2).sum(axis=1).mean())


def log_loss(xi_pred: np.ndarray, y_onehot: np.ndarray, eps: float = 1e-15) -> float:
    """Log-loss multiclase = -(1/T) Σ_t Σ_k y_t(k) log ξ_{t|t-1}(k). Menor es
    mejor. Se recorta la probabilidad a [eps, 1] para no explotar en log 0."""
    p = np.clip(_as_2d(xi_pred), eps, 1.0)
    y = _as_2d(y_onehot)
    return float(-(y * np.log(p)).sum(axis=1).mean())


def log_loss_baseline(y_onehot: np.ndarray, eps: float = 1e-15) -> float:
    """Log-loss de la prediccion climatologica (frecuencias marginales)."""
    y = _as_2d(y_onehot)
    freq = np.clip(y.mean(axis=0, keepdims=True), eps, 1.0)
    return float(-(y * np.log(freq)).sum(axis=1).mean())


def reliability_curve(
    xi_pred: np.ndarray, y_onehot: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Datos del diagrama de fiabilidad de la clase mas probable (top-label).

    Para cada dia se toma la probabilidad predicha del argmax (la "confianza") y
    si esa clase resulto ser la realizada-proxy. Se agrupa la confianza en n_bins
    y, por bin, se reporta confianza media vs. acierto empirico. Bien calibrado =>
    ambos coinciden (la diagonal). El GRAFICO se dibuja en la app; aqui solo datos
    (R9: la interfaz no calcula, solo lee).
    """
    p = _as_2d(xi_pred)
    y = _as_2d(y_onehot)
    conf = p.max(axis=1)
    pred_cls = p.argmax(axis=1)
    true_cls = y.argmax(axis=1)
    hit = (pred_cls == true_cls).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize: bin index 1..n_bins; clip para incluir conf==1.0 en el ultimo.
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        sel = idx == b
        count = int(sel.sum())
        rows.append(
            {
                "bin_lo": edges[b],
                "bin_hi": edges[b + 1],
                "bin_mid": 0.5 * (edges[b] + edges[b + 1]),
                "mean_confidence": float(conf[sel].mean()) if count else np.nan,
                "empirical_accuracy": float(hit[sel].mean()) if count else np.nan,
                "count": count,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    xi_pred: np.ndarray, y_onehot: np.ndarray, n_bins: int = 10
) -> float:
    """ECE = Σ_bins (n_bin/T) |confianza_media - acierto_empirico|. Resumen escalar
    de la curva de fiabilidad: 0 = perfectamente calibrado."""
    curve = reliability_curve(xi_pred, y_onehot, n_bins)
    valid = curve.dropna(subset=["mean_confidence", "empirical_accuracy"])
    total = valid["count"].sum()
    if total == 0:
        return float("nan")
    w = valid["count"] / total
    return float((w * (valid["mean_confidence"] - valid["empirical_accuracy"]).abs()).sum())


def pit_returns(
    r: np.ndarray,
    xi_pred: np.ndarray,
    mu: np.ndarray,
    sigma2: np.ndarray,
) -> np.ndarray:
    """Transformada integral de probabilidad de r_t bajo la densidad predictiva
    de la mezcla: F_t(r_t) = Σ_k ξ_{t|t-1}(k) Φ((r_t - μ_k)/σ_{k,t}).

    Si el modelo predictivo esta bien calibrado, {F_t} ~ Uniforme(0,1). Esta es la
    calibracion SIN etiquetas latentes (sobre el retorno observado). σ² debe ser
    la varianza condicional PREDICHA con info hasta t-1, que en la especificacion
    Haas es σ²_{k,t} = f(r_{t-1}, ...) -> ya es predecible en t-1.
    """
    r = np.asarray(r, dtype=float)
    p = _as_2d(xi_pred)
    mu = np.asarray(mu, dtype=float)[None, :]
    sd = np.sqrt(np.asarray(sigma2, dtype=float))
    cdf = norm.cdf((r[:, None] - mu) / sd)          # (T, K)
    return (p * cdf).sum(axis=1)


def summarize(xi_pred: np.ndarray, xi_filtered: np.ndarray, n_bins: int = 10) -> dict:
    """Resumen de calibracion listo para el reporte y el artefacto de validacion.

    Usa el proxy y_t = argmax ξ_{t|t}. Devuelve Brier y log-loss del modelo y de
    la climatologia (para saber si el condicional aporta), ECE, y la curva de
    fiabilidad como lista de dicts.
    """
    y = onehot_from_filtered(xi_filtered)
    curve = reliability_curve(xi_pred, y, n_bins)
    return {
        "brier": brier_score(xi_pred, y),
        "brier_baseline": brier_baseline(y),
        "log_loss": log_loss(xi_pred, y),
        "log_loss_baseline": log_loss_baseline(y),
        "ece": expected_calibration_error(xi_pred, y, n_bins),
        "reliability_curve": curve.to_dict(orient="records"),
        "n_obs": int(len(xi_pred)),
        "target_note": (
            "y_t = argmax ξ_{t|t} es un PROXY del estado latente, no la verdad. "
            "La calibracion mide consistencia predicho(t|t-1) vs. concluido(t|t)."
        ),
    }

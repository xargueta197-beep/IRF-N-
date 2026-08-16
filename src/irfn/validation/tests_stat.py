"""Tests estadisticos de V1: seleccion de K, Diebold-Mariano, Pesaran-Timmermann.

--------------------------------------------------------------------------------
Seleccion de K: por que el LR estandar NO sirve y que se usa en su lugar
--------------------------------------------------------------------------------
Bajo H0: K-1 regimenes, los parametros del regimen extra del modelo alternativo
NO estan identificados (con K=1, la fila de transicion y el GARCH del "segundo
regimen" pueden ser cualquier cosa sin cambiar la verosimilitud nula). Es el
problema de Davies (1977, 1987) de parametros de molestia presentes solo bajo la
alternativa: la teoria asintotica clasica se cae y 2*(llK - llK-1) NO es chi2.
Usar el LR con p-values de chi2 aqui daria significancia inventada.

Alternativas correctas:
  * Hansen (1992): cota superior para el LR estandarizado tomando supremo sobre
    una malla del espacio de parametros de molestia, con el proceso empirico
    evaluado en cada punto. Correcto, pero exige una malla multidimensional
    (mu, v, P del regimen extra) y una optimizacion por punto: para un
    MS-GJR-GARCH con multistart (R6) el costo es prohibitivo -- una corrida
    honesta seria dias de CPU, no horas.
  * BOOTSTRAP PARAMETRICO (lo que se usa AQUI, documentado como exige la guia):
    se ajustan el nulo y el alternativo sobre los datos -> LR_obs; se simulan B
    series desde el modelo NULO ajustado (mismo T); en cada replica se ajustan
    ambos modelos y se calcula LR_b. El p-value es
        p = (1 + #{LR_b >= LR_obs}) / (B + 1),
    que es exacto en nivel bajo H0 salvo error de Monte Carlo (Davidson &
    MacKinnon). La distribucion nula del LR se APRENDE simulando, en lugar de
    suponer una chi2 que no aplica.
    Costo asumido y reportado: cada replica re-estima con multistart REDUCIDO
    (config v1.ktest.boot_n_starts); eso puede subestimar levemente el optimo de
    cada replica en ambos modelos. Se reporta ademas cuantas replicas fallaron.

--------------------------------------------------------------------------------
Diebold-Mariano: sobre la PERDIDA PREDICTIVA, no sobre el ajuste
--------------------------------------------------------------------------------
Compara dos series de perdidas out-of-sample EMPAREJADAS por fecha; aqui la
perdida es -loglik predictiva por observacion (el denominador de Hamilton en el
tramo de test del walk-forward, jamas el ajuste in-sample). Varianza HAC
(Newey-West) del diferencial y correccion de muestra pequena de Harvey,
Leybourne & Newbold (1997), con p-value de la t de Student.

Pesaran-Timmermann (1992): precision direccional. Evalua si sign(prediccion) y
sign(realizado) estan asociados mas alla de lo que dictan las frecuencias
marginales de cada signo. Estadistico asintoticamente N(0,1).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy import stats

from irfn.models.estimate import fit
from irfn.models.msgarch import simulate
from irfn.models.params import n_params
from irfn.validation import calibration as _cal
from irfn.validation import checkpoint as _ckpt

logger = logging.getLogger("irfn.tests_stat")


# --------------------------------------------------------------------------- #
# Seleccion de K
# --------------------------------------------------------------------------- #
def bic_table(
    r: np.ndarray,
    *,
    k_candidates: list[int],
    dists: list[str],
    n_starts: int,
    seed: int,
    checkpoint_path: Path | None = None,
) -> list[dict]:
    """Tabla COMPLETA de BIC para cada (K, dist). No se esconden los perdedores:
    la tabla entera va al reporte y al artefacto de validacion.

    El BIC es un criterio IN-SAMPLE (verosimilitud sobre la muestra de ajuste
    penalizada por parametros); la seleccion definitiva se contrasta ademas
    fuera de muestra en la ablacion walk-forward (la apuesta K=3 vs K=4 del
    director se dirime alli, no aqui).

    Si `checkpoint_path` se da, cada fila (K, dist) ya computada se guarda de
    inmediato (validation/checkpoint.py); una corrida interrumpida reanuda sin
    repetir las filas ya calculadas (mismos k_candidates/dists/n_starts/seed/r;
    si algo no calza, el checkpoint se ignora y arranca de cero).
    """
    sig = (tuple(k_candidates), tuple(dists), n_starts, seed, r.shape[0], float(r.sum()))
    done: dict[tuple[int, str], dict] = {}
    if checkpoint_path is not None:
        payload = _ckpt.load(checkpoint_path, expected_sig=sig)
        if payload is not None:
            done = payload["rows"]
            logger.info("bic_table: reanudando, %d/%d filas ya en checkpoint.",
                        len(done), len(k_candidates) * len(dists))

    rows: list[dict] = []
    for dist in dists:
        for K in k_candidates:
            key = (K, dist)
            if key in done:
                rows.append(done[key])
                continue
            fr = fit(r, K, n_starts=n_starts, seed=seed, compute_se=False, dist=dist)
            row = {
                "K": K,
                "dist": dist,
                "loglik": fr.loglik,
                "k_free": n_params(K, dist=dist),
                "bic": fr.bic,
                "aic": fr.aic,
                "n_converged": fr.n_converged,
                "n_starts": fr.n_starts,
                "nu": fr.params["nu"].tolist() if dist == "t" else None,
            }
            rows.append(row)
            done[key] = row
            logger.info(
                "BIC: K=%d dist=%s loglik=%.2f bic=%.2f (%d/%d arranques en el optimo)",
                K, dist, fr.loglik, fr.bic, fr.n_converged, fr.n_starts,
            )
            if checkpoint_path is not None:
                _ckpt.save(checkpoint_path, {"sig": sig, "rows": done})
    return rows


def bootstrap_lr_test(
    r: np.ndarray,
    *,
    K_null: int,
    K_alt: int,
    dist: str,
    n_boot: int,
    n_starts_data: int,
    n_starts_boot: int,
    seed: int,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 5,
) -> dict:
    """Test de numero de regimenes K_alt vs K_null por BOOTSTRAP PARAMETRICO.

    Ver el docstring del modulo para por que NO se usa el LR con chi2 (Davies /
    parametros no identificados bajo H0) y por que bootstrap en vez del test de
    Hansen (1992) exacto (costo computacional; decision documentada, guia P8).

    Devuelve dict con lr_obs, replicas, p_value y contadores de fallos. La
    semilla gobierna datos->modelo->replicas: reproducible de punta a punta.

    Si `checkpoint_path` se da, el progreso de las replicas (la parte cara: 49
    replicas x hasta 2 ajustes c/u) se guarda cada `checkpoint_every` replicas
    (validation/checkpoint.py); una corrida interrumpida reanuda desde la
    replica siguiente a la ultima guardada, con el MISMO seed+b por replica
    (reproducible bit a bit, R6). `fit_null`/`fit_alt` (los ajustes obreservados,
    baratos frente al loop) siempre se recalculan.
    """
    if K_alt <= K_null:
        raise ValueError("K_alt debe ser > K_null.")
    r = np.asarray(r, dtype=float)
    T = r.shape[0]

    fit_null = fit(r, K_null, n_starts=n_starts_data, seed=seed, compute_se=False, dist=dist)
    fit_alt = fit(r, K_alt, n_starts=n_starts_data, seed=seed, compute_se=False, dist=dist)
    lr_obs = 2.0 * (fit_alt.loglik - fit_null.loglik)

    from irfn.models.params import pack

    theta_null = pack(fit_null.params, K_null, dist=dist)

    sig = (K_null, K_alt, dist, n_boot, n_starts_boot, seed, T, float(r.sum()))
    lr_boot: list[float] = []
    n_failed = 0
    b_start = 0
    if checkpoint_path is not None:
        payload = _ckpt.load(checkpoint_path, expected_sig=sig)
        if payload is not None:
            lr_boot = payload["lr_boot"]
            n_failed = payload["n_failed"]
            b_start = payload["n_done"]
            logger.info("bootstrap K=%d vs %d: reanudando desde replica %d/%d.",
                        K_alt, K_null, b_start, n_boot)

    for b in range(b_start, n_boot):
        r_b, _ = simulate(theta_null, K_null, T=T, seed=seed + 1000 + b, dist=dist)
        try:
            f0 = fit(r_b, K_null, n_starts=n_starts_boot, seed=seed + b, compute_se=False, dist=dist)
            f1 = fit(r_b, K_alt, n_starts=n_starts_boot, seed=seed + b, compute_se=False, dist=dist)
        except RuntimeError:
            n_failed += 1
        else:
            # El LR de una replica puede salir levemente negativo si el multistart
            # reducido no encuentra el optimo del modelo grande; se recorta a 0 (el
            # LR verdadero es >= 0 porque los modelos estan anidados). Recortar
            # ENDURECE el test contra el alternativo, no lo ablanda.
            lr_boot.append(max(0.0, 2.0 * (f1.loglik - f0.loglik)))
        n_done = b + 1
        if checkpoint_path is not None and (n_done % checkpoint_every == 0 or n_done == n_boot):
            _ckpt.save(checkpoint_path, {
                "sig": sig, "lr_boot": lr_boot, "n_failed": n_failed, "n_done": n_done,
            })

    lr_arr = np.array(lr_boot)
    n_eff = len(lr_boot)
    p_value = float((1 + int((lr_arr >= lr_obs).sum())) / (n_eff + 1)) if n_eff else float("nan")

    if n_failed:
        logger.warning(
            "bootstrap K=%d vs %d: %d/%d replicas fallaron al estimar.",
            K_alt, K_null, n_failed, n_boot,
        )

    return {
        "K_null": K_null,
        "K_alt": K_alt,
        "dist": dist,
        "lr_obs": float(lr_obs),
        "loglik_null": float(fit_null.loglik),
        "loglik_alt": float(fit_alt.loglik),
        "n_boot": n_boot,
        "n_boot_ok": n_eff,
        "n_boot_failed": n_failed,
        "p_value": p_value,
        "lr_boot_q50": float(np.median(lr_arr)) if n_eff else float("nan"),
        "lr_boot_q95": float(np.quantile(lr_arr, 0.95)) if n_eff else float("nan"),
        "method": (
            "bootstrap parametrico (LR estandar invalido: Davies/parametros no "
            "identificados bajo H0; Hansen 1992 exacto descartado por costo, "
            "decision documentada en validation/tests_stat.py)"
        ),
    }


# --------------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------------- #
def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    h: int = 1,
    n_lags: int | None = None,
) -> dict:
    """DM sobre dos series de perdidas EMPAREJADAS (misma fecha, mismo objetivo).

    d_t = loss_a_t - loss_b_t. H0: E[d_t] = 0. DM < 0 => el modelo A (el primero)
    pierde MENOS (mejor). Varianza HAC de Newey-West con n_lags rezagos (default:
    floor(1.5 * T^(1/3)), la regla clasica; para pronosticos a 1 paso el
    diferencial deberia ser ~no autocorrelado, pero el HAC protege si no lo es).
    Correccion de muestra pequena de Harvey-Leybourne-Newbold para horizonte `h`
    y p-value con t de Student (T-1 gl), como recomienda HLN.

    Claves del dict: `statistic` y `dm_stat` son el mismo estadistico (el segundo
    se conserva por compatibilidad con validation/ablation.py); `better` es True
    si el modelo A pierde menos (DM<0). El llamador de la ablacion pasa (curr,
    prev) para leer si el peldano nuevo mejora al anterior.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Las series de perdida deben tener la misma longitud (emparejadas).")
    d = a - b
    T = d.shape[0]
    if T < 10:
        raise ValueError("Muy pocas observaciones para DM.")
    if n_lags is None:
        n_lags = max(h - 1, int(np.floor(1.5 * T ** (1.0 / 3.0))))

    d_bar = float(d.mean())
    dc = d - d_bar
    # Newey-West: gamma_0 + 2*sum_j w_j gamma_j, w_j = 1 - j/(L+1)
    var_hac = float(dc @ dc) / T
    for j in range(1, n_lags + 1):
        w = 1.0 - j / (n_lags + 1.0)
        var_hac += 2.0 * w * float(dc[j:] @ dc[:-j]) / T

    if var_hac <= 0:
        return {"statistic": float("nan"), "dm_stat": float("nan"),
                "p_value": float("nan"), "mean_diff": d_bar, "better": None,
                "n_obs": T, "n_lags": n_lags, "h": h, "note": "varianza HAC no positiva"}

    dm = d_bar / np.sqrt(var_hac / T)
    # correccion HLN para horizonte h: sqrt((T + 1 - 2h + h(h-1)/T)/T).
    # h=1 => sqrt((T-1)/T), el caso clasico.
    hln_factor = np.sqrt((T + 1.0 - 2.0 * h + h * (h - 1.0) / T) / T)
    dm_hln = dm * hln_factor
    p = 2.0 * float(stats.t.sf(abs(dm_hln), df=T - 1))

    return {
        "statistic": float(dm_hln),
        "dm_stat": float(dm_hln),
        "p_value": p,
        "mean_diff": d_bar,
        "better": bool(d_bar < 0.0),
        "n_obs": T,
        "n_lags": n_lags,
        "h": h,
        "note": "DM<0 => el primer modelo tiene MENOR perdida (mejor); HAC Newey-West + HLN",
    }


# --------------------------------------------------------------------------- #
# Pesaran-Timmermann
# --------------------------------------------------------------------------- #
def pesaran_timmermann(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    """Test de precision direccional de Pesaran-Timmermann (1992).

    Evalua H0: la prediccion no tiene informacion direccional (los aciertos de
    signo son los que producirian dos monedas independientes con las frecuencias
    marginales observadas). Estadistico ~ N(0,1) bajo H0. Se pasan los NIVELES
    (retorno realizado y media predictiva); los signos se toman aqui.
    """
    y = np.sign(np.asarray(y_real, dtype=float))
    x = np.sign(np.asarray(y_pred, dtype=float))
    if y.shape != x.shape:
        raise ValueError("y_real y y_pred deben tener la misma longitud.")
    T = y.shape[0]

    hit = float((y == x).mean())                 # P_hat: proporcion de aciertos
    py = float((y > 0).mean())                   # P(realizado positivo)
    px = float((x > 0).mean())                   # P(predicho positivo)
    p_star = py * px + (1.0 - py) * (1.0 - px)   # aciertos esperados bajo independencia

    var_p = p_star * (1.0 - p_star) / T
    var_star = (
        (2.0 * py - 1.0) ** 2 * px * (1.0 - px) / T
        + (2.0 * px - 1.0) ** 2 * py * (1.0 - py) / T
        + 4.0 * py * px * (1.0 - py) * (1.0 - px) / T**2
    )
    denom = var_p - var_star
    if denom <= 0:
        # prediccion (o realizado) de signo constante: el test degenera; se
        # reporta tal cual, no se maquilla con un p-value inventado.
        return {"pt_stat": float("nan"), "statistic": float("nan"),
                "p_value": float("nan"), "hit_rate": hit,
                "expected_hit_rate": p_star, "n_obs": T,
                "interpretation": ("El test degenera (signo constante en prediccion "
                                   "o realizado): no es interpretable."),
                "note": "degenerado: signo constante en prediccion o realizado"}

    pt = (hit - p_star) / np.sqrt(denom)
    p = float(stats.norm.sf(pt))                 # una cola: H1 = acierta MAS que el azar
    if p < 0.05:
        interpretation = (
            f"El modelo predice el signo del retorno mejor que el azar (p={p:.2f})."
        )
    else:
        interpretation = (
            f"No hay evidencia de que el modelo prediga el signo mejor que el azar (p={p:.2f})."
        )
    return {
        "pt_stat": float(pt),
        "statistic": float(pt),
        "p_value": p,
        "hit_rate": hit,
        "expected_hit_rate": p_star,
        "n_obs": T,
        "interpretation": interpretation,
        "note": "H1 unilateral: precision direccional mayor que bajo independencia",
    }


# --------------------------------------------------------------------------- #
# Calibracion (Test 2 V4): SIEMPRE sobre xi_PREDICHA (ξ_{t|t-1}), jamas filtrada
# --------------------------------------------------------------------------- #
# Por que ξ_{t|t-1} y nunca ξ_{t|t}: la filtrada ya incorporo r_t en la
# actualizacion de Bayes. Evaluar calibracion sobre ella es preguntarle al
# modelo "acertaste?" DESPUES de que ya vio la respuesta -> el resultado parece
# perfecto por construccion y no dice nada util. La delegacion de metricas va a
# validation/calibration.py (fuente unica); aqui se expone la API del reporte V4
# con la guardia que impide el error de pasar la filtrada por la predicha.
def _onehot(realized: np.ndarray, K: int) -> np.ndarray:
    """Convierte `realized` a one-hot (T, K). Acepta ya-one-hot (2d) o etiquetas
    de clase enteras (1d, el proxy y_t = argmax ξ_{t|t})."""
    y = np.asarray(realized)
    if y.ndim == 2:
        return y.astype(float)
    oh = np.zeros((y.shape[0], K))
    oh[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return oh


def brier_score_multiclass(xi_predicted: np.ndarray, realized: np.ndarray) -> float:
    """Brier multiclase = (1/T) Σ_t Σ_k (ξ_{t|t-1}(k) - y_t(k))². Menor es mejor."""
    p = np.asarray(xi_predicted, dtype=float)
    return _cal.brier_score(p, _onehot(realized, p.shape[1]))


def log_loss(xi_predicted: np.ndarray, realized: np.ndarray, eps: float = 1e-15) -> float:
    """Log-loss multiclase = -(1/T) Σ_t Σ_k y_t(k) log ξ_{t|t-1}(k). Menor es mejor."""
    p = np.asarray(xi_predicted, dtype=float)
    return _cal.log_loss(p, _onehot(realized, p.shape[1]), eps=eps)


def reliability_diagram_data(
    xi_predicted: np.ndarray, realized: np.ndarray, n_bins: int = 10
) -> dict:
    """Datos del diagrama de fiabilidad (el grafico lo hace la app, R9). Devuelve
    bin_centers, mean_predicted, freq_observed y n_obs_per_bin de la clase mas
    probable (top-label): confianza predicha vs. acierto empirico por bin."""
    p = np.asarray(xi_predicted, dtype=float)
    curve = _cal.reliability_curve(p, _onehot(realized, p.shape[1]), n_bins)
    return {
        "bin_centers": curve["bin_mid"].tolist(),
        "mean_predicted": curve["mean_confidence"].tolist(),
        "freq_observed": curve["empirical_accuracy"].tolist(),
        "n_obs_per_bin": curve["count"].astype(int).tolist(),
    }


def calibration_metrics(
    xi_predicted: np.ndarray, xi_filtered: np.ndarray, realized: np.ndarray
) -> dict:
    """Metricas de calibracion sobre ξ_{t|t-1} (PREDICHA). El assert impide el
    error clasico de pasar la FILTRADA donde se espera la predicha: eso produciria
    calibracion artificialmente perfecta (ver comentario del bloque). `realized`
    es el proxy del estado (argmax ξ_{t|t}) en enteros o one-hot."""
    xi_predicted = np.asarray(xi_predicted, dtype=float)
    xi_filtered = np.asarray(xi_filtered, dtype=float)
    assert not np.allclose(xi_predicted, xi_filtered), (
        "Se paso xi_filtered donde se esperaba xi_predicted. "
        "Esto produciria calibracion artificialmente perfecta. "
        "Revisa el llamador."
    )
    return {
        "brier": brier_score_multiclass(xi_predicted, realized),
        "log_loss": log_loss(xi_predicted, realized),
        "reliability": reliability_diagram_data(xi_predicted, realized),
        "n_obs": int(xi_predicted.shape[0]),
    }

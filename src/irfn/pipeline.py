"""Pipeline: retornos (+ covariables en V1) -> estimacion -> filtro de Hamilton
-> tabla filtrada.

Este modulo vive en src/ (no en scripts/) a proposito: es la logica que tanto el
CLI (scripts/run_pipeline.py) como el test anti-look-ahead (tests/test_prefix_
invariance.py) necesitan importar, y src/ nunca importa de scripts/ (CLAUDE.md).

V1 agrega TVTP (covariables rezagadas x_{t-1}, ver models/tvtp.py) e innovaciones
t de Student. La ESTANDARIZACION de las covariables es parte del modelo y por eso
se maneja aqui con la misma disciplina que los parametros: media y desviacion se
calculan SOLO sobre la ventana de entrenamiento (jamas sobre el test ni sobre el
futuro) y viajan junto a los parametros como `scaler`. Estandarizar con momentos
de toda la muestra seria look-ahead silencioso: la media "conocida" en 2015
incluiria datos de 2024.

--------------------------------------------------------------------------------
Por que este pipeline es prefix-invariante (la propiedad que protege R1)
--------------------------------------------------------------------------------
`run_pipeline` separa DOS cosas:

  1. De donde salen los parametros (`params` / `train_len`). En walk-forward y en
     el test PIT los parametros se fijan a partir de una ventana de entrenamiento
     que NO depende del punto de truncamiento. Por tanto, truncar la serie en t
     (con t despues de esa ventana) deja los parametros identicos.

  2. El filtro de Hamilton, que es CAUSAL: ξ_{s|s} usa solo r_1..r_s. Ademas la
     inicializacion es funcion solo de los parametros (σ²_{k,0}=v_k y ξ_0 = π
     estacionaria), nunca de los datos. Un arranque data-dependiente (backcast,
     burn-in muestral) meteria look-ahead disfrazado de inicializacion.

Con (1) y (2), correr sobre data[:T] y sobre data[:t] produce EXACTAMENTE el mismo
ξ_{s|s} para s <= t. Eso es lo que verifica audit.pit.prefix_invariance_check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from irfn.models.estimate import FitResult, fit
from irfn.models.hamilton import hamilton_filter


def regime_labels(K: int) -> list[str]:
    """Etiquetas de regimen en el orden estructural de v (R5: v_1 < ... < v_K).

    El indice 0 es SIEMPRE el de menor varianza incondicional y el indice K-1 el
    de mayor. Ese orden es fijo entre bloques (por eso no hay label switching),
    asi que las etiquetas son estables. Para K=2: baja / alta volatilidad.
    """
    if K == 2:
        return ["baja volatilidad", "alta volatilidad"]
    if K == 1:
        return ["unico regimen"]
    labels = ["baja volatilidad"]
    labels += [f"volatilidad intermedia {i}" for i in range(1, K - 1)]
    labels += ["alta volatilidad"]
    return labels


def row_entropy(xi: np.ndarray) -> np.ndarray:
    """Entropia de Shannon H(ξ) = -Σ_k ξ_k ln ξ_k por fila. xi es (T,K) o (K,).

    En log natural, por lo que H_max = ln(K). 0 ln 0 := 0 (limite correcto).
    """
    xi = np.asarray(xi, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(xi > 0.0, xi * np.log(xi), 0.0)
    return -terms.sum(axis=-1)


@dataclass
class PipelineResult:
    """Salida de run_pipeline. `frame` esta indexado por fecha e incluye, para
    cada regimen k: xi_filtered_k, xi_predicted_k; y columnas escalares r,
    entropy (H), entropy_norm (H/ln K) y argmax (etiqueta del regimen mas
    probable segun ξ_{t|t}).
    """

    frame: pd.DataFrame
    fit: FitResult
    K: int
    train_end: pd.Timestamp
    loglik_full: float
    scaler: tuple[np.ndarray, np.ndarray] | None = None   # (media, std) del train
    covariates: list[str] | None = None                   # nombres de columnas de X


def standardize_with(X: pd.DataFrame, scaler: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Aplica un scaler (media, std) YA calculado. Separado de su calculo a
    proposito: en walk-forward y en el chequeo PIT el scaler viene del train y
    se aplica a fechas posteriores sin recalcularse (eso seria look-ahead)."""
    mean, std = scaler
    return (X.to_numpy(dtype=float) - mean) / std


def train_scaler(X: pd.DataFrame, train_len: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Media y std por columna calculadas SOLO sobre la ventana de entrenamiento
    (las primeras train_len filas; toda la muestra si train_len es None, que es
    el caso "artefacto de hoy": todo el pasado hasta asof no es look-ahead)."""
    Xt = X.iloc[:train_len] if train_len is not None else X
    mean = Xt.to_numpy(dtype=float).mean(axis=0)
    std = Xt.to_numpy(dtype=float).std(axis=0, ddof=1)
    if np.any(std <= 0) or not np.all(np.isfinite(std)):
        raise ValueError("Covariable degenerada en el train (std<=0 o no finita).")
    return mean, std


def _filter_frame(
    returns: pd.Series, params: dict, K: int, X_std: np.ndarray | None = None
) -> pd.DataFrame:
    r = returns.to_numpy(dtype=float)
    xi_filt, xi_pred, loglik = hamilton_filter(r, params, K, X_lagged=X_std)

    cols: dict[str, np.ndarray] = {"r": r}
    for k in range(K):
        cols[f"xi_filtered_{k}"] = xi_filt[:, k]
    for k in range(K):
        cols[f"xi_predicted_{k}"] = xi_pred[:, k]

    frame = pd.DataFrame(cols, index=returns.index)
    H = row_entropy(xi_filt)
    frame["entropy"] = H
    frame["entropy_norm"] = H / np.log(K) if K > 1 else 0.0

    labels = regime_labels(K)
    argmax_idx = xi_filt.argmax(axis=1)
    frame["argmax_idx"] = argmax_idx
    frame["argmax"] = [labels[i] for i in argmax_idx]
    frame.attrs["loglik"] = float(loglik)
    return frame


def run_pipeline(
    returns: pd.Series,
    *,
    K: int,
    seed: int,
    n_starts: int = 30,
    params: dict | None = None,
    train_len: int | None = None,
    compute_se: bool = True,
    X: pd.DataFrame | None = None,
    scaler: tuple[np.ndarray, np.ndarray] | None = None,
    dist: str = "normal",
    l1_lambda: float = 0.0,
    l1_smooth_eps: float | None = None,
) -> PipelineResult:
    """Corre el pipeline sobre una serie de retornos indexada por fecha.

    Parametros
    ----------
    returns : pd.Series
        Retornos (escala %), indice de fechas. Es la serie que se FILTRA completa.
    K, seed, n_starts : configuracion del modelo/estimacion (R6).
    params : dict de parametros naturales ya estimados. Si se pasa, NO se
        re-estima: se filtra directamente. Es la ruta que usa el test PIT para
        fijar los parametros y aislar la causalidad del filtro.
    train_len : si params es None, numero de observaciones INICIALES usadas para
        estimar (ventana de entrenamiento fija). Si tambien es None, se estima
        sobre TODA la serie (caso "artefacto de hoy": usar todo el historico hasta
        asof NO es look-ahead para hoy).
    compute_se : pasa a estimate.fit; False acelera walk-forward (SE por bloque
        no se reporta).
    X : covariables del TVTP YA REZAGADAS por la capa de features (fila t =
        x_{t-1}; R3, lag_ledger). Mismo indice que returns, sin NaN. None = V0.
    scaler : (media, std) de estandarizacion. Si params es None se IGNORA y se
        calcula del train (train_scaler); si params se pasa con X, es
        OBLIGATORIO pasar el scaler del mismo entrenamiento -- recalcularlo
        sobre la serie truncada seria look-ahead y rompe la invarianza PIT.
    dist : "normal" | "t" (innovaciones; decide el BIC en V1, no un default).
    l1_lambda, l1_smooth_eps : penalizacion L1 de los beta TVTP (config/CV).

    Devuelve PipelineResult. `frame` cubre TODA la serie `returns`.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("run_pipeline espera un pd.Series de retornos indexado por fecha.")
    if returns.isna().any():
        raise ValueError("returns contiene NaN; limpiar antes de correr el pipeline.")

    cov_names: list[str] | None = None
    if X is not None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un pd.DataFrame de covariables rezagadas.")
        if not X.index.equals(returns.index):
            raise ValueError("X debe tener exactamente el mismo indice que returns.")
        if X.isna().any().any():
            raise ValueError("X contiene NaN; alinear/dropna antes del pipeline.")
        cov_names = list(X.columns)

    if params is None:
        train = returns if train_len is None else returns.iloc[:train_len]
        if X is not None:
            # Estandarizacion SOLO con el train (look-ahead si no; ver docstring).
            scaler = train_scaler(X, train_len)
            X_std_full = standardize_with(X, scaler)
            X_train = X_std_full[: len(train)]
        else:
            scaler, X_std_full, X_train = None, None, None
        fit_res = fit(
            train.to_numpy(dtype=float),
            K=K,
            n_starts=n_starts,
            seed=seed,
            compute_se=compute_se,
            X_lagged=X_train,
            dist=dist,
            l1_lambda=l1_lambda,
            l1_smooth_eps=l1_smooth_eps,
        )
        params = fit_res.params
        train_end = train.index[-1]
    else:
        if X is not None:
            if scaler is None:
                raise ValueError(
                    "params dados con X requieren el scaler del MISMO entrenamiento "
                    "(recalcularlo aqui seria look-ahead)."
                )
            X_std_full = standardize_with(X, scaler)
        else:
            X_std_full = None
        # params dados: sin re-estimacion. Se sintetiza un FitResult minimo para
        # que aguas abajo (contrato, reportes) tengan la metadata de semilla/K.
        fit_res = FitResult(
            K=K,
            params=params,
            theta=np.array([]),
            loglik=float("nan"),
            se={},
            bic=float("nan"),
            aic=float("nan"),
            n_converged=0,
            n_starts=0,
            seed=seed,
            hessian_ok=None,
            P=params["P"],
            se_P=None,
            dist="t" if "nu" in params else "normal",
            n_cov=0 if X is None else len(cov_names),
            l1_lambda=l1_lambda,
        )
        train_end = returns.index[-1]

    frame = _filter_frame(returns, params, K, X_std=X_std_full)
    return PipelineResult(
        frame=frame,
        fit=fit_res,
        K=K,
        train_end=train_end,
        loglik_full=float(frame.attrs["loglik"]),
        scaler=scaler,
        covariates=cov_names,
    )

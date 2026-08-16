"""Walk-forward con re-estimacion por bloque (R2). Minimo 6 bloques de test.

Reglas que este modulo materializa:

  R2  Cada bloque RE-ESTIMA los parametros desde cero, usando solo datos de
      entrenamiento de ese bloque. Prohibido estimar una vez sobre todo el
      historico. audit.pit.block_reestimation_check verifica despues que los
      bloques tengan parametros DISTINTOS.

  R8  Ninguna version avanza sin walk-forward documentado, aunque el resultado
      sea negativo. Este modulo produce los numeros; el reporte los escribe tal
      cual.

--------------------------------------------------------------------------------
El punto sutil: arrastrar el ultimo ξ_{t|t} del entrenamiento, NO reiniciar
--------------------------------------------------------------------------------
Al empezar el TEST de un bloque hay que arrastrar el ultimo ξ_{t|t} (y el ultimo
σ² de cada regimen) del entrenamiento. NO reiniciar a la distribucion estacionaria.

Por que importa: reiniciar en cada frontera introduce un artefacto. El modelo
"olvida" lo que sabia y arranca el test desde π estacionaria y σ²=v_k, como si el
mercado no tuviera historia el 1 de enero. El filtro tarda decenas de dias en
re-converger, y esos dias de transitorio contaminan justo el principio de cada
ventana de test -> el desempeno out-of-sample se ve PEOR de lo que realmente es,
por una decision de implementacion, no por el modelo.

Como se implementa aqui SIN codigo especial: en vez de filtrar la ventana de test
en aislamiento, se filtra la ventana CONTINUA [train_start, test_end] con los
parametros del bloque, y se registran solo las fechas de test. La recursion de
Hamilton (y las K recursiones GARCH) arrastran su estado a traves de la frontera
de forma natural: el ξ_{t|t} y el σ²_{k,t} con que arranca el test son EXACTAMENTE
los que dejo el ultimo dia de entrenamiento. El tramo de entrenamiento actua de
"warm-up" y no se evalua out-of-sample.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from irfn.models.estimate import cv_l1_lambda, fit
from irfn.models.hamilton import hamilton_filter, predictive_loglik_per_obs
from irfn.pipeline import regime_labels, row_entropy, standardize_with, train_scaler
from irfn.validation import checkpoint as _ckpt

logger = logging.getLogger("irfn.walkforward")


def per_obs_loglik(
    r: np.ndarray, params: dict, K: int, X_lagged: np.ndarray | None = None
) -> np.ndarray:
    """Contribucion a la log-verosimilitud de CADA observacion.

    ll_t = log Σ_m ξ_{t|t-1}(m) f(r_t | S_t=m)  -- el denominador de la
    actualizacion de Hamilton, que es la densidad predictiva un-paso-adelante.
    Sumando sobre t se recupera exactamente la loglik total del filtro.
    Delegado a models.hamilton.predictive_loglik_per_obs (V1: la misma serie
    alimenta el CV del lambda y la perdida del Diebold-Mariano)."""
    return predictive_loglik_per_obs(r, params, K, X_lagged=X_lagged)


@dataclass
class BlockResult:
    block_id: int
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    theta: np.ndarray                 # vector sin restricciones (para el check R2)
    params: dict                      # parametros naturales del bloque
    P: np.ndarray
    kappa: np.ndarray                 # persistencia por regimen (diagnostico)
    loglik_train_per_obs: float       # loglik media por observacion (comparable entre bloques)
    loglik_test_per_obs: float
    n_converged: int
    test_frame: pd.DataFrame          # ξ out-of-sample de este bloque
    l1_lambda: float = 0.0            # lambda elegido por CV DENTRO del train (V1)
    cv_table: list[dict] | None = None  # tabla completa del CV (no se esconden perdedores)
    scaler: tuple | None = None       # (media, std) de covariables, del train del bloque


@dataclass
class WalkForwardResult:
    K: int
    seed: int
    n_starts: int
    blocks: list[BlockResult]
    oos_frame: pd.DataFrame           # ξ filtrado out-of-sample concatenado, todos los bloques
    block_boundaries: list[pd.Timestamp] = field(default_factory=list)
    dist: str = "normal"              # innovaciones de la especificacion (V1)
    covariates: list[str] | None = None  # covariables TVTP; None = P constante

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)


def _make_windows(
    index: pd.DatetimeIndex, train_years: int, test_months: int, n_blocks_min: int
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Construye ventanas (train_start, test_start, test_end) NO solapadas en test,
    de la mas antigua a la mas reciente, cada una con train_years completos de
    entrenamiento previo. Devuelve TODAS las que caben (>= n_blocks_min o falla).
    """
    start, end = index[0], index[-1]
    train_off = pd.DateOffset(years=train_years)
    test_off = pd.DateOffset(months=test_months)

    windows = []
    test_start = start + train_off
    while test_start + test_off <= end:
        windows.append((test_start - train_off, test_start, test_start + test_off))
        test_start = test_start + test_off

    if len(windows) < n_blocks_min:
        raise ValueError(
            f"Solo caben {len(windows)} bloques de walk-forward con train={train_years}a "
            f"y test={test_months}m sobre {start.date()}..{end.date()}; se requieren "
            f">= {n_blocks_min} (R2, R8). Amplia el historico o reduce las ventanas."
        )
    return windows


def _init_worker() -> None:
    """Inicializador de los procesos del pool: fija a 1 hilo las libs BLAS/OMP.

    Con n_jobs procesos corriendo NumPy/SciPy en paralelo, dejar que cada uno
    lance su propio pool de hilos BLAS sobre-suscribe la CPU (n_jobs x hilos).
    El filtro de Hamilton es de matrices K x K minusculas (no es BLAS-bound),
    asi que 1 hilo por proceso no cuesta nada y evita la contencion. Redundante
    con la herencia de entorno del padre (spawn), pero explicito no hace dano.
    """
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"


def _compute_block(
    b: int,
    train_start: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    returns: pd.Series,
    K: int,
    seed: int,
    n_starts: int,
    dist: str,
    l1_grid: list[float] | None,
    cv_val_frac: float | None,
    cv_n_starts: int | None,
    l1_smooth_eps: float | None,
    X: pd.DataFrame | None,
) -> tuple[int, BlockResult | None]:
    """Resuelve UN bloque del walk-forward: pura funcion de sus entradas (R2).

    Es determinista dada (ventana, K, seed, n_starts, dist, covariables): no lee
    ni escribe estado compartido, asi que correrla en serie o en un proceso del
    pool produce EXACTAMENTE el mismo BlockResult. Ese es el invariante que
    permite paralelizar sobre bloques sin alterar los numeros (la firma del
    checkpoint no incluye n_jobs: serial y paralelo comparten checkpoints).

    Devuelve (b, BlockResult) o (b, None) si la ventana quedo vacia (mismo
    salto que el bucle serial original).
    """
    index = returns.index
    labels = regime_labels(K)

    train_mask = (index >= train_start) & (index < test_start)
    window_mask = (index >= train_start) & (index < test_end)
    test_mask_full = (index >= test_start) & (index < test_end)

    train_r = returns[train_mask]
    window_r = returns[window_mask]
    if train_r.empty or test_mask_full.sum() == 0:
        return b, None

    # --- Covariables del bloque: scaler SOLO del train (V1) ---
    if X is not None:
        X_win = X.loc[window_mask]
        n_train_rows = int(train_mask.sum())
        scaler = train_scaler(X_win, n_train_rows)
        X_win_std = standardize_with(X_win, scaler)
        X_train_std = X_win_std[:n_train_rows]
    else:
        scaler, X_win_std, X_train_std = None, None, None

    # --- Lambda de la L1 por CV DENTRO del train del bloque (V1) ---
    lam, cv_table = 0.0, None
    if X is not None and l1_grid:
        if len(l1_grid) == 1:
            lam = float(l1_grid[0])
        else:
            lam, cv_table = cv_l1_lambda(
                train_r.to_numpy(dtype=float),
                X_train_std,
                K,
                dist=dist,
                l1_grid=l1_grid,
                val_frac=cv_val_frac,
                n_starts=cv_n_starts,
                seed=seed,
                l1_smooth_eps=l1_smooth_eps,
            )

    # R2: re-estimacion desde cero, solo con datos de entrenamiento del bloque.
    # compute_se=False: el walk-forward no reporta SE por bloque; ahorra el
    # hessiano numerico (caro) sin afectar las metricas out-of-sample.
    fit_res = fit(
        train_r.to_numpy(dtype=float),
        K=K,
        n_starts=n_starts,
        seed=seed,
        compute_se=False,
        X_lagged=X_train_std,
        dist=dist,
        l1_lambda=lam,
        l1_smooth_eps=l1_smooth_eps,
    )
    params = fit_res.params

    # Filtro sobre la ventana continua: el test arrastra el estado del train.
    r_win = window_r.to_numpy(dtype=float)
    xi_filt, xi_pred, _ = hamilton_filter(r_win, params, K, X_lagged=X_win_std)
    ll_win = per_obs_loglik(r_win, params, K, X_lagged=X_win_std)

    win_index = window_r.index
    is_test = (win_index >= test_start) & (win_index < test_end)
    is_train = ~is_test

    ll_train = float(ll_win[is_train].mean()) if is_train.any() else float("nan")
    ll_test = float(ll_win[is_test].mean()) if is_test.any() else float("nan")

    # Tabla out-of-sample de este bloque (solo fechas de test).
    test_idx = win_index[is_test]
    xi_f_test = xi_filt[is_test]
    xi_p_test = xi_pred[is_test]
    H = row_entropy(xi_f_test)
    cols = {"r": r_win[is_test]}
    for k in range(K):
        cols[f"xi_filtered_{k}"] = xi_f_test[:, k]
    for k in range(K):
        cols[f"xi_predicted_{k}"] = xi_p_test[:, k]
    tf = pd.DataFrame(cols, index=test_idx)
    tf["entropy"] = H
    tf["entropy_norm"] = H / np.log(K) if K > 1 else 0.0
    am = xi_f_test.argmax(axis=1)
    tf["argmax_idx"] = am
    tf["argmax"] = [labels[i] for i in am]
    tf["block_id"] = b
    # loglik predictiva POR OBSERVACION del tramo test: la serie de perdidas
    # del Diebold-Mariano (V1 vs V0 se comparan emparejando estas fechas).
    tf["loglik_obs"] = ll_win[is_test]
    # media predictiva un-paso-adelante del retorno (mezcla): el signo es lo
    # que evalua Pesaran-Timmermann. Con TVTP y t, mu_k no cambia, asi que
    # E[r_t | t-1] = sum_k xi_{t|t-1}(k) mu_k en ambas especificaciones.
    tf["r_pred_mean"] = xi_p_test @ params["mu"]

    kappa = params["alpha"] + params["gamma"] / 2.0 + params["beta"]
    br = BlockResult(
        block_id=b,
        train_start=train_start,
        test_start=test_start,
        test_end=test_end,
        n_train=int(is_train.sum()),
        n_test=int(is_test.sum()),
        theta=fit_res.theta,
        params=params,
        P=params["P"],
        kappa=kappa,
        loglik_train_per_obs=ll_train,
        loglik_test_per_obs=ll_test,
        n_converged=fit_res.n_converged,
        test_frame=tf,
        l1_lambda=lam,
        cv_table=cv_table,
        scaler=scaler,
    )
    return b, br


def walk_forward(
    returns: pd.Series,
    *,
    K: int,
    seed: int,
    n_starts: int,
    train_years: int,
    test_months: int,
    n_blocks_min: int = 6,
    X: pd.DataFrame | None = None,
    dist: str = "normal",
    l1_grid: list[float] | None = None,
    cv_val_frac: float | None = None,
    cv_n_starts: int | None = None,
    l1_smooth_eps: float | None = None,
    checkpoint_path: Path | None = None,
    n_jobs: int = 1,
) -> WalkForwardResult:
    """Corre el walk-forward completo sobre `returns` (Series indexada por fecha).

    Para cada bloque: re-estima desde cero sobre su ventana de entrenamiento (R2),
    filtra la ventana CONTINUA train+test (arrastre de estado, ver docstring del
    modulo) y registra las metricas y el ξ out-of-sample del tramo de test.

    V1 (X no None): TVTP. Por bloque, y usando SOLO su ventana de entrenamiento:
      1. se estandarizan las covariables con media/std del train del bloque
         (pipeline.train_scaler; estandarizar con toda la muestra es look-ahead);
      2. si l1_grid tiene mas de un valor, se elige el lambda de la L1 por CV
         temporal interno (estimate.cv_l1_lambda) -- el test JAMAS participa;
      3. se re-estima con el lambda elegido y multistart completo (R6).
    El tramo de test consume el scaler y los parametros del train tal cual.

    Si `checkpoint_path` se da, cada bloque ya resuelto (BlockResult completo)
    se guarda de inmediato (validation/checkpoint.py, ~1/n_blocks del trabajo
    cada vez): una corrida interrumpida reanuda sin re-estimar los bloques ya
    hechos, siempre que K/seed/n_starts/ventanas/covariables coincidan.

    `n_jobs` (default 1 = serie, camino de referencia) reparte los bloques
    PENDIENTES en un pool de procesos. Los bloques son independientes (R2:
    cada uno re-estima desde cero), asi que el resultado es identico al serial;
    n_jobs NO entra en la firma del checkpoint, de modo que una corrida serial
    y una paralela comparten y reanudan el mismo checkpoint. Es solo velocidad,
    nunca cambia los numeros.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("walk_forward espera un pd.Series indexado por fecha.")
    returns = returns.dropna()
    index = returns.index
    if X is not None:
        if not X.index.equals(index):
            raise ValueError("X debe tener exactamente el mismo indice que returns (alinear antes).")
        if X.isna().any().any():
            raise ValueError("X contiene NaN; alinear/dropna antes del walk-forward.")
        if l1_grid and len(l1_grid) > 1 and (cv_val_frac is None or cv_n_starts is None or l1_smooth_eps is None):
            raise ValueError("CV de lambda requiere cv_val_frac, cv_n_starts y l1_smooth_eps (config).")

    windows = _make_windows(index, train_years, test_months, n_blocks_min)

    sig = (
        K, seed, n_starts, train_years, test_months, n_blocks_min, dist,
        tuple(X.columns) if X is not None else None,
        tuple((str(w[0]), str(w[1]), str(w[2])) for w in windows),
        tuple(l1_grid) if l1_grid else None, cv_val_frac, cv_n_starts, l1_smooth_eps,
    )
    done: dict[int, BlockResult] = {}
    if checkpoint_path is not None:
        payload = _ckpt.load(checkpoint_path, expected_sig=sig)
        if payload is not None:
            done = payload["blocks"]
            logger.info("walk_forward: reanudando, %d/%d bloques ya en checkpoint.",
                        len(done), len(windows))

    # Bloques pendientes (no resueltos en un checkpoint previo). Cada uno es una
    # tarea independiente (R2): se resuelven en serie (n_jobs=1, referencia
    # exacta) o repartidos en un pool de procesos. El resultado es identico en
    # ambos caminos porque _compute_block es pura y determinista dada la semilla.
    pending = [
        (b, ts0, ts1, ts2)
        for b, (ts0, ts1, ts2) in enumerate(windows)
        if b not in done
    ]

    def _store(b: int, br: BlockResult | None) -> None:
        if br is None:
            return
        done[b] = br
        if checkpoint_path is not None:
            _ckpt.save(checkpoint_path, {"sig": sig, "blocks": done})

    if n_jobs and n_jobs > 1 and len(pending) > 1:
        # Sobre-suscripcion BLAS: fijamos 1 hilo por proceso ANTES de crear el
        # pool para que los hijos (spawn en Windows) hereden el entorno en su
        # import fresco de NumPy; _init_worker lo repite dentro por si acaso.
        for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(_v, "1")
        max_workers = min(int(n_jobs), len(pending))
        logger.info("walk_forward: %d bloques pendientes en pool de %d procesos.",
                    len(pending), max_workers)
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as ex:
            futs = {
                ex.submit(
                    _compute_block, b, ts0, ts1, ts2, returns, K, seed, n_starts,
                    dist, l1_grid, cv_val_frac, cv_n_starts, l1_smooth_eps, X,
                ): b
                for (b, ts0, ts1, ts2) in pending
            }
            for fut in as_completed(futs):
                b, br = fut.result()
                _store(b, br)
    else:
        for (b, ts0, ts1, ts2) in pending:
            _store(*_compute_block(
                b, ts0, ts1, ts2, returns, K, seed, n_starts,
                dist, l1_grid, cv_val_frac, cv_n_starts, l1_smooth_eps, X,
            ))

    # Ensamblar en orden temporal (as_completed no preserva el orden de bloques).
    blocks: list[BlockResult] = []
    oos_parts: list[pd.DataFrame] = []
    boundaries: list[pd.Timestamp] = []
    for b in range(len(windows)):
        if b in done:
            br = done[b]
            blocks.append(br)
            oos_parts.append(br.test_frame)
            boundaries.append(br.test_start)

    if len(blocks) < n_blocks_min:
        raise RuntimeError(
            f"Solo se completaron {len(blocks)} bloques (< {n_blocks_min}). "
            f"Revisa cobertura de datos."
        )

    oos_frame = pd.concat(oos_parts).sort_index()
    return WalkForwardResult(
        K=K,
        seed=seed,
        n_starts=n_starts,
        blocks=blocks,
        oos_frame=oos_frame,
        block_boundaries=boundaries,
        dist=dist,
        covariates=None if X is None else list(X.columns),
    )

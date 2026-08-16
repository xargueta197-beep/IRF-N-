"""Auditoria point-in-time. La joya de la corona: si el prefijo no es invariante,
hay look-ahead y todo lo demas sobra.

Tres chequeos, los tres devuelven DataFrames que la app lee de artifacts/ (R9):

  1. prefix_invariance_check  -- correr sobre data[:t] y sobre data[:T] debe dar
     el MISMO ξ_{s|s} para s <= t (atol=1e-10). El mas importante del repo.
  2. lag_ledger               -- cada covariable con su .shift aplicado; sin shift
     documentado se marca en rojo (R3).
  3. block_reestimation_check -- los N bloques de walk-forward deben tener
     parametros DISTINTOS; si son iguales, alguien estimo una sola vez (viola R2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from irfn.pipeline import run_pipeline

# Tolerancia del test de invarianza. No es "tolerancia numerica" laxa: el filtro
# es determinista y las mismas operaciones sobre el mismo prefijo deben dar bit a
# bit casi lo mismo. 1e-10 solo absorbe el reordenamiento de sumas en punto
# flotante; cualquier fuga de futuro real produce diferencias >> 1e-10.
PREFIX_ATOL = 1e-10


def prefix_invariance_check(
    returns: pd.Series,
    *,
    K: int,
    seed: int,
    n_starts: int = 8,
    train_len: int | None = None,
    dates_to_test: int = 10,
    atol: float = PREFIX_ATOL,
    X: pd.DataFrame | None = None,
    dist: str = "normal",
) -> pd.DataFrame:
    """Verifica invarianza de prefijo del ξ filtrado.

    Estrategia (y por que es honesta): los parametros se estiman UNA vez sobre la
    ventana de entrenamiento (`train_len` observaciones iniciales) y se FIJAN. Las
    fechas de prueba se muestrean DESPUES de esa ventana, asi que truncar la serie
    no cambia los parametros. Con los parametros fijos, el test aisla exactamente
    la superficie de look-ahead que importa: la CAUSALIDAD DEL FILTRO. Si el filtro
    usara cualquier r_s con s > t (o una inicializacion data-dependiente), ξ_{·|·}
    en fechas <= t cambiaria al ver mas datos, y max_abs_diff se disparараria.

    Para cada fecha t muestreada compara ξ_filtered en TODAS las fechas <= t entre:
      - la corrida completa sobre returns
      - la corrida truncada sobre returns[:t]
    Devuelve DataFrame(fecha, max_abs_diff, n_dates, passed).

    V1 (X no None): el mismo chequeo con TVTP. Los parametros Y el scaler de
    estandarizacion se fijan del entrenamiento y las corridas truncadas reciben
    X truncado con ese scaler fijo. La superficie de look-ahead nueva que esto
    aisla: que la trayectoria de matrices P(x_{t-1}) en fechas <= t no cambie al
    ver mas datos (si el rezago de las covariables estuviera mal aplicado o el
    scaler se recalculara, cambiaria). La causalidad de las VENTANAS moviles de
    los features se cubre aparte en test_features_prefix_invariance, que trunca
    los PRECIOS crudos.
    """
    returns = returns.dropna()
    T = len(returns)
    if train_len is None:
        train_len = T // 2
    if train_len >= T - 2:
        raise ValueError("train_len deja demasiado pocas fechas de test para el chequeo PIT.")

    xi_cols = [f"xi_filtered_{k}" for k in range(K)]

    # 1) Estimacion unica sobre la ventana de entrenamiento -> parametros fijos
    #    (y scaler fijo del train si hay covariables).
    full = run_pipeline(
        returns, K=K, seed=seed, n_starts=n_starts, train_len=train_len,
        compute_se=False, X=X, dist=dist,
    )
    params = full.fit.params
    scaler = full.scaler
    full_frame = full.frame

    # 2) Fechas de prueba: posiciones despues de la ventana de entrenamiento.
    rng = np.random.default_rng(seed)
    lo, hi = train_len + 1, T - 1          # posiciones candidatas (excluye la ultima)
    n = min(dates_to_test, hi - lo)
    positions = np.sort(rng.choice(np.arange(lo, hi), size=n, replace=False))

    rows = []
    for pos in positions:
        t = returns.index[pos]
        truncated = run_pipeline(
            returns.iloc[: pos + 1], K=K, seed=seed, params=params,
            X=None if X is None else X.iloc[: pos + 1], scaler=scaler,
        )
        a = full_frame.loc[:t, xi_cols].to_numpy()
        b = truncated.frame.loc[:t, xi_cols].to_numpy()
        max_abs_diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        rows.append(
            {
                "fecha": t,
                "max_abs_diff": max_abs_diff,
                "n_dates": int(a.shape[0]),
                "passed": bool(max_abs_diff < atol),
            }
        )

    df = pd.DataFrame(rows)
    df.attrs["passed"] = bool(df["passed"].all())
    df.attrs["atol"] = atol
    return df


# Registro de features de V0. En V0 el modelo NO tiene covariables (TVTP y
# noticias son V1+): la unica serie que entra es el retorno r_t, que es la
# OBSERVACION que se modela, no un predictor -> se usa contemporaneo (shift 0) y
# eso es correcto. Cuando lleguen las covariables (sma_gap, bb_width_z,
# slope_2s10y, hy_oas_z, surprise_index, lambda_N_z) cada una debe registrarse
# aqui con shift 1 (R3); una covariable con shift < 1 es look-ahead.
V0_FEATURE_REGISTRY: list[dict] = [
    {
        "feature": "r_t (retorno)",
        "role": "observacion",
        "applied_shift": 0,
        "note": "es la variable modelada, no un predictor; contemporaneo es correcto.",
    },
]


def v1_feature_registry(
    include_macro: bool = True, include_news: bool = False, include_hawkes: bool = False
) -> list[dict]:
    """Registro completo de features para el lag_ledger: la observacion r_t
    (V0), las covariables tecnicas (shift(1) explicito en features/technical.py),
    las macro (lag de publicacion real + margen, features/macro.py, V1), --
    V2, `include_news=True` -- la de sorpresa (shift(1) explicito en features/
    surprise.py) y -- V3, `include_hawkes=True` -- lambda_N_z (shift(1)
    explicito en features/hawkes_features.py). Cada modulo de features es dueno
    de sus propias entradas; aqui solo se concatenan -- si un feature nuevo no
    registra su rezago, aparece "SIN DOCUMENTAR" y el ledger se pone rojo (R3).
    """
    from irfn.features.technical import FEATURE_REGISTRY as TECH_REGISTRY

    reg = list(V0_FEATURE_REGISTRY) + list(TECH_REGISTRY)
    if include_macro:
        from irfn.features.macro import FEATURE_REGISTRY as MACRO_REGISTRY

        reg += list(MACRO_REGISTRY)
    if include_news:
        from irfn.features.surprise import FEATURE_REGISTRY as NEWS_REGISTRY

        reg += list(NEWS_REGISTRY)
    if include_hawkes:
        from irfn.features.hawkes_features import FEATURE_REGISTRY as HAWKES_REGISTRY

        reg += list(HAWKES_REGISTRY)
    return reg


def lag_ledger(registry: list[dict] | None = None) -> pd.DataFrame:
    """Ledger de rezagos: cada feature con su .shift aplicado y su veredicto.

    Regla (R3): toda COVARIABLE entra rezagada x_{t-1}, es decir shift >= 1. La
    observacion r_t es la excepcion legitima (shift 0). Cualquier covariable con
    shift < 1 o sin shift documentado se marca en rojo.
    """
    reg = V0_FEATURE_REGISTRY if registry is None else registry
    rows = []
    for entry in reg:
        role = entry.get("role", "covariable")
        shift = entry.get("applied_shift", None)
        if role == "observacion":
            status = "verde" if shift == 0 else "rojo"
        else:
            status = "verde" if isinstance(shift, int) and shift >= 1 else "rojo"
        rows.append(
            {
                "feature": entry.get("feature", "?"),
                "role": role,
                "applied_shift": shift if shift is not None else "SIN DOCUMENTAR",
                "status": status,
                "note": entry.get("note", ""),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["passed"] = bool((df["status"] == "verde").all())
    return df


def block_reestimation_check(blocks: list, atol: float = 1e-8) -> pd.DataFrame:
    """Verifica que los bloques de walk-forward tengan parametros DISTINTOS (R2).

    Si dos bloques comparten el mismo theta (a tolerancia atol), alguien estimo una
    sola vez y reciclo parametros -> viola R2. Recibe la lista de BlockResult del
    walk-forward (cada uno con .theta). Devuelve DataFrame por bloque con la norma
    de su theta, la minima distancia a otro bloque y si es distinto de todos.
    """
    thetas = [np.asarray(b.theta, dtype=float) for b in blocks]
    rows = []
    for i, b in enumerate(blocks):
        dists = [
            float(np.max(np.abs(thetas[i] - thetas[j])))
            for j in range(len(blocks))
            if j != i and thetas[j].shape == thetas[i].shape
        ]
        min_dist = min(dists) if dists else np.inf
        rows.append(
            {
                "block_id": b.block_id,
                "train_start": b.train_start,
                "test_start": b.test_start,
                "theta_norm": float(np.linalg.norm(thetas[i])),
                "min_dist_a_otro_bloque": min_dist,
                "distinto": bool(min_dist > atol),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        # Sin bloques que re-estimar (p.ej. activo K=1: no hay escalera de
        # walk-forward): el check es VACUO -> pasa por vacuidad (no hay R2 que
        # violar). Antes esto reventaba con KeyError('distinto') sobre un df sin
        # columnas; el pase vacuo es la lectura honesta, analoga a los otros checks
        # PIT vacuos (expanding_window_check, timestamp_audit).
        df = pd.DataFrame(columns=["block_id", "train_start", "test_start",
                                   "theta_norm", "min_dist_a_otro_bloque", "distinto"])
        df.attrs["passed"] = True
        df.attrs["vacuous"] = True
        return df
    df.attrs["passed"] = bool(df["distinto"].all())
    return df


# --------------------------------------------------------------------------- #
# V2: auditoria de la capa de sorpresa (pantalla 6, secciones 4-5)
# --------------------------------------------------------------------------- #
def consensus_vintage_ledger(calendar: pd.DataFrame) -> pd.DataFrame:
    """Ledger de vintages del calendario macro: fecha_evento vs la fecha en que
    ESTE proyecto vio el consenso por primera vez (`captured_at` de la captura
    diaria mas temprana que lo registro; ver data.calendar.load_local_snapshots
    -- point-in-time por construccion, nunca se deja que una captura posterior
    reescriba un consenso ya visto).

    `ok` es False si `captured_at` es POSTERIOR a `fecha_evento`: eso significa
    que la primera vez que este proyecto vio ese consenso fue DESPUES de que el
    dato se publicara, es decir no hay garantia de que sea el consenso
    pre-release genuino (podria ya reflejar el ajuste post-sorpresa de los
    pronosticadores). Es un hallazgo, no un error silencioso (R8).

    Con calendario vacio (la condicion de hoy, ver reports/data_audit.md)
    devuelve un DataFrame vacio con passed=True: no hay ninguna fila que
    contradiga la propiedad -- un pase VACUO, marcado como tal via
    `df.attrs["n_events"] == 0`, no una verificacion sustantiva.
    """
    cols = ["indicator", "fecha_evento", "fecha_publicacion_consenso", "lag_dias", "ok"]
    if calendar is None or calendar.empty:
        df = pd.DataFrame(columns=cols)
        df.attrs["passed"] = True
        df.attrs["n_events"] = 0
        return df

    rows = []
    for ts, row in calendar.iterrows():
        captured = pd.to_datetime(row.get("captured_at"), utc=True, errors="coerce")
        has_consensus = pd.notna(row.get("consensus"))
        if pd.isna(captured):
            lag_days, ok = float("nan"), not has_consensus
        else:
            lag_days = (pd.Timestamp(ts) - captured).total_seconds() / 86400.0
            ok = bool((lag_days >= 0) or not has_consensus)
        rows.append(
            {
                "indicator": row["indicator"],
                "fecha_evento": str(pd.Timestamp(ts).date()),
                "fecha_publicacion_consenso": (str(captured.date()) if pd.notna(captured) else None),
                "lag_dias": lag_days,
                "ok": ok,
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["passed"] = bool(df["ok"].all())
    df.attrs["n_events"] = int(len(df))
    return df


def expanding_window_check(
    calendar: pd.DataFrame, *, min_obs: int, n_checks: int = 5, seed: int = 0, atol: float = 1e-10
) -> pd.DataFrame:
    """Verifica EMPIRICAMENTE, sobre el calendario REAL disponible, que sigma_i
    (y por tanto z_i) en cada fecha t no cambia al truncar el calendario en t --
    el analogo de prefix_invariance_check para la capa de sorpresa. Reutiliza
    features.surprise.expanding_surprise_z (no se reimplementa el calculo).

    Con menos de 2 releases con consenso (la condicion de hoy) no hay nada que
    truncar de forma no trivial: el chequeo pasa VACUAMENTE y lo declara via
    `df.attrs["vacuous"] = True` -- no se finge una verificacion sustantiva que
    no ocurrio.
    """
    from irfn.features.surprise import expanding_surprise_z

    cols = ["fecha_corte", "max_abs_diff", "ok"]
    with_cons = calendar[calendar["consensus"].notna()] if (calendar is not None and len(calendar)) else calendar
    n = 0 if with_cons is None else len(with_cons)
    if n < 2:
        df = pd.DataFrame(columns=cols)
        df.attrs["passed"] = True
        df.attrs["vacuous"] = True
        df.attrs["n_events_con_consenso"] = n
        return df

    full_z, _ = expanding_surprise_z(calendar, min_obs=min_obs)
    dates = with_cons.index.unique().sort_values()
    rng = np.random.default_rng(seed)
    k = min(n_checks, len(dates) - 1)
    positions = np.sort(rng.choice(np.arange(1, len(dates)), size=k, replace=False)) if k > 0 else []

    rows = []
    for pos in positions:
        cutoff = dates[pos]
        trunc_z, _ = expanding_surprise_z(calendar.loc[:cutoff], min_obs=min_obs)
        common_cols = [c for c in trunc_z.columns if c in full_z.columns]
        a = full_z.loc[trunc_z.index, common_cols].to_numpy(dtype=float)
        b = trunc_z[common_cols].to_numpy(dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        diff = float(np.max(np.abs(a[mask] - b[mask]))) if mask.any() else 0.0
        rows.append({"fecha_corte": str(cutoff), "max_abs_diff": diff, "ok": bool(diff < atol)})

    df = pd.DataFrame(rows)
    df.attrs["passed"] = bool(df["ok"].all()) if len(df) else True
    df.attrs["vacuous"] = False
    df.attrs["n_events_con_consenso"] = n
    return df

"""Nucleo matematico de la capa de sorpresa macro (V2): z_i, w_i (R7), SI_t.

SIN NOTICIAS-Hawkes aqui (eso es V3) y SIN ingesta de datos: este modulo consume
un calendario YA parseado y devuelve features. Las tres piezas del spec (CLAUDE.md):

    z_i  = (actual_i - consenso_i) / sigma_i        # sigma_i ESPECIFICO del indicador
    w_i  <- por regresion de impacto:  |r_ventana| = a + w_i*|z_i| + u   (R7)
    SI_t = sum_{i: t_i <= t} w_i * z_i * exp(-delta*(t - t_i))           # delta por MLE

--------------------------------------------------------------------------------
DOS DECISIONES DE DISENO (confirmadas con el director)
--------------------------------------------------------------------------------
1. sigma_i con poca historia: sigma_i es un std EXPANDING (nunca sobre toda la
   muestra: eso usaria el futuro para normalizar el pasado). Hasta que un
   indicador acumula >= `sigma_min_obs` (config, default 12) de sus PROPIAS
   sorpresas, su z_i es NaN y NO contribuye a SI_t. No se pooleа un sigma global
   entre indicadores: sus escalas naturales son incomparables (sorpresa de NFP
   en miles de empleos vs CPI en 0.1 pp) y poolear las desescalaria. Warm-up
   honesto, igual que las ventanas de features/technical.py.

2. Release sin consenso: z_i requiere consenso. Si un indicador publica sin
   consenso disponible, ese evento se OMITE (no aporta termino a SI_t) y se
   CUENTA en el reporte de cobertura. JAMAS se imputa consenso=actual (z=0):
   eso fabricaria la afirmacion "no hubo sorpresa" y sesgaria SI_t hacia cero,
   subestimando la presion de noticias. Un consenso ausente != sorpresa nula.

--------------------------------------------------------------------------------
R3 (rezago): SI_t entra al logit de transicion como SI_{t-1}
--------------------------------------------------------------------------------
`surprise_feature` aplica un .shift(1) EXPLICITO al SI_t antes de devolverlo como
covariable, exactamente como features/technical.py: el valor fechado en t es el
indice de sorpresa conocido al cierre de t-1. Registrado en FEATURE_REGISTRY para
el lag_ledger. El decaimiento exp(-delta*(t-t_i)) ya es causal (solo eventos
pasados), pero el shift protege el limite: sin el, un release del propio dia t
entraria a predecir la transicion HACIA t (look-ahead intradia).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("irfn.surprise")

# Registro para el lag_ledger (R3). El shift aplicado es el del codigo de
# surprise_feature; si alguien cambia uno sin el otro, la auditoria lo delata.
FEATURE_REGISTRY: list[dict] = [
    {
        "feature": "surprise_index",
        "role": "covariable",
        "applied_shift": 1,
        "note": (
            "SI_t = sum w_i z_i exp(-delta (t - t_i)); shift(1) explicito en "
            "features/surprise.surprise_feature. delta se estima por MLE del modelo completo."
        ),
    },
]


# --------------------------------------------------------------------------- #
# (a) Estandarizacion por indicador: sigma_i EXPANDING, z_i, cobertura
# --------------------------------------------------------------------------- #
def expanding_surprise_z(
    calendar: pd.DataFrame, *, min_obs: int
) -> tuple[pd.DataFrame, dict]:
    """z_i por indicador con sigma_i EXPANDING (causal) y warm-up de `min_obs`.

    Parametros
    ----------
    calendar : DataFrame de releases YA parseado, con columnas obligatorias
        ['indicator', 'actual', 'consensus'] e indice DatetimeIndex de la fecha
        (y hora, si la hay) de publicacion. Un consenso NaN marca "release sin
        consenso": se omite y se cuenta (ver decision 2 del docstring del modulo).
    min_obs : minimo de sorpresas VALIDAS de un indicador antes de emitir su z_i
        (config news.sigma_min_obs; default del proyecto = 12). Con menos, z_i=NaN.

    Devuelve
    --------
    z_wide : DataFrame indexado por las fechas de release (ordenadas), una columna
        por indicador. z_wide.loc[t, ind] es el z-score de la sorpresa de `ind` si
        publico en t con consenso y ya paso el warm-up; NaN en cualquier otro caso
        (ese indicador no publico en t, o falto consenso, o aun no hay historia).
    coverage : dict {indicador: {n_releases, n_with_consensus, n_z_valid,
        n_skipped_no_consensus, n_warmup}} -- la cobertura del consenso como
        cantidad de primera clase (el consenso real puede estar bloqueado; que la
        escasez se vea, no se esconda).

    Causalidad (PIT): sigma_i(t) es el std EXPANDING de las sorpresas del
    indicador en releases con fecha <= t (incluido el de t, que ya es conocido en
    t; jamas releases futuros). Truncar el calendario en t y recomputar da el
    MISMO z_i(t): eso lo verifica test_sigma_expanding / test_si_pit.
    """
    required = {"indicator", "actual", "consensus"}
    missing = required - set(calendar.columns)
    if missing:
        raise ValueError(f"calendar necesita columnas {required}; faltan {missing}.")
    if not isinstance(calendar.index, pd.DatetimeIndex):
        raise TypeError("calendar debe estar indexado por DatetimeIndex de fecha de release.")

    cal = calendar.sort_index()
    indicators = list(pd.unique(cal["indicator"]))
    all_dates = cal.index.unique()

    z_cols: dict[str, pd.Series] = {}
    coverage: dict[str, dict] = {}

    for ind in indicators:
        sub = cal[cal["indicator"] == ind]
        n_releases = len(sub)
        # sorpresa cruda solo donde HAY consenso; el resto se omite (no se imputa).
        has_cons = sub["consensus"].notna()
        surprise = (sub["actual"] - sub["consensus"]).where(has_cons)
        valid = surprise.dropna()                       # releases con consenso
        n_with_cons = len(valid)
        n_skipped = n_releases - n_with_cons

        # sigma_i EXPANDING (incluye el punto actual, que ya es conocido) sobre las
        # sorpresas validas, en orden temporal. ddof=1 -> la 1a es NaN.
        sigma = valid.expanding(min_periods=2).std(ddof=1)
        count = valid.expanding().count()
        # warm-up: z solo cuando ya hay >= min_obs sorpresas validas acumuladas.
        z_valid = valid / sigma
        z_valid = z_valid.where(count >= min_obs)
        n_zvalid = int(z_valid.notna().sum())
        n_warmup = n_with_cons - n_zvalid

        # reindexar al calendario completo: NaN donde el indicador no publico.
        z_cols[ind] = z_valid.reindex(all_dates)
        coverage[ind] = {
            "n_releases": int(n_releases),
            "n_with_consensus": int(n_with_cons),
            "n_z_valid": n_zvalid,
            "n_skipped_no_consensus": int(n_skipped),
            "n_warmup": int(n_warmup),
        }
        if n_skipped:
            logger.info("%s: %d/%d releases sin consenso, omitidos (no imputados).",
                        ind, n_skipped, n_releases)

    z_wide = pd.DataFrame(z_cols, index=all_dates).sort_index()
    return z_wide, coverage


# --------------------------------------------------------------------------- #
# (b) Estimacion de w_i por regresion de impacto (R7)
# --------------------------------------------------------------------------- #
def _ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """OLS y = a + w x + u. Devuelve (w, se_w, n). SE clasico (homocedastico)."""
    n = x.shape[0]
    X = np.column_stack([np.ones(n), x])
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), n
    coef = XtX_inv @ (X.T @ y)
    resid = y - X @ coef
    dof = n - 2
    if dof <= 0:
        return float(coef[1]), float("nan"), n
    sigma2 = float(resid @ resid) / dof
    var_w = sigma2 * XtX_inv[1, 1]
    se_w = float(np.sqrt(var_w)) if var_w > 0 else float("nan")
    return float(coef[1]), se_w, n


def estimate_impact_weights(
    abs_window_ret: pd.Series,
    abs_z: dict[str, pd.Series],
    *,
    t_threshold: float = 2.0,
    min_events: int = 3,
    use_intraday: bool = True,
) -> dict[str, dict]:
    """Estima w_i por indicador con la regresion de impacto |r| = a + w_i|z_i| + u.

    R7 (corazon de la sesion): los w_i NO se asignan a mano; se ESTIMAN, con error
    estandar. Un w_i no distinguible de cero (|t| < t_threshold) entra con peso
    ~0 -- pero eso es un HALLAZGO que se marca (`distinguible_de_cero=False`), no
    se descarta el indicador en silencio.

    Parametros
    ----------
    abs_window_ret : |retorno de la ventana alrededor del release|, indexado por
        fecha de release. use_intraday documenta que fuente es.
    abs_z : dict {indicador: |z_i|} indexado por las fechas VALIDAS de ese
        indicador (salida de expanding_surprise_z tras abs()+dropna por columna).
    t_threshold : umbral de |t| para "distinguible de cero" (config; default 2).
    min_events : minimo de eventos para intentar la regresion; con menos, w=NaN.
    use_intraday : True si `abs_window_ret` es el |retorno| de una ventana
        INTRADIA ajustada al release (lo correcto: aisla la reaccion al dato).

        PLAN B documentado (use_intraday=False): si no hay datos intradia, se pasa
        el |retorno DIARIO| del dia del release. Degradacion: el retorno diario
        mezcla la reaccion al dato con todo lo demas que movio al mercado ese dia
        (otros datos, flujo, sesion externa), asi que w_i se estima con mas ruido y
        SESGADO A LA BAJA (mas varianza no explicada -> pendiente atenuada). El w_i
        sigue siendo interpretable pero su magnitud es un piso, no el efecto limpio.
        Esta funcion NO decide la ventana: usa la serie que se le pasa y registra
        `use_intraday` en cada resultado para que la degradacion viaje con el numero.

    Devuelve
    --------
    dict {indicador: {w, se, t_stat, distinguible_de_cero, n_events, use_intraday}}.
    """
    out: dict[str, dict] = {}
    for ind, z in abs_z.items():
        z = z.dropna()
        common = z.index.intersection(abs_window_ret.dropna().index)
        n = len(common)
        if n < min_events:
            out[ind] = {
                "w": float("nan"), "se": float("nan"), "t_stat": float("nan"),
                "distinguible_de_cero": False, "n_events": int(n),
                "use_intraday": use_intraday,
                "note": f"insuficientes eventos ({n} < {min_events}) para la regresion.",
            }
            continue
        x = np.abs(z.loc[common].to_numpy(dtype=float))
        y = np.abs(abs_window_ret.loc[common].to_numpy(dtype=float))
        w, se, _ = _ols_slope(x, y)
        t_stat = (w / se) if (se and np.isfinite(se) and se > 0) else float("nan")
        distinguible = bool(np.isfinite(t_stat) and abs(t_stat) >= t_threshold)
        out[ind] = {
            "w": w, "se": se, "t_stat": t_stat,
            "distinguible_de_cero": distinguible, "n_events": int(n),
            "use_intraday": use_intraday,
        }
        if not distinguible:
            logger.info("w[%s]=%.4g no distinguible de cero (t=%.2f): entra con peso ~0 (hallazgo, no descarte).",
                        ind, w, t_stat)
    return out


# --------------------------------------------------------------------------- #
# (c) Indice de sorpresa agregado SI_t con decaimiento exponencial
# --------------------------------------------------------------------------- #
DEFAULT_DELTA = np.log(2.0) / 25.0   # vida media ~25 dias; PUNTO DE PARTIDA del
                                     # optimizador, NO un valor fijo (delta se estima).


def surprise_index(
    weighted_events: pd.Series,
    target_index: pd.DatetimeIndex,
    *,
    delta: float,
) -> pd.Series:
    """SI_t = sum_{t_i <= t} c_i * exp(-delta*(t - t_i)) sobre `target_index`.

    Parametros
    ----------
    weighted_events : Series de contribuciones c_i = w_i * z_i por evento, indexada
        por la fecha del release. Varios indicadores el mismo dia se SUMAN (misma
        fecha -> sus c_i se agregan). El signo de z_i se conserva (SI_t puede ser
        negativo: sorpresas a la baja empujan distinto que al alza).
    target_index : calendario destino (fechas de los retornos del activo).
    delta : tasa de decaimiento (1/dias). NO es libre de esta funcion: la fija el
        optimizador del modelo completo (estimate.py). El default del proyecto es
        DEFAULT_DELTA = ln(2)/25 como arranque, no como verdad.

    El decaimiento usa la DISTANCIA EN DIAS CALENDARIO (t - t_i). Recursion O(N):
        SI_{j} = exp(-delta * dias(t_{j-1}, t_j)) * SI_{j-1} + (nuevos c_i en t_j]
    Solo eventos con t_i <= t entran en SI_t: causal por construccion.
    """
    if delta < 0:
        raise ValueError("delta debe ser >= 0.")
    target = pd.DatetimeIndex(target_index).sort_values()

    ev = weighted_events.dropna()
    if len(ev):
        ev = ev.groupby(level=0).sum().sort_index()     # suma same-day

    # Rejilla union: fechas de eventos + fechas destino, para arrastrar el estado.
    grid = target.union(pd.DatetimeIndex(ev.index)).sort_values() if len(ev) else target
    ev_on_grid = ev.reindex(grid).fillna(0.0) if len(ev) else pd.Series(0.0, index=grid)

    si = np.empty(len(grid))
    acc = 0.0
    prev = None
    grid_vals = grid.to_numpy()
    for j, tj in enumerate(grid_vals):
        if prev is not None:
            dt_days = (tj - prev) / np.timedelta64(1, "D")
            acc *= np.exp(-delta * dt_days)
        acc += float(ev_on_grid.iloc[j])
        si[j] = acc
        prev = tj

    si_series = pd.Series(si, index=grid)
    return si_series.reindex(target)


def surprise_feature(
    weighted_events: pd.Series,
    target_index: pd.DatetimeIndex,
    *,
    delta: float,
) -> pd.Series:
    """Covariable de sorpresa YA rezagada (R3) para el logit de transicion.

    Devuelve SI_{t-1}: SI_t calculado por surprise_index y luego .shift(1)
    EXPLICITO. El valor fechado en t es el indice de sorpresa conocido al cierre
    de t-1, que es el x_{t-1} que el logit usa para predecir la transicion HACIA t
    (ver models/tvtp.py y FEATURE_REGISTRY para el lag_ledger). Aplicar el shift
    aqui y solo aqui evita rezagar dos veces aguas abajo.
    """
    si = surprise_index(weighted_events, target_index, delta=delta)
    # ----------------------------------------------------------------------
    # R3: shift(1) EXPLICITO. Sin esta linea un release del propio dia t
    # entraria a predecir la transicion hacia t: look-ahead intradia.
    # ----------------------------------------------------------------------
    return si.shift(1).rename("surprise_index")


def weighted_events_from(z_wide: pd.DataFrame, weights: dict[str, dict]) -> pd.Series:
    """Construye la serie de contribuciones c_i = w_i * z_i por evento a partir de
    los z por indicador (expanding_surprise_z) y los pesos estimados (R7).

    Suma sobre indicadores por fecha: c(t) = sum_i w_i * z_i(t) para los z_i(t) no
    NaN. Un indicador con w_i no distinguible de cero entra con su w_i estimado
    (tipicamente ~0): NO se pone a cero a mano (R7), el propio w_i lo atenua.
    Indicadores sin w estimado (NaN por pocos eventos) se excluyen del agregado.
    """
    contrib = pd.Series(0.0, index=z_wide.index)
    any_event = pd.Series(False, index=z_wide.index)
    for ind in z_wide.columns:
        w = weights.get(ind, {}).get("w", np.nan)
        if not np.isfinite(w):
            continue
        zi = z_wide[ind]
        mask = zi.notna()
        contrib[mask] = contrib[mask] + w * zi[mask]
        any_event = any_event | mask
    # dias sin ningun evento valido -> se dejan como 0 (no aportan termino nuevo;
    # el decaimiento de surprise_index los arrastra). Devolvemos solo los dias con
    # evento para que surprise_index agregue correctamente.
    return contrib[any_event]


# --------------------------------------------------------------------------- #
# (d) Atribucion precio vs sorpresa
# --------------------------------------------------------------------------- #
def attribution(
    xi_prev: np.ndarray,
    xi_curr: np.ndarray,
    delta_SI: float | None,
    delta_price_features: np.ndarray | None,
    *,
    sens_SI: float | None = None,
    sens_price: np.ndarray | None = None,
) -> dict:
    """Reparte el movimiento de xi entre la covariable de sorpresa y las de precio.

    Devuelve {"surprise": s, "price": p} con s + p = 1 (y "no_news_in_window" si
    no hubo sorpresa en la ventana).

    Implementacion (sensibilidad numerica, normalizada): la contribucion de una
    covariable al cambio de xi se aproxima por |sensibilidad * cambio_covariable|,
    donde la sensibilidad es la derivada numerica de xi respecto a esa covariable
    (columna del jacobiano del filtro/transicion, evaluada perturbando la
    covariable y recomputando xi). El reparto es esa contribucion normalizada a 1.

    sens_SI / sens_price son esas sensibilidades numericas (||d xi / d cov||),
    tipicamente calculadas por el llamador perturbando el modelo. Si NO se
    proporcionan, se cae a un PROXY documentado: como surprise_index y las features
    de precio entran al MISMO logit estandarizadas (ambas en unidades de desviacion
    estandar), se usa la magnitud del movimiento estandarizado como base del
    reparto (|delta_SI| vs sum|delta_price|). El proxy ignora los betas (una
    covariable que se movio mucho pero pesa ~0 recibiria credito de mas): por eso
    lo correcto es pasar las sensibilidades; el proxy es el fallback honesto.

    Sin sorpresa en la ventana (delta_SI None/0 o sin evento) => toda la atribucion
    es de precio y se marca la flag, en vez de repartir un cambio que la sorpresa
    no causo.
    """
    xi_prev = np.asarray(xi_prev, dtype=float)
    xi_curr = np.asarray(xi_curr, dtype=float)
    no_news = delta_SI is None or (isinstance(delta_SI, float) and delta_SI == 0.0)

    dprice = (np.zeros(0) if delta_price_features is None
              else np.asarray(delta_price_features, dtype=float).ravel())

    if no_news:
        return {"surprise": 0.0, "price": 1.0, "no_news_in_window": True}

    if sens_SI is not None and sens_price is not None:
        sp = np.asarray(sens_price, dtype=float).ravel()
        contrib_s = abs(float(sens_SI) * float(delta_SI))
        contrib_p = float(np.sum(np.abs(sp * dprice)))
        basis = "sensibilidad numerica del modelo"
    else:
        contrib_s = abs(float(delta_SI))
        contrib_p = float(np.sum(np.abs(dprice)))
        basis = "proxy: magnitud del movimiento estandarizado (betas ignorados)"

    total = contrib_s + contrib_p
    if total <= 0:
        return {"surprise": 0.0, "price": 1.0, "no_news_in_window": True}
    return {
        "surprise": contrib_s / total,
        "price": contrib_p / total,
        "no_news_in_window": False,
        "basis": basis,
    }


def attribution_nway(
    deltas: dict[str, float | np.ndarray | None],
    sens: dict[str, float | np.ndarray] | None = None,
) -> dict:
    """Atribucion del movimiento de xi entre N componentes (V3: la pantalla 3
    reparte en 3 vias surprise / hawkes / price).

    Generaliza `attribution` (V2, 2 vias) con la misma semantica y el mismo
    proxy documentado: la contribucion de cada componente es
    sum(|sens_c * delta_c|) normalizada a 1. Si no se pasan sensibilidades se
    usa el proxy honesto de magnitud estandarizada (todas las covariables
    entran al MISMO logit en unidades de desviacion estandar); el proxy ignora
    los betas y se declara en "basis".

    `deltas[c]` es el cambio del componente c entre t-1 y t (float o vector;
    None = componente inactivo hoy, recibe 0 y se marca en inactive_components
    en vez de repartirle credito de un cambio que no se midio). "price" debe
    estar siempre presente; si TODOS los cambios son 0, la atribucion es 100%
    precio por convencion (no hubo nada de noticias que repartir).
    """
    if "price" not in deltas:
        raise ValueError("attribution_nway requiere el componente 'price'.")

    def _mag(name: str) -> float:
        d = deltas[name]
        if d is None:
            return 0.0
        d_arr = np.abs(np.asarray(d, dtype=float).ravel())
        if sens is not None and name in sens:
            s_arr = np.abs(np.asarray(sens[name], dtype=float).ravel())
            return float(np.sum(s_arr * d_arr))
        return float(np.sum(d_arr))

    inactive = [k for k, v in deltas.items() if v is None]
    contribs = {k: _mag(k) for k in deltas}
    total = sum(contribs.values())
    basis = ("sensibilidad numerica del modelo" if sens
             else "proxy: magnitud del movimiento estandarizado (betas ignorados)")
    if total <= 0:
        out = {k: 0.0 for k in deltas}
        out["price"] = 1.0
    else:
        out = {k: v / total for k, v in contribs.items()}
    out["basis"] = basis
    out["inactive_components"] = inactive
    return out

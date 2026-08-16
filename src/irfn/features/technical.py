"""Features tecnicos del TVTP: sma_gap y bb_width_z (V1).

Construccion (ventanas desde config/base.yaml, cero constantes magicas aqui):

    sma_gap    = (SMA_corta - SMA_larga) / std_movil_corta(close)
    bb_width   = (banda_sup - banda_inf) / SMA_bb
               = 2 * k * std_movil_bb(close) / SMA_bb
    bb_width_z = z-score movil de bb_width sobre z_window

Nota sobre k (multiplicador de Bollinger): en bb_width_z el k CANCELA (el
z-score es invariante a escala multiplicativa), asi que el valor de k no afecta
al modelo; se toma de config igualmente para que bb_width crudo sea el estandar.

--------------------------------------------------------------------------------
R3: TODA COVARIABLE ENTRA REZAGADA -- el shift(1) vive AQUI y SOLO aqui
--------------------------------------------------------------------------------
Cada columna del DataFrame devuelto lleva un .shift(1) EXPLICITO al final de su
construccion: el valor fechado en t es el feature calculado con precios hasta el
CIERRE de t-1. Ese es el x_{t-1} que el logit de transicion usa para predecir la
transicion HACIA t (p_ij(t) = softmax(beta'_ij x_{t-1}); ver models/tvtp.py).

El rezago se aplica UNA sola vez y en UN solo lugar (aqui), y queda registrado
en FEATURE_REGISTRY para el lag_ledger (audit/pit.py). Aguas abajo NADIE vuelve
a rezagar: hamilton_filter consume estas columnas tal cual (su argumento se
llama X_lagged para que el contrato sea imposible de ignorar).

Causalidad de las ventanas moviles: rolling(w) de pandas usa las w observaciones
que TERMINAN en t (pasado + presente, nunca futuro), y el z-score movil usa
media/std de esa misma ventana que termina en t. Por eso el feature crudo en t
depende solo de close[<=t], y tras el shift(1) la columna en t depende solo de
close[<=t-1]. test_features_prefix_invariance verifica esto empiricamente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Registro para el lag_ledger (R3). El shift aplicado es EXACTAMENTE el del
# codigo de abajo; si alguien cambia uno sin el otro, la auditoria lo delata.
FEATURE_REGISTRY: list[dict] = [
    {
        "feature": "sma_gap",
        "role": "covariable",
        "applied_shift": 1,
        "note": "(SMA corta - SMA larga) / std movil corta; shift(1) explicito en features/technical.py.",
    },
    {
        "feature": "bb_width_z",
        "role": "covariable",
        "applied_shift": 1,
        "note": "z-score movil del ancho Bollinger relativo; shift(1) explicito en features/technical.py.",
    },
]


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """z-score movil CAUSAL: (s_t - media_[t-w+1..t]) / std_[t-w+1..t].

    La ventana termina en t (pasado y presente; jamas futuro). ddof=1 (pandas).
    """
    m = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - m) / sd


def technical_features(
    close: pd.Series,
    *,
    sma_short: int,
    sma_long: int,
    bb_window: int,
    bb_k: float,
    z_window: int,
) -> pd.DataFrame:
    """Covariables tecnicas del TVTP, YA rezagadas (R3), indexadas como `close`.

    Devuelve DataFrame con columnas [sma_gap, bb_width_z]. Las primeras
    max(sma_long, bb_window + z_window) + 1 filas son NaN (warm-up de las
    ventanas + el rezago); el llamador decide el recorte con dropna, alineando
    con los retornos. Ventanas: desde config/base.yaml via el llamador.
    """
    if not isinstance(close, pd.Series):
        raise TypeError("technical_features espera un pd.Series de precios de cierre.")
    close = close.astype(float)

    sma_s = close.rolling(sma_short).mean()
    sma_l = close.rolling(sma_long).mean()
    std_s = close.rolling(sma_short).std()

    # sma_gap crudo en t: usa close[<= t] (ventanas causales).
    sma_gap_raw = (sma_s - sma_l) / std_s

    # Ancho de Bollinger relativo: (sup - inf)/SMA = 2k*std/SMA. El k cancela en
    # el z-score, pero se construye el ancho estandar por legibilidad.
    sma_bb = close.rolling(bb_window).mean()
    std_bb = close.rolling(bb_window).std()
    bb_width_raw = (2.0 * bb_k * std_bb) / sma_bb
    bb_width_z_raw = rolling_zscore(bb_width_raw, z_window)

    # ----------------------------------------------------------------------
    # R3: shift(1) EXPLICITO. El valor fechado en t pasa a ser el feature del
    # cierre de t-1: es el x_{t-1} del logit de transicion (models/tvtp.py).
    # Sin esta linea, el modelo "predeciria" la transicion hacia t usando el
    # cierre de t: look-ahead. Registrado en FEATURE_REGISTRY -> lag_ledger.
    # ----------------------------------------------------------------------
    sma_gap = sma_gap_raw.shift(1)
    bb_width_z = bb_width_z_raw.shift(1)

    out = pd.DataFrame({"sma_gap": sma_gap, "bb_width_z": bb_width_z}, index=close.index)
    # std==0 en ventanas de precio plano produciria +-inf; se degrada a NaN para
    # que la alineacion (dropna) lo excluya en vez de envenenar el optimizador.
    return out.replace([np.inf, -np.inf], np.nan)

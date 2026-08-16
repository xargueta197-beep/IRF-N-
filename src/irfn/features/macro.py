"""Features macro VINTAGE-AWARE del TVTP: slope_2s10y y hy_oas_z (V1, R4).

    slope_2s10y = DGS10 - DGS2          (pendiente 2s10y de la curva Treasury)
    hy_oas_z    = z-score movil de BAMLH0A0HYM2 (ICE BofA HY OAS)

--------------------------------------------------------------------------------
El rezago de esta capa NO es un shift(1) arbitrario (R3 + R4)
--------------------------------------------------------------------------------
Cada valor entra al modelo en la fecha en que REALMENTE estuvo disponible: su
realtime_start de ALFRED (fecha de publicacion) mas un margen de disponibilidad
de config (macro.availability_margin_days). Un dato de enero publicado el 25 de
febrero existe para el modelo desde el 25 de febrero, no antes. Eso lo hace
data/alfred.py (point_in_time_series); aqui NO se aplica ningun shift adicional
-- aplicarlo seria rezagar dos veces.

Para las series de esta capa el data_audit documenta que DGS2/DGS10 (H.15) no se
revisan tras publicarse: el vintage no cambia el VALOR, pero la fecha de
publicacion real sigue importando (el dato del martes se publica al final del
martes; el margen lo hace usable el miercoles, como el cierre tecnico). Para
BAMLH0A0HYM2 ademas aplica el hallazgo de la ventana rodante de 3 anios (abril
2026): la cobertura real la reporta el vintage ledger, y si no alcanza para el
walk-forward se declara bloqueante -- jamas se rellena con FRED revisado.

El z-score de hy_oas es una ventana movil CAUSAL (termina en t) sobre la serie
point-in-time: cada valor dentro de la ventana es el que se conocia en su fecha.
"""

from __future__ import annotations

import pandas as pd

from irfn.features.technical import rolling_zscore

# Registro para el lag_ledger (R3/R4). applied_shift=1 es el margen de
# disponibilidad en dias habiles (config macro.availability_margin_days) que se
# suma al rezago REAL de publicacion (realtime_start del vintage ALFRED), que es
# variable por serie y fecha y queda documentado en el vintage ledger.
FEATURE_REGISTRY: list[dict] = [
    {
        "feature": "slope_2s10y",
        "role": "covariable",
        "applied_shift": 1,
        "note": (
            "DGS10 - DGS2 desde vintages ALFRED; rezago = fecha de publicacion real "
            "(realtime_start) + margen de 1 dia habil. Sin shift adicional."
        ),
    },
    {
        "feature": "hy_oas_z",
        "role": "covariable",
        "applied_shift": 1,
        "note": (
            "z-score movil causal de BAMLH0A0HYM2 point-in-time (ALFRED); rezago = "
            "publicacion real + margen de 1 dia habil. Cobertura en vintage ledger."
        ),
    },
]


def macro_features(
    pit_series: dict[str, pd.Series],
    target_index: pd.DatetimeIndex,
    *,
    z_window: int,
) -> pd.DataFrame:
    """Construye las covariables macro alineadas al calendario del activo.

    Parametros
    ----------
    pit_series : dict con las series point-in-time de data/alfred.py, indexadas
        por su fecha de DISPONIBILIDAD real. Claves esperadas: "DGS10", "DGS2",
        "BAMLH0A0HYM2" (las que falten producen columnas ausentes, no errores
        silenciosos: el llamador decide si puede seguir sin ellas).
    target_index : calendario del activo ancla (fechas de los retornos).
    z_window : ventana del z-score movil (config windows.z_window).

    La alineacion usa reindex + ffill: entre publicaciones, el ultimo valor
    conocido sigue siendo el conocido (eso es informacion point-in-time
    legitima, no relleno cosmetico). NO se aplica shift adicional: el rezago ya
    viene en el indice de disponibilidad (ver docstring del modulo).
    """
    cols: dict[str, pd.Series] = {}

    def _align(s: pd.Series) -> pd.Series:
        # union de calendarios y arrastre del ultimo valor CONOCIDO; luego se
        # recorta al calendario del activo.
        return s.reindex(s.index.union(target_index)).ffill().reindex(target_index)

    if "DGS10" in pit_series and "DGS2" in pit_series:
        cols["slope_2s10y"] = _align(pit_series["DGS10"]) - _align(pit_series["DGS2"])

    if "BAMLH0A0HYM2" in pit_series:
        oas = _align(pit_series["BAMLH0A0HYM2"])
        cols["hy_oas_z"] = rolling_zscore(oas, z_window)

    return pd.DataFrame(cols, index=target_index)

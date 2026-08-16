"""Walk-forward economico pre-registrado (Test 5 de V4, parte economica).

Implementa EXACTAMENTE la regla congelada en docs/preregistro_regla_trading.md
(aprobada por el director 2026-07-18). Cualquier cambio de regla, umbral o
costo despues de ver resultados esta prohibido por el pre-registro (R8): las
variantes serian un segundo pre-registro, corrido y etiquetado por separado.

Regla (des-riesgo binario, sin senal direccional):

    w_{t}   = 1           si  P_{t-1}(alta vol) <= p_threshold
    w_{t}   = w_reduced   si  P_{t-1}(alta vol) >  p_threshold

P es la probabilidad FILTRADA xi_{t|t} del regimen de mayor varianza
incondicional (ultimo indice bajo R5) — nunca la suavizada (R1) — y entra con
shift(1) explicito (R3): la posicion que rige el retorno de t se decide con lo
conocido al cierre de t-1. El primer dia del periodo OOS arranca comprado
(w=1), documentado en el pre-registro.

Costos: cost_bps por unidad de turnover (|delta w|), cargados solo cuando la
posicion cambia. r esta en % (log-retornos x 100), asi que 1 pb = 0.01 en esas
unidades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from irfn.validation.bootstrap import sharpe_ci

# Dias habiles por anio para anualizar (convencion estandar de mercado diario;
# mismo valor que usa el resto de la validacion V4).
TRADING_DAYS = 252


def _sharpe_annualized(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS))


def _max_drawdown(r_pct: np.ndarray) -> float:
    """Max drawdown (fraccion positiva) del equity exp(cumsum(r/100))."""
    equity = np.exp(np.cumsum(np.asarray(r_pct, dtype=float) / 100.0))
    peak = np.maximum.accumulate(equity)
    return float((1.0 - equity / peak).max())


def strategy_returns(
    history: pd.DataFrame,
    *,
    p_threshold: float,
    w_reduced: float,
    cost_bps: float,
) -> pd.DataFrame:
    """Serie diaria de la estrategia sobre el frame OOS del walk-forward.

    history requiere columnas: r (%, log-retorno del dia), xi_filtered_* (la
    ultima es el regimen de mayor varianza por R5). Devuelve DataFrame con
    w (posicion vigente el dia t), turnover, cost (en %), r_strategy (%).
    """
    xi_cols = sorted(
        (c for c in history.columns if c.startswith("xi_filtered_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    if not xi_cols:
        raise ValueError("history no tiene columnas xi_filtered_*: no hay regimen que leer.")
    p_high = history[xi_cols[-1]]  # ultimo indice = mayor varianza (R5)

    # shift(1) EXPLICITO (R3): la posicion del dia t se decide con xi_{t-1|t-1}.
    # El primer dia no tiene ayer dentro del periodo OOS: arranca comprado
    # (w=1), tal como esta congelado en el pre-registro.
    signal = p_high.shift(1)
    w = pd.Series(
        np.where(signal.isna(), 1.0, np.where(signal <= p_threshold, 1.0, w_reduced)),
        index=history.index,
        name="w",
    )
    # turnover del dia t = |w_t - w_{t-1}|; la posicion previa al primer dia es
    # el mismo w=1 inicial, asi que el arranque no carga costo.
    turnover = (w - w.shift(1).fillna(1.0)).abs()
    cost = turnover * cost_bps * 0.01  # pb -> unidades de r (%)
    r_strategy = w * history["r"] - cost

    return pd.DataFrame(
        {"w": w, "turnover": turnover, "cost": cost, "r_strategy": r_strategy}
    )


def economic_walkforward(
    history: pd.DataFrame,
    *,
    p_threshold: float,
    w_reduced: float,
    cost_bps: float,
    cost_bps_sensitivity: list[float],
    n_boot: int,
    seed: int,
) -> dict:
    """Corre la evaluacion economica pre-registrada y devuelve el dict del
    artefacto (validation.json). El veredicto principal es a cost_bps; las
    sensibilidades son informativas y no cambian el veredicto (pre-registro
    seccion 3)."""
    r_bh = history["r"].to_numpy(dtype=float)

    def evaluate(cost: float) -> dict:
        strat = strategy_returns(
            history, p_threshold=p_threshold, w_reduced=w_reduced, cost_bps=cost
        )
        r_st = strat["r_strategy"].to_numpy(dtype=float)
        diff = r_st - r_bh
        ci = sharpe_ci(diff, n_boot=n_boot, seed=seed)
        return {
            "cost_bps": cost,
            "sharpe_strategy": _sharpe_annualized(r_st),
            "sharpe_buyhold": _sharpe_annualized(r_bh),
            "sharpe_diff": ci["sharpe"],
            "sharpe_diff_ci95": [ci["ci_lower"], ci["ci_upper"]],
            "sharpe_diff_block_length": ci["block_length"],
            "sharpe_diff_includes_zero": ci["includes_zero"],
            "ann_return_strategy_pct": float(r_st.mean() * TRADING_DAYS),
            "ann_return_buyhold_pct": float(r_bh.mean() * TRADING_DAYS),
            "ann_vol_strategy_pct": float(r_st.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "ann_vol_buyhold_pct": float(r_bh.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "max_drawdown_strategy": _max_drawdown(r_st),
            "max_drawdown_buyhold": _max_drawdown(r_bh),
            "exposure_fraction": float((strat["w"] > 0).mean()),
            "n_position_changes": int((strat["turnover"] > 0).sum()),
            "total_turnover": float(strat["turnover"].sum()),
        }

    main = evaluate(cost_bps)
    # exito SOLO si el IC de la diferencia excluye 0 POR ARRIBA a cost_bps
    # (pre-registro seccion 5).
    lo = main["sharpe_diff_ci95"][0]
    success = bool(lo is not None and np.isfinite(lo) and lo > 0.0)

    per_block = []
    if "block_id" in history.columns:
        strat_main = strategy_returns(
            history, p_threshold=p_threshold, w_reduced=w_reduced, cost_bps=cost_bps
        )
        diff_all = strat_main["r_strategy"] - history["r"]
        for block_id, idx in history.groupby("block_id").groups.items():
            per_block.append(
                {
                    "block_id": int(block_id),
                    "n_days": int(len(idx)),
                    "sharpe_diff": _sharpe_annualized(diff_all.loc[idx].to_numpy()),
                    "exposure_fraction": float((strat_main.loc[idx, "w"] > 0).mean()),
                }
            )

    return {
        "preregistration": "docs/preregistro_regla_trading.md",
        "rule": {
            "type": "binary_derisk_high_vol",
            "p_threshold": p_threshold,
            "w_reduced": w_reduced,
            "cost_bps": cost_bps,
        },
        "success_criterion": (
            "IC 95% bootstrap del Sharpe anualizado de (estrategia - buyhold) "
            f"excluye 0 por arriba, a {cost_bps} pb de costo"
        ),
        "success": success,
        "main": main,
        "cost_sensitivity": [evaluate(c) for c in cost_bps_sensitivity],
        "per_block": per_block,
        "n_obs": int(len(history)),
    }

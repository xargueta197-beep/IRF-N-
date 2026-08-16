"""Checkpoint generico para corridas largas (kselect/ablation/v2/v3).

No es parte del contrato de salida (artifacts/latest/*.json): son archivos
efimeros de reanudacion, en pickle, para que un corte del proceso (sistema
suspendido, terminal cerrada, etc.) no obligue a repetir horas de multistart
ya computadas. Escritura atomica (tmp + replace) para que un corte A MITAD de
un guardado nunca deje un checkpoint corrupto que parezca valido.

El resume es exacto: cada pieza reanudada (replica de bootstrap, bloque de
walk-forward, fila de la tabla BIC) se recalcularia con la MISMA semilla si no
hubiera checkpoint, asi que reanudar produce bit a bit el mismo resultado que
una corrida sin cortes (R6: reproducibilidad).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger("irfn.checkpoint")


def save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)  # rename es atomico en el mismo filesystem


def load(path: Path, *, expected_sig: Any) -> dict[str, Any] | None:
    """Devuelve el payload si el checkpoint existe, no esta corrupto, y su
    `sig` coincide con `expected_sig` (mismos parametros de la corrida que lo
    generaria desde cero). Si algo no calza, se ignora silenciosamente (log
    info) y se arranca de cero -- nunca se usa un checkpoint que no se puede
    verificar."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        logger.warning("checkpoint %s corrupto o ilegible; se ignora, arranca de cero.", path)
        return None
    if payload.get("sig") != expected_sig:
        logger.info("checkpoint %s no coincide con los parametros actuales; se ignora.", path)
        return None
    return payload

"""Relevancia de titulares s_i in [0,1] con FinBERT LOCAL.

--------------------------------------------------------------------------------
TRAMPA 4 (guia 7.2) -- RESTRICCION DURA, NO NEGOCIABLE EN ESTE MODULO
--------------------------------------------------------------------------------
PROHIBIDO preguntarle al clasificador si la noticia es alcista o bajista.

Un modelo entrenado DESPUES de los titulares que clasifica CONOCE EL DESENLACE:
un LLM de 2026 etiquetando "Lehman files for bankruptcy" como bajista no esta
leyendo el texto, esta recordando 2008. Es look-ahead sutil pero real, y no se
cura con disciplina de uso: se cura restringiendo LA PREGUNTA.

La unica pregunta permitida es: "que tan relevante/intensa es esta noticia?"
-> s_i in [0,1]. Eso es lo que el proceso de Hawkes necesita (la marca s_i
modula cuanto excita el titular la llegada de titulares siguientes), y es una
propiedad del TEXTO, no del futuro.

Implementacion de esa restriccion con FinBERT (ProsusAI/finbert, cabeza de
sentimiento financiero con clases positive/negative/neutral):

    s_i = 1 - P(neutral | titular)

  - USA la probabilidad de "neutral" como medida inversa de intensidad: un
    titular sin carga financiera reparte masa hacia neutral; uno intenso
    (en cualquier direccion) se la quita.
  - NUNCA usa P(positive) - P(negative), ni el argmax, ni el signo: la
    DIRECCION se descarta a proposito y este modulo no la expone. Si alguien
    agrega una funcion que devuelva el signo del sentimiento, esta violando la
    Trampa 4 y el codigo se revierte (CLAUDE.md).
  - FinBERT ademas fue entrenado (Financial PhraseBank, 2013-2019) ANTES del
    final de la muestra, lo que reduce -- no elimina -- el riesgo de que "el
    modelo recuerde el desenlace". La mitigacion principal sigue siendo la
    restriccion de la pregunta.

El modelo corre LOCAL (transformers + torch CPU): sin API externa, sin que los
titulares salgan de la maquina, reproducible con la version pineada del modelo.

LIMITACION MEDIDA (smoke test de esta sesion): FinBERT esta entrenado SOLO en
texto financiero; fuera de dominio ("local bakery wins bread competition") su
distribucion de clases no es fiable y 1 - P(neutral) puede salir alto. El
filtro de dominio NO es este modulo: es la consulta GDELT (config/news.yaml
headlines.query), que restringe el corpus a titulares financieros. Dentro del
dominio el score discrimina como se espera (crash de mercado ~0.9-0.97; "Fed
leaves rates unchanged, in line with expectations" ~0.26).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("irfn.relevance")

ROOT = Path(__file__).resolve().parents[3]
CACHE_PATH = ROOT / "data" / "interim" / "headline_relevance.parquet"


class RelevanceModelUnavailable(RuntimeError):
    """FinBERT local no disponible (dependencias o descarga del modelo). El
    orquestador documenta esto como bloqueante (R8); no hay fallback heuristico
    a proposito: un score de relevancia inventado a mano seria un peso a mano
    (el espiritu de R7 aplicado a las marcas)."""


def _title_key(title: str) -> str:
    return hashlib.sha1(title.strip().lower().encode("utf-8")).hexdigest()


class FinbertRelevance:
    """Scorer de relevancia. Carga perezosa: el modelo (~440 MB la primera vez)
    solo se descarga/carga si de verdad hay titulares que puntuar."""

    def __init__(self, model_id: str, batch_size: int):
        self.model_id = model_id
        self.batch_size = int(batch_size)
        self._tokenizer = None
        self._model = None
        self._neutral_idx: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RelevanceModelUnavailable(
                "transformers/torch no instalados en el venv; "
                "pip install transformers torch (CPU). Sin FinBERT no hay s_i "
                "y la capa de Hawkes queda bloqueada (no hay fallback heuristico)."
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            self._model.eval()
        except Exception as exc:  # noqa: BLE001 -- descarga/HF hub: bloqueante documentado
            raise RelevanceModelUnavailable(
                f"no se pudo cargar el modelo local '{self.model_id}': {exc}"
            ) from exc
        id2label = {int(k): v.lower() for k, v in self._model.config.id2label.items()}
        neutral = [i for i, lbl in id2label.items() if lbl == "neutral"]
        if not neutral:
            raise RelevanceModelUnavailable(
                f"el modelo '{self.model_id}' no tiene clase 'neutral' "
                f"(labels={id2label}); s_i = 1 - P(neutral) no es computable."
            )
        self._neutral_idx = neutral[0]

    def score(self, titles: list[str]) -> np.ndarray:
        """s_i = 1 - P(neutral) por titular. Ver docstring del modulo: la
        direccion (positive vs negative) se descarta a proposito (Trampa 4)."""
        self._load()
        import torch

        out = np.empty(len(titles))
        with torch.no_grad():
            for lo in range(0, len(titles), self.batch_size):
                batch = titles[lo: lo + self.batch_size]
                enc = self._tokenizer(
                    batch, padding=True, truncation=True, max_length=64, return_tensors="pt",
                )
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                # 1 - P(neutral): intensidad sin direccion. NO tocar las otras
                # dos columnas por separado (eso seria preguntar la direccion).
                out[lo: lo + len(batch)] = (1.0 - probs[:, self._neutral_idx]).cpu().numpy()
        return np.clip(out, 0.0, 1.0)


def score_headlines(
    headlines: pd.DataFrame,
    *,
    model_id: str,
    batch_size: int,
    cache_path: Path = CACHE_PATH,
) -> pd.DataFrame:
    """Agrega la columna `s` (relevancia) a un DataFrame de titulares.

    Cache por hash del titulo en data/interim/ (fuera de artifacts/): puntuar
    cientos de miles de titulares en CPU es caro y determinista, asi que un
    titular ya visto no se re-puntua. La cache guarda tambien el model_id: si
    cambia el modelo, los scores viejos NO se reutilizan (serian de otra
    funcion s_i) y se recalcula todo.
    """
    df = headlines.copy()
    if df.empty:
        df["s"] = pd.Series(dtype=float)
        return df

    df["_key"] = df["title"].map(_title_key)

    cached = None
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if not (cached["model_id"] == model_id).all():
            logger.warning("cache de relevancia con model_id distinto; se descarta y recalcula.")
            cached = None
    if cached is not None:
        df = df.merge(cached[["_key", "s"]], on="_key", how="left")
    else:
        df["s"] = np.nan

    todo = df["s"].isna()
    n_todo = int(todo.sum())
    if n_todo:
        logger.info("puntuando %d titulares nuevos con %s (batch=%d, CPU)...",
                    n_todo, model_id, batch_size)
        scorer = FinbertRelevance(model_id, batch_size)
        titles = df.loc[todo, "title"].tolist()
        df.loc[todo, "s"] = scorer.score(titles)
        new_cache = df[["_key", "s"]].drop_duplicates("_key").assign(model_id=model_id)
        if cached is not None:
            new_cache = pd.concat([cached, new_cache]).drop_duplicates("_key")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        new_cache.to_parquet(cache_path, index=False)

    return df.drop(columns="_key")

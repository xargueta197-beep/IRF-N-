"""Contrato de salida: artifacts/latest/irfn.json.

Este esquema se escribe ANTES que el modelo, a proposito: el contrato manda.
Si el esquema cambia, la interfaz rompe a proposito (ver CLAUDE.md, seccion
CONTRATO DE SALIDA). extra="forbid" en cada modelo para que un campo agregado
o renombrado sin querer falle ruidosamente, no en silencio.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EstimationInfo(_Strict):
    method: str
    multistart: int
    seed: int
    converged: bool


class NewsIndicatorWeight(_Strict):
    """w_i estimado por regresion de impacto (R7), un indicador del calendario."""

    indicator: str
    w: float | None
    se: float | None
    t_stat: float | None
    distinguible_de_cero: bool
    n_events: int


class NewsLayerParams(_Strict):
    """Parametros de la capa de sorpresa (V2). Todo estimado (R7), nada a mano.

    `active` refleja config/news.yaml surprise_layer; puede ser False aunque
    haya w_i/delta estimados (p.ej. datos insuficientes para el walk-forward
    pero suficientes para una estimacion preliminar). `blocker` es la razon
    documentada (R8) cuando la capa no pudo activarse; None si no hay bloqueo.
    """

    active: bool
    delta: float | None
    delta_se: float | None
    surprise_start_date: date | None
    indicators: list[NewsIndicatorWeight]
    coverage: dict[str, dict[str, int | str | None]]
    blocker: str | None


class HawkesCoverage(_Strict):
    """Cobertura del corpus de titulares (data/raw/headlines/): la cobertura es
    cantidad de primera clase -- sin ella, un lambda_N bajo es indistinguible
    de 'no descargue los titulares'."""

    first_day: str | None
    last_day: str | None
    n_days: int
    n_missing_days: int
    n_censored_days: int


class HawkesLayerParams(_Strict):
    """Parametros de la capa de Hawkes (V3). Todo estimado por MLE (R7).

    `stationary` es el chequeo n = alpha*E[s]/beta < 1: si es False el proceso
    es explosivo y NO se publica como indicador (blocker lo documenta). El KS
    de re-escalamiento temporal se reporta PASE O FALLE (guia 6.6: si falla no
    se fuerza el modelo, se documenta y se considera kernel power-law).
    """

    active: bool
    mu_N: float | None
    alpha: float | None
    beta: float | None
    se_mu_N: float | None
    se_alpha: float | None
    se_beta: float | None
    mean_mark: float | None
    branching_ratio: float | None
    # IC del branching ratio (metodo delta, Parte B). expected_cascade = None cuando
    # se censura (IC superior >= cascade_ci_trigger); expected_cascade_bounded lo marca.
    branching_ratio_se: float | None
    branching_ratio_ci_low: float | None
    branching_ratio_ci_high: float | None
    expected_cascade: float | None
    expected_cascade_bounded: bool | None
    stationary: bool | None
    ks_stat: float | None
    ks_pvalue: float | None
    ks_passed: bool | None
    n_events: int
    n_starts: int
    starts_at_best: int
    coverage: HawkesCoverage
    # umbral de la alerta "modo reflexivo" (config news.hawkes.reflexive_threshold);
    # viaja en el artefacto para que la app no lea config (R9).
    reflexive_threshold: float | None
    blocker: str | None


class ModelBlock(_Strict):
    K: int
    spec: str
    tvtp: bool
    covariates: list[str]
    news_layer: list[str]
    estimation: EstimationInfo
    news_layer_params: NewsLayerParams
    hawkes_layer_params: HawkesLayerParams


class RegimeBlock(_Strict):
    labels: list[str]
    xi_filtered: list[float]
    entropy: float
    entropy_max: float
    confidence: str
    expected_duration_days: list[float]
    argmax: str
    xi_momentum_5d: list[float]


class NewsBlock(_Strict):
    surprise_index: float
    lambda_N: float
    lambda_N_z: float
    branching_ratio: float
    branching_ratio_ci_low: float | None
    branching_ratio_ci_high: float | None
    # None = "no acotada" (censura de la Parte B cuando el IC superior de n >= trigger).
    expected_cascade: float | None
    expected_cascade_bounded: bool
    attribution: dict[str, float]


class MetricWithCI(_Strict):
    """Estadistico puntual + IC por bootstrap estacionario (Politis-Romano).

    ci_low/ci_high son None cuando no hay suficientes observaciones en el
    regimen para intentar el bootstrap (config v2.bootstrap.min_obs): la celda
    se muestra vacia en pantalla 4, no se inventa un intervalo de dos datos.

    value es None cuando la metrica es una anualizada (mean_ann/sharpe/maxdd)
    y el regimen no es anualizable de forma honesta (F2.c, auditoria
    2026-08-18): E[D] < degenerate_duration_days (no persiste, "un anio" no
    tiene sentido) O n_obs < bootstrap_min_obs (muy poca precision). n_obs
    siempre viaja, anualizable o no, para que la pantalla pueda mostrar
    "no anualizable -- N obs" en vez de imprimir un punto sin contexto.
    """

    value: float | None
    ci_low: float | None
    ci_high: float | None
    n_obs: int


class ConditionalStatsEntry(_Strict):
    mean_ann: MetricWithCI
    vol_ann: MetricWithCI
    sharpe: MetricWithCI
    maxdd: MetricWithCI


class IRFNOutput(_Strict):
    run_id: str
    generated_at: datetime
    git_commit: str
    config_hash: str
    asof: date
    version: str

    # El contrato usa la clave literal "model" (ver CLAUDE.md); se mapea via
    # alias porque "model" como nombre de atributo de instancia sombrearia
    # metodos heredados de BaseModel en algunas versiones de pydantic.
    model_block: ModelBlock = Field(alias="model")
    regime: RegimeBlock
    transition_matrix_today: list[list[float]]
    news: NewsBlock
    conditional_stats: dict[str, dict[str, ConditionalStatsEntry]]

    warnings: list[str]
    validation_ref: str
    disclaimer: str

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    def to_contract_dict(self) -> dict:
        """Serializa con las claves literales del contrato (incluye "model")."""
        return self.model_dump(mode="json", by_alias=True)

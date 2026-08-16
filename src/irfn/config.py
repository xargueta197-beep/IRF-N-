"""Carga y valida config/*.yaml. Cero constantes magicas fuera de este modulo
y de los YAML que lee: si un numero vive en otro archivo .py sin venir de aqui
ni de una estimacion, es un bug (ver CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class ModelConfig(BaseModel):
    K: int
    seed: int
    n_multistart: int


class WindowsConfig(BaseModel):
    sma_short: int
    sma_long: int
    bb_window: int
    bb_k: float
    z_window: int


class WalkforwardConfig(BaseModel):
    n_blocks: int = Field(ge=6, description="Minimo 6 bloques de test (R2, R8)")
    train_years: int
    test_months: int


class RegimeConfig(BaseModel):
    entropy_threshold: float = Field(description="H/ln(K) >= este => sin senal clara")
    entropy_mid: float = Field(description="H/ln(K) >= este => media; por debajo => alta")


class V0Config(BaseModel):
    anchor_asset: str
    returns_start: str
    wf_n_starts: int = Field(ge=20, le=50, description="arranques por bloque WF (R6)")


class TVTPConfig(BaseModel):
    l1_grid: list[float] = Field(description="malla del lambda L1; CV dentro del train")
    cv_val_frac: float = Field(gt=0.0, lt=1.0)
    cv_n_starts: int = Field(ge=1)
    l1_smooth_eps: float = Field(gt=0.0)


class KTestConfig(BaseModel):
    n_boot: int = Field(ge=19, description="replicas del bootstrap parametrico K vs K-1")
    boot_n_starts: int = Field(ge=1)
    bic_n_starts: int = Field(ge=20, le=50, description="multistart de la tabla BIC (R6)")


class V1Config(BaseModel):
    prices_start: str
    k_candidates: list[int]
    dist_candidates: list[str]
    covariates: list[str]
    covariates_ablation_m2: list[str]
    tvtp: TVTPConfig
    ktest: KTestConfig


class MacroSeries(BaseModel):
    series_id: str
    name: str


class MacroConfig(BaseModel):
    availability_margin_days: int = Field(ge=0)
    use_recovered_prefix: bool = Field(
        default=False,
        description=(
            "antepone el historico recuperado de data/raw/recovered/ (Wayback) a las "
            "vintages ALFRED truncadas; decision del director 2026-07-18, data_audit seccion 3"
        ),
    )
    series: list[MacroSeries]


class BootstrapConfig(BaseModel):
    n_boot: int = Field(ge=99, description="replicas del bootstrap estacionario (Politis-Romano)")
    block_len: int = Field(gt=0, description="longitud media del bloque geometrico, en dias")
    ci_level: float = Field(gt=0.0, lt=1.0)
    min_obs: int = Field(ge=2, description="minimo de dias en el regimen para intentar el bootstrap")


class DeltaMLEConfig(BaseModel):
    n_starts: int = Field(ge=1, description="multistart del fit dedicado que estima delta (R6)")
    min_events_total: int = Field(ge=1, description="eventos validos minimos para intentar el fit conjunto")


class V2Config(BaseModel):
    # M3 = M2 + macro (guia 8.1); M4 = M3 + surprise_index (se agrega en
    # validation/ablation.py, no aqui, para no duplicar el nombre de la
    # covariable en dos lugares de config).
    covariates_ablation_m3: list[str]
    news_covariate_name: str = "surprise_index"
    bootstrap: BootstrapConfig
    delta_mle: DeltaMLEConfig


class EconomicRuleConfig(BaseModel):
    # Regla economica PRE-REGISTRADA (docs/preregistro_regla_trading.md,
    # aprobada 2026-07-18). Estos valores se congelaron ANTES de correr el test
    # y no se tocan despues de ver resultados (R8).
    p_threshold: float = Field(gt=0.0, lt=1.0)
    w_reduced: float = Field(ge=0.0, le=1.0)
    cost_bps: float = Field(ge=0.0)
    cost_bps_sensitivity: list[float]
    n_boot: int = Field(ge=99)


class V4Config(BaseModel):
    economic: EconomicRuleConfig


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(frozen=True)

    model: ModelConfig
    windows: WindowsConfig
    walkforward: WalkforwardConfig
    regime: RegimeConfig
    v0: V0Config
    v1: V1Config
    v2: V2Config
    v4: V4Config
    macro: MacroConfig

    @classmethod
    def load(cls, path: Path = CONFIG_DIR / "base.yaml") -> "BaseConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


class AssetConfig(BaseModel):
    name: str
    source: str
    start_date: str
    symbol: str | None = None
    ticker: str | None = None
    interval: str | None = None

    @property
    def identifier(self) -> str:
        """El symbol/ticker que se pasa a data/prices.py, segun `source`."""
        ident = self.symbol or self.ticker
        if ident is None:
            raise ValueError(f"activo {self.name!r} no tiene symbol ni ticker en assets.yaml")
        return ident


class AssetsConfig(BaseSettings):
    model_config = SettingsConfigDict(frozen=True)

    assets: list[AssetConfig]

    @classmethod
    def load(cls, path: Path = CONFIG_DIR / "assets.yaml") -> "AssetsConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    def get(self, name: str) -> AssetConfig:
        for a in self.assets:
            if a.name == name:
                return a
        raise KeyError(f"activo {name!r} no esta en config/assets.yaml")


class NewsIndicator(BaseModel):
    name: str
    series_id: str
    w_i_placeholder: float


class DecayConfig(BaseModel):
    delta_placeholder: float


class HeadlinesConfig(BaseModel):
    """Ingesta de titulares GDELT (V3). Cambiar `query` invalida el historico
    acumulado en data/raw/headlines/ (otra poblacion de eventos)."""

    enabled: bool = True
    query: str
    start_date: str
    max_records: int = Field(ge=1, le=250, description="cap duro del DOC 2.0 API por consulta")
    request_spacing_seconds: float = Field(gt=0.0)
    rate_limit_backoff_seconds: float = Field(gt=0.0)
    max_backoff_retries: int = Field(ge=0)
    min_window_hours: int = Field(ge=1, description="tramo minimo antes de aceptar censura por cap")
    match_window_hours: int = Field(ge=1, description="ventana titular<->release para la auditoria de timestamps")
    min_events_fit: int = Field(ge=10, description="titulares minimos con s_i para intentar el MLE de Hawkes")


class RelevanceConfig(BaseModel):
    """Clasificador de relevancia s_i (V3, Trampa 4: relevancia, jamas direccion)."""

    finbert_model: str
    batch_size: int = Field(ge=1)


class HawkesLayerConfig(BaseModel):
    n_starts: int = Field(ge=20, le=50, description="multistart del MLE de Hawkes (R6)")
    reflexive_threshold: float = Field(gt=0.0, lt=1.0, description="n >= este => alerta modo reflexivo")
    cascade_ci_trigger: float = Field(
        gt=0.0, le=1.0,
        description="si el IC superior de n alcanza este umbral, la cascada 1/(1-n) se "
                    "reporta 'no acotada' en vez de un valor puntual (Parte B, 2026-08-15)",
    )


class NewsConfig(BaseSettings):
    model_config = SettingsConfigDict(frozen=True)

    # Capa de sorpresa (V2). Cero constantes magicas: viven en config/news.yaml.
    surprise_layer: bool = False
    sigma_min_obs: int = Field(default=12, ge=2, description="min. sorpresas validas para z_i")
    impact_t_threshold: float = Field(default=2.0, gt=0.0, description="|t| para w_i distinguible")
    # null mientras no haya ni un dia de consenso historico legitimo capturado
    # (ver reports/data_audit.md secciones 4 y 7). No se imputa hacia atras.
    surprise_start_date: str | None = None
    decay: DecayConfig
    indicators: list[NewsIndicator]
    # Capa de flujo de titulares (V3): GDELT -> FinBERT (s_i) -> Hawkes.
    headlines: HeadlinesConfig
    relevance: RelevanceConfig
    hawkes: HawkesLayerConfig

    @classmethod
    def load(cls, path: Path = CONFIG_DIR / "news.yaml") -> "NewsConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)

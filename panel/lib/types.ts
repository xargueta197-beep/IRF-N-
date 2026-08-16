/** Subconjunto del contrato de salida (src/irfn/outputs/schema.py) que el panel
 * consume. El panel NUNCA calcula: si un campo falta aqui, se agrega al
 * contrato y al script de exportacion, no se inventa en el frontend (R9). */

export interface MetricCI {
  value: number;
  ci_low: number | null;
  ci_high: number | null;
}

export interface ConditionalStatsEntry {
  mean_ann: MetricCI;
  vol_ann: MetricCI;
  sharpe: MetricCI;
  maxdd: MetricCI;
}

export interface RegimeBlock {
  labels: string[];
  xi_filtered: number[];
  entropy: number;
  entropy_max: number;
  confidence: string; // "alta" | "media" | "el modelo no distingue"
  expected_duration_days: number[];
  argmax: string;
  xi_momentum_5d: number[];
}

export interface ModelBlock {
  K: number;
  spec: string;
  tvtp: boolean;
  covariates: string[];
}

export interface IRFNData {
  run_id: string;
  generated_at: string;
  asof: string;
  version: string;
  model: ModelBlock;
  regime: RegimeBlock;
  transition_matrix_today: number[][];
  conditional_stats: Record<string, Record<string, ConditionalStatsEntry>>;
  warnings: string[];
  validation_ref: string;
  disclaimer: string;
}

export interface HistoryPoint {
  date: string;
  xi: number[];
  entropy: number;
  price: number;
  block_boundary: boolean;
  argmax: string;
}

export interface ValidationRow {
  test: string;
  result: string;
  verdict: string;
}

export interface ValidationSummary {
  generated_at: string | null;
  rows: ValidationRow[];
  source: string;
  // Coherencia con el indicador publicado (F6). El reporte de validacion valida
  // un run concreto (`validates_run_id`); si difiere del run publicado hoy
  // (`published_run_id`), `stale` es true y quien renderice esta validacion DEBE
  // avisar que describe una corrida distinta a la de "hoy".
  validates_run_id: string | null;
  published_run_id: string | null;
  stale: boolean;
}

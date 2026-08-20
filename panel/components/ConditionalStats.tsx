import { colors } from "@/lib/theme";
import type { ConditionalStatsEntry, MetricCI } from "@/lib/types";

export interface ConditionalStatsProps {
  stats: Record<string, Record<string, ConditionalStatsEntry>>;
  assets: string[];
}

const METRICS: { key: keyof ConditionalStatsEntry; label: string; fmt: (v: number) => string }[] = [
  { key: "mean_ann", label: "Retorno anualizado", fmt: (v) => `${(v * 100).toFixed(1)}%` },
  { key: "vol_ann", label: "Volatilidad anualizada", fmt: (v) => `${(v * 100).toFixed(1)}%` },
  { key: "sharpe", label: "Sharpe", fmt: (v) => v.toFixed(2) },
  { key: "maxdd", label: "Máximo drawdown", fmt: (v) => `${(v * 100).toFixed(1)}%` },
];

function cell(m: MetricCI, fmt: (v: number) => string) {
  // Punto suprimido por el pipeline (regimen que no persiste o con pocas obs):
  // se muestra "no anualizable", NUNCA fmt(null) -> "0.0%" (numero falso).
  // Paridad con app/pages/4_Retornos_condicionales.py (_fmt_cell).
  if (m.value === null) {
    return (
      <span style={{ color: colors.muted }}>
        no anualizable{typeof m.n_obs === "number" ? ` — ${m.n_obs} obs` : ""}
      </span>
    );
  }
  const includesZero = m.ci_low !== null && m.ci_high !== null && m.ci_low <= 0 && m.ci_high >= 0;
  const hasCI = m.ci_low !== null && m.ci_high !== null;
  return (
    <span style={{ color: includesZero ? colors.muted : colors.foreground }}>
      {fmt(m.value)}
      {hasCI && (
        <>
          {" "}
          ({fmt(m.ci_low as number)}, {fmt(m.ci_high as number)})
        </>
      )}
      {includesZero && <sup>ⁱ</sup>}
    </span>
  );
}

export function ConditionalStats({ stats, assets }: ConditionalStatsProps) {
  const regimeSet = new Set<string>();
  for (const asset of assets) {
    for (const regime of Object.keys(stats[asset] ?? {})) regimeSet.add(regime);
  }
  const regimes = Array.from(regimeSet);

  return (
    <div className="w-full">
      <div className="grid gap-6 md:grid-cols-2">
        {METRICS.map((metric) => (
          <div key={metric.key}>
            <h3 className="font-body mb-2 text-sm font-medium" style={{ color: colors.foreground }}>
              {metric.label}
            </h3>
            <table className="w-full border-separate border-spacing-1 font-body text-sm">
              <thead>
                <tr>
                  <th className="p-1 text-left text-xs font-normal" style={{ color: colors.muted }}>
                    activo
                  </th>
                  {regimes.map((r) => (
                    <th key={r} className="p-1 text-left text-xs font-normal" style={{ color: colors.muted }}>
                      {r}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {assets.map((asset) => (
                  <tr key={asset}>
                    <td className="p-1 font-medium" style={{ color: colors.foreground }}>
                      {asset}
                    </td>
                    {regimes.map((regime) => {
                      const entry = stats[asset]?.[regime];
                      return (
                        <td key={regime} className="p-1 tabular-nums">
                          {entry ? cell(entry[metric.key], metric.fmt) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
      <p className="font-body mt-4 text-xs" style={{ color: colors.muted }}>
        Estadísticas sobre períodos donde el modelo asignó probabilidad &gt; 50% al
        régimen. Intervalos de confianza por bootstrap estacionario (Politis-Romano).
        <sup>ⁱ</sup> el intervalo incluye el cero: no hay evidencia estadística de que
        el valor sea distinto de cero.
      </p>
    </div>
  );
}

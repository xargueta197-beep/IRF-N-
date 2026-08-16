import { colors, regimeColor } from "@/lib/theme";

export interface TransitionMatrixProps {
  matrix: number[][];
  labels: string[];
  /**
   * Indice del regimen de HOY (argmax de xi_filtered). No esta en la lista de
   * props del encargo original, pero la frase generada ("la probabilidad de
   * mantenerse en [regimen actual]...") no se puede escribir sin saber cual
   * fila es "hoy". El caller lo pasa (labels.indexOf(regime.argmax)).
   */
  currentIndex: number;
}

function hexToRgba(hex: string, alpha: number): string {
  const v = hex.replace("#", "");
  const r = parseInt(v.slice(0, 2), 16);
  const g = parseInt(v.slice(2, 4), 16);
  const b = parseInt(v.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function TransitionMatrix({ matrix, labels, currentIndex }: TransitionMatrixProps) {
  const K = labels.length;
  const pStay = matrix[currentIndex]?.[currentIndex] ?? 0;
  const todayLabel = labels[currentIndex] ?? "?";

  return (
    <div className="w-full">
      <table className="w-full border-separate border-spacing-1 font-body text-sm">
        <thead>
          <tr>
            <th className="p-1 text-left text-xs font-normal" style={{ color: colors.muted }}>
              desde \ hacia
            </th>
            {labels.map((l) => (
              <th key={l} className="p-1 text-center text-xs font-normal" style={{ color: colors.muted }}>
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i] ?? i}>
              <td className="p-1 text-xs font-medium" style={{ color: colors.foreground }}>
                {labels[i]}
              </td>
              {row.map((p, j) => {
                const destColor = regimeColor(j, K);
                return (
                  <td
                    key={j}
                    className="rounded p-2 text-center tabular-nums"
                    style={{ backgroundColor: hexToRgba(destColor, 0.12 + 0.75 * Math.min(1, Math.max(0, p))) }}
                  >
                    <span style={{ color: p > 0.55 ? colors.background : colors.foreground }}>
                      {(p * 100).toFixed(0)}%
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="font-body mt-3 text-sm" style={{ color: colors.foreground }}>
        Con las condiciones de hoy, la probabilidad de mantenerse en{" "}
        <strong>{todayLabel}</strong> es <strong>{(pStay * 100).toFixed(0)}%</strong>.
      </p>
    </div>
  );
}

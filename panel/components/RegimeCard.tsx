import { colors, regimeColor, textOn } from "@/lib/theme";
import { HatchDefs } from "./HatchPattern";

export interface RegimeCardProps {
  regime: string;
  xi: number[];
  confidence: string;
  expected_duration: number;
  momentum: number[];
}

/**
 * Minigrafico de xi_momentum_5d: NO es una serie temporal de 5 dias (el
 * contrato solo publica el snapshot de hoy, R1: nada de historico dentro del
 * payload diario), es un vector de K valores -- el cambio de probabilidad de
 * CADA regimen en los ultimos 5 dias (ξ(t) - ξ(t-5)). Se dibuja honestamente
 * como barras divergentes desde cero (una por regimen), no como una linea de
 * tendencia inventada sobre datos que no la tienen.
 */
function MomentumBars({ values, barColor }: { values: number[]; barColor: string }) {
  if (values.length === 0) return null;
  const w = 120;
  const h = 32;
  const mid = h / 2;
  const max = Math.max(...values.map((v) => Math.abs(v)), 1e-9);
  const barW = w / values.length;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
      <line x1={0} y1={mid} x2={w} y2={mid} stroke={barColor} strokeOpacity={0.35} strokeWidth={1} />
      {values.map((v, i) => {
        const barH = (Math.abs(v) / max) * mid;
        const x = i * barW + barW * 0.2;
        const y = v >= 0 ? mid - barH : mid;
        return <rect key={i} x={x} y={y} width={barW * 0.6} height={Math.max(barH, 1)} fill={barColor} />;
      })}
    </svg>
  );
}

export function RegimeCard({ regime, xi, confidence, expected_duration, momentum }: RegimeCardProps) {
  const noSignal = confidence === "el modelo no distingue";
  const K = xi.length;
  const argmaxIndex = xi.indexOf(Math.max(...xi));
  const bg = regimeColor(argmaxIndex, K);
  const hatchId = "hatch-regimecard";

  if (noSignal) {
    return (
      <div className="relative overflow-hidden rounded-lg border border-border" style={{ backgroundColor: colors.background }}>
        <svg className="absolute inset-0 h-full w-full" preserveAspectRatio="none" aria-hidden="true">
          <HatchDefs id={hatchId} color={colors.foreground} />
          <rect width="100%" height="100%" fill={`url(#${hatchId})`} opacity={0.14} />
        </svg>
        <div className="relative flex min-h-[220px] flex-col justify-center gap-3 p-8">
          <p className="font-display text-5xl" style={{ color: colors.accent }}>
            Sin señal clara
          </p>
          <p className="font-body text-sm" style={{ color: colors.muted }}>
            La entropía del estado es alta: el modelo no distingue el régimen con
            confianza suficiente para declararse por uno.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-[220px] flex-col justify-between gap-4 rounded-lg p-8" style={{ backgroundColor: bg }}>
      <div>
        <p className="font-display text-5xl leading-tight" style={{ color: textOn(bg) }}>
          {regime}
        </p>
        <p className="font-body mt-2 text-sm" style={{ color: textOn(bg), opacity: 0.85 }}>
          Este régimen dura en promedio {expected_duration.toFixed(1)} días más.
        </p>
      </div>
      <div className="flex items-center justify-between">
        <span className="font-body text-xs uppercase tracking-wide" style={{ color: textOn(bg), opacity: 0.7 }}>
          momentum por régimen (5d)
        </span>
        <MomentumBars values={momentum} barColor={textOn(bg)} />
      </div>
    </div>
  );
}

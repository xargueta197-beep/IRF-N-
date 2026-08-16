"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { colors, regimeColor } from "@/lib/theme";
import type { HistoryPoint } from "@/lib/types";
import { HatchDefs } from "./HatchPattern";

export interface RegimeHistoryProps {
  history: HistoryPoint[];
}

const CHART_MARGIN = { top: 8, right: 48, left: 40, bottom: 8 };
const BAND_HATCH_ID = "hatch-regimehistory-band";

type ChartRow = Record<string, number | string | boolean> & { date: string };

export function RegimeHistory({ history }: RegimeHistoryProps) {
  if (history.length === 0) return null;
  const K = history[0].xi.length;
  const entropyMax = Math.log(K); // R5: H_max = ln(K), constante para todo K fijo

  const data: ChartRow[] = history.map((p) => {
    const row: ChartRow = { date: p.date, price: p.price, entropy: p.entropy, block_boundary: p.block_boundary };
    p.xi.forEach((v, k) => {
      row[`xi_${k}`] = v;
    });
    return row;
  });

  // Franjas de fondo por argmax(xi): agrupa corridas CONTIGUAS del mismo
  // indice dominante. Es agrupacion de presentacion sobre datos ya
  // calculados por el modelo (no recalcula nada, R9).
  const argmaxIdx = history.map((p) => p.xi.indexOf(Math.max(...p.xi)));
  const bgSegments: { x1: string; x2: string; idx: number }[] = [];
  for (let i = 0; i < history.length; i++) {
    const idx = argmaxIdx[i];
    const last = bgSegments[bgSegments.length - 1];
    if (last && last.idx === idx) {
      last.x2 = history[i].date;
    } else {
      bgSegments.push({ x1: history[i].date, x2: history[i].date, idx });
    }
  }

  const boundaries = history.filter((p) => p.block_boundary).map((p) => p.date);

  // Franja de entropia (debajo del grafico): corridas contiguas donde
  // entropy > 0.5 * entropy_max, representadas por posicion PORCENTUAL del
  // indice (eje categorico: cada punto ocupa un ancho igual, igual que el
  // eje X de arriba). Aproximacion deliberada: no reproduce pixel a pixel el
  // escalado interno de Recharts, alineada por el mismo margen izq/der.
  const n = history.length;
  const highEntropySegments: { startPct: number; widthPct: number }[] = [];
  let runStart: number | null = null;
  history.forEach((p, i) => {
    const high = p.entropy > 0.5 * entropyMax;
    if (high && runStart === null) runStart = i;
    if (!high && runStart !== null) {
      highEntropySegments.push({ startPct: (runStart / n) * 100, widthPct: ((i - runStart) / n) * 100 });
      runStart = null;
    }
  });
  if (runStart !== null) {
    highEntropySegments.push({ startPct: (runStart / n) * 100, widthPct: ((n - runStart) / n) * 100 });
  }

  return (
    <div className="w-full">
      <p className="font-body mb-2 text-xs" style={{ color: colors.muted }}>
        Probabilidad filtrada — solo información disponible en cada fecha.
      </p>
      <ResponsiveContainer width="100%" height={360}>
        <ComposedChart data={data} margin={CHART_MARGIN}>
          <defs>
            <HatchDefs id={BAND_HATCH_ID} color={colors.foreground} />
          </defs>
          <CartesianGrid stroke={colors.border} strokeDasharray="3 3" vertical={false} />
          {bgSegments.map((seg, i) => (
            <ReferenceArea
              key={i}
              x1={seg.x1}
              x2={seg.x2}
              yAxisId="xi"
              fill={regimeColor(seg.idx, K)}
              fillOpacity={0.06}
              ifOverflow="visible"
              stroke="none"
            />
          ))}
          <XAxis dataKey="date" tick={{ fontSize: 11, fill: colors.muted }} minTickGap={40} />
          <YAxis
            yAxisId="xi"
            domain={[0, 1]}
            width={CHART_MARGIN.left}
            tick={{ fontSize: 11, fill: colors.muted }}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          />
          <YAxis
            yAxisId="price"
            orientation="right"
            width={CHART_MARGIN.right}
            tick={{ fontSize: 11, fill: colors.muted }}
          />
          <Tooltip
            contentStyle={{ backgroundColor: colors.background, border: `1px solid ${colors.border}`, fontSize: 12 }}
            formatter={(value: number, name: string) => {
              if (name === "price") return [value.toFixed(2), "índice del activo"];
              if (name.startsWith("xi_")) return [`${(value * 100).toFixed(1)}%`, `régimen ${name.slice(3)}`];
              return [value, name];
            }}
          />
          {Array.from({ length: K }).map((_, k) => (
            <Area
              key={k}
              yAxisId="xi"
              type="monotone"
              dataKey={`xi_${k}`}
              stackId="xi"
              stroke="none"
              fill={regimeColor(k, K)}
              fillOpacity={0.55}
              isAnimationActive={false}
            />
          ))}
          <Line
            yAxisId="price"
            type="monotone"
            dataKey="price"
            stroke={colors.foreground}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          {boundaries.map((d) => (
            <ReferenceLine key={d} x={d} yAxisId="xi" stroke={colors.muted} strokeDasharray="3 3" />
          ))}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Franja de entropia: aproximada por indice (ver comentario arriba),
          alineada con el mismo margen izquierdo/derecho del grafico. */}
      <div className="relative h-10 w-full" style={{ paddingLeft: CHART_MARGIN.left, paddingRight: CHART_MARGIN.right }}>
        <svg width="100%" height="100%" viewBox="0 0 100 40" preserveAspectRatio="none" className="h-full w-full">
          <defs>
            <HatchDefs id={`${BAND_HATCH_ID}-inner`} color={colors.foreground} />
          </defs>
          <rect x={0} y={0} width={100} height={40} fill={colors.background} />
          {highEntropySegments.map((seg, i) => (
            <rect key={i} x={seg.startPct} y={0} width={seg.widthPct} height={40} fill={`url(#${BAND_HATCH_ID}-inner)`} opacity={0.5} />
          ))}
        </svg>
      </div>
      <p className="font-body mt-1 text-xs" style={{ color: colors.muted }}>
        Zonas rayadas = modelo sin señal clara
      </p>
    </div>
  );
}

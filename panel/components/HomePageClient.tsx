"use client";

import { useState } from "react";
import { AssetTabs } from "@/components/AssetTabs";
import { ConditionalStats } from "@/components/ConditionalStats";
import { EntropyBar } from "@/components/EntropyBar";
import { RegimeCard } from "@/components/RegimeCard";
import { TransitionMatrix } from "@/components/TransitionMatrix";
import type { IRFNData } from "@/lib/types";
import { colors, regimeColor } from "@/lib/theme";

export interface HomePageClientProps {
  datasets: Record<string, IRFNData>;
}

/** Contenido de la pantalla 1 (estado de hoy), identico al que existia antes
 * de anadir el selector de activo -- solo se parametrizo por `data` en vez de
 * leerla directo de loadIrfn(). El selector decide QUE dataset (ya cargado
 * en build por el server component de app/page.tsx) se muestra; no hay
 * fetch ni calculo en el cliente (R9). */
export function HomePageClient({ datasets }: HomePageClientProps) {
  const assets = Object.keys(datasets);
  const [selected, setSelected] = useState(assets[0]);
  const data = datasets[selected] ?? datasets[assets[0]];

  const { regime, model, transition_matrix_today, conditional_stats, warnings, asof, version } = data;
  const K = regime.labels.length;
  const argmaxIndex = regime.labels.indexOf(regime.argmax);
  const expectedDuration = regime.expected_duration_days[argmaxIndex] ?? 0;
  const currentColor = regimeColor(argmaxIndex, K);

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-8">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <p className="font-body text-xs uppercase tracking-wide" style={{ color: colors.muted }}>
            {version} · asof {asof}
          </p>
          <p className="font-body text-xs" style={{ color: colors.muted }}>
            {model.spec}
          </p>
        </div>
        <AssetTabs assets={assets} selected={selected} onSelect={setSelected} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <RegimeCard
          regime={regime.argmax}
          xi={regime.xi_filtered}
          confidence={regime.confidence}
          expected_duration={expectedDuration}
          momentum={regime.xi_momentum_5d}
        />
        <div className="flex flex-col gap-6">
          <EntropyBar entropy={regime.entropy} entropy_max={regime.entropy_max} color={currentColor} />
          <TransitionMatrix matrix={transition_matrix_today} labels={regime.labels} currentIndex={argmaxIndex} />
        </div>
      </div>

      <ConditionalStats stats={conditional_stats} assets={Object.keys(conditional_stats)} />

      {warnings.length > 0 && (
        <div className="rounded border p-4" style={{ borderColor: colors.border }}>
          <p className="font-body mb-2 text-xs font-medium uppercase tracking-wide" style={{ color: colors.muted }}>
            Avisos de esta corrida
          </p>
          <ul className="font-body flex flex-col gap-1 text-xs" style={{ color: colors.muted }}>
            {warnings.map((w, i) => (
              <li key={i}>· {w}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

"use client";

import { useState } from "react";
import { AssetTabs } from "@/components/AssetTabs";
import { EntropyBar } from "@/components/EntropyBar";
import { RegimeHistory } from "@/components/RegimeHistory";
import type { HistoryPoint, IRFNData } from "@/lib/types";
import { regimeColor } from "@/lib/theme";

export interface HistoricoDataset {
  irfn: IRFNData;
  history: HistoryPoint[];
}

export interface HistoricoPageClientProps {
  datasets: Record<string, HistoricoDataset>;
}

/** Identico al contenido previo de app/historico/page.tsx, parametrizado por
 * el activo seleccionado (ambos datasets ya cargados en build; el selector
 * solo decide cual se muestra, sin fetch ni calculo en el cliente, R9). */
export function HistoricoPageClient({ datasets }: HistoricoPageClientProps) {
  const assets = Object.keys(datasets);
  const [selected, setSelected] = useState(assets[0]);
  const { irfn: data, history } = datasets[selected] ?? datasets[assets[0]];
  const { regime } = data;
  const K = regime.labels.length;
  const argmaxIndex = regime.labels.indexOf(regime.argmax);

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="font-display mb-4 text-3xl">Histórico</h1>
          <EntropyBar entropy={regime.entropy} entropy_max={regime.entropy_max} color={regimeColor(argmaxIndex, K)} />
        </div>
        <AssetTabs assets={assets} selected={selected} onSelect={setSelected} />
      </div>
      <RegimeHistory history={history} />
    </div>
  );
}

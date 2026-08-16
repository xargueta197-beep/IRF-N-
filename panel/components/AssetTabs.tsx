"use client";

import { colors } from "@/lib/theme";

export interface AssetTabsProps {
  assets: string[];
  selected: string;
  onSelect: (asset: string) => void;
}

/** Selector de activo (SPY/BTC/...). Se oculta solo si hay un unico activo
 * exportado (p.ej. BTC aun no corrido): nunca un selector vacio o con una
 * sola opcion inutil. */
export function AssetTabs({ assets, selected, onSelect }: AssetTabsProps) {
  if (assets.length <= 1) return null;
  return (
    <div className="flex gap-2" role="tablist" aria-label="Activo">
      {assets.map((a) => {
        const active = a === selected;
        return (
          <button
            key={a}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onSelect(a)}
            className="font-body rounded px-3 py-1 text-xs font-medium uppercase tracking-wide transition-colors"
            style={{
              border: `1px solid ${active ? colors.foreground : colors.border}`,
              background: active ? colors.foreground : "transparent",
              color: active ? colors.background : colors.muted,
            }}
          >
            {a}
          </button>
        );
      })}
    </div>
  );
}

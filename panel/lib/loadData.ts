import fs from "node:fs";
import path from "node:path";
import type { HistoryPoint, IRFNData, ValidationSummary } from "./types";

/** Lecturas de build/render (server components, static export). El panel
 * SIEMPRE lee de public/data/*.json (generados por scripts/export_panel_data.py
 * desde artifacts/), nunca de artifacts/ directamente (R9: la interfaz no
 * calcula, solo lee).
 *
 * Segundo activo (BTC, en paralelo a SPY): export_panel_data.py --asset BTC
 * escribe en public/data/btc/*.json, una subcarpeta aditiva por activo que
 * NUNCA pisa los archivos raiz de SPY. `asset` aqui es el slug en minuscula
 * ("btc"); undefined/"" = raiz = SPY, identico al comportamiento historico. */
function dataPath(asset: string | undefined, file: string): string {
  return asset
    ? path.join(process.cwd(), "public", "data", asset, file)
    : path.join(process.cwd(), "public", "data", file);
}

function readJson<T>(asset: string | undefined, file: string): T {
  return JSON.parse(fs.readFileSync(dataPath(asset, file), "utf-8")) as T;
}

/** true si existe un export de panel para ese activo (p.ej. "btc"). Permite
 * que el build no falle si alguna vez se corre sin haber exportado ese
 * activo todavia -- se degrada a solo-SPY en vez de romper (R8). */
export function hasAssetData(asset: string): boolean {
  return fs.existsSync(dataPath(asset, "irfn.json"));
}

export function loadIrfn(asset?: string): IRFNData {
  return readJson<IRFNData>(asset, "irfn.json");
}

export function loadHistory(asset?: string): HistoryPoint[] {
  return readJson<HistoryPoint[]>(asset, "history.json");
}

export function loadValidation(): ValidationSummary {
  return readJson<ValidationSummary>(undefined, "validation.json");
}

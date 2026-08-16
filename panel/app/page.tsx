import { HomePageClient } from "@/components/HomePageClient";
import { hasAssetData, loadIrfn } from "@/lib/loadData";
import type { IRFNData } from "@/lib/types";

export default function HomePage() {
  const datasets: Record<string, IRFNData> = { SPY: loadIrfn() };
  if (hasAssetData("btc")) datasets.BTC = loadIrfn("btc");
  return <HomePageClient datasets={datasets} />;
}

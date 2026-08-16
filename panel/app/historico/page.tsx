import { HistoricoPageClient, type HistoricoDataset } from "@/components/HistoricoPageClient";
import { hasAssetData, loadHistory, loadIrfn } from "@/lib/loadData";

export default function HistoricoPage() {
  const datasets: Record<string, HistoricoDataset> = {
    SPY: { irfn: loadIrfn(), history: loadHistory() },
  };
  if (hasAssetData("btc")) {
    datasets.BTC = { irfn: loadIrfn("btc"), history: loadHistory("btc") };
  }
  return <HistoricoPageClient datasets={datasets} />;
}

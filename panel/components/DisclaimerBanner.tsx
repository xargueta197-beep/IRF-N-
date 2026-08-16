import { colors } from "@/lib/theme";

/** Footer fijo, presente en las 3 paginas via app/layout.tsx (root layout):
 * no hay boton de cierre, no hay estado, no hay animacion de entrada. */
export function DisclaimerBanner() {
  return (
    <footer className="font-body w-full px-6 py-4 text-center text-xs" style={{ backgroundColor: colors.foreground, color: colors.background }}>
      IRF-N es un indicador de investigación. No constituye recomendación de
      inversión ni promesa de rendimiento. Araht Analytics.
    </footer>
  );
}

import { colors, textOn } from "@/lib/theme";
import { HatchDefs } from "./HatchPattern";

export interface EntropyBarProps {
  entropy: number;
  entropy_max: number;
  /**
   * Color solido del regimen actual (en H=0, certeza total). No forma parte
   * de la lista de props del encargo original (solo entropy/entropy_max):
   * se agrega porque sin el es imposible cumplir "En H=0: color solido del
   * regimen actual" -- el color del regimen vive en RegimeBlock, no en la
   * entropia. El caller (page.tsx) lo pasa via theme.regimeColor(argmax, K).
   */
  color?: string;
}

const HATCH_ID = "hatch-entropybar";

/** El elemento visual de firma del indicador: una barra que va de "color
 * solido" (certeza) a "hachurado" (el modelo no distingue), con interpolacion
 * visual entre medias via un gradiente de opacidad sobre el overlay
 * hachurado -- todo definido en <defs> SVG, nada de CSS. */
export function EntropyBar({ entropy, entropy_max, color = colors.foreground }: EntropyBarProps) {
  const ratio = entropy_max > 0 ? Math.min(1, Math.max(0, entropy / entropy_max)) : 0;
  const warn = ratio > 0.8;
  const gradId = "entropybar-fade";
  const markerX = ratio * 100;

  return (
    <div className="w-full">
      <div className="flex items-center justify-between font-body text-xs" style={{ color: colors.muted }}>
        <span>Certeza total</span>
        <span>El modelo no distingue</span>
      </div>
      <div className="relative mt-1 h-6 w-full overflow-hidden rounded-full border" style={{ borderColor: colors.border }}>
        <svg width="100%" height="100%" viewBox="0 0 100 24" preserveAspectRatio="none" className="absolute inset-0">
          <defs>
            <HatchDefs id={HATCH_ID} color={colors.foreground} />
            {/* opacidad del overlay hachurado: 0 en x=0 (certeza), 1 en x=100 (ignorancia) */}
            <linearGradient id={gradId} x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="white" stopOpacity="0" />
              <stop offset="100%" stopColor="white" stopOpacity="1" />
            </linearGradient>
            <mask id="entropybar-mask">
              <rect x="0" y="0" width="100" height="24" fill={`url(#${gradId})`} />
            </mask>
          </defs>
          {/* base: color solido del regimen actual */}
          <rect x="0" y="0" width="100" height="24" fill={color} />
          {/* overlay: hachurado, mas presente hacia la derecha (mascara del gradiente) */}
          <rect x="0" y="0" width="100" height="24" fill={`url(#${HATCH_ID})`} mask="url(#entropybar-mask)" />
          {/* marcador de la posicion actual de H */}
          <rect x={Math.max(0, markerX - 0.6)} y="0" width="1.2" height="24" fill={textOn(color)} />
        </svg>
      </div>
      {warn && (
        <p className="font-body mt-1 text-xs font-medium" style={{ color: colors.accent }}>
          El modelo no distingue el régimen con confianza.
        </p>
      )}
    </div>
  );
}

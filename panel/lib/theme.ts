/**
 * Design system del panel IRF-N. Fuente unica de colores y tipografia:
 * ningun componente hardcodea un hex fuera de aqui.
 *
 * Los tres colores de regimen (risk_on/transicion/risk_off) se ajustaron desde
 * la paleta de marca original para que fueran distinguibles en escala de
 * grises (verificado con la formula de luminancia relativa sRGB): grises
 * aproximados 94 / 133 / 45 sobre 255, separacion minima de 39 puntos entre
 * cualquier par. El matiz (verde desaturado / naranja / azul-tinta) se
 * conserva; solo cambio la luminosidad.
 */
export const colors = {
  background: "#F7F6F2",
  foreground: "#1A1A1A",
  accent: "#E8570A",
  regimes: {
    risk_on: "#336849",
    transicion: "#E8570A",
    risk_off: "#132A46",
  },
  muted: "#6B6B6B",
  border: "#E2E1DC",
} as const;

export const typography = {
  display: "var(--font-instrument-serif)",
  body: "var(--font-dm-sans)",
} as const;

/** Paleta ordenada de regimen, de menor a mayor riesgo (coincide con el orden
 * estructural R5: v_1 < v_2 < ... < v_K, asi que el indice 0 es SIEMPRE el
 * regimen de menor varianza). */
const REGIME_PALETTE = [colors.regimes.risk_on, colors.regimes.transicion, colors.regimes.risk_off];

function hexToRgb(hex: string): [number, number, number] {
  const v = hex.replace("#", "");
  return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16)];
}

function rgbToHex([r, g, b]: [number, number, number]): string {
  const c = (n: number) => Math.round(Math.min(255, Math.max(0, n))).toString(16).padStart(2, "0");
  return `#${c(r)}${c(g)}${c(b)}`;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Interpola linealmente en RGB entre los 3 anclas de REGIME_PALETTE segun
 * t in [0,1]. Solo se usa cuando K > 3 (fuera de la escalera actual de
 * k_candidates=[1,2,3,4], pero sin asumir que K nunca crecera). */
function interpolatePalette(t: number): string {
  const clamped = Math.min(1, Math.max(0, t));
  const seg = clamped * (REGIME_PALETTE.length - 1);
  const i = Math.min(REGIME_PALETTE.length - 2, Math.floor(seg));
  const localT = seg - i;
  const a = hexToRgb(REGIME_PALETTE[i]);
  const b = hexToRgb(REGIME_PALETTE[i + 1]);
  return rgbToHex([lerp(a[0], b[0], localT), lerp(a[1], b[1], localT), lerp(a[2], b[2], localT)]);
}

/**
 * Color de un regimen por su INDICE (no por su nombre): el orden estructural
 * R5 garantiza que el indice 0 es siempre el de menor varianza (risk_on-like)
 * y el indice K-1 el de mayor varianza (risk_off-like). Con K=2 (el modelo
 * publicado hoy) no hay "transicion"; con K=3 los tres nombres se usan tal
 * cual; con K=4+ se interpola sobre la misma paleta de 3 anclas.
 */
export function regimeColor(index: number, K: number): string {
  if (K <= 1) return colors.regimes.risk_on;
  if (K === 2) return index === 0 ? colors.regimes.risk_on : colors.regimes.risk_off;
  if (K === 3) return REGIME_PALETTE[index] ?? colors.muted;
  return interpolatePalette(index / (K - 1));
}

/**
 * Patron SVG de hachurado diagonal (lineas a 45 grados), definido UNA vez por
 * <svg> que lo usa (id local a ese documento SVG). Es el lenguaje visual
 * compartido de "el modelo no distingue": aparece en RegimeCard, EntropyBar y
 * la franja de entropia de RegimeHistory. Deliberadamente SVG <pattern>, no un
 * repeating-linear-gradient de CSS.
 */
export function HatchDefs({ id, color = "#1A1A1A" }: { id: string; color?: string }) {
  return (
    <defs>
      <pattern id={id} width="8" height="8" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
        <line x1="0" y1="0" x2="0" y2="8" stroke={color} strokeWidth="2" />
      </pattern>
    </defs>
  );
}

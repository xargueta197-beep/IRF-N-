// GitHub Pages sirve un PROJECT PAGE bajo /<repo>/ (aqui /IRF-N-), asi que en CI
// se necesita basePath/assetPrefix o los assets de _next/ dan 404. En local NO se
// pone prefijo (para servir panel/out/ en la raiz sin romper rutas). El workflow
// de Pages exporta GITHUB_PAGES=true; cualquier otro entorno queda en la raiz.
const isPages = process.env.GITHUB_PAGES === "true";
const repo = "IRF-N-";

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  // Exporta cada ruta como <ruta>/index.html en vez de <ruta>.html, para que
  // GitHub Pages sirva tanto /historico como /historico/ (sin esto, la variante
  // con slash da 404). Estandar para Next static export en Pages.
  trailingSlash: true,
  images: { unoptimized: true },
  // El panel es puramente estatico: lee de public/data/*.json en build, jamas
  // hace fetch en runtime (ver scripts/export_panel_data.py).
  eslint: { ignoreDuringBuilds: true },
  ...(isPages ? { basePath: `/${repo}`, assetPrefix: `/${repo}/` } : {}),
};

export default nextConfig;

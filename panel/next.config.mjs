/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  // El panel es puramente estatico: lee de public/data/*.json en build, jamas
  // hace fetch en runtime (ver scripts/export_panel_data.py).
  eslint: { ignoreDuringBuilds: true },
};

export default nextConfig;

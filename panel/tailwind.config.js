/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#F7F6F2",
        foreground: "#1A1A1A",
        accent: "#E8570A",
        muted: "#6B6B6B",
        border: "#E2E1DC",
        "regime-risk-on": "#336849",
        "regime-transicion": "#E8570A",
        "regime-risk-off": "#132A46",
      },
      fontFamily: {
        display: ["var(--font-instrument-serif)", "serif"],
        body: ["var(--font-dm-sans)", "sans-serif"],
      },
    },
  },
  plugins: [],
};

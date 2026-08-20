/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0E1117",
        surface: "#161A23",
        foreground: "#D7DCE5",
        accent: "#5B8DEF",
        muted: "#8B93A5",
        border: "#262C38",
        "regime-risk-on": "#31688E",
        "regime-transicion": "#1F9E89",
        "regime-risk-off": "#FDE725",
      },
      fontFamily: {
        display: ["var(--font-instrument-serif)", "serif"],
        body: ["var(--font-dm-sans)", "sans-serif"],
      },
    },
  },
  plugins: [],
};

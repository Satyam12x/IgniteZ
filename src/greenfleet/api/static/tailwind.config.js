/** Tailwind config for the dashboard. Build: npx tailwindcss@3 -c tailwind.config.js -i src/tailwind.in.css -o tailwind.css --minify */
module.exports = {
  content: ["./index.html", "./app.js"],
  theme: {
    extend: {
      colors: {
        ground: "#f4f6f8", side: "#0f1f2a", ink: "#0f1c26", ink2: "#4a5866", muted: "#7d8791", line: "#e3e8ec", surface2: "#eef2f4",
        accent: { DEFAULT: "#1f6f8b", 2: "#2a9ac0", soft: "#e3f0f5" },
        good: { DEFAULT: "#168f65", soft: "#e2f3ec", ink: "#0e5f43" },
        warn: { DEFAULT: "#b7791f", soft: "#fbf1dc", ink: "#7a4f0e" },
        signal: { DEFAULT: "#d9482b", soft: "#fbe9e4", ink: "#8b2a17" },
      },
      fontFamily: {
        display: ["Poppins", "Segoe UI", "system-ui", "sans-serif"],
        sans: ["Poppins", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["Poppins", "Segoe UI", "system-ui", "sans-serif"],
      },
      boxShadow: { card: "0 1px 2px rgba(15,28,38,.04), 0 6px 20px rgba(15,28,38,.05)" },
    },
  },
  plugins: [],
};

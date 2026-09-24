import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: proxy /api to the visionist-webui backend (WEBUI_PORT, default 8080).
const BACKEND = process.env.WEBUI_BACKEND || "http://127.0.0.1:8080";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: BACKEND, changeOrigin: true } },
  },
  build: { outDir: "dist", sourcemap: false },
});

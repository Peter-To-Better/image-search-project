import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 開發模式：前端跑在 5173，API 與圖片轉給 FastAPI (8000)
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/data": "http://127.0.0.1:8000",
    },
  },
});

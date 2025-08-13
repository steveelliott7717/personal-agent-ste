// frontend/vite.config.js
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
    plugins: [vue()],
    base: '/app/',
    server: {
        proxy: {
            // Dev only: proxy unified API to backend
            '/app/api': {
                target: 'http://localhost:8000',
                changeOrigin: true
            }
        }
    },
    build: { outDir: 'dist', emptyOutDir: true }
})

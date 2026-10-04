import react, { reactCompilerPreset } from '@vitejs/plugin-react'
import babel from '@rolldown/plugin-babel'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

const backendUrl = process.env.HIVE_API_URL || 'http://127.0.0.1:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    tailwindcss(),
    react(),
    babel({ presets: [reactCompilerPreset()] })
  ],
  server: {
    proxy: {
      '^/pool(?:/|$)': { target: backendUrl, ws: true },
      '/shared': backendUrl,
      '/auth': backendUrl,
      '/api': { target: backendUrl, ws: true },
      '/kernels': backendUrl,
      '/jobs': backendUrl,
      '^/node(?:/|$)': backendUrl,
      '^/legacy-node(?:/|$)': backendUrl,
      '^/nodes$': { target: backendUrl, ws: true },
    },
  },
})

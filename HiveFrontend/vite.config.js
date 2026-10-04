import react, { reactCompilerPreset } from '@vitejs/plugin-react'
import babel from '@rolldown/plugin-babel'
import { defineConfig } from 'vite'

const backendUrl = process.env.HIVE_API_URL || 'http://127.0.0.1:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    babel({ presets: [reactCompilerPreset()] })
  ],
  server: {
    proxy: {
      '^/pool(?:/|$)': { target: backendUrl, ws: true },
      '/shared': backendUrl,
      '/kernels': backendUrl,
      '/jobs': backendUrl,
      '^/node(?:/|$)': backendUrl,
      '^/legacy-node(?:/|$)': backendUrl,
      '^/nodes$': { target: backendUrl, ws: true },
    },
  },
})

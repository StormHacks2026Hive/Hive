import react, { reactCompilerPreset } from '@vitejs/plugin-react'
import babel from '@rolldown/plugin-babel'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath } from 'node:url'
import { defineConfig, loadEnv } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Expose only Google's public client ID from the existing root .env.
  const rootEnv = loadEnv(mode, fileURLToPath(new URL('../', import.meta.url)), 'GOOGLE_CLIENT_ID')
  const frontendEnv = loadEnv(mode, fileURLToPath(new URL('./', import.meta.url)), 'VITE_GOOGLE_CLIENT_ID')
  return {
    define: {
      'import.meta.env.VITE_GOOGLE_CLIENT_ID': JSON.stringify(
        frontendEnv.VITE_GOOGLE_CLIENT_ID || rootEnv.GOOGLE_CLIENT_ID || '',
      ),
    },
    server: {
      port: 5173,
      strictPort: true,
    },
    plugins: [
      react(),
      tailwindcss(),
      babel({ presets: [reactCompilerPreset()] })
    ],
  }
})

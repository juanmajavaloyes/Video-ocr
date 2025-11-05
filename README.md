# MP4 → PDF (estricto + OCR) — Web (Vite + React)

Convierte un MP4 de páginas de libro en un PDF con:

- Detección estricta de páginas (sin perder ninguna).
- Selección del fotograma más nítido y filtro de duplicados.
- Verificación de numeración por **OCR de dígitos** (configurable).
- **OCR completo** y PDF buscable (capa de texto), 100% en el navegador.
- Despliegue *functionless* en Vercel.

## Requisitos locales
- Node 18+

## Ejecutar en local
```bash
npm i
npm run dev
```
Abre `http://localhost:5173`.

## Despliegue en Vercel (GitHub → Import Project)
1. Sube este repo a GitHub.
2. En Vercel, **Add New Project** → **Import Git Repository** → selecciona tu repo.
3. Configuración de Build:
   - Framework Preset: **Vite**
   - Build Command: `npm run build`
   - Output Directory: `dist`
4. Deploy (no usa funciones ni backend).

## Notas
- El OCR usa `tesseract.js` desde CDN; no necesitas servidor.
- Si el vídeo es muy grande, baja **Max ancho** o **FPS muestreo** desde la UI.
- Para idioma del OCR completo, ajusta en la UI (por defecto `spa`).


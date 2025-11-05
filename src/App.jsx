import React, { useRef, useState } from "react";

/**
 * WebApp: MP4 → PDF de páginas de libro
 * - Detección de cambio de página (histograma + diferencia de cuadros)
 * - Selección del frame más nítido por segmento (varianza del Laplaciano)
 * - Filtro de duplicados (pHash)
 * - Modo estricto: refinado iterativo de segmentos largos
 * - OCR de Nº de página (zona configurable) y OCR completo para PDF buscable
 * - Generación de PDF con capa de texto (pdf-lib) — 100% en el navegador
 *
 * Despliegue en Vercel (functionless):
 * 1) Crea un proyecto Vite + React o Next.js estático (sin APIs).
 * 2) Copia este componente como `src/App.jsx` (Vite) o en una página en Next (p.ej. `app/page.jsx`).
 * 3) Asegúrate de incluir los scripts ESM desde CDN en `index.html` (Vite) o en `<Head/>` (Next):
 *    <script type="module" src="https://cdn.jsdelivr.net/npm/pdf-lib@1.17.1/dist/pdf-lib.esm.min.js"></script>
 *    <script type="module" src="https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.esm.min.js"></script>
 * 4) Este componente hace import() dinámico de esas libs vía window.PDFLib y window.Tesseract
 * 5) `npm run build` y sube a Vercel con adaptación estática.
 *
 * Nota: Para rendimiento en móviles, usa videos < 1080p y `fpsSample` 6–10.
 */

// ---------- Utilidades matemáticas/imagen (sin OpenCV) ---------- //

function toImageData(canvas, ctx) {
  const { width, height } = canvas;
  const data = ctx.getImageData(0, 0, width, height);
  return { data: data.data, width, height };
}

function grayFromImageData({ data, width, height }) {
  const g = new Float32Array(width * height);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    // luma BT.709
    g[p] = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
  }
  return { g, width, height };
}

function histogram64({ g }) {
  const hist = new Float32Array(64);
  for (let i = 0; i < g.length; i++) hist[Math.min(63, (g[i] * 64) >>> 8)]++;
  // normaliza
  let s = 0;
  for (let i = 0; i < 64; i++) s += hist[i] * hist[i];
  const norm = Math.sqrt(s) || 1;
  for (let i = 0; i < 64; i++) hist[i] /= norm;
  return hist;
}

function histDiffScore(a, b) {
  // 1 - correlación
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const denom = Math.sqrt(na * nb) || 1;
  const corr = dot / denom;
  return 1 - Math.max(-1, Math.min(1, corr));
}

function boxBlurGray({ g, width, height }) {
  const out = new Float32Array(g.length);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let s = 0; let c = 0;
      for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) { s += g[(y + j) * width + (x + i)]; c++; }
      out[y * width + x] = s / c;
    }
  }
  return { g: out, width, height };
}

function frameDiffMag(a, b) { // media de |a-b|
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s / a.length / 255; // ~0..1
}

function laplacianVariance({ g, width, height }) {
  // kernel Laplaciano 3x3
  let sum = 0, sum2 = 0, n = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const p = y * width + x;
      const val = -4 * g[p] + g[p - 1] + g[p + 1] + g[p - width] + g[p + width];
      sum += val; sum2 += val * val; n++;
    }
  }
  const mean = sum / n;
  const variance = sum2 / n - mean * mean;
  return variance;
}

// pHash (DCT simple 32x32 → 8x8)
function dct2D(matrix, N) {
  const C = new Array(N).fill(0).map(() => new Array(N).fill(0));
  const PI = Math.PI;
  for (let u = 0; u < N; u++) {
    for (let v = 0; v < N; v++) {
      let sum = 0;
      for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) sum += matrix[x][y] * Math.cos(((2 * x + 1) * u * PI) / (2 * N)) * Math.cos(((2 * y + 1) * v * PI) / (2 * N));
      const cu = u === 0 ? Math.SQRT1_2 : 1;
      const cv = v === 0 ? Math.SQRT1_2 : 1;
      C[u][v] = 2 / N * cu * cv * sum;
    }
  }
  return C;
}

function pHashFromGray({ g, width, height }) {
  // remuestrea a 32x32
  const N = 32;
  const m = new Array(N).fill(0).map(() => new Array(N).fill(0));
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const x = Math.floor((i * width) / N);
      const y = Math.floor((j * height) / N);
      m[i][j] = g[y * width + x];
    }
  }
  const D = dct2D(m, N);
  const size = 8;
  let vals = [];
  for (let i = 0; i < size; i++) for (let j = 0; j < size; j++) if (i !== 0 || j !== 0) vals.push(D[i][j]);
  const median = vals.slice().sort((a, b) => a - b)[(vals.length / 2) | 0];
  let bits = 0n;
  for (let k = 0; k < vals.length; k++) if (vals[k] > median) bits |= (1n << BigInt(k));
  return bits; // 63 bits
}

function hammingDistance64(a, b) {
  let x = a ^ b;
  let count = 0;
  while (x) { count += Number(x & 1n); x >>= 1n; }
  return count;
}

// ---------- OCR helpers ---------- //
async function ocrNumber(imageBlob, { lang = "eng", region = "inferior", frac = 0.22 }) {
  const Tesseract = window.Tesseract;
  const img = await createImageBitmap(imageBlob);
  const canvas = new OffscreenCanvas(img.width, img.height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const { width: w, height: h } = canvas;
  let sx = 0, sy = 0, sw = w, sh = h;
  const f = Math.max(0.05, Math.min(0.5, frac));
  if (region === "inferior") { sy = Math.floor(h * (1 - f)); sh = Math.floor(h * f); }
  else if (region === "superior") { sy = 0; sh = Math.floor(h * f); }
  else if (region === "izquierda") { sx = 0; sw = Math.floor(w * f); }
  else if (region === "derecha") { sx = Math.floor(w * (1 - f)); sw = Math.floor(w * f); }
  else if (region === "centro") { sx = Math.floor(w * 0.2); sw = Math.floor(w * 0.6); sy = Math.floor(h * (0.5 - f / 2)); sh = Math.floor(h * f); }
  const roi = ctx.getImageData(sx, sy, sw, sh);
  const tmp = new OffscreenCanvas(sw, sh); const tctx = tmp.getContext("2d");
  tctx.putImageData(roi, 0, 0);
  const blob = await tmp.convertToBlob({ type: "image/png" });
  const { data: { text } } = await Tesseract.recognize(blob, lang, { tessedit_char_whitelist: "0123456789", logger: () => {} });
  const m = (text || "").match(/\d{1,4}/g);
  if (!m) return null;
  m.sort((a, b) => b.length - a.length);
  const n = parseInt(m[0], 10);
  return isFinite(n) && n > 0 ? n : null;
}

async function ocrFull(imageBlob, { lang = "eng" }) {
  const Tesseract = window.Tesseract;
  const { data } = await Tesseract.recognize(imageBlob, lang, { logger: () => {} });
  // data.words => [{text, bbox: {x0, x1, y0, y1}}]
  return data;
}

// ---------- PDF (pdf-lib) helpers ---------- //
async function imageToSearchablePDF(pages, { lang = "eng" }) {
  const { PDFDocument, StandardFonts, rgb } = window.PDFLib;
  const pdfDoc = await PDFDocument.create();
  const font = await pdfDoc.embedFont(StandardFonts.Helvetica);

  for (const p of pages) {
    const imgBytes = await p.image.arrayBuffer();
    const img = await pdfDoc.embedPng(imgBytes).catch(async () => pdfDoc.embedJpg(imgBytes));
    const w = img.width; const h = img.height;
    const page = pdfDoc.addPage([w, h]);
    page.drawImage(img, { x: 0, y: 0, width: w, height: h });

    if (p.ocr && p.ocr.words) {
      // dibuja texto con baja opacidad para capa OCR
      for (const wd of p.ocr.words) {
        const t = (wd.text || "").trim();
        if (!t) continue;
        const { x0, y0, x1, y1 } = wd.bbox || {};
        if (x0 == null) continue;
        const x = x0; const y = h - y1; // coord PDF: origen abajo-izq
        const bw = x1 - x0; const bh = y1 - y0;
        // tamaño aproximado a la altura del bloque
        const fontSize = Math.max(6, Math.min(24, bh));
        page.drawText(t, { x, y, size: fontSize, font, color: rgb(0, 0, 0), opacity: 0.01 });
      }
    }
  }
  return await pdfDoc.save({ addDefaultPage: false });
}

// ---------- Pipeline de video ---------- //
async function extractKeyFramesFromVideo(file, opts, progressCb) {
  const {
    scaleMaxW = 1600,
    fpsSample = 8,
    changeThreshold = 0.42,
    minGapSec = 0.6,
    settleSec = 0.25,
    dupHash = 6,
    strictPasses = 3,
    longMult = 1.75,
  } = opts;

  // Carga el video en <video>
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.src = url;
  video.crossOrigin = "anonymous";
  video.muted = true;
  video.playsInline = true;
  video.setAttribute("playsinline", "");
  video.setAttribute("muted", "");
  video.load();
  await video.play().catch(() => {});
  await new Promise((res) => (video.onloadedmetadata = res));
  video.pause();

  const duration = video.duration;
  const width0 = video.videoWidth; const height0 = video.videoHeight;
  const scale = scaleMaxW && width0 > scaleMaxW ? scaleMaxW / width0 : 1;
  const W = Math.floor(width0 * scale), H = Math.floor(height0 * scale);
  const canvas = document.createElement("canvas"); canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  // muestreo
  const dt = 1 / fpsSample;
  let t = 0, lastChangeT = 0;
  let prevGray = null, prevHist = null;
  const changePoints = [0];
  const allFrames = []; // {t, gray, hist, sharp, blob}

  while (t <= duration + 1e-3) {
    video.currentTime = Math.min(duration, t);
    await new Promise((res) => (video.onseeked = res));
    ctx.drawImage(video, 0, 0, W, H);
    const img = toImageData(canvas, ctx);
    const gray = grayFromImageData(img);
    const hist = histogram64(gray);
    const blur = boxBlurGray(gray);
    const sharp = laplacianVariance(blur);

    let score = 0;
    if (prevGray && prevHist) {
      const hdiff = histDiffScore(prevHist, hist);
      const mdiff = frameDiffMag(prevGray.g, gray.g);
      score = 0.65 * hdiff + 0.35 * Math.tanh(mdiff * 5);
      if (score >= changeThreshold && (t - lastChangeT) >= minGapSec) {
        changePoints.push(allFrames.length);
        lastChangeT = t;
      }
    }

    allFrames.push({ t, gray, hist, sharp });
    prevGray = gray; prevHist = hist;
    if (progressCb) progressCb((t / duration) * 40);
    t += dt;
  }

  // segmenta por índices de muestreo
  function segmentsFromChangePoints(cps, end = allFrames.length) {
    const starts = cps;
    const ends = cps.slice(1).concat([end]);
    return starts.map((s, i) => [s, ends[i]]);
  }

  function pickBestInSegment([s, e]) {
    const settle = Math.floor(settleSec * fpsSample);
    const start = Math.min(s + settle, e - 1);
    let best = start, bestSharp = -1;
    for (let i = start; i < e; i++) {
      const sh = allFrames[i].sharp;
      if (sh > bestSharp) { bestSharp = sh; best = i; }
    }
    return best;
  }

  // Modo estricto: refina segmentos largos con umbral decreciente
  let cps = [...changePoints];
  let added = new Set();
  for (let pass = 0; pass < strictPasses; pass++) {
    const segs = segmentsFromChangePoints(cps);
    const durs = segs.map(([s, e]) => allFrames[e - 1].t - allFrames[s].t);
    const med = durs.length ? durs.slice().sort((a, b) => a - b)[(durs.length / 2) | 0] : 0;
    const longSegs = segs.filter((_, i) => durs[i] > longMult * med);
    if (!longSegs.length) break;
    const lowThr = Math.max(0.12, changeThreshold * Math.pow(0.82, pass + 1));

    for (const [s, e] of longSegs) {
      let lastT = allFrames[s].t;
      for (let i = s + 1; i < e; i++) {
        const hdiff = histDiffScore(allFrames[i - 1].hist, allFrames[i].hist);
        const mdiff = frameDiffMag(allFrames[i - 1].gray.g, allFrames[i].gray.g);
        const sc = 0.65 * hdiff + 0.35 * Math.tanh(mdiff * 5);
        if (sc >= lowThr && (allFrames[i].t - lastT) >= minGapSec * 0.75) {
          if (!cps.includes(i)) { cps.push(i); added.add(i); }
          lastT = allFrames[i].t;
        }
      }
    }
    cps.sort((a, b) => a - b);
    if (progressCb) progressCb(40 + (pass + 1) * (15 / strictPasses));
  }

  // elige mejores frames
  const segs = segmentsFromChangePoints(cps);
  const chosenIdx = segs.map(pickBestInSegment);

  // filtra duplicados por pHash
  const pages = [];
  const hashes = [];
  const tmpCanvas = document.createElement("canvas"); tmpCanvas.width = W; tmpCanvas.height = H;
  const tmpCtx = tmpCanvas.getContext("2d");

  for (const i of chosenIdx) {
    video.currentTime = Math.min(duration, allFrames[i].t);
    await new Promise((res) => (video.onseeked = res));
    tmpCtx.drawImage(video, 0, 0, W, H);
    const blob = await new Promise((res) => tmpCanvas.toBlob(res, "image/png", 0.9));
    // pHash
    const id = tmpCtx.getImageData(0, 0, W, H);
    const gray = grayFromImageData(id);
    const bits = pHashFromGray(gray);
    const dup = hashes.some((h) => hammingDistance64(h, bits) <= dupHash);
    if (dup) continue;
    hashes.push(bits);
    pages.push({ t: allFrames[i].t, image: blob });
    if (progressCb) progressCb(60 + (pages.length / chosenIdx.length) * 10);
  }

  URL.revokeObjectURL(url);
  return { width: W, height: H, pages, added: Array.from(added), duration };
}

export default function App() {
  const fileRef = useRef(null);
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [pages, setPages] = useState([]); // {image: Blob, t}
  const [addedCount, setAddedCount] = useState(0);
  const [params, setParams] = useState({
    fpsSample: 8,
    scaleMaxW: 1600,
    changeThreshold: 0.42,
    minGapSec: 0.6,
    settleSec: 0.25,
    dupHash: 6,
    strictPasses: 3,
    longMult: 1.75,
    ocrDigits: true,
    ocrLangDigits: "eng",
    ocrRegion: "inferior",
    ocrFrac: 0.22,
    ocrFull: true,
    ocrLangFull: "spa",
  });

  const [digitSeqReport, setDigitSeqReport] = useState(null);

  async function ensureLibs() {
    if (!window.PDFLib) {
      await import("https://cdn.jsdelivr.net/npm/pdf-lib@1.17.1/dist/pdf-lib.esm.min.js");
    }
    if (!window.Tesseract) {
      await import("https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.esm.min.js");
    }
  }

  const runPreview = async () => {
    if (!file) return;
    setBusy(true); setProgress(0); setDigitSeqReport(null);
    try {
      await ensureLibs();
      const result = await extractKeyFramesFromVideo(
        file,
        params,
        (p) => setProgress(Math.round(p))
      );
      setPages(result.pages);
      setAddedCount(result.added.length);
      setProgress(70);

      // OCR de números (secuencia)
      if (params.ocrDigits) {
        const nums = [];
        for (let i = 0; i < result.pages.length; i++) {
          const n = await ocrNumber(result.pages[i].image, {
            lang: params.ocrLangDigits,
            region: params.ocrRegion,
            frac: params.ocrFrac,
          });
          nums.push(n);
          setProgress(70 + Math.round((i / result.pages.length) * 10));
        }
        // report sencillo
        let ok = true; const gaps = []; const dups = []; const unread = [];
        for (let i = 0; i < nums.length; i++) if (nums[i] == null) unread.push(i + 1);
        for (let i = 1; i < nums.length; i++) {
          const a = nums[i - 1], b = nums[i];
          if (a != null && b != null) {
            if (b === a) dups.push([i, a]);
            else if (b !== a + 1) gaps.push([i, a, b]);
          }
        }
        ok = !gaps.length && !dups.length;
        setDigitSeqReport({ ok, gaps, dups, unread, nums });
      }

      setProgress(80);
    } catch (e) {
      alert("Error en vista previa: " + e.message);
    } finally {
      setBusy(false);
      setProgress(0);
    }
  };

  const exportPDF = async () => {
    if (!pages.length) return;
    setBusy(true); setProgress(0);
    try {
      await ensureLibs();
      const pagesWithOCR = [];
      for (let i = 0; i < pages.length; i++) {
        let ocr = null;
        if (params.ocrFull) {
          ocr = await ocrFull(pages[i].image, { lang: params.ocrLangFull });
        }
        pagesWithOCR.push({ image: pages[i].image, ocr });
        setProgress(Math.round((i / pages.length) * 90));
      }
      const pdfBytes = await imageToSearchablePDF(pagesWithOCR, { lang: params.ocrLangFull });
      const blob = new Blob([pdfBytes], { type: "application/pdf" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "libro_ocr.pdf";
      link.click();
      URL.revokeObjectURL(link.href);
      setProgress(100);
    } catch (e) {
      alert("Error exportando PDF: " + e.message);
    } finally {
      setBusy(false); setProgress(0);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 p-6">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-2xl font-bold mb-2">MP4 → PDF de libro (estricto + OCR, 100% web)</h1>
        <p className="text-sm mb-4">Sube un vídeo MP4 con páginas del libro. Detecto cada cambio, escojo el mejor fotograma por página, verifico numeración y genero un PDF buscable con capa OCR.</p>

        <div className="grid md:grid-cols-3 gap-4 mb-4">
          <div className="md:col-span-3 bg-white rounded-2xl shadow p-4">
            <div className="flex items-center gap-3">
              <input ref={fileRef} type="file" accept="video/mp4" onChange={(e) => setFile(e.target.files?.[0] || null)} />
              <button disabled={!file || busy} onClick={runPreview} className="px-4 py-2 rounded-xl bg-slate-900 text-white disabled:opacity-50">Vista previa</button>
              <button disabled={!pages.length || busy} onClick={exportPDF} className="px-4 py-2 rounded-xl bg-emerald-600 text-white disabled:opacity-50">Exportar PDF (OCR)</button>
              {busy && <div className="ml-auto">Progreso: {progress}%</div>}
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow p-4">
            <h2 className="font-semibold mb-2">Detección</h2>
            <label className="block text-sm">FPS muestreo: {params.fpsSample}
              <input type="range" min={4} max={12} value={params.fpsSample} onChange={(e)=>setParams(p=>({...p, fpsSample:+e.target.value}))} className="w-full"/>
            </label>
            <label className="block text-sm mt-2">Umbral cambio: {params.changeThreshold.toFixed(2)}
              <input type="range" min={0.2} max={0.8} step={0.01} value={params.changeThreshold} onChange={(e)=>setParams(p=>({...p, changeThreshold:+e.target.value}))} className="w-full"/>
            </label>
            <div className="grid grid-cols-2 gap-2 mt-2 text-sm">
              <label>Min gap (s)
                <input type="number" step="0.1" value={params.minGapSec} onChange={(e)=>setParams(p=>({...p, minGapSec:+e.target.value}))} className="w-full border rounded px-2 py-1"/>
              </label>
              <label>Settle (s)
                <input type="number" step="0.05" value={params.settleSec} onChange={(e)=>setParams(p=>({...p, settleSec:+e.target.value}))} className="w-full border rounded px-2 py-1"/>
              </label>
              <label>Max ancho (px)
                <input type="number" value={params.scaleMaxW} onChange={(e)=>setParams(p=>({...p, scaleMaxW:+e.target.value}))} className="w-full border rounded px-2 py-1"/>
              </label>
              <label>pHash dup ≤
                <input type="number" value={params.dupHash} onChange={(e)=>setParams(p=>({...p, dupHash:+e.target.value}))} className="w-full border rounded px-2 py-1"/>
              </label>
              <label>Pasadas estrictas
                <input type="number" min={0} max={5} value={params.strictPasses} onChange={(e)=>setParams(p=>({...p, strictPasses:+e.target.value}))} className="w-full border rounded px-2 py-1"/>
              </label>
              <label>Tramo largo × mediana
                <input type="number" step="0.1" value={params.longMult} onChange={(e)=>setParams(p=>({...p, longMult:+e.target.value}))} className="w-full border rounded px-2 py-1"/>
              </label>
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow p-4">
            <h2 className="font-semibold mb-2">OCR nº de página</h2>
            <label className="inline-flex items-center gap-2 text-sm">
              <input type="checkbox" checked={params.ocrDigits} onChange={(e)=>setParams(p=>({...p, ocrDigits:e.target.checked}))}/> Activar
            </label>
            <div className="grid grid-cols-2 gap-2 mt-2 text-sm">
              <label>Idioma
                <input type="text" value={params.ocrLangDigits} onChange={(e)=>setParams(p=>({...p, ocrLangDigits:e.target.value}))} className="w-full border rounded px-2 py-1"/>
              </label>
              <label>Región
                <select value={params.ocrRegion} onChange={(e)=>setParams(p=>({...p, ocrRegion:e.target.value}))} className="w-full border rounded px-2 py-1">
                  <option>inferior</option><option>superior</option><option>izquierda</option><option>derecha</option><option>centro</option><option>todo</option>
                </select>
              </label>
              <label>Fracción
                <input type="number" step="0.01" value={params.ocrFrac} onChange={(e)=>setParams(p=>({...p, ocrFrac:+e.target.value}))} className="w-full border rounded px-2 py-1"/>
              </label>
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow p-4">
            <h2 className="font-semibold mb-2">OCR completo (PDF buscable)</h2>
            <label className="inline-flex items-center gap-2 text-sm">
              <input type="checkbox" checked={params.ocrFull} onChange={(e)=>setParams(p=>({...p, ocrFull:e.target.checked}))}/> Activar
            </label>
            <div className="grid grid-cols-2 gap-2 mt-2 text-sm">
              <label>Idioma OCR
                <input type="text" value={params.ocrLangFull} onChange={(e)=>setParams(p=>({...p, ocrLangFull:e.target.value}))} className="w-full border rounded px-2 py-1"/>
              </label>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow p-4 mb-4">
          <div className="flex items-center justify-between mb-2">
            <h2 className="font-semibold">Páginas detectadas ({pages.length}) {addedCount>0 && <span className="text-emerald-600 text-sm">+{addedCount} refinadas</span>}</h2>
          </div>
          {digitSeqReport && (
            <div className={`p-3 rounded-xl mb-3 ${digitSeqReport.ok ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-800'}`}>
              {digitSeqReport.ok ? (
                <span>Numeración coherente (sin saltos/duplicados).</span>
              ) : (
                <div>
                  <div className="font-medium mb-1">Incidencias OCR nº:</div>
                  {!!digitSeqReport.gaps.length && <div>Saltos: {digitSeqReport.gaps.map(([i,a,b])=>` pág ${i} (${a})→${i+1} (${b})`).join(', ')}</div>}
                  {!!digitSeqReport.dups.length && <div>Duplicados: {digitSeqReport.dups.map(([i,a])=>` pág ${i} = ${i+1} (${a})`).join(', ')}</div>}
                  {!!digitSeqReport.unread.length && <div>Ilegibles: {digitSeqReport.unread.join(', ')}</div>}
                </div>
              )}
            </div>
          )}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {pages.map((p, idx) => (
              <div key={idx} className="border rounded-xl overflow-hidden">
                <img className="block w-full" src={URL.createObjectURL(p.image)} alt={`p${idx+1}`} onLoad={(e)=>URL.revokeObjectURL(e.currentTarget.src)} />
                <div className="text-xs p-2 text-slate-600">p.{idx+1}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="text-xs text-slate-500">
          <p>Consejos: si faltan páginas, baja el umbral (0.36–0.40), sube pasadas estrictas o aumenta FPS muestreo. Si el PDF pesa, reduce "Max ancho" y desactiva OCR completo.</p>
        </div>
      </div>
    </div>
  );
}


---

# Proyecto listo para GitHub → Vercel (Vite + React)

A continuación tienes **todo el esqueleto del proyecto**. Ya tienes `src/App.jsx` en este canvas. Añade los demás ficheros tal cual y súbelo a GitHub.

## Estructura
```
libro-ocr/
├─ index.html
├─ package.json
├─ vite.config.js
├─ .gitignore
├─ README.md
└─ src/
   ├─ main.jsx
   └─ App.jsx   ← (usa el que ya está en este canvas)
```

## `index.html`
```html
<!doctype html>
<html lang="es">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>MP4 → PDF (estricto + OCR)</title>
    <!-- Librerías ESM desde CDN (functionless, todo cliente) -->
    <script type="module" src="https://cdn.jsdelivr.net/npm/pdf-lib@1.17.1/dist/pdf-lib.esm.min.js"></script>
    <script type="module" src="https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.esm.min.js"></script>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

## `package.json`
```json
{
  "name": "libro-ocr",
  "private": true,
  "version": "0.0.1",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview --port 4173"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "vite": "^5.4.0"
  }
}
```

## `vite.config.js`
```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Si no deseas JSX transform de React, puedes omitir el plugin y usar sólo Vite.
export default defineConfig({
  plugins: [react()],
})
```

> Si no quieres instalar `@vitejs/plugin-react`, borra el import y `plugins: [react()]` y añade en `package.json` sólo `vite` como devDependency. Funciona igual con React 18 usando JSX si tu editor transpila, pero lo normal es mantener el plugin.

## `src/main.jsx`
```jsx
import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```

## `src/App.jsx`
> Usa aquí **exactamente** el componente que ya tienes en el canvas (arriba). Copia/pega todo su contenido.

## `.gitignore`
```gitignore
# Vite/Node
node_modules/
dist/
.env
.DS_Store
```

## `README.md`
```md
# MP4 → PDF (estricto + OCR) — Web (Vite + React)

Convierte un MP4 de páginas de libro en un PDF con:
- Detección estricta de páginas (sin perder ninguna).
- Selección del fotograma más nítido y filtro de duplicados.
- Verificación de numeración por **OCR de dígitos**.
- **OCR completo** y PDF buscable (capa de texto), 100% en el navegador.

## Requisitos locales
- Node 18+
- Vite 5

## Ejecutar en local
```bash
npm i
npm run dev
```
Visita `http://localhost:5173`.

## Despliegue en Vercel (GitHub → Import Project)
1. Sube este repo a GitHub.
2. En Vercel, **Add New Project** → **Import Git Repository** → elige tu repo.
3. Configuración de Build:
   - Framework Preset: **Vite**
   - Build Command: `npm run build`
   - Output Directory: `dist`
   - (sin funciones; todo es estático)
4. Deploy. Al terminar, tendrás tu URL pública.

## Notas
- El OCR usa `tesseract.js` desde CDN; no necesitas servidor.
- Si el vídeo es muy grande, baja **Max ancho** o **FPS muestreo** desde la UI.
- Para idioma del OCR completo, ajusta en la UI (por defecto `spa`).
```

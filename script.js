/**
 * Currency Vision — Inference Pipeline (onnxruntime-web)
 */

// --- CONFIGURATION ---
const MODEL_PATH = './best.onnx';
const INPUT_WIDTH = 640;
const INPUT_HEIGHT = 640;

// Filters to eliminate false positives
const CONF_THRESHOLD = 0.70;      // 70%+ confidence threshold
const NMS_THRESHOLD = 0.45;       // IoU threshold for Non-Maximum Suppression
const MIN_ASPECT_RATIO = 1.5;    // Minimum long-side/short-side aspect ratio
const MAX_ASPECT_RATIO = 2.7;    // Maximum aspect ratio
const MIN_AREA_RATIO = 0.015;     // Minimum box area relative to image (1.5%)

const LABELS = [
  '10_New', '10_Old', '20', '50_New', '50_Old', 
  '100_New', '100_Old', '200', '500'
];

let session = null;

// DOM Elements
const imageInput = document.getElementById('imageInput');
const dropZone = document.getElementById('dropZone');
const canvas = document.getElementById('outputCanvas');
const canvasPlaceholder = document.getElementById('canvasPlaceholder');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const resultsContainer = document.getElementById('resultsContainer');
const noteCount = document.getElementById('noteCount');
const resetBtn = document.getElementById('resetBtn');

// --- INITIALIZE ONNX MODEL ---
async function initModel() {
  try {
    updateStatus('Loading AI Model...', 'loading');
    
    // Explicitly configure WASM path for onnxruntime-web
    if (window.ort) {
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/';
      ort.env.wasm.numThreads = 1;
    }

    console.log('Loading ONNX model from:', MODEL_PATH);
    
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ['wasm'],
    });
    
    console.log('ONNX Model loaded successfully.');
    updateStatus('Model Ready', 'ready');
  } catch (error) {
    console.error('Failed to load ONNX model:', error);
    updateStatus('Model Load Failed', 'error');
  }
}

function updateStatus(text, state) {
  if (statusText) statusText.innerText = text;
  if (!statusBadge) return;

  if (state === 'ready') {
    statusBadge.className = "flex items-center gap-2 px-4 py-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 text-sm font-medium";
  } else if (state === 'error') {
    statusBadge.className = "flex items-center gap-2 px-4 py-1.5 rounded-full border border-red-500/30 bg-red-500/10 text-red-400 text-sm font-medium";
  } else {
    statusBadge.className = "flex items-center gap-2 px-4 py-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 text-amber-400 text-sm font-medium";
  }
}

initModel();

// --- EVENT LISTENERS ---
if (dropZone) {
  dropZone.addEventListener('click', () => imageInput && imageInput.click());
  dropZone.addEventListener('dragover', (e) => e.preventDefault());
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processImage(e.dataTransfer.files[0]);
    }
  });
}

if (imageInput) {
  imageInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      processImage(e.target.files[0]);
    }
  });
}

if (resetBtn) {
  resetBtn.addEventListener('click', resetUI);
}

function resetUI() {
  if (canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  if (canvasPlaceholder) canvasPlaceholder.style.display = 'flex';
  if (resultsContainer) resultsContainer.innerHTML = '<p class="text-sm text-slate-500 italic">Upload an image to perform real-time detection.</p>';
  if (noteCount) noteCount.innerText = '0 Notes';
  if (imageInput) imageInput.value = '';
}

// --- IMAGE PIPELINE ---
async function processImage(file) {
  if (!session) {
    alert('Model is still loading or failed to initialize. Please check browser console.');
    return;
  }

  const img = new Image();
  img.src = URL.createObjectURL(file);

  img.onload = async () => {
    if (canvasPlaceholder) canvasPlaceholder.style.display = 'none';

    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    // 1. Preprocess
    const [tensor, scale, padX, padY] = preprocess(img);

    // 2. Inference
    const feeds = {};
    feeds[session.inputNames[0]] = tensor;
    const results = await session.run(feeds);
    const outputTensor = results[session.outputNames[0]];

    // 3. Postprocess
    const detections = postprocess(outputTensor, scale, padX, padY, img.width, img.height);

    // 4. Render
    renderDetections(img, detections);
  };
}

// --- PREPROCESSING ---
function preprocess(img) {
  const canvasPre = document.createElement('canvas');
  canvasPre.width = INPUT_WIDTH;
  canvasPre.height = INPUT_HEIGHT;
  const ctxPre = canvasPre.getContext('2d');

  const scale = Math.min(INPUT_WIDTH / img.width, INPUT_HEIGHT / img.height);
  const newW = img.width * scale;
  const newH = img.height * scale;
  const padX = (INPUT_WIDTH - newW) / 2;
  const padY = (INPUT_HEIGHT - newH) / 2;

  ctxPre.fillStyle = '#111827';
  ctxPre.fillRect(0, 0, INPUT_WIDTH, INPUT_HEIGHT);
  ctxPre.drawImage(img, padX, padY, newW, newH);

  const imgData = ctxPre.getImageData(0, 0, INPUT_WIDTH, INPUT_HEIGHT);
  const { data } = imgData;

  const float32Data = new Float32Array(3 * INPUT_WIDTH * INPUT_HEIGHT);
  const channelLength = INPUT_WIDTH * INPUT_HEIGHT;

  for (let i = 0; i < channelLength; i++) {
    float32Data[i] = data[i * 4] / 255.0;                   // Red
    float32Data[channelLength + i] = data[i * 4 + 1] / 255.0; // Green
    float32Data[2 * channelLength + i] = data[i * 4 + 2] / 255.0; // Blue
  }

  const tensor = new ort.Tensor('float32', float32Data, [1, 3, INPUT_HEIGHT, INPUT_WIDTH]);
  return [tensor, scale, padX, padY];
}

// --- POSTPROCESSING ---
function postprocess(outputTensor, scale, padX, padY, origW, origH) {
  const rawData = outputTensor.data;
  const [batch, channels, numBoxes] = outputTensor.dims; // e.g., [1, 13, 8400]
  const numClasses = channels - 4;

  let candidates = [];

  for (let i = 0; i < numBoxes; i++) {
    let maxClassScore = 0;
    let classId = -1;

    for (let c = 0; c < numClasses; c++) {
      const score = rawData[(4 + c) * numBoxes + i];
      if (score > maxClassScore) {
        maxClassScore = score;
        classId = c;
      }
    }

    // 1. Confidence Threshold
    if (maxClassScore < CONF_THRESHOLD) continue;

    const cx = rawData[0 * numBoxes + i];
    const cy = rawData[1 * numBoxes + i];
    const w = rawData[2 * numBoxes + i];
    const h = rawData[3 * numBoxes + i];

    let x1 = (cx - w / 2 - padX) / scale;
    let y1 = (cy - h / 2 - padY) / scale;
    let boxW = w / scale;
    let boxH = h / scale;

    x1 = Math.max(0, Math.min(x1, origW));
    y1 = Math.max(0, Math.min(y1, origH));
    boxW = Math.min(boxW, origW - x1);
    boxH = Math.min(boxH, origH - y1);

    // 2. Aspect Ratio Filter
    const longSide = Math.max(boxW, boxH);
    const shortSide = Math.min(boxW, boxH);
    const aspectRatio = longSide / Math.max(shortSide, 1e-6);

    if (aspectRatio < MIN_ASPECT_RATIO || aspectRatio > MAX_ASPECT_RATIO) continue;

    // 3. Minimum Area Filter
    const boxAreaRatio = (boxW * boxH) / (origW * origH);
    if (boxAreaRatio < MIN_AREA_RATIO) continue;

    candidates.push({
      box: [x1, y1, boxW, boxH],
      score: maxClassScore,
      classId: classId,
      label: LABELS[classId] || 'Currency'
    });
  }

  return nms(candidates, NMS_THRESHOLD);
}

// --- NMS ---
function nms(boxes, iouThreshold) {
  boxes.sort((a, b) => b.score - a.score);
  const selected = [];
  const active = new Array(boxes.length).fill(true);

  for (let i = 0; i < boxes.length; i++) {
    if (!active[i]) continue;
    selected.push(boxes[i]);

    for (let j = i + 1; j < boxes.length; j++) {
      if (!active[j]) continue;
      if (calculateIoU(boxes[i].box, boxes[j].box) > iouThreshold) {
        active[j] = false;
      }
    }
  }
  return selected;
}

function calculateIoU(boxA, boxB) {
  const [x1, y1, w1, h1] = boxA;
  const [x2, y2, w2, h2] = boxB;

  const interX1 = Math.max(x1, x2);
  const interY1 = Math.max(y1, y2);
  const interX2 = Math.min(x1 + w1, x2 + w2);
  const interY2 = Math.min(y1 + h1, y2 + h2);

  const interWidth = Math.max(0, interX2 - interX1);
  const interHeight = Math.max(0, interY2 - interY1);
  const interArea = interWidth * interHeight;

  return interArea / (w1 * h1 + w2 * h2 - interArea);
}

// --- RENDERING ---
function renderDetections(img, detections) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);

  if (resultsContainer) resultsContainer.innerHTML = '';
  if (noteCount) noteCount.innerText = `${detections.length} Notes`;

  if (detections.length === 0) {
    if (resultsContainer) {
      resultsContainer.innerHTML = '<p class="text-sm text-slate-400">No valid banknote detected.</p>';
    }
    return;
  }

  detections.forEach((det) => {
    const [x, y, w, h] = det.box;
    const cleanLabel = det.label.replace('_', ' ');
    const labelText = `₹${cleanLabel} (${(det.score * 100).toFixed(1)}%)`;

    // Draw Bounding Box
    ctx.strokeStyle = '#10B981';
    ctx.lineWidth = Math.max(3, Math.round(canvas.width / 250));
    ctx.strokeRect(x, y, w, h);

    // Draw Text Background
    ctx.font = '600 16px Inter, sans-serif';
    const textWidth = ctx.measureText(labelText).width;
    ctx.fillStyle = '#10B981';
    ctx.fillRect(x, y > 30 ? y - 30 : y, textWidth + 12, 30);

    // Draw Text
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(labelText, x + 6, y > 30 ? y - 9 : y + 20);

    // Results Badge UI
    if (resultsContainer) {
      const badge = document.createElement('div');
      badge.className = 'inline-flex items-center gap-1.5 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 px-3 py-1.5 rounded-xl text-sm font-semibold';
      badge.innerText = labelText;
      resultsContainer.appendChild(badge);
    }
  });
}
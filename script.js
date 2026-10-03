/**
 * Currency Detector - Inference Script (onnxruntime-web)
 */

// --- CONFIGURATION & THRESHOLDS ---
const MODEL_PATH = './best.onnx';
const INPUT_WIDTH = 640;
const INPUT_HEIGHT = 640;

// Increased confidence threshold to suppress weak false positives
const CONF_THRESHOLD = 0.75;
const NMS_THRESHOLD = 0.45;

// Indian banknote physical aspect ratio bounds (~2.0 to ~2.4)
const MIN_ASPECT_RATIO = 1.6;
const MAX_ASPECT_RATIO = 2.6;

// Minimum bounding box area relative to image (e.g. at least 1.5% of total area)
const MIN_AREA_RATIO = 0.015;

const LABELS = [
  '10_New', '10_Old', '20', '50_New', '50_Old', 
  '100_New', '100_Old', '200', '500'
];

let session = null;

// UI Elements
const imageInput = document.getElementById('imageInput');
const canvas = document.getElementById('outputCanvas');
const ctx = canvas ? canvas.getContext('2d') : null;
const resultsContainer = document.getElementById('resultsContainer');
const dropZone = document.getElementById('dropZone');

// --- INITIALIZATION ---
async function initModel() {
  try {
    console.log('Loading ONNX model...');
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ['wasm'],
    });
    console.log('ONNX Model Loaded Successfully.');
  } catch (e) {
    console.error('Failed to load ONNX model:', e);
  }
}

initModel();

// --- EVENT LISTENERS ---
if (imageInput) {
  imageInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      processImage(e.target.files[0]);
    }
  });
}

if (dropZone) {
  dropZone.addEventListener('dragover', (e) => e.preventDefault());
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processImage(e.dataTransfer.files[0]);
    }
  });
}

// --- IMAGE PROCESSING & INFERENCE ---
async function processImage(file) {
  if (!session) {
    alert('Model is still loading. Please wait a moment and try again.');
    return;
  }

  const img = new Image();
  img.src = URL.createObjectURL(file);

  img.onload = async () => {
    // 1. Prepare Canvas
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    // 2. Preprocess tensor (640x640 RGB float32 normalized [0, 1])
    const [tensor, scale, padX, padY] = preprocess(img);

    // 3. Run Inference
    const feeds = {};
    feeds[session.inputNames[0]] = tensor;
    const results = await session.run(feeds);
    const outputTensor = results[session.outputNames[0]];

    // 4. Postprocess Detections
    const detections = postprocess(
      outputTensor, 
      scale, 
      padX, 
      padY, 
      img.width, 
      img.height
    );

    // 5. Render Output
    renderDetections(img, detections);
  };
}

// --- PREPROCESSING ---
function preprocess(img) {
  const canvasPre = document.createElement('canvas');
  canvasPre.width = INPUT_WIDTH;
  canvasPre.height = INPUT_HEIGHT;
  const ctxPre = canvasPre.getContext('2d');

  // Letterboxing (preserve aspect ratio)
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

// --- POSTPROCESSING & FILTERING ---
function postprocess(outputTensor, scale, padX, padY, origW, origH) {
  const rawData = outputTensor.data;
  const [batch, channels, numBoxes] = outputTensor.dims; // e.g. [1, 13, 8400]
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

    // 1. Confidence Threshold Filter
    if (maxClassScore < CONF_THRESHOLD) continue;

    // Center coordinates & dimensions in 640x640 space
    const cx = rawData[0 * numBoxes + i];
    const cy = rawData[1 * numBoxes + i];
    const w = rawData[2 * numBoxes + i];
    const h = rawData[3 * numBoxes + i];

    // Scale back to original image coordinates
    let x1 = (cx - w / 2 - padX) / scale;
    let y1 = (cy - h / 2 - padY) / scale;
    let boxW = w / scale;
    let boxH = h / scale;

    // Clamp coordinates
    x1 = Math.max(0, Math.min(x1, origW));
    y1 = Math.max(0, Math.min(y1, origH));
    boxW = Math.min(boxW, origW - x1);
    boxH = Math.min(boxH, origH - y1);

    // 2. Aspect Ratio Filter (Width / Height or Height / Width)
    const longSide = Math.max(boxW, boxH);
    const shortSide = Math.min(boxW, boxH);
    const aspectRatio = longSide / Math.max(shortSide, 1e-6);

    if (aspectRatio < MIN_ASPECT_RATIO || aspectRatio > MAX_ASPECT_RATIO) {
      continue; // Discard non-rectangular detections (e.g., circular badges, square icons)
    }

    // 3. Minimum Area Filter
    const boxAreaRatio = (boxW * boxH) / (origW * origH);
    if (boxAreaRatio < MIN_AREA_RATIO) {
      continue; // Discard tiny noise fragments
    }

    candidates.push({
      box: [x1, y1, boxW, boxH],
      score: maxClassScore,
      classId: classId,
      label: LABELS[classId] || 'Currency'
    });
  }

  // 4. Non-Maximum Suppression (NMS)
  return nms(candidates, NMS_THRESHOLD);
}

// --- NON-MAXIMUM SUPPRESSION (NMS) ---
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

  const areaA = w1 * h1;
  const areaB = w2 * h2;

  return interArea / (areaA + areaB - interArea);
}

// --- RENDERING ---
function renderDetections(img, detections) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);

  if (resultsContainer) {
    resultsContainer.innerHTML = '';
  }

  if (detections.length === 0) {
    if (resultsContainer) {
      resultsContainer.innerHTML = '<p class="text-gray-400">No currency detected.</p>';
    }
    return;
  }

  detections.forEach((det) => {
    const [x, y, w, h] = det.box;
    const labelText = `₹${det.label.replace('_', ' ')} (${(det.score * 100).toFixed(1)}%)`;

    // Draw Bounding Box
    ctx.strokeStyle = '#10B981';
    ctx.lineWidth = Math.max(2, Math.round(canvas.width / 300));
    ctx.strokeRect(x, y, w, h);

    // Draw Label Background
    ctx.font = '16px Inter, sans-serif';
    const textWidth = ctx.measureText(labelText).width;
    ctx.fillStyle = '#10B981';
    ctx.fillRect(x, y > 25 ? y - 25 : y, textWidth + 10, 25);

    // Draw Label Text
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(labelText, x + 5, y > 25 ? y - 7 : y + 18);

    // Render summary badge UI
    if (resultsContainer) {
      const badge = document.createElement('div');
      badge.className = 'inline-block bg-emerald-900/40 border border-emerald-500/50 text-emerald-400 px-3 py-1 rounded-lg text-sm font-medium mr-2 mb-2';
      badge.innerText = labelText;
      resultsContainer.appendChild(badge);
    }
  });
}
/**
 * Currency Vision - ONNX Web Runtime Script
 */

// --- CONFIGURATION ---
const MODEL_PATH = './best.onnx';
const INPUT_WIDTH = 640;
const INPUT_HEIGHT = 640;

// Filtering Thresholds to reduce false positives
const CONF_THRESHOLD = 0.70;      // Confidence cutoff (0.70+)
const NMS_THRESHOLD = 0.45;       // Non-Maximum Suppression overlap threshold
const MIN_ASPECT_RATIO = 1.5;    // Indian notes aspect ratio min (W/H or H/W)
const MAX_ASPECT_RATIO = 2.7;    // Indian notes aspect ratio max
const MIN_AREA_RATIO = 0.015;     // Minimum box area relative to image (1.5%)

const LABELS = [
  '10_New', '10_Old', '20', '50_New', '50_Old', 
  '100_New', '100_Old', '200', '500'
];

let session = null;

// Configure ONNX Web WASM path explicitly
if (window.ort) {
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
}

// UI Elements
const imageInput = document.getElementById('imageInput') || document.querySelector('input[type="file"]');
const canvas = document.getElementById('outputCanvas') || document.querySelector('canvas');
const dropZone = document.getElementById('dropZone') || document.querySelector('.upload-box') || document.body;
const statusBadge = document.querySelector('.status-badge') || document.getElementById('statusBadge');
const resultsContainer = document.getElementById('resultsContainer');

// --- INITIALIZE MODEL ---
async function initModel() {
  try {
    updateStatus('Loading AI Model...', 'loading');
    console.log('Loading ONNX model...');
    
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
  if (!statusBadge) return;
  statusBadge.innerText = text;
  
  if (state === 'ready') {
    statusBadge.style.color = '#10B981';
    statusBadge.style.borderColor = '#10B981';
  } else if (state === 'error') {
    statusBadge.style.color = '#EF4444';
    statusBadge.style.borderColor = '#EF4444';
  } else {
    statusBadge.style.color = '#F59E0B';
    statusBadge.style.borderColor = '#F59E0B';
  }
}

// Start model initialization on load
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

// --- MAIN IMAGE PIPELINE ---
async function processImage(file) {
  if (!session) {
    alert('Model is still loading or failed to load. Please refresh the page.');
    return;
  }

  const img = new Image();
  img.src = URL.createObjectURL(file);

  img.onload = async () => {
    if (canvas) {
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
    }

    // 1. Preprocess Image to Tensor
    const [tensor, scale, padX, padY] = preprocess(img);

    // 2. Run Inference
    const feeds = {};
    feeds[session.inputNames[0]] = tensor;
    const results = await session.run(feeds);
    const outputTensor = results[session.outputNames[0]];

    // 3. Postprocess Output
    const detections = postprocess(
      outputTensor, 
      scale, 
      padX, 
      padY, 
      img.width, 
      img.height
    );

    // 4. Render Detections
    renderDetections(img, detections);
  };
}

// --- PREPROCESSING ---
function preprocess(img) {
  const canvasPre = document.createElement('canvas');
  canvasPre.width = INPUT_WIDTH;
  canvasPre.height = INPUT_HEIGHT;
  const ctxPre = canvasPre.getContext('2d');

  // Letterbox scaling
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

    // Confidence filter
    if (maxClassScore < CONF_THRESHOLD) continue;

    // Dimensions
    const cx = rawData[0 * numBoxes + i];
    const cy = rawData[1 * numBoxes + i];
    const w = rawData[2 * numBoxes + i];
    const h = rawData[3 * numBoxes + i];

    // Scale back to original coordinates
    let x1 = (cx - w / 2 - padX) / scale;
    let y1 = (cy - h / 2 - padY) / scale;
    let boxW = w / scale;
    let boxH = h / scale;

    x1 = Math.max(0, Math.min(x1, origW));
    y1 = Math.max(0, Math.min(y1, origH));
    boxW = Math.min(boxW, origW - x1);
    boxH = Math.min(boxH, origH - y1);

    // Aspect Ratio Filter (Long side / Short side)
    const longSide = Math.max(boxW, boxH);
    const shortSide = Math.min(boxW, boxH);
    const aspectRatio = longSide / Math.max(shortSide, 1e-6);

    if (aspectRatio < MIN_ASPECT_RATIO || aspectRatio > MAX_ASPECT_RATIO) {
      continue; // Filter circular icons/badges or irregular shapes
    }

    // Minimum Area Filter
    const boxAreaRatio = (boxW * boxH) / (origW * origH);
    if (boxAreaRatio < MIN_AREA_RATIO) {
      continue;
    }

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
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);

  if (resultsContainer) {
    resultsContainer.innerHTML = '';
  }

  if (detections.length === 0) {
    if (resultsContainer) {
      resultsContainer.innerHTML = '<p class="text-gray-400">No valid currency detected.</p>';
    }
    return;
  }

  detections.forEach((det) => {
    const [x, y, w, h] = det.box;
    const labelText = `₹${det.label.replace('_', ' ')} (${(det.score * 100).toFixed(1)}%)`;

    ctx.strokeStyle = '#10B981';
    ctx.lineWidth = Math.max(2, Math.round(canvas.width / 300));
    ctx.strokeRect(x, y, w, h);

    ctx.font = '16px Inter, sans-serif';
    const textWidth = ctx.measureText(labelText).width;
    ctx.fillStyle = '#10B981';
    ctx.fillRect(x, y > 25 ? y - 25 : y, textWidth + 10, 25);

    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(labelText, x + 5, y > 25 ? y - 7 : y + 18);

    if (resultsContainer) {
      const badge = document.createElement('div');
      badge.className = 'inline-block bg-emerald-900/40 border border-emerald-500/50 text-emerald-400 px-3 py-1 rounded-lg text-sm font-medium mr-2 mb-2';
      badge.innerText = labelText;
      resultsContainer.appendChild(badge);
    }
  });
}
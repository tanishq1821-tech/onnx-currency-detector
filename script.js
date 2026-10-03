let session;

const classNames = [
    '10_New',   // 0
    '10_Old',   // 1
    '20',       // 2
    '50_New',   // 3
    '50_Old',   // 4
    '100_New',  // 5
    '100_Old',  // 6
    '200',      // 7
    '500'       // 8
];

const dropZone = document.getElementById('dropZone');
const uploadInput = document.getElementById('uploadInput');
const modelStatus = document.getElementById('modelStatus');
const statusText = document.getElementById('statusText');
const resultText = document.getElementById('resultText');
const detectionCount = document.getElementById('detectionCount');
const resetBtn = document.getElementById('resetBtn');
const canvas = document.getElementById('canvas');
const emptyState = document.getElementById('emptyState');

// Initialize ONNX Runtime Session
async function initModel() {
    try {
        session = await ort.InferenceSession.create('./best.onnx', { executionProviders: ['wasm'] });
        
        // Update UI status to ready
        modelStatus.querySelector('.dot').className = "dot green";
        statusText.innerText = "Model Ready (Client-Side WASM)";
    } catch (e) {
        console.error("Failed to load ONNX model:", e);
        statusText.innerText = "Model Load Failed";
    }
}
initModel();

// File Input & Drag and Drop Handlers
dropZone.addEventListener('click', () => uploadInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

uploadInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

resetBtn.addEventListener('click', resetApp);

function handleFile(file) {
    if (!file || !session) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
        emptyState.style.display = 'none';
        canvas.style.display = 'block';

        const ctx = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        const tensor = preprocessImage(img);
        const feeds = { images: tensor };
        const results = await session.run(feeds);
        const outputTensor = results[Object.keys(results)[0]];

        const boxes = processOutput(outputTensor.data, img.width, img.height);
        drawBoxes(ctx, boxes);
    };
}

function preprocessImage(img) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 640;
    tempCanvas.height = 640;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 640, 640);

    const imgData = ctx.getImageData(0, 0, 640, 640).data;
    const float32Data = new Float32Array(1 * 3 * 640 * 640);

    for (let i = 0; i < 640 * 640; i++) {
        float32Data[i] = imgData[i * 4] / 255.0;
        float32Data[i + 640 * 640] = imgData[i * 4 + 1] / 255.0;
        float32Data[i + 2 * 640 * 640] = imgData[i * 4 + 2] / 255.0;
    }

    return new ort.Tensor('float32', float32Data, [1, 3, 640, 640]);
}

function processOutput(output, imgWidth, imgHeight) {
    const numClasses = classNames.length;
    const numAnchors = 8400;
    const boxes = [];
    const confThreshold = 0.70;

    for (let i = 0; i < numAnchors; i++) {
        let maxScore = -Infinity;
        let classId = -1;

        for (let c = 0; c < numClasses; c++) {
            const score = output[(4 + c) * numAnchors + i];
            if (score > maxScore) {
                maxScore = score;
                classId = c;
            }
        }

        if (maxScore >= confThreshold) {
            const cx = output[0 * numAnchors + i];
            const cy = output[1 * numAnchors + i];
            const w = output[2 * numAnchors + i];
            const h = output[3 * numAnchors + i];

            const x1 = ((cx - w / 2) / 640) * imgWidth;
            const y1 = ((cy - h / 2) / 640) * imgHeight;
            const boxW = (w / 640) * imgWidth;
            const boxH = (h / 640) * imgHeight;
            const aspectRatio = boxW / boxH;

        // Ensure detected box meets minimum dimensions and currency-like aspect ratio
        if (boxW > 60 && boxH > 40 && (aspectRatio >= 1.2 || aspectRatio <= 0.83)) {
            boxes.push({
                x: x1,
                y: y1,
                w: boxW,
                h: boxH,
                classId: classId,
                className: classNames[classId],
                score: maxScore
            });
        }

            
            
        }
    }

    return applyNMS(boxes, 0.45);
}

function applyNMS(boxes, iouThreshold) {
    boxes.sort((a, b) => b.score - a.score);
    const selected = [];

    while (boxes.length > 0) {
        const current = boxes.shift();
        selected.push(current);

        boxes = boxes.filter(box => calculateIoU(current, box) < iouThreshold);
    }

    return selected;
}

function calculateIoU(boxA, boxB) {
    const xA = Math.max(boxA.x, boxB.x);
    const yA = Math.max(boxA.y, boxB.y);
    const xB = Math.min(boxA.x + boxA.w, boxB.x + boxB.w);
    const yB = Math.min(boxA.y + boxA.h, boxB.y + boxB.h);

    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    return interArea / (boxA.w * boxA.h + boxB.w * boxB.h - interArea);
}

function drawBoxes(ctx, boxes) {
    ctx.lineWidth = 4;
    ctx.font = "bold 20px 'Plus Jakarta Sans', sans-serif";

    detectionCount.innerText = `${boxes.length} Notes`;

    if (boxes.length === 0) {
        resultText.innerHTML = '<span class="placeholder-text">No Indian banknote detected in image.</span>';
        return;
    }

    // Render results UI badges
    resultText.innerHTML = boxes.map(b => `
        <div class="detection-tag">
            ₹${b.className.replace('_', ' ')} (${(b.score * 100).toFixed(1)}%)
        </div>
    `).join('');

    // Draw on Canvas
    boxes.forEach(box => {
        ctx.strokeStyle = "#10B981";
        ctx.fillStyle = "#10B981";
        ctx.strokeRect(box.x, box.y, box.w, box.h);

        const label = `₹${box.className.replace('_', ' ')} ${(box.score * 100).toFixed(0)}%`;
        const textWidth = ctx.measureText(label).width;
        
        ctx.fillRect(box.x, box.y > 30 ? box.y - 30 : box.y, textWidth + 16, 30);
        ctx.fillStyle = "#000000";
        ctx.fillText(label, box.x + 8, box.y > 30 ? box.y - 8 : box.y + 22);
    });
}

function resetApp() {
    uploadInput.value = "";
    emptyState.style.display = 'flex';
    canvas.style.display = 'none';
    resultText.innerHTML = '<span class="placeholder-text">Upload an image to perform real-time detection.</span>';
    detectionCount.innerText = "0 Notes";
}
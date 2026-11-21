/**
 * ===========================================================================
 * Handwritten Multi-Digit Recognizer (Fast Training + Multi-Digit Support)
 * ===========================================================================
 */

// 모델 구조가 변경되어 v2로 키 변경 (최초 1회 빠른 학습 후 브라우저에 저장됨)
const MODEL_STORAGE_KEY = 'indexeddb://digit-recognizer-mnist-cnn-v2';

// ---------- Training HYPERPARAMETERS (속도 최적화) ----------
const EPOCHS = 1;        // 2 -> 1 (학습 시간 절반 단축)
const BATCH_SIZE = 512;  // 256 -> 512 (배치 업데이트 횟수 단축)
const BRUSH_WIDTH = 14;  // 22 -> 14 (여러 자릿수를 겹치지 않게 쓰기 위해 얇게 조정)

// ---------- DOM elements ----------
const statusEl = document.getElementById('status');
const progressBarEl = document.getElementById('progress-bar');
const drawCanvas = document.getElementById('draw-canvas');
const drawCtx = drawCanvas.getContext('2d', { willReadFrequently: true });
const previewCanvas = document.getElementById('preview-canvas');
const previewCtx = previewCanvas.getContext('2d');
const predictionEl = document.getElementById('prediction');
const confidenceEl = document.getElementById('confidence');
const barsEl = document.getElementById('bars');
const clearBtn = document.getElementById('clear-btn');
const retrainBtn = document.getElementById('retrain-btn');

let model = null;

// =====================================================================
// Status helpers
// =====================================================================

function setStatus(message) {
  statusEl.textContent = message;
}

function setProgress(fraction) {
  progressBarEl.style.width = `${Math.round(fraction * 100)}%`;
}

// =====================================================================
// Model design (경량화 CNN: 파라미터 수를 대폭 줄여 학습 속도 향상)
// =====================================================================

function createModel() {
  const m = tf.sequential();

  m.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 16,
    kernelSize: 3,
    activation: 'relu',
  }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  m.add(tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
  }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  m.add(tf.layers.flatten());
  m.add(tf.layers.dropout({ rate: 0.2 }));
  m.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  m.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  m.compile({
    optimizer: tf.train.adam(0.002), // 학습률을 살짝 높여 1 epoch 만에 빠르게 수렴
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return m;
}

// =====================================================================
// Train, evaluate, save
// =====================================================================

async function trainNewModel() {
  setProgress(0);
  setStatus('Downloading MNIST dataset (about 11 MB)...');
  const data = new MnistData();
  await data.load();

  const newModel = createModel();
  const train = data.getTrainData();
  const test = data.getTestData();

  const batchesPerEpoch = Math.ceil(train.xs.shape[0] / BATCH_SIZE);
  const totalBatches = batchesPerEpoch * EPOCHS;
  let batchesDone = 0;
  let currentEpoch = 1;

  setStatus(`Training model (epoch 1 of ${EPOCHS})...`);
  await newModel.fit(train.xs, train.ys, {
    batchSize: BATCH_SIZE,
    epochs: EPOCHS,
    shuffle: true,
    callbacks: {
      onEpochBegin: async (epoch) => {
        currentEpoch = epoch + 1;
      },
      onBatchEnd: async (batch, logs) => {
        batchesDone++;
        setProgress(batchesDone / totalBatches);
        if (batch % 5 === 0) {
          const accuracy = logs.acc ?? logs.accuracy ?? 0;
          setStatus(
            `Training model (epoch ${currentEpoch} of ${EPOCHS}) - ` +
            `batch accuracy ${(accuracy * 100).toFixed(1)}%`
          );
        }
      },
    },
  });

  setStatus('Evaluating on test images...');
  const [lossTensor, accTensor] = newModel.evaluate(test.xs, test.ys, { batchSize: 1000 });
  const testAccuracy = (await accTensor.data())[0];

  tf.dispose([train.xs, train.ys, test.xs, test.ys, lossTensor, accTensor]);

  let saveNote = '';
  try {
    await newModel.save(MODEL_STORAGE_KEY);
  } catch (err) {
    saveNote = ' (Could not save the model; it will retrain next time.)';
  }

  setProgress(1);
  setStatus(
    `Model ready. Test accuracy: ${(testAccuracy * 100).toFixed(2)}%. ` +
    `Draw one or more digits!${saveNote}`
  );
  return newModel;
}

async function loadOrTrainModel() {
  try {
    const saved = await tf.loadLayersModel(MODEL_STORAGE_KEY);
    setProgress(1);
    setStatus('Loaded saved model from the browser. Draw one or more digits!');
    return saved;
  } catch (err) {
    return trainNewModel();
  }
}

// =====================================================================
// Drawing on the canvas
// =====================================================================

let isDrawing = false;
let lastPoint = null;

function clearCanvas() {
  drawCtx.fillStyle = '#000000';
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  previewCtx.fillStyle = '#000000';
  previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
  predictionEl.textContent = '?';
  confidenceEl.innerHTML = '&nbsp;';
  renderBars(null);
}

function getCanvasPoint(event) {
  const rect = drawCanvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * drawCanvas.width,
    y: ((event.clientY - rect.top) / rect.height) * drawCanvas.height,
  };
}

function drawLine(from, to) {
  drawCtx.strokeStyle = '#ffffff';
  drawCtx.lineWidth = BRUSH_WIDTH;
  drawCtx.lineCap = 'round';
  drawCtx.lineJoin = 'round';
  drawCtx.beginPath();
  drawCtx.moveTo(from.x, from.y);
  drawCtx.lineTo(to.x, to.y);
  drawCtx.stroke();
}

drawCanvas.addEventListener('pointerdown', (event) => {
  isDrawing = true;
  drawCanvas.setPointerCapture(event.pointerId);
  lastPoint = getCanvasPoint(event);
  drawLine(lastPoint, lastPoint);
});

drawCanvas.addEventListener('pointermove', (event) => {
  if (!isDrawing) return;
  const point = getCanvasPoint(event);
  drawLine(lastPoint, point);
  lastPoint = point;
});

function stopDrawing() {
  if (!isDrawing) return;
  isDrawing = false;
  predict();
}

drawCanvas.addEventListener('pointerup', stopDrawing);
drawCanvas.addEventListener('pointercancel', stopDrawing);

// =====================================================================
// Preprocessing: 여러 자릿수 자동 분리 (수직 투영 방식)
// =====================================================================

/**
 * 캔버스에서 왼쪽->오른쪽 순서로 각 숫자의 영역(Bounding Box)을 분리합니다.
 */
function findDigitBoxes() {
  const { width, height } = drawCanvas;
  const pixels = drawCtx.getImageData(0, 0, width, height).data;

  // 각 x열(세로줄)에 흰색 픽셀이 있는지 검사
  const colHasInk = new Uint8Array(width);
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      if (pixels[(y * width + x) * 4] > 20) {
        colHasInk[x] = 1;
        break;
      }
    }
  }

  // 연속된 x 구간들을 찾아 각각의 숫자 박스로 분리
  const boxes = [];
  let inDigit = false;
  let startX = 0;

  for (let x = 0; x < width; x++) {
    if (colHasInk[x] && !inDigit) {
      inDigit = true;
      startX = x;
    } else if (!colHasInk[x] && inDigit) {
      inDigit = false;
      boxes.push({ minX: startX, maxX: x - 1 });
    }
  }
  if (inDigit) {
    boxes.push({ minX: startX, maxX: width - 1 });
  }

  // 각 x 구간 내에서 실제 y 범위(minY, maxY) 계산
  const result = [];
  for (const box of boxes) {
    let minY = height, maxY = -1;
    for (let y = 0; y < height; y++) {
      for (let x = box.minX; x <= box.maxX; x++) {
        if (pixels[(y * width + x) * 4] > 20) {
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxY >= 0) {
      result.push({ minX: box.minX, maxX: box.maxX, minY, maxY });
    }
  }

  return result;
}

/**
 * 분리된 숫자 영역 1개를 28x28 MNIST 형식의 Float32Array(784)로 변환합니다.
 */
function preprocessBox(box, drawToPreview = false) {
  const boxW = box.maxX - box.minX + 1;
  const boxH = box.maxY - box.minY + 1;
  const scale = 20 / Math.max(boxW, boxH);
  const targetW = boxW * scale;
  const targetH = boxH * scale;

  const scaled = document.createElement('canvas');
  scaled.width = 28;
  scaled.height = 28;
  const scaledCtx = scaled.getContext('2d', { willReadFrequently: true });
  scaledCtx.fillStyle = '#000000';
  scaledCtx.fillRect(0, 0, 28, 28);
  scaledCtx.imageSmoothingEnabled = true;
  scaledCtx.imageSmoothingQuality = 'high';
  scaledCtx.drawImage(
    drawCanvas,
    box.minX, box.minY, boxW, boxH,
    (28 - targetW) / 2, (28 - targetH) / 2, targetW, targetH
  );

  // Center of mass 계산
  const small = scaledCtx.getImageData(0, 0, 28, 28).data;
  let total = 0, sumX = 0, sumY = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const value = small[(y * 28 + x) * 4];
      total += value;
      sumX += x * value;
      sumY += y * value;
    }
  }
  const shiftX = total > 0 ? Math.round(14 - sumX / total) : 0;
  const shiftY = total > 0 ? Math.round(14 - sumY / total) : 0;

  const centered = document.createElement('canvas');
  centered.width = 28;
  centered.height = 28;
  const centeredCtx = centered.getContext('2d', { willReadFrequently: true });
  centeredCtx.fillStyle = '#000000';
  centeredCtx.fillRect(0, 0, 28, 28);
  centeredCtx.drawImage(scaled, shiftX, shiftY);

  if (drawToPreview) {
    previewCtx.fillStyle = '#000000';
    previewCtx.fillRect(0, 0, 28, 28);
    previewCtx.drawImage(centered, 0, 0);
  }

  const finalPixels = centeredCtx.getImageData(0, 0, 28, 28).data;
  const input = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    input[i] = finalPixels[i * 4] / 255;
  }
  return input;
}

// =====================================================================
// Prediction (다중 자릿수 일괄 예측)
// =====================================================================

function predict() {
  const boxes = findDigitBoxes();
  if (boxes.length === 0) return;

  if (!model) {
    confidenceEl.textContent = 'The model is not ready yet. Please wait.';
    return;
  }

  // 모든 분리된 숫자를 하나의 배치 텐서 [N, 28, 28, 1]로 묶어 한 번에 예측
  const count = boxes.length;
  const batchData = new Float32Array(count * 784);
  for (let i = 0; i < count; i++) {
    // 마지막 숫자를 preview 캔버스에 표시
    const input = preprocessBox(boxes[i], i === count - 1);
    batchData.set(input, i * 784);
  }

  const allProbs = tf.tidy(() => {
    const x = tf.tensor4d(batchData, [count, 28, 28, 1]);
    return model.predict(x).dataSync();
  });

  let resultStr = '';
  let totalConf = 0;
  let lastProbs = null;
  let lastBest = 0;

  for (let i = 0; i < count; i++) {
    const probs = allProbs.slice(i * 10, (i + 1) * 10);
    let best = 0;
    for (let digit = 1; digit < 10; digit++) {
      if (probs[digit] > probs[best]) best = digit;
    }
    resultStr += best;
    totalConf += probs[best];

    if (i === count - 1) {
      lastProbs = probs;
      lastBest = best;
    }
  }

  const avgConf = (totalConf / count) * 100;
  predictionEl.textContent = resultStr;
  confidenceEl.textContent =
    count === 1
      ? `Confidence: ${avgConf.toFixed(1)}%`
      : `Avg Confidence (${count} digits): ${avgConf.toFixed(1)}%`;

  // 확률 막대는 가장 마지막(오른쪽) 숫자의 확률을 표시
  renderBars(lastProbs, lastBest);
}

function renderBars(probabilities, best) {
  barsEl.innerHTML = '';
  for (let digit = 0; digit < 10; digit++) {
    const p = probabilities ? probabilities[digit] : 0;

    const row = document.createElement('div');
    row.className = 'bar-row' + (digit === best ? ' top' : '');
    row.innerHTML = `
      <span class="bar-label">${digit}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${p * 100}%"></div></div>
      <span class="bar-value">${(p * 100).toFixed(1)}%</span>
    `;
    barsEl.appendChild(row);
  }
}

// =====================================================================
// Buttons and startup
// =====================================================================

clearBtn.addEventListener('click', clearCanvas);

retrainBtn.addEventListener('click', async () => {
  retrainBtn.disabled = true;
  try {
    const oldModel = model;
    model = null;
    if (oldModel) oldModel.dispose();
    model = await trainNewModel();
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    retrainBtn.disabled = false;
  }
});

async function main() {
  clearCanvas();
  if (typeof tf === 'undefined') {
    setStatus('Error: TensorFlow.js could not be loaded. Check your internet connection.');
    return;
  }

  retrainBtn.disabled = true;
  try {
    model = await loadOrTrainModel();
    predict();
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    retrainBtn.disabled = false;
  }
}

main();
